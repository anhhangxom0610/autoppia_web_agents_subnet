# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): Daryxx
# Copyright © 2023 Autoppia

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt
from typing import List, Union, Dict, Optional
from traceback import print_exception
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from autoppia_web_agents_subnet.base.neuron import BaseNeuron
from autoppia_web_agents_subnet.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from autoppia_web_agents_subnet.utils.config import add_validator_args


class BaseValidatorNeuron(BaseNeuron):
    """
    Optimized base class for Bittensor validators with improved performance and async operations.
    """

    neuron_type: str = "ValidatorNeuron"
    _score_cache: Dict[int, float] = {}
    _weight_cache: Dict[str, np.ndarray] = {}
    _executor = ThreadPoolExecutor(max_workers=4)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Initialize with optimized data structures
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        
        # Initialize async components
        self.loop = asyncio.get_event_loop()
        self.lock = asyncio.Lock()
        self._forward_semaphore = asyncio.Semaphore(self.config.neuron.num_concurrent_forwards)
        
        # State management
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        
        # Initialize network components
        self._initialize_network()
        
        # Initial sync
        asyncio.create_task(self.sync())

    def _initialize_network(self):
        """Initialize network components with error handling."""
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("Axon disabled, not serving IP to chain.")

    def serve_axon(self):
        """Serve axon with improved error handling and retry logic."""
        bt.logging.info("Serving IP to chain...")
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                self.axon = bt.axon(wallet=self.wallet, config=self.config)
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} "
                    f"with netuid: {self.config.netuid}"
                )
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    bt.logging.error(f"Failed to serve axon after {max_retries} attempts: {e}")
                    raise
                bt.logging.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)

    async def concurrent_forward(self):
        """Execute multiple forwards concurrently with rate limiting."""
        async with self._forward_semaphore:
            coroutines = [
                self.forward()
                for _ in range(self.config.neuron.num_concurrent_forwards)
            ]
            return await asyncio.gather(*coroutines, return_exceptions=True)

    async def run(self):
        """Asynchronous main loop with improved error handling."""
        try:
            await self.sync()
            bt.logging.info(f"Validator starting at block: {self.block}")

            while not self.should_exit:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run forwards with error handling
                try:
                    results = await self.concurrent_forward()
                    for result in results:
                        if isinstance(result, Exception):
                            bt.logging.error(f"Forward failed: {result}")
                except Exception as e:
                    bt.logging.error(f"Concurrent forward failed: {e}")

                # Sync with error handling
                try:
                    await self.sync()
                except Exception as e:
                    bt.logging.error(f"Sync failed: {e}")

                self.step += 1
                await asyncio.sleep(1)  # Prevent tight loop

        except KeyboardInterrupt:
            await self.cleanup()
            bt.logging.success("Validator stopped by keyboard interrupt.")
        except Exception as e:
            bt.logging.error(f"Validator error: {e}")
            bt.logging.debug(str(print_exception(type(e), e, e.__traceback__)))
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'axon'):
            self.axon.stop()
        self.should_exit = True
        if self.is_running:
            self.is_running = False

    def run_in_background_thread(self):
        """Start validator in background with improved thread management."""
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(
                target=lambda: asyncio.run(self.run()),
                daemon=True
            )
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    async def set_weights(self):
        """Optimized weight setting with caching and error handling."""
        try:
            # Check for NaN values
            if np.isnan(self.scores).any():
                bt.logging.warning("Scores contain NaN values. This may indicate issues with reward functions.")
                self.scores = np.nan_to_num(self.scores)

            # Compute weights with caching
            cache_key = hash(self.scores.tobytes())
            if cache_key in self._weight_cache:
                return self._weight_cache[cache_key]

            # Compute norm safely
            norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)
            if np.any(norm == 0) or np.isnan(norm).any():
                norm = np.ones_like(norm)

            # Compute and process weights
            raw_weights = self.scores / norm
            processed_weight_uids, processed_weights = process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=raw_weights,
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )

            # Cache results
            self._weight_cache[cache_key] = (processed_weight_uids, processed_weights)
            
            # Convert and emit weights
            uint_uids, uint_weights = convert_weights_and_uids_for_emit(
                uids=processed_weight_uids,
                weights=processed_weights
            )
            
            # Emit weights asynchronously
            await asyncio.to_thread(
                self.subtensor.set_weights,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_inclusion=False
            )

        except Exception as e:
            bt.logging.error(f"Failed to set weights: {e}")
            raise

    @lru_cache(maxsize=1)
    def resync_metagraph(self):
        """Cached metagraph resync with error handling."""
        try:
            self.metagraph.sync()
            self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        except Exception as e:
            bt.logging.error(f"Failed to resync metagraph: {e}")
            raise

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Update scores with improved error handling and validation."""
        try:
            if not isinstance(rewards, np.ndarray):
                rewards = np.array(rewards, dtype=np.float32)
            
            if not isinstance(uids, (list, np.ndarray)):
                uids = np.array(uids, dtype=np.int64)
            
            # Validate inputs
            if len(rewards) != len(uids):
                raise ValueError("Rewards and UIDs must have the same length")
            
            # Update scores with bounds checking
            for reward, uid in zip(rewards, uids):
                if 0 <= uid < len(self.scores):
                    self.scores[uid] = reward
                else:
                    bt.logging.warning(f"Invalid UID {uid} received")
                    
        except Exception as e:
            bt.logging.error(f"Failed to update scores: {e}")
            raise

    async def save_state(self):
        """Save validator state with error handling."""
        try:
            async with self.lock:
                # Implement state saving logic here
                pass
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")

    async def load_state(self):
        """Load validator state with error handling."""
        try:
            async with self.lock:
                # Implement state loading logic here
                pass
        except Exception as e:
            bt.logging.error(f"Failed to load state: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.run_in_background_thread()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
        if self.thread:
            self.thread.join(timeout=5)
