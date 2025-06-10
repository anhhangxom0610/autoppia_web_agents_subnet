# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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

import time
import asyncio
import threading
import argparse
import traceback
from typing import Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import bittensor as bt

from autoppia_web_agents_subnet.base.neuron import BaseNeuron
from autoppia_web_agents_subnet.utils.config import add_miner_args


class BaseMinerNeuron(BaseNeuron):
    """
    Optimized base class for Bittensor miners with improved performance and async operations.
    """

    neuron_type: str = "MinerNeuron"
    _request_cache: Dict[str, Any] = {}
    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Security checks
        self._check_security_config()

        # Initialize axon with retry mechanism
        self._initialize_axon()

        # State management
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = asyncio.Lock()
        self._last_sync_time = 0
        self._sync_interval = 60  # seconds

    def _check_security_config(self):
        """Check security configuration and log warnings."""
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "Security risk: Allowing non-validators to send requests to your miner."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "Security risk: Allowing non-registered entities to send requests to your miner."
            )

    def _initialize_axon(self, max_retries=3, retry_delay=5):
        """Initialize axon with retry mechanism."""
        for attempt in range(max_retries):
            try:
                self.axon = bt.axon(
                    wallet=self.wallet,
                    config=self.config() if callable(self.config) else self.config,
                )

                # Attach handlers with error checking
                self._attach_axon_handlers()
                
                bt.logging.info(f"Axon created: {self.axon}")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    bt.logging.error(f"Failed to initialize axon after {max_retries} attempts: {e}")
                    raise
                bt.logging.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)

    def _attach_axon_handlers(self):
        """Attach handlers to axon with error checking."""
        handlers = [
            {
                'forward_fn': self.forward,
                'blacklist_fn': self.blacklist,
                'priority_fn': self.priority
            },
            {
                'forward_fn': self.forward_feedback,
                'blacklist_fn': self.blacklist_feedback,
                'priority_fn': self.priority_feedback
            },
            {
                'forward_fn': self.forward_set_organic_endpoint,
                'blacklist_fn': self.blacklist_set_organic_endpoint,
                'priority_fn': self.priority_set_organic_endpoint
            }
        ]

        for handler in handlers:
            try:
                self.axon.attach(**handler)
            except Exception as e:
                bt.logging.error(f"Failed to attach handler: {e}")
                raise

    async def run(self):
        """Asynchronous main loop with improved error handling."""
        try:
            await self.sync()
            
            # Start axon with error handling
            await self._start_axon()
            
            bt.logging.info(f"Miner starting at block: {self.block}")

            while not self.should_exit:
                # Check if we need to sync
                if self._should_sync():
                    try:
                        await self.sync()
                        self.step += 1
                    except Exception as e:
                        bt.logging.error(f"Sync failed: {e}")

                # Prevent tight loop
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            await self.cleanup()
            bt.logging.success("Miner stopped by keyboard interrupt.")
        except Exception as e:
            bt.logging.error(f"Miner error: {e}")
            bt.logging.debug(str(traceback.format_exc()))
            await self.cleanup()

    async def _start_axon(self):
        """Start axon with error handling."""
        try:
            bt.logging.info(
                f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} "
                f"with netuid: {self.config.netuid}"
            )
            await asyncio.to_thread(
                self.axon.serve,
                netuid=self.config.netuid,
                subtensor=self.subtensor
            )
            await asyncio.to_thread(self.axon.start)
        except Exception as e:
            bt.logging.error(f"Failed to start axon: {e}")
            raise

    def _should_sync(self) -> bool:
        """Check if sync is needed with caching."""
        current_time = time.time()
        if current_time - self._last_sync_time < self._sync_interval:
            return False
            
        should_sync = (
            self.block - self.metagraph.last_update[self.uid]
            >= self.config.neuron.epoch_length
        )
        
        if should_sync:
            self._last_sync_time = current_time
            
        return should_sync

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'axon'):
            await asyncio.to_thread(self.axon.stop)
        self.should_exit = True
        if self.is_running:
            self.is_running = False

    def run_in_background_thread(self):
        """Start miner in background with improved thread management."""
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(
                target=lambda: asyncio.run(self.run()),
                daemon=True
            )
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    @lru_cache(maxsize=1)
    def resync_metagraph(self):
        """Cached metagraph resync with error handling."""
        try:
            bt.logging.info("Resyncing metagraph...")
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception as e:
            bt.logging.error(f"Failed to resync metagraph: {e}")
            raise

    def set_weights(self):
        """Empty implementation of set_weights for miners."""
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    async def __aenter__(self):
        """Async context manager entry."""
        self.run_in_background_thread()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
        if self.thread:
            self.thread.join(timeout=5)
