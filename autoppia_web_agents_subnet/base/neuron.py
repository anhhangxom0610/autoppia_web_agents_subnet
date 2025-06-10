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

import copy
import bittensor as bt
from abc import ABC, abstractmethod
import time
import traceback
import requests
import re
import asyncio
from functools import lru_cache
from typing import Optional, Dict, Any
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from autoppia_web_agents_subnet.utils.config import check_config, add_args, config
from autoppia_web_agents_subnet.utils.misc import ttl_get_block
from autoppia_web_agents_subnet import version_url
from autoppia_web_agents_subnet import __version__, __least_acceptable_version__, __spec_version__

class BaseNeuron(ABC):
    """
    Optimized base class for Bittensor neurons with improved performance and error handling.
    """
    neuron_type: str = "BaseNeuron"
    _version_cache: Dict[str, str] = {}
    _session: Optional[aiohttp.ClientSession] = None
    _executor = ThreadPoolExecutor(max_workers=4)

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = __spec_version__

    @property
    @lru_cache(maxsize=1)
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # Initialize version information
        self.version = __version__
        self.least_acceptable_version = __least_acceptable_version__

        # Set up logging with the provided configuration
        bt.logging.set_config(config=self.config.logging)
        self.device = self.config.neuron.device
        bt.logging.info(self.config)

        # Initialize Bittensor objects with retry mechanism
        self._initialize_bittensor_objects()

        # Initialize state
        self.step = 0
        self.last_update = 0
        self._last_sync_time = 0
        self._sync_interval = 60  # seconds

    def _initialize_bittensor_objects(self, max_retries=5, retry_delay=5):
        """Initialize Bittensor objects with retry mechanism."""
        self.wallet = bt.wallet(config=self.config)
        
        for attempt in range(max_retries):
            try:
                bt.logging.info("Initializing subtensor and metagraph")
                self.subtensor = bt.subtensor(config=self.config)
                self.metagraph = self.subtensor.metagraph(self.config.netuid)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    bt.logging.error(f"Failed to initialize after {max_retries} attempts: {e}")
                    raise
                bt.logging.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)

        self._verify_registration()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

    def _verify_registration(self):
        """Verify registration with improved error handling."""
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            error_msg = (
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}. "
                "Please register the hotkey using `btcli subnets register` before trying again"
            )
            bt.logging.error(error_msg)
            raise RuntimeError(error_msg)

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def resync_metagraph(self): ...

    @abstractmethod
    def set_weights(self): ...

    async def sync(self):
        """Optimized sync method with rate limiting and error handling."""
        current_time = time.time()
        if current_time - self._last_sync_time < self._sync_interval:
            return

        try:
            self._verify_registration()

            if self.should_sync_metagraph():
                self.last_update = self.block
                await asyncio.to_thread(self.resync_metagraph)

            if self.should_set_weights():
                await asyncio.to_thread(self.set_weights)

            self.save_state()
            self._last_sync_time = current_time

        except Exception as e:
            bt.logging.error(f"Sync failed: {traceback.format_exc()}")
            if "public RPC endpoint" in str(e).lower():
                bt.logging.error("Consider using a local node for better performance")
            await asyncio.sleep(5)

    def should_sync_metagraph(self) -> bool:
        """Check if metagraph sync is needed with caching."""
        last_update = (
            self.last_update if self.neuron_type == "MinerNeuron"
            else self.metagraph.last_update[self.uid]
        )
        return (self.block - last_update) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        """Check if weights should be set with improved logic."""
        if self.step == 0 or self.config.neuron.disable_set_weights:
            return False

        return (
            (self.block - self.metagraph.last_update[self.uid]) > self.config.neuron.epoch_length
            and self.neuron_type != "MinerNeuron"
        )

    async def parse_versions(self):
        """Asynchronously parse versions with caching."""
        if not self._version_cache:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(version_url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
                            least_acceptable_match = re.search(
                                r"__least_acceptable_version__\s*=\s*['\"]([^'\"]+)['\"]", 
                                content
                            )
                            
                            if version_match and least_acceptable_match:
                                self._version_cache = {
                                    'version': version_match.group(1),
                                    'least_acceptable': least_acceptable_match.group(1)
                                }
                                
                                self.version = self._version_cache['version']
                                self.least_acceptable_version = self._version_cache['least_acceptable']
                                
            except Exception as e:
                bt.logging.error(f"Version parsing failed: {e}")
                # Fall back to default versions
                self.version = __version__
                self.least_acceptable_version = __least_acceptable_version__

    def save_state(self):
        """Save neuron state with error handling."""
        try:
            bt.logging.trace("Saving neuron state...")
            # Implement state saving logic here
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load neuron state with error handling."""
        try:
            bt.logging.trace("Loading neuron state...")
            # Implement state loading logic here
        except Exception as e:
            bt.logging.error(f"Failed to load state: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if self._session:
            await self._session.close()
        self._executor.shutdown(wait=True)
