"""LocalBackend for single-machine GPU management."""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Set, Union

from gpudispatch.backends.base import Backend
from gpudispatch.core.resources import GPU, Memory
from gpudispatch.utils.gpu import detect_gpus, is_gpu_available

logger = logging.getLogger(__name__)


class LocalBackend(Backend):
    """Backend for managing GPUs on a single local machine.

    LocalBackend provides GPU allocation and management for single-machine
    workloads. It supports automatic GPU detection or explicit GPU lists,
    memory threshold-based availability checking, and thread-safe allocation.

    Example:
        >>> with LocalBackend(gpus="auto", memory_threshold="500MB") as backend:
        ...     gpus = backend.allocate_gpus(2)
        ...     # use GPUs...
        ...     backend.release_gpus(gpus)

    Args:
        gpus: GPU configuration. Either "auto" for automatic detection,
            or a list of GPU indices to manage. Default: "auto".
        memory_threshold: Consider a GPU "free" if its used memory is below
            this threshold. Accepts string like "500MB" or "2GB", or integer
            for megabytes. Default: "500MB".
        polling_interval: Seconds between availability checks when waiting
            for GPUs. Default: 5.
        process_mode: Execution mode for jobs. Either "subprocess" or "thread".
            Default: "subprocess".
    """

    def __init__(
        self,
        gpus: Union[str, List[int]] = "auto",
        memory_threshold: Union[str, int] = "500MB",
        polling_interval: int = 5,
        process_mode: str = "subprocess",
    ):
        # Validate process_mode
        if process_mode not in ("subprocess", "thread"):
            raise ValueError(
                f"Invalid process_mode '{process_mode}'. "
                "Must be 'subprocess' or 'thread'."
            )

        self._gpus_config = gpus
        self._process_mode = process_mode
        self._polling_interval = polling_interval

        # Parse memory threshold
        if isinstance(memory_threshold, str):
            self._memory_threshold_mb = Memory.from_string(memory_threshold).mb
        else:
            self._memory_threshold_mb = memory_threshold

        # Internal state
        self._gpu_pool: List[GPU] = []
        self._occupied_gpus: Set[int] = set()
        self._running = False
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "local"

    @property
    def is_running(self) -> bool:
        """Whether the backend is currently active."""
        return self._running

    def start(self) -> None:
        """Initialize and start the backend.

        Discovers GPUs (if gpus="auto") and initializes the GPU pool.
        """
        with self._lock:
            if self._running:
                return

            # Initialize GPU pool
            if self._gpus_config == "auto":
                detected = detect_gpus()
                self._gpu_pool = [
                    GPU(index=gpu.index, memory=gpu.memory_total_mb)
                    for gpu in detected
                ]
                logger.info(
                    f"Auto-detected {len(self._gpu_pool)} GPUs: "
                    f"{[g.index for g in self._gpu_pool]}"
                )
            else:
                # Explicit list of GPU indices
                self._gpu_pool = [GPU(index=idx) for idx in self._gpus_config]
                logger.info(
                    f"Using configured GPUs: {[g.index for g in self._gpu_pool]}"
                )

            self._occupied_gpus.clear()
            self._running = True

    def shutdown(self) -> None:
        """Gracefully shutdown the backend.

        Releases all allocated GPUs and stops the backend.
        """
        with self._lock:
            if not self._running:
                return

            # Release all occupied GPUs
            self._occupied_gpus.clear()
            self._running = False
            logger.info("LocalBackend shutdown complete")

    def allocate_gpus(
        self, count: int, memory: Optional[Memory] = None
    ) -> List[GPU]:
        """Allocate GPUs for a job.

        Allocates the requested number of GPUs that are:
        1. Not already allocated
        2. Have memory usage below the configured threshold

        Args:
            count: Number of GPUs to allocate.
            memory: Optional minimum memory requirement per GPU (stored
                with allocation but not validated against actual GPU memory).

        Returns:
            List of allocated GPU objects. Returns empty list if the
            requested number of available GPUs cannot be found.
        """
        with self._lock:
            # Find available GPUs
            available = self._get_available_gpu_indices()

            if len(available) < count:
                logger.debug(
                    f"Cannot allocate {count} GPUs, only {len(available)} available"
                )
                return []

            # Allocate first 'count' available GPUs
            to_allocate = available[:count]
            self._occupied_gpus.update(to_allocate)

            # Create GPU objects with optional memory requirement
            allocated = []
            for idx in to_allocate:
                gpu = GPU(index=idx, memory=memory.mb if memory else None)
                allocated.append(gpu)

            logger.debug(f"Allocated GPUs: {[g.index for g in allocated]}")
            return allocated

    def release_gpus(self, gpus: List[GPU]) -> None:
        """Release allocated GPUs.

        Released GPUs become available for allocation again.
        Releasing GPUs that were not allocated is a no-op.

        Args:
            gpus: List of GPU objects to release.
        """
        with self._lock:
            for gpu in gpus:
                if gpu.index in self._occupied_gpus:
                    self._occupied_gpus.discard(gpu.index)
                    logger.debug(f"Released GPU {gpu.index}")

    def list_available(self) -> List[GPU]:
        """List currently available GPUs.

        Returns GPUs that are:
        1. In the configured pool
        2. Not currently allocated
        3. Have memory usage below the configured threshold

        Returns:
            List of available GPU objects.
        """
        with self._lock:
            available_indices = self._get_available_gpu_indices()
            return [GPU(index=idx) for idx in available_indices]

    def health_check(self) -> bool:
        """Verify backend is operational.

        Returns:
            True if the backend is running, False otherwise.
        """
        return self._running

    def _get_available_gpu_indices(self) -> List[int]:
        """Get indices of available GPUs.

        A GPU is considered available if:
        1. It is in the GPU pool
        2. It is not currently occupied
        3. Its memory usage is below the threshold

        Returns:
            List of available GPU indices.
        """
        available = []
        for gpu in self._gpu_pool:
            # Skip if already occupied
            if gpu.index in self._occupied_gpus:
                continue

            # Check memory threshold
            if is_gpu_available(gpu.index, self._memory_threshold_mb):
                available.append(gpu.index)

        return available
