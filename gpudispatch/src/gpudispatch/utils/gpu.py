"""GPU detection and monitoring utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    import gpustat
except ImportError:
    gpustat = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: int

    @property
    def memory_free_mb(self) -> int:
        """Get free memory in MB."""
        return self.memory_total_mb - self.memory_used_mb


def detect_gpus() -> list[GPUInfo]:
    """Detect all available GPUs using gpustat.

    Returns:
        List of GPUInfo objects for each detected GPU.
        Empty list if no GPUs found or gpustat unavailable.
    """
    if gpustat is None:
        logger.warning("gpustat not installed. GPU detection unavailable.")
        return []

    try:
        stats = gpustat.GPUStatCollection.new_query()
        gpus = []

        for gpu in stats.gpus:
            gpus.append(GPUInfo(
                index=gpu.index,
                name=gpu.name,
                memory_total_mb=gpu.memory_total,
                memory_used_mb=gpu.memory_used,
                utilization_percent=gpu.utilization or 0,
            ))

        return gpus

    except Exception as e:
        logger.warning(f"Failed to detect GPUs: {e}")
        return []


def get_gpu_memory_usage(gpu_index: int) -> Optional[tuple[int, int]]:
    """Get memory usage for a specific GPU.

    Args:
        gpu_index: GPU index to query.

    Returns:
        Tuple of (used_mb, total_mb) or None if GPU not found.
    """
    gpus = detect_gpus()

    for gpu in gpus:
        if gpu.index == gpu_index:
            return (gpu.memory_used_mb, gpu.memory_total_mb)

    return None


def is_gpu_available(
    gpu_index: int,
    memory_threshold_mb: int = 500,
) -> bool:
    """Check if a GPU is available (memory usage below threshold).

    Args:
        gpu_index: GPU index to check.
        memory_threshold_mb: Consider GPU free if used memory is below this.

    Returns:
        True if GPU exists and has memory usage below threshold.
    """
    gpus = detect_gpus()

    for gpu in gpus:
        if gpu.index == gpu_index:
            return gpu.memory_used_mb < memory_threshold_mb

    return False
