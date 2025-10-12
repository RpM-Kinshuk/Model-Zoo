"""
GPU scheduling utilities for batch ESD analysis.
Integrates with shells/gputracker for dynamic GPU allocation.
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional
import gpustat

# Add shells directory to path for gputracker import
sys.path.insert(0, str(Path(__file__).parent.parent / "shells"))
from gputracker.gputracker import get_free_gpu_indices, mark_occupied

logger = logging.getLogger(__name__)

# Global tracking of occupied GPUs
occupied_gpus_esd = []

def get_available_gpus_for_model(
    num_gpus_needed: int = 1,
    gpu_pool: Optional[List[int]] = None,
    max_gpus: Optional[int] = None,
    memory_threshold: int = 500,
    max_checks: int = 5,
) -> List[int]:
    """
    Get available GPUs for a single model processing job.
    
    Args:
        num_gpus_needed: Number of GPUs needed for this model
        gpu_pool: List of GPU IDs we can use (e.g., [0,1,2,3,4,5,6,7])
        max_gpus: Maximum total GPUs we can use from the pool
        memory_threshold: GPU memory threshold in MB to consider GPU as free
        max_checks: Number of checks before confirming GPU is free
    
    Returns:
        List of GPU IDs to use for this model
    """
    global occupied_gpus_esd
    
    # Default to all GPUs if not specified
    if gpu_pool is None:
        stats = gpustat.GPUStatCollection.new_query()
        gpu_pool = list(range(len(stats.gpus)))
    
    # Apply max_gpus constraint
    if max_gpus is not None:
        available_pool = [gpu for gpu in gpu_pool if gpu not in occupied_gpus_esd]
        if len(occupied_gpus_esd) >= max_gpus:
            # We've reached our limit, wait for GPUs to free up
            logger.info(f"Reached max GPU limit ({max_gpus}), waiting for GPUs to free up...")
            while len(occupied_gpus_esd) >= max_gpus:
                time.sleep(10)
                # Clean up any GPUs that might have been freed externally
                occupied_gpus_esd = [gpu for gpu in occupied_gpus_esd if is_gpu_occupied(gpu)]
    
    counter = {}
    
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        available_gpus = []
        
        for i, stat in enumerate(stats.gpus):
            # Check if GPU is in our allowed pool and not occupied
            if i not in gpu_pool:
                continue
            if i in occupied_gpus_esd:
                continue
                
            memory_used = stat['memory.used']
            
            if memory_used < memory_threshold:
                if i not in counter:
                    counter[i] = 0
                else:
                    counter[i] += 1
                
                if counter[i] >= max_checks:
                    available_gpus.append(i)
                    if len(available_gpus) == num_gpus_needed:
                        # Mark as occupied
                        occupied_gpus_esd.extend(available_gpus)
                        logger.info(f"Allocated GPUs {available_gpus} for model processing")
                        return available_gpus
            else:
                counter[i] = 0
        
        # Log waiting status
        if available_gpus:
            logger.debug(f"Found {len(available_gpus)}/{num_gpus_needed} GPUs, waiting for more...")
        else:
            logger.debug(f"No free GPUs found in pool {gpu_pool}, waiting...")
        
        time.sleep(10)

def release_gpus(gpu_ids: List[int]):
    """Release GPUs after model processing is complete."""
    global occupied_gpus_esd
    for gpu_id in gpu_ids:
        if gpu_id in occupied_gpus_esd:
            occupied_gpus_esd.remove(gpu_id)
    logger.info(f"Released GPUs {gpu_ids}")

def is_gpu_occupied(gpu_id: int, memory_threshold: int = 500) -> bool:
    """Check if a GPU is currently occupied."""
    try:
        stats = gpustat.GPUStatCollection.new_query()
        if gpu_id < len(stats.gpus):
            return stats.gpus[gpu_id]['memory.used'] >= memory_threshold
    except:
        pass
    return False

class GPUScheduledProcessor:
    """Wrapper to add GPU scheduling to any processor."""
    
    def __init__(
        self,
        gpu_pool: Optional[List[int]] = None,
        max_total_gpus: Optional[int] = None,
        gpus_per_model: int = 1,
        memory_threshold: int = 500,
        max_checks: int = 10,
    ):
        """
        Initialize GPU scheduled processor.
        
        Args:
            gpu_pool: List of GPU IDs we're allowed to use (e.g., [0,1,2,3,4,5,6,7])
            max_total_gpus: Maximum number of GPUs we can use at once from the pool
            gpus_per_model: Number of GPUs to allocate per model
            memory_threshold: GPU memory threshold in MB
            max_checks: Number of checks before confirming GPU is free
        """
        self.gpu_pool = gpu_pool
        self.max_total_gpus = max_total_gpus
        self.gpus_per_model = gpus_per_model
        self.memory_threshold = memory_threshold
        self.max_checks = max_checks
    
    def allocate_gpus(self) -> List[int]:
        """Allocate GPUs for a model."""
        return get_available_gpus_for_model(
            num_gpus_needed=self.gpus_per_model,
            gpu_pool=self.gpu_pool,
            max_gpus=self.max_total_gpus,
            memory_threshold=self.memory_threshold,
            max_checks=self.max_checks,
        )
    
    def release_gpus(self, gpu_ids: List[int]):
        """Release allocated GPUs."""
        release_gpus(gpu_ids)
