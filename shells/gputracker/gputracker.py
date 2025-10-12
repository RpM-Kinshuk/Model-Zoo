#!/usr/bin/env python3
"""
GPU Tracker - Intelligent GPU resource management for shared environments.

This module provides thread-safe GPU allocation and management for running
multiple jobs in shared GPU environments. It monitors GPU memory usage,
respects other users' jobs, and automatically allocates/deallocates GPUs.

Features:
- Non-exclusive GPU usage with memory threshold monitoring
- Automatic GPU allocation and release
- Thread-safe operation for parallel job dispatch
- Configurable memory thresholds and check intervals
- Support for multi-GPU jobs
"""


import os
import sys
import time
import threading
import logging
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import gpustat
import psutil
import json
from datetime import datetime

# Global configuration
GPU_MEMORY_THRESHOLD = 500  # MB - Default memory threshold
AVAILABLE_GPUS = []  # Will be set during initialization
MAX_NCHECK = 10  # Number of checks before considering GPU free
CHECK_INTERVAL = 10  # Seconds between checks

# Tracking state
occupied_gpus = []  # GPUs currently allocated by this process
all_empty = {"ind": True}  # Whether to wait for all GPUs to be empty initially
gpu_allocation_lock = threading.Lock()  # Thread safety for GPU allocation

# Performance metrics
metrics = {
    "total_jobs": 0,
    "successful_jobs": 0,
    "failed_jobs": 0,
    "total_wait_time": 0,
    "start_time": None,
}

@dataclass
class GPUStatus:
    """Status information for a GPU."""
    gpu_id: int
    memory_used: float
    memory_total: float
    utilization: float
    temperature: int
    processes: List[Dict[str, Any]]
    
    @property
    def memory_free(self) -> float:
        return self.memory_total - self.memory_used
    
    @property
    def is_available(self, threshold_mb: float = 500) -> bool:
        return self.memory_used < threshold_mb


def mark_occupied(gpu_ids: List[int]) -> None:
    """Mark GPUs as occupied by this process."""
    global occupied_gpus
    with gpu_allocation_lock:
        occupied_gpus.extend(gpu_ids)
        logging.debug(f"Marked GPUs {gpu_ids} as occupied. Total occupied: {occupied_gpus}")


def release_gpus(gpu_ids: List[int]) -> None:
    """Release GPUs back to the pool."""
    global occupied_gpus
    with gpu_allocation_lock:
        for gpu_id in gpu_ids:
            if gpu_id in occupied_gpus:
                occupied_gpus.remove(gpu_id)
        logging.debug(f"Released GPUs {gpu_ids}. Remaining occupied: {occupied_gpus}")


def get_gpu_status(gpu_id: int) -> Optional[GPUStatus]:
    """Get detailed status for a specific GPU."""
    try:
        stats = gpustat.GPUStatCollection.new_query()
        if gpu_id < len(stats.gpus):
            stat = stats.gpus[gpu_id]
            return GPUStatus(
                gpu_id=gpu_id,
                memory_used=stat['memory.used'],
                memory_total=stat['memory.total'],
                utilization=stat['utilization.gpu'],
                temperature=stat['temperature.gpu'],
                processes=stat['processes']
            )
    except Exception as e:
        logging.warning(f"Failed to get GPU {gpu_id} status: {e}")
    return None


def num_available_GPUs(gpus, threshold: float = 100) -> int:
    """Count number of available GPUs based on memory threshold."""
    available = 0
    for stat in gpus:
        if stat['memory.used'] < threshold:
            available += 1
    return available


def estimate_wait_time(counter: Dict[int, int]) -> float:
    """Estimate remaining wait time based on check counters."""
    if not counter:
        return float('inf')
    max_checks = max(counter.values())
    remaining_checks = MAX_NCHECK - max_checks
    return max(0, remaining_checks * CHECK_INTERVAL)


def get_free_gpu_indices(logger: logging.Logger, num_gpus_needed: int = 1,
                        priority_gpus: Optional[List[int]] = None) -> List[int]:
    """
    Return a list of available GPU indices.
    
    Args:
        logger: Logger instance for output
        num_gpus_needed: Number of GPUs required
        priority_gpus: Optional list of preferred GPU IDs
        
    Returns:
        List of allocated GPU IDs
    """
    counter = {}
    start_wait = time.time()
    check_count = 0
    
    while True:
        check_count += 1
        try:
            stats = gpustat.GPUStatCollection.new_query()
        except Exception as e:
            logger.error(f"Failed to query GPU stats: {e}")
            time.sleep(CHECK_INTERVAL)
            continue
        
        # Check if enough GPUs are potentially available
        if num_available_GPUs(stats.gpus, GPU_MEMORY_THRESHOLD) >= num_gpus_needed:
            all_empty["ind"] = True
        
        if not all_empty["ind"]:
            logger.info("Waiting for previous experiments to finish...")
            time.sleep(CHECK_INTERVAL)
            continue
        
        max_checks = 0
        max_gpu_id = -1
        available_gpus = []
        candidates = priority_gpus if priority_gpus else AVAILABLE_GPUS
        
        # Check each GPU
        for gpu_stat in stats.gpus:
            gpu_id = gpu_stat['index']
            memory_used = gpu_stat['memory.used']
            
            # Skip if not in allowed list or already occupied
            if gpu_id not in candidates:
                continue
                
            with gpu_allocation_lock:
                if gpu_id in occupied_gpus:
                    continue
            
            # Check if GPU meets memory requirements
            if memory_used < GPU_MEMORY_THRESHOLD:
                # Increment counter for this GPU
                if gpu_id not in counter:
                    counter[gpu_id] = 1
                else:
                    counter[gpu_id] += 1
                
                # Add to available list if consistently free
                if counter[gpu_id] >= MAX_NCHECK:
                    available_gpus.append(gpu_id)
                    if len(available_gpus) == num_gpus_needed:
                        # Allocate these GPUs
                        mark_occupied(available_gpus)
                        wait_time = time.time() - start_wait
                        logger.info(f"Allocated GPUs {available_gpus} after {wait_time:.1f}s")
                        
                        # Update metrics
                        global metrics
                        metrics["total_wait_time"] += wait_time
                        
                        return available_gpus
            else:
                # Reset counter if GPU is busy
                counter[gpu_id] = 0
            
            # Track GPU with most checks
            if gpu_id in counter and counter[gpu_id] > max_checks:
                max_checks = counter[gpu_id]
                max_gpu_id = gpu_id
        
        # Log progress
        if max_gpu_id != -1 and check_count % 3 == 0:  # Log every 3rd check
            est_time = estimate_wait_time(counter)
            logger.info(f"Waiting for GPUs... Best candidate: GPU {max_gpu_id} "
                       f"({max_checks}/{MAX_NCHECK} checks, ~{est_time:.0f}s remaining)")
            
            # Also show current GPU status
            if check_count % 10 == 0:  # Detailed status every 10 checks
                logger.debug(f"GPU status - Available: {candidates}, "
                           f"Occupied: {occupied_gpus}, Counter: {counter}")
        
        time.sleep(CHECK_INTERVAL)

class DispatchThread(threading.Thread):
    """Main dispatcher thread for managing job execution across GPUs."""
    
    def __init__(self, name: str, bash_command_list: List[str], logger: logging.Logger,
                 gpu_m_th: int = 500, gpu_list: Optional[List[int]] = None, 
                 maxcheck: int = 10, num_gpus_needed: int = 1):
        """
        Initialize dispatcher thread.
        
        Args:
            name: Thread name
            bash_command_list: List of commands to execute
            logger: Logger instance
            gpu_m_th: GPU memory threshold in MB
            gpu_list: List of allowed GPU IDs
            maxcheck: Number of checks before considering GPU free
            num_gpus_needed: GPUs required per job
        """
        threading.Thread.__init__(self)
        self.name = name
        self.bash_command_list = bash_command_list
        self.logger = logger
        self.num_gpus_needed = num_gpus_needed
        
        # Update global configuration
        global GPU_MEMORY_THRESHOLD, AVAILABLE_GPUS, MAX_NCHECK, CHECK_INTERVAL
        GPU_MEMORY_THRESHOLD = gpu_m_th
        
        # Auto-detect GPUs if not specified
        if gpu_list:
            AVAILABLE_GPUS = gpu_list
        else:
            try:
                import torch
                AVAILABLE_GPUS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            except ImportError:
                # Fallback to gpustat
                stats = gpustat.GPUStatCollection.new_query()
                AVAILABLE_GPUS = list(range(len(stats.gpus)))
        MAX_NCHECK = max(1, maxcheck)  # At least 1 check
        
        # Initialize metrics
        global metrics
        metrics["start_time"] = time.time()
        
        self.logger.info(f"Initialized {self.name} with {len(self.bash_command_list)} jobs")
        self.logger.info(f"Configuration: GPUs={AVAILABLE_GPUS}, Memory threshold={GPU_MEMORY_THRESHOLD}MB, "
                        f"Checks={MAX_NCHECK}, GPUs per job={self.num_gpus_needed}")

    def run(self):
        """Run all jobs with GPU management."""
        self.logger.info(f"Starting {self.name}")
        threads = []
        
        # Process each command
        for i, bash_command in enumerate(self.bash_command_list):
            # Small delay between job submissions
            time.sleep(0.5)
            
            # Skip empty commands
            if not bash_command or not bash_command.strip():
                self.logger.warning(f"Skipping empty command at index {i}")
                continue
            
            # Get job name (truncate if too long)
            job_name = f"Job_{i}"
            if len(bash_command) < 100:
                job_name = f"Job_{i}: {bash_command[:50]}..."
            
            self.logger.info(f"Scheduling {job_name}")
            
            # Wait for free GPUs
            try:
                cuda_devices = get_free_gpu_indices(self.logger, self.num_gpus_needed)
            except Exception as e:
                self.logger.error(f"Failed to allocate GPUs for {job_name}: {e}")
                continue
            
            # Create and start worker thread
            worker = ChildThread(
                name=job_name,
                counter=i,
                cuda_devices=cuda_devices,
                bash_command=bash_command,
                logger=self.logger
            )
            worker.start()
            threads.append(worker)
            
            # Stagger job starts
            time.sleep(5)
        
        # Wait for all jobs to complete
        self.logger.info(f"All jobs scheduled. Waiting for completion...")
        for thread in threads:
            thread.join()
        
        # Save metrics
        metrics_file = "gpu_metrics.json"
        save_metrics(metrics_file)
        
        # Print summary
        self.logger.info("="*60)
        self.logger.info(f"Completed {self.name}")
        self.logger.info(f"Total jobs: {metrics['total_jobs']}")
        self.logger.info(f"Successful: {metrics['successful_jobs']}")
        self.logger.info(f"Failed: {metrics['failed_jobs']}")
        if metrics["total_jobs"] > 0:
            self.logger.info(f"Average wait time: {metrics['total_wait_time']/metrics['total_jobs']:.1f}s")
        self.logger.info("="*60)

class ChildThread(threading.Thread):
    """Thread for executing a single job with allocated GPUs."""
    
    def __init__(self, name: str, counter: int, cuda_devices: List[int], 
                 bash_command: str, logger: logging.Logger):
        threading.Thread.__init__(self)
        self.name = name
        self.counter = counter
        self.cuda_devices = cuda_devices
        self.bash_command = bash_command
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.success = False

    def run(self):
        """Execute the command with allocated GPUs."""
        self.start_time = time.time()
        
        # Set CUDA devices for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.cuda_devices))
        
        # Log start
        self.logger.info(f"Starting job '{self.name}' on GPUs {self.cuda_devices}")
        self.logger.debug(f"Command: {self.bash_command}")
        
        # Update metrics
        global metrics
        metrics["total_jobs"] += 1
        
        try:
            # Execute command
            return_code = os.system(self.bash_command)
            
            # Check success
            if return_code == 0:
                self.success = True
                metrics["successful_jobs"] += 1
                self.logger.info(f"Job '{self.name}' completed successfully")
            else:
                metrics["failed_jobs"] += 1
                self.logger.error(f"Job '{self.name}' failed with return code {return_code}")
                
        except Exception as e:
            metrics["failed_jobs"] += 1
            self.logger.error(f"Job '{self.name}' failed with exception: {e}")
            
        finally:
            # Small random delay to prevent race conditions
            time.sleep(random.random() * 2)
            
            # Release GPUs
            release_gpus(self.cuda_devices)
            
            # Log completion
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.logger.info(f"Finished job '{self.name}' in {duration:.1f}s\n")


def get_logger(path: str, fname: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create and configure logger instance.
    
    Args:
        path: Directory for log file
        fname: Log filename
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    os.makedirs(path, exist_ok=True)
    
    # Create unique logger name to avoid conflicts
    logger_name = f"gputracker_{os.getpid()}_{threading.current_thread().ident}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    file_log_handler = logging.FileHandler(os.path.join(path, fname))
    file_log_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Less verbose on console
    
    # Formatter
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)-15s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        "%H:%M:%S"
    )
    
    file_log_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_log_handler)
    logger.addHandler(console_handler)
    
    sys.stdout.flush()
    return logger


def save_metrics(output_path: str) -> None:
    """Save performance metrics to JSON file."""
    global metrics
    
    if metrics["start_time"]:
        metrics["total_runtime"] = time.time() - metrics["start_time"]
        metrics["avg_wait_time"] = (metrics["total_wait_time"] / max(1, metrics["total_jobs"]))
    
    metrics["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Saved metrics to {output_path}")

