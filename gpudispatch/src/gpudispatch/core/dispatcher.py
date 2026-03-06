"""Core Dispatcher for GPU job orchestration."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional, Sequence, Union

from gpudispatch.core.job import Job, JobStatus
from gpudispatch.core.queue import JobQueue, PriorityQueue
from gpudispatch.core.resources import GPU, Memory
from gpudispatch.utils.gpu import detect_gpus, is_gpu_available

logger = logging.getLogger(__name__)


@dataclass
class DispatcherStats:
    """Statistics about dispatcher state."""

    jobs_queued: int = 0
    jobs_running: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    gpus_available: int = 0
    gpus_total: int = 0


class Dispatcher:
    """Core dispatcher for GPU job orchestration.

    The Dispatcher manages a pool of GPUs and a queue of jobs. It monitors
    GPU availability and dispatches jobs when resources become available.

    Example:
        >>> with Dispatcher(gpus=[0, 1]) as dispatcher:
        ...     job = dispatcher.submit(my_function, args=(1, 2))
        ...     # Job will be executed when GPU is available

    Args:
        gpus: List of GPU indices to use. If None, auto-detects available GPUs.
        memory_threshold: Memory threshold for considering a GPU "free".
            Can be string like "500MB" or integer in MB.
        queue: Custom JobQueue implementation. Defaults to PriorityQueue.
        polling_interval: Seconds between queue checks. Default 5.0.
    """

    def __init__(
        self,
        gpus: Optional[list[int]] = None,
        memory_threshold: Union[str, int] = "500MB",
        queue: Optional[JobQueue] = None,
        polling_interval: float = 5.0,
    ):
        # Parse memory threshold
        if isinstance(memory_threshold, str):
            self.memory_threshold_mb = Memory.from_string(memory_threshold).mb
        else:
            self.memory_threshold_mb = memory_threshold

        # Initialize GPU pool
        if gpus is not None:
            self._gpu_indices = list(gpus)
        else:
            detected = detect_gpus()
            self._gpu_indices = [g.index for g in detected]

        self.available_gpus = [GPU(i) for i in self._gpu_indices]

        # Job management
        self._queue = queue or PriorityQueue()
        self._jobs: dict[str, Job] = {}
        self._completed_jobs: set[str] = set()
        self._running_jobs: dict[str, Job] = {}

        # GPU tracking
        self._occupied_gpus: set[int] = set()
        self._lock = threading.Lock()

        # Dispatcher state
        self._running = False
        self._dispatch_thread: Optional[threading.Thread] = None
        self._polling_interval = polling_interval

        # Control events
        self._shutdown_event = threading.Event()
        self._drain_event = threading.Event()

    @property
    def is_running(self) -> bool:
        """Check if dispatcher is currently running."""
        return self._running

    def submit(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        gpu: int = 1,
        memory: Optional[str] = None,
        priority: int = 0,
        name: Optional[str] = None,
        after: Optional[Sequence[Job]] = None,
    ) -> Job:
        """Submit a job to the dispatcher queue.

        Args:
            fn: The function to execute.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
            gpu: Number of GPUs required (default 1).
            memory: Memory requirement (e.g., "16GB").
            priority: Job priority (higher = more important).
            name: Optional job name for logging.
            after: Jobs that must complete before this one starts.

        Returns:
            The created Job object.
        """
        job = Job(
            fn=fn,
            args=args,
            kwargs=kwargs,
            gpu=gpu,
            memory=memory,
            priority=priority,
            name=name,
            after=after,
        )

        with self._lock:
            self._jobs[job.id] = job
            job.status = JobStatus.QUEUED
            job.queued_at = datetime.now()
            self._queue.put(job)

        logger.info(f"Job {job.id} ({job.name}) submitted")
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a job by ID.

        Args:
            job_id: The ID of the job to cancel.

        Returns:
            True if job was found and cancelled, False otherwise.
        """
        with self._lock:
            if job_id not in self._jobs:
                return False

            job = self._jobs[job_id]

            if job.status.is_terminal:
                return False

            self._queue.remove(job_id)
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

        logger.info(f"Job {job_id} cancelled")
        return True

    def stats(self) -> DispatcherStats:
        """Get current dispatcher statistics.

        Returns:
            DispatcherStats with current state information.
        """
        with self._lock:
            queued = len(self._queue)
            running = len(self._running_jobs)
            completed = sum(
                1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED
            )
            failed = sum(
                1 for j in self._jobs.values() if j.status == JobStatus.FAILED
            )
            available = sum(
                1 for idx in self._gpu_indices if idx not in self._occupied_gpus
            )

        return DispatcherStats(
            jobs_queued=queued,
            jobs_running=running,
            jobs_completed=completed,
            jobs_failed=failed,
            gpus_available=available,
            gpus_total=len(self._gpu_indices),
        )

    def start(self) -> None:
        """Start the dispatcher's job processing loop.

        This spawns a background thread that monitors the queue and
        dispatches jobs to available GPUs.
        """
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()
        self._drain_event.clear()

        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="gpudispatch-dispatcher",
        )
        self._dispatch_thread.start()
        logger.info("Dispatcher started")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the dispatcher.

        Args:
            wait: If True, wait for the dispatch thread to exit.
        """
        if not self._running:
            return

        self._shutdown_event.set()

        if wait and self._dispatch_thread:
            self._dispatch_thread.join(timeout=30)

        self._running = False
        logger.info("Dispatcher shutdown")

    def drain(self) -> None:
        """Enter drain mode - stop accepting new jobs but finish running ones."""
        self._drain_event.set()
        logger.info("Dispatcher entering drain mode")

    def _dispatch_loop(self) -> None:
        """Main dispatch loop - runs in background thread."""
        while not self._shutdown_event.is_set():
            try:
                self._process_queue()
            except Exception as e:
                logger.exception(f"Error in dispatch loop: {e}")

            time.sleep(self._polling_interval)

    def _process_queue(self) -> None:
        """Process jobs from the queue if resources available."""
        if self._drain_event.is_set():
            return

        with self._lock:
            job = self._queue.peek()
            if job is None:
                return

            # Check dependencies
            if not job.can_run(self._completed_jobs):
                return

            # Find available GPUs
            available = self._find_available_gpus(job.gpu_count)
            if not available:
                return

            # Start the job
            self._queue.get()
            self._start_job(job, available)

    def _find_available_gpus(self, count: int) -> list[int]:
        """Find available GPUs for a job.

        Args:
            count: Number of GPUs needed.

        Returns:
            List of GPU indices, or empty list if not enough available.
        """
        available = []

        for idx in self._gpu_indices:
            if idx in self._occupied_gpus:
                continue

            if is_gpu_available(idx, self.memory_threshold_mb):
                available.append(idx)
                if len(available) >= count:
                    return available

        return []

    def _start_job(self, job: Job, gpus: list[int]) -> None:
        """Start executing a job on the given GPUs.

        Args:
            job: The job to start.
            gpus: GPU indices allocated to this job.
        """
        job.status = JobStatus.RUNNING
        self._running_jobs[job.id] = job

        for gpu in gpus:
            self._occupied_gpus.add(gpu)

        logger.info(f"Starting job {job.id} on GPUs {gpus}")

        thread = threading.Thread(
            target=self._execute_job,
            args=(job, gpus),
            daemon=True,
        )
        thread.start()

    def _execute_job(self, job: Job, gpus: list[int]) -> None:
        """Execute a job (runs in worker thread).

        Args:
            job: The job to execute.
            gpus: GPU indices allocated to this job.
        """
        job.started_at = datetime.now()

        # Set CUDA_VISIBLE_DEVICES for the worker
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

        try:
            result = job.fn(*job.args, **job.kwargs)
            job.result = result
            job.status = JobStatus.COMPLETED
            logger.info(f"Job {job.id} completed")

        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            logger.error(f"Job {job.id} failed: {e}")

        finally:
            job.completed_at = datetime.now()

            with self._lock:
                del self._running_jobs[job.id]
                self._completed_jobs.add(job.id)

                for gpu in gpus:
                    self._occupied_gpus.discard(gpu)

    def __enter__(self) -> Dispatcher:
        """Enter context manager - start the dispatcher."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - shutdown the dispatcher."""
        self.shutdown()
