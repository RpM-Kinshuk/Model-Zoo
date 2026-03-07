"""Core Dispatcher for GPU job orchestration."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from gpudispatch.core.job import CommandResult, Job, JobStatus
from gpudispatch.core.queue import JobQueue, PriorityQueue
from gpudispatch.core.resources import GPU, Memory
from gpudispatch.observability import hooks
from gpudispatch.utils.gpu import detect_gpus, is_gpu_available

if TYPE_CHECKING:
    from gpudispatch.core.signals import SignalHandler

logger = logging.getLogger(__name__)

# Type alias for config dict
ConfigDict = Dict[str, Any]


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
        default_command_timeout: Default timeout in seconds for command/script
            jobs submitted without an explicit timeout.
        default_command_env: Default environment variables merged into every
            command/script job's environment.
    """

    def __init__(
        self,
        gpus: Optional[list[int]] = None,
        memory_threshold: Union[str, int] = "500MB",
        queue: Optional[JobQueue] = None,
        polling_interval: float = 5.0,
        default_command_timeout: Optional[float] = None,
        default_command_env: Optional[dict[str, str]] = None,
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
        self._env_lock = threading.Lock()

        # Dispatcher state
        self._running = False
        self._dispatch_thread: Optional[threading.Thread] = None
        self._polling_interval = polling_interval

        # Command/script defaults
        self._default_command_timeout = default_command_timeout
        self._default_command_env = {
            str(k): str(v) for k, v in (default_command_env or {}).items()
        }

        # Control events
        self._shutdown_event = threading.Event()
        self._drain_event = threading.Event()

        # Signal handler (set by setup_signals())
        self._signal_handler: Optional[SignalHandler] = None

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

        return self._enqueue_job(job)

    def submit_command(
        self,
        command: Union[str, Sequence[str]],
        *,
        shell: Optional[bool] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        gpu: int = 1,
        memory: Optional[str] = None,
        priority: int = 0,
        name: Optional[str] = None,
        after: Optional[Sequence[Job]] = None,
    ) -> Job:
        """Submit a command/script job to run in an isolated subprocess.

        Args:
            command: Command string or argv sequence.
            shell: Whether to execute via shell. Defaults to True for string
                commands and False for sequence commands.
            cwd: Optional working directory for command execution.
            env: Optional environment variable overrides for this job.
            timeout: Optional execution timeout in seconds.
            gpu: Number of GPUs required.
            memory: Memory requirement (e.g., "16GB").
            priority: Job priority (higher = more important).
            name: Optional job name for logging.
            after: Jobs that must complete before this one starts.

        Returns:
            The created Job object.
        """
        merged_env = dict(self._default_command_env)
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})

        effective_timeout = self._default_command_timeout if timeout is None else timeout

        job = Job(
            command=command,
            shell=shell,
            cwd=cwd,
            env=merged_env,
            timeout=effective_timeout,
            gpu=gpu,
            memory=memory,
            priority=priority,
            name=name,
            after=after,
        )

        return self._enqueue_job(job)

    def submit_script(
        self,
        script_path: str,
        script_args: Optional[Sequence[str]] = None,
        *,
        interpreter: Optional[Union[str, Sequence[str]]] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        gpu: int = 1,
        memory: Optional[str] = None,
        priority: int = 0,
        name: Optional[str] = None,
        after: Optional[Sequence[Job]] = None,
    ) -> Job:
        """Submit a script file as a command job.

        This helper is intended for plugging in existing experiment scripts
        (bash, python, etc.) with dispatcher controls.

        Args:
            script_path: Path to the script file to execute.
            script_args: Optional command-line arguments for the script.
            interpreter: Optional interpreter command. If omitted and script_path
                ends with ".py", uses the current Python executable.
            cwd: Optional working directory for command execution.
            env: Optional environment variable overrides for this job.
            timeout: Optional execution timeout in seconds.
            gpu: Number of GPUs required.
            memory: Memory requirement (e.g., "16GB").
            priority: Job priority (higher = more important).
            name: Optional job name for logging.
            after: Jobs that must complete before this one starts.

        Returns:
            The created Job object.
        """
        normalized_args = [str(arg) for arg in (script_args or ())]
        script = str(script_path)

        if interpreter is None:
            if script.endswith(".py"):
                command: list[str] = [sys.executable, script, *normalized_args]
            else:
                command = [script, *normalized_args]
        elif isinstance(interpreter, str):
            command = [interpreter, script, *normalized_args]
        else:
            command = [*(str(part) for part in interpreter), script, *normalized_args]

        return self.submit_command(
            command=command,
            shell=False,
            cwd=cwd,
            env=env,
            timeout=timeout,
            gpu=gpu,
            memory=memory,
            priority=priority,
            name=name,
            after=after,
        )

    def _enqueue_job(self, job: Job) -> Job:
        """Register a job and enqueue it for scheduling."""

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

    def wait(
        self,
        job: Job,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> Any:
        """Wait for a submitted job to finish and return its result.

        Args:
            job: Job returned by :meth:`submit`.
            timeout: Optional timeout in seconds. If exceeded, raises TimeoutError.
            poll_interval: How often to check job status.

        Returns:
            The job return value.

        Raises:
            ValueError: If the job is unknown to this dispatcher.
            RuntimeError: If the job failed or was cancelled.
            TimeoutError: If timeout is exceeded before completion.
        """
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")

        with self._lock:
            if job.id not in self._jobs:
                raise ValueError(f"Unknown job: {job.id}")

        # Allow submit()+wait() usage without requiring explicit start().
        if not self._running and not job.status.is_terminal:
            self.start()

        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            status = job.status

            if status == JobStatus.COMPLETED:
                return job.result

            if status == JobStatus.FAILED:
                error_msg = job.error or "Unknown error"
                raise RuntimeError(f"Job {job.id} failed: {error_msg}")

            if status == JobStatus.CANCELLED:
                raise RuntimeError(f"Job {job.id} was cancelled")

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for job {job.id} after {timeout:.1f}s"
                )

            time.sleep(poll_interval)

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

    def reload_config(self, config: Optional[ConfigDict]) -> None:
        """Reload configuration from a config dictionary.

        This method updates the dispatcher's configuration based on the
        provided config dictionary. It's typically called by the SignalHandler
        on SIGHUP.

        Args:
            config: Configuration dictionary with optional keys:
                - available_gpus: List of GPU indices to use
                - memory_threshold_mb: Memory threshold in MB
                - max_checks: (reserved for future use)

        Example config:
            {
                "available_gpus": [0, 1, 2, 3],
                "max_checks": 5,
                "memory_threshold_mb": 500
            }
        """
        if config is None:
            logger.warning("reload_config called with None config, ignoring")
            return

        with self._lock:
            # Update GPU pool if specified
            if "available_gpus" in config:
                new_gpus = [int(g) for g in config["available_gpus"]]
                self._gpu_indices = new_gpus
                self.available_gpus = [GPU(i) for i in new_gpus]
                logger.info(f"GPU pool updated to: {new_gpus}")

            # Update memory threshold if specified
            if "memory_threshold_mb" in config:
                self.memory_threshold_mb = int(config["memory_threshold_mb"])
                logger.info(f"Memory threshold updated to: {self.memory_threshold_mb}MB")

        logger.info("Configuration reloaded successfully")

    def setup_signals(self, config_path: Optional[str] = None) -> "SignalHandler":
        """Set up Unix signal handling for runtime control.

        This method creates and installs a SignalHandler that connects
        signals to dispatcher methods:
        - SIGHUP: Calls reload_config() with config from config_path
        - SIGUSR1: Calls drain()
        - SIGTERM/SIGINT: Calls shutdown()

        Args:
            config_path: Optional path to JSON config file for SIGHUP reloads.

        Returns:
            The installed SignalHandler instance.

        Note:
            Signal handling only works on Unix systems. On Windows, this
            returns a handler that does nothing.

        Example:
            >>> dispatcher = Dispatcher(gpus=[0, 1])
            >>> handler = dispatcher.setup_signals("gpu_config.json")
            >>> dispatcher.start()
            >>> # Now you can send signals:
            >>> # kill -HUP <pid>   # Reload config
            >>> # kill -USR1 <pid>  # Enter drain mode
            >>> # kill -TERM <pid>  # Shutdown
        """
        # Import here to avoid circular imports
        from gpudispatch.core.signals import SignalHandler

        handler = SignalHandler(
            dispatcher=self,
            config_path=config_path,
            on_reload=self.reload_config,
        )
        handler.install()

        # Store reference so it doesn't get garbage collected
        self._signal_handler = handler

        return handler

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
        hooks.emit(
            "on_job_start",
            job_id=job.id,
            job_name=job.name or "anonymous",
            gpus=list(gpus),
            is_command=job.is_command,
        )

        thread = threading.Thread(
            target=self._execute_job,
            args=(job, gpus),
            daemon=True,
            name=f"gpudispatch-job-{job.id}",
        )
        thread.start()

    def _execute_job(self, job: Job, gpus: list[int]) -> None:
        """Execute a job (runs in worker thread).

        Args:
            job: The job to execute.
            gpus: GPU indices allocated to this job.
        """
        job.started_at = datetime.now()
        job_env = self._build_job_env(gpus, job.env)

        try:
            if job.is_command:
                result = self._execute_command_job(job, job_env)
            else:
                result = self._execute_callable_job(job, job_env)

            job.result = result
            job.status = JobStatus.COMPLETED
            logger.info(f"Job {job.id} completed")
            runtime_seconds = 0.0
            if job.started_at is not None:
                runtime_seconds = (datetime.now() - job.started_at).total_seconds()
            hooks.emit(
                "on_job_complete",
                job_id=job.id,
                job_name=job.name or "anonymous",
                runtime_seconds=runtime_seconds,
                gpus=list(gpus),
                is_command=job.is_command,
                result=result,
            )

        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            logger.error(f"Job {job.id} failed: {e}")
            hooks.emit(
                "on_job_failed",
                job_id=job.id,
                job_name=job.name or "anonymous",
                error=job.error,
                gpus=list(gpus),
                is_command=job.is_command,
            )

        finally:
            job.completed_at = datetime.now()

            with self._lock:
                del self._running_jobs[job.id]
                self._completed_jobs.add(job.id)

                for gpu in gpus:
                    self._occupied_gpus.discard(gpu)

    def _build_job_env(self, gpus: list[int], extra_env: dict[str, str]) -> dict[str, str]:
        """Build environment variables for a running job."""
        assigned_gpus = ",".join(str(g) for g in gpus)
        job_env = os.environ.copy()
        job_env["CUDA_VISIBLE_DEVICES"] = assigned_gpus
        job_env["GPUDISPATCH_ASSIGNED_GPUS"] = assigned_gpus
        job_env.update(extra_env)
        return job_env

    def _execute_callable_job(self, job: Job, job_env: dict[str, str]) -> Any:
        """Execute an in-process callable job with temporary env overrides."""
        if job.fn is None:
            raise RuntimeError("Callable job is missing function")

        env_overrides = {
            "CUDA_VISIBLE_DEVICES": job_env["CUDA_VISIBLE_DEVICES"],
            "GPUDISPATCH_ASSIGNED_GPUS": job_env["GPUDISPATCH_ASSIGNED_GPUS"],
            **job.env,
        }

        # Process environment is global. Guard and restore to prevent leakage
        # between jobs while preserving compatibility for callable workflows.
        with self._env_lock:
            previous = {key: os.environ.get(key) for key in env_overrides}
            os.environ.update(env_overrides)
            try:
                return job.fn(*job.args, **job.kwargs)
            finally:
                for key, old_value in previous.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

    def _execute_command_job(self, job: Job, job_env: dict[str, str]) -> CommandResult:
        """Execute a command job in a subprocess with isolated environment."""
        if job.command is None:
            raise RuntimeError("Command job is missing command")

        try:
            completed = subprocess.run(
                job.command,
                shell=job.shell,
                cwd=job.cwd,
                env=job_env,
                capture_output=True,
                text=True,
                timeout=job.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Command timed out after {job.timeout}s") from e

        result = CommandResult(
            command=job.command,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

        if not result.is_success:
            stderr_preview = (result.stderr.strip() or "command exited with non-zero status")
            raise RuntimeError(
                f"Command failed (exit={result.returncode}): {stderr_preview}"
            )

        return result

    def __enter__(self) -> Dispatcher:
        """Enter context manager - start the dispatcher."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - shutdown the dispatcher."""
        self.shutdown()
