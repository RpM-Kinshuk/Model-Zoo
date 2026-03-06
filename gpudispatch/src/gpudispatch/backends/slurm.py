"""SLURM cluster backend stub for GPU orchestration."""

from __future__ import annotations

from typing import List, Optional

from gpudispatch.backends.base import Backend
from gpudispatch.core.resources import GPU, Memory


class SLURMBackend(Backend):
    """SLURM cluster backend (stub - extensible).

    This is a stub implementation for SLURM-based GPU orchestration.
    To implement a full SLURM backend, subclass this and override:
        - _submit_job(): Submit a job to the SLURM scheduler
        - _check_job_status(): Check the status of a submitted job
        - _cancel_job(): Cancel a running/pending job

    Example:
        >>> class MySLURMBackend(SLURMBackend):
        ...     def _submit_job(self, script: str) -> str:
        ...         # Submit via sbatch, return job ID
        ...         pass
        ...     def _check_job_status(self, job_id: str) -> str:
        ...         # Check via squeue/sacct, return status
        ...         pass

    Args:
        partition: SLURM partition to submit jobs to. Default: "gpu".
        account: SLURM account for job accounting. Default: None.
        time_limit: Maximum job runtime (HH:MM:SS format). Default: "24:00:00".
        nodes: Number of nodes to request. Default: 1.
        gpus_per_node: Number of GPUs per node. Default: 1.
    """

    def __init__(
        self,
        partition: str = "gpu",
        account: Optional[str] = None,
        time_limit: str = "24:00:00",
        nodes: int = 1,
        gpus_per_node: int = 1,
        **kwargs,
    ):
        self._partition = partition
        self._account = account
        self._time_limit = time_limit
        self._nodes = nodes
        self._gpus_per_node = gpus_per_node
        self._extra_config = kwargs
        self._running = False

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "slurm"

    @property
    def is_running(self) -> bool:
        """Whether the backend is currently active."""
        return self._running

    @property
    def partition(self) -> str:
        """SLURM partition name."""
        return self._partition

    @property
    def account(self) -> Optional[str]:
        """SLURM account name."""
        return self._account

    @property
    def time_limit(self) -> str:
        """Job time limit in HH:MM:SS format."""
        return self._time_limit

    @property
    def nodes(self) -> int:
        """Number of nodes requested."""
        return self._nodes

    @property
    def gpus_per_node(self) -> int:
        """Number of GPUs per node."""
        return self._gpus_per_node

    def start(self) -> None:
        """Initialize and start the backend.

        For SLURM, this would typically verify connectivity to the scheduler.
        """
        self._running = True

    def shutdown(self) -> None:
        """Gracefully shutdown the backend.

        For SLURM, this would cancel pending jobs and cleanup.
        """
        self._running = False

    def allocate_gpus(
        self, count: int, memory: Optional[Memory] = None
    ) -> List[GPU]:
        """Allocate GPUs for a job.

        Not implemented in stub. Override _submit_job() to implement.

        Raises:
            NotImplementedError: SLURM job submission not implemented.
        """
        raise NotImplementedError(
            "SLURM GPU allocation requires job submission. "
            "Subclass SLURMBackend and override _submit_job() to implement."
        )

    def release_gpus(self, gpus: List[GPU]) -> None:
        """Release allocated GPUs.

        Not implemented in stub. Override _cancel_job() to implement.

        Raises:
            NotImplementedError: SLURM job cancellation not implemented.
        """
        raise NotImplementedError(
            "SLURM GPU release requires job cancellation. "
            "Subclass SLURMBackend and override _cancel_job() to implement."
        )

    def list_available(self) -> List[GPU]:
        """List currently available GPUs.

        Not implemented in stub. Would require querying SLURM scheduler.

        Raises:
            NotImplementedError: SLURM resource querying not implemented.
        """
        raise NotImplementedError(
            "SLURM GPU listing requires scheduler queries (sinfo/squeue). "
            "Subclass SLURMBackend and implement cluster resource discovery."
        )

    def health_check(self) -> bool:
        """Verify backend is operational.

        Returns:
            bool: True if backend is running, False otherwise.
            A full implementation would check SLURM scheduler connectivity.
        """
        return self._running

    # Extension points for subclasses

    def _submit_job(self, script: str) -> str:
        """Submit a job to SLURM scheduler.

        Override this method to implement job submission via sbatch.

        Args:
            script: SLURM job script content.

        Returns:
            str: Job ID returned by SLURM.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "_submit_job() must be implemented in subclass"
        )

    def _check_job_status(self, job_id: str) -> str:
        """Check status of a SLURM job.

        Override this method to implement status checking via squeue/sacct.

        Args:
            job_id: SLURM job ID.

        Returns:
            str: Job status (e.g., "PENDING", "RUNNING", "COMPLETED").

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "_check_job_status() must be implemented in subclass"
        )

    def _cancel_job(self, job_id: str) -> None:
        """Cancel a SLURM job.

        Override this method to implement job cancellation via scancel.

        Args:
            job_id: SLURM job ID to cancel.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "_cancel_job() must be implemented in subclass"
        )
