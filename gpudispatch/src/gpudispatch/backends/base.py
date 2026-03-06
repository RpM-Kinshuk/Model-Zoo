"""Abstract base class for GPU orchestration backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from gpudispatch.core.resources import GPU, Memory


class Backend(ABC):
    """Abstract base class for GPU orchestration backends.

    A Backend is responsible for managing GPU resources in a specific
    environment (local machine, SLURM cluster, Kubernetes, cloud, etc.).
    It provides a common interface for allocating and releasing GPUs,
    checking availability, and managing the backend lifecycle.

    All backends must implement:
        - name: Identifier for the backend (e.g., "local", "slurm", "k8s")
        - is_running: Whether the backend is active
        - allocate_gpus(): Allocate GPUs for a job
        - release_gpus(): Release allocated GPUs
        - list_available(): List currently available GPUs
        - health_check(): Verify backend is operational
        - start(): Initialize the backend
        - shutdown(): Graceful shutdown

    Example:
        >>> class LocalBackend(Backend):
        ...     @property
        ...     def name(self) -> str:
        ...         return "local"
        ...     # ... implement other methods
        ...
        >>> with LocalBackend() as backend:
        ...     gpus = backend.allocate_gpus(2)
        ...     # use GPUs
        ...     backend.release_gpus(gpus)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier.

        Returns:
            str: A unique identifier for the backend type,
                e.g., "local", "slurm", "k8s", "aws".
        """
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the backend is currently active.

        Returns:
            bool: True if the backend has been started and not yet
                shut down, False otherwise.
        """
        pass

    @abstractmethod
    def allocate_gpus(
        self, count: int, memory: Optional[Memory] = None
    ) -> List[GPU]:
        """Allocate GPUs for a job.

        Args:
            count: Number of GPUs to allocate.
            memory: Optional minimum memory requirement per GPU.
                If specified, only GPUs with at least this much
                memory will be allocated.

        Returns:
            List[GPU]: List of allocated GPU objects. Returns an
                empty list if the requested number of GPUs cannot
                be allocated (either not enough available or not
                enough with sufficient memory).
        """
        pass

    @abstractmethod
    def release_gpus(self, gpus: List[GPU]) -> None:
        """Release allocated GPUs.

        Released GPUs become available for allocation again.
        Releasing GPUs that were not allocated is a no-op.

        Args:
            gpus: List of GPU objects to release.
        """
        pass

    @abstractmethod
    def list_available(self) -> List[GPU]:
        """List currently available GPUs.

        Returns:
            List[GPU]: List of GPUs that are available for allocation.
                Does not include GPUs that are currently allocated.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify backend is operational.

        Performs a health check to verify the backend is functioning
        correctly. This may include checking connectivity to external
        services, verifying GPU availability, etc.

        Returns:
            bool: True if the backend is healthy and operational,
                False otherwise.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the backend.

        This method should perform any necessary initialization,
        such as connecting to external services, discovering GPUs,
        or setting up monitoring.

        After start() returns, is_running should be True.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shutdown the backend.

        This method should release any resources, close connections,
        and perform cleanup. Any GPUs that are still allocated should
        be released.

        After shutdown() returns, is_running should be False.
        """
        pass

    def __enter__(self) -> Backend:
        """Enter context manager - start the backend.

        Returns:
            Backend: self, after calling start().
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - shutdown the backend."""
        self.shutdown()
