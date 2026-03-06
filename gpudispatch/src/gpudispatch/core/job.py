"""Job representation and lifecycle management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Union

from gpudispatch.core.resources import Memory, ResourceRequirements


class JobStatus(Enum):
    """Job lifecycle status."""

    PENDING = "pending"      # Created, not yet submitted
    QUEUED = "queued"        # In queue, waiting for resources
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Finished with error
    CANCELLED = "cancelled"  # Cancelled by user

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) status."""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass
class Job:
    """A unit of work to be executed on GPU resources."""

    fn: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    # Resource requirements
    gpu_count: int = 1
    memory: Optional[Memory] = None

    # Scheduling
    priority: int = 0
    name: Optional[str] = None

    # Dependencies
    dependencies: set[str] = field(default_factory=set)

    # Internal state
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        gpu: int = 1,
        memory: Optional[Union[str, Memory]] = None,
        priority: int = 0,
        name: Optional[str] = None,
        after: Optional[Sequence[Job]] = None,
    ):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}
        self.gpu_count = gpu

        if memory is None:
            self.memory = None
        elif isinstance(memory, str):
            self.memory = Memory.from_string(memory)
        else:
            self.memory = memory

        self.priority = priority
        self.name = name or getattr(fn, "__name__", "anonymous")

        self.dependencies = set()
        if after:
            for dep in after:
                self.dependencies.add(dep.id)

        self.id = str(uuid.uuid4())[:8]
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.queued_at = None
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None

    def can_run(self, completed_jobs: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.dependencies.issubset(completed_jobs)

    @property
    def requirements(self) -> ResourceRequirements:
        """Get resource requirements for this job."""
        return ResourceRequirements(gpu=self.gpu_count, memory=self.memory)

    def __repr__(self) -> str:
        return f"Job(id={self.id}, name={self.name}, status={self.status.value})"


@dataclass
class JobResult:
    """Result of a completed job."""

    value: Any
    status: JobStatus
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED

    @property
    def runtime_seconds(self) -> Optional[float]:
        """Get job runtime in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
