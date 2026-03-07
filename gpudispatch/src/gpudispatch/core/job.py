"""Job representation and lifecycle management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Union

from gpudispatch.core.resources import Memory, ResourceRequirements

CommandType = Union[str, Sequence[str]]


def _command_display_name(command: CommandType) -> str:
    """Build a compact display name for command jobs."""
    if isinstance(command, str):
        compact = " ".join(command.strip().split())
        return compact[:80] if compact else "command"

    parts = list(command)
    if not parts:
        return "command"
    return str(parts[0])


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

    fn: Optional[Callable[..., Any]]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    # Command execution (for script/binary workloads)
    command: Optional[CommandType] = None
    shell: bool = False
    cwd: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None

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
        fn: Optional[Callable[..., Any]] = None,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        command: Optional[CommandType] = None,
        shell: Optional[bool] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        gpu: int = 1,
        memory: Optional[Union[str, Memory]] = None,
        priority: int = 0,
        name: Optional[str] = None,
        after: Optional[Sequence[Job]] = None,
    ):
        if fn is None and command is None:
            raise ValueError("Job requires either fn or command")
        if fn is not None and command is not None:
            raise ValueError("Job cannot specify both fn and command")

        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs or {}

        normalized_command: Optional[CommandType] = None
        if command is not None:
            if isinstance(command, str):
                normalized_command = command
            else:
                normalized_command = tuple(str(part) for part in command)
                if not normalized_command:
                    raise ValueError("Command sequence cannot be empty")

        self.command = normalized_command
        self.shell = isinstance(self.command, str) if shell is None else shell
        if self.shell and self.command is not None and not isinstance(self.command, str):
            raise ValueError("shell=True requires command to be a string")

        self.cwd = cwd
        self.env = {str(k): str(v) for k, v in (env or {}).items()}
        self.timeout = timeout
        self.gpu_count = gpu

        if memory is None:
            self.memory = None
        elif isinstance(memory, str):
            self.memory = Memory.from_string(memory)
        else:
            self.memory = memory

        self.priority = priority
        if name is not None:
            self.name = name
        elif self.command is not None:
            self.name = _command_display_name(self.command)
        elif fn is not None:
            self.name = getattr(fn, "__name__", "anonymous")
        else:
            self.name = "anonymous"

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

    @property
    def is_command(self) -> bool:
        """Whether this job executes a shell command instead of a callable."""
        return self.command is not None

    def __repr__(self) -> str:
        mode = "command" if self.is_command else "callable"
        return (
            f"Job(id={self.id}, name={self.name}, mode={mode}, "
            f"status={self.status.value})"
        )


@dataclass
class CommandResult:
    """Result payload for command-based jobs."""

    command: CommandType
    returncode: int
    stdout: str
    stderr: str

    @property
    def is_success(self) -> bool:
        """Check if command exited successfully."""
        return self.returncode == 0


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
