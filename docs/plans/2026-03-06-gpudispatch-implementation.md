# gpudispatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade, researcher-friendly GPU orchestration library that becomes the universal solution for ML workload management.

**Architecture:** Layered architecture with plugin system - Core dispatcher handles job lifecycle, backend plugins implement environment-specific logic (Local, SLURM, K8s, Cloud), experiment layer provides Grid/Sweep/Bayesian optimization, observability layer provides metrics/traces/logging.

**Tech Stack:** Python 3.10+, PyTorch, gpustat, pydantic, click, rich, prometheus-client, opentelemetry, pytest, mypy

---

## Phase 1: Project Scaffolding & Core Foundation

### Task 1.1: Initialize Package Structure

**Files:**
- Create: `gpudispatch/pyproject.toml`
- Create: `gpudispatch/src/gpudispatch/__init__.py`
- Create: `gpudispatch/src/gpudispatch/py.typed`
- Create: `gpudispatch/README.md`
- Create: `gpudispatch/LICENSE`

**Step 1: Create gpudispatch directory**

```bash
mkdir -p gpudispatch/src/gpudispatch
mkdir -p gpudispatch/tests/unit
mkdir -p gpudispatch/tests/integration
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gpudispatch"
version = "0.1.0"
description = "Universal GPU orchestration - from laptop to supercomputer"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    { name = "Model-Zoo Team" }
]
keywords = ["gpu", "orchestration", "distributed", "machine-learning", "hpc"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "gpustat>=1.1.0",
    "pydantic>=2.0.0",
    "click>=8.0.0",
    "rich>=13.0.0",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
all = [
    "prometheus-client>=0.17.0",
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "optuna>=3.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
    "hypothesis>=6.0.0",
]
slurm = []
kubernetes = ["kubernetes>=28.0.0"]
aws = ["boto3>=1.28.0"]
gcp = ["google-cloud-compute>=1.14.0"]

[project.scripts]
gpudispatch = "gpudispatch.cli.main:main"

[project.urls]
Homepage = "https://github.com/model-zoo/gpudispatch"
Documentation = "https://gpudispatch.readthedocs.io"
Repository = "https://github.com/model-zoo/gpudispatch"

[tool.hatch.build.targets.wheel]
packages = ["src/gpudispatch"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 3: Create __init__.py with public API**

```python
"""gpudispatch: Universal GPU orchestration - from laptop to supercomputer."""

__version__ = "0.1.0"

from gpudispatch.core.dispatcher import Dispatcher
from gpudispatch.core.job import Job, JobStatus
from gpudispatch.core.resources import GPU, Resource
from gpudispatch.decorators import gpu

__all__ = [
    "Dispatcher",
    "Job",
    "JobStatus",
    "GPU",
    "Resource",
    "gpu",
    "__version__",
]
```

**Step 4: Create py.typed marker and LICENSE**

```bash
touch gpudispatch/src/gpudispatch/py.typed
```

LICENSE (Apache 2.0):
```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[Full Apache 2.0 license text]
```

**Step 5: Commit**

```bash
cd gpudispatch
git init
git add .
git commit -m "chore: initialize gpudispatch package structure"
```

---

### Task 1.2: Core Resource Abstractions

**Files:**
- Create: `gpudispatch/src/gpudispatch/core/__init__.py`
- Create: `gpudispatch/src/gpudispatch/core/resources.py`
- Create: `gpudispatch/tests/unit/core/__init__.py`
- Create: `gpudispatch/tests/unit/core/test_resources.py`

**Step 1: Write failing tests for Resource classes**

```python
# tests/unit/core/test_resources.py
"""Tests for resource abstractions."""

import pytest
from gpudispatch.core.resources import GPU, Memory, Resource, ResourceRequirements


class TestGPU:
    def test_gpu_creation_with_index(self):
        gpu = GPU(index=0)
        assert gpu.index == 0
        assert gpu.memory is None

    def test_gpu_creation_with_memory(self):
        gpu = GPU(index=1, memory="16GB")
        assert gpu.index == 1
        assert gpu.memory == 16 * 1024  # Stored in MB

    def test_gpu_memory_parsing_gb(self):
        gpu = GPU(index=0, memory="24GB")
        assert gpu.memory == 24 * 1024

    def test_gpu_memory_parsing_mb(self):
        gpu = GPU(index=0, memory="8192MB")
        assert gpu.memory == 8192

    def test_gpu_equality(self):
        gpu1 = GPU(index=0)
        gpu2 = GPU(index=0)
        gpu3 = GPU(index=1)
        assert gpu1 == gpu2
        assert gpu1 != gpu3

    def test_gpu_hash(self):
        gpu1 = GPU(index=0)
        gpu2 = GPU(index=0)
        assert hash(gpu1) == hash(gpu2)
        gpu_set = {gpu1, gpu2}
        assert len(gpu_set) == 1


class TestMemory:
    def test_memory_from_string_gb(self):
        mem = Memory.from_string("16GB")
        assert mem.mb == 16 * 1024

    def test_memory_from_string_mb(self):
        mem = Memory.from_string("4096MB")
        assert mem.mb == 4096

    def test_memory_from_int_mb(self):
        mem = Memory(mb=8192)
        assert mem.mb == 8192

    def test_memory_comparison(self):
        mem1 = Memory(mb=8192)
        mem2 = Memory(mb=16384)
        assert mem1 < mem2
        assert mem2 > mem1

    def test_memory_str(self):
        mem = Memory(mb=16384)
        assert str(mem) == "16.0GB"


class TestResourceRequirements:
    def test_requirements_simple(self):
        req = ResourceRequirements(gpu=1)
        assert req.gpu_count == 1
        assert req.memory is None

    def test_requirements_with_memory(self):
        req = ResourceRequirements(gpu=2, memory="32GB")
        assert req.gpu_count == 2
        assert req.memory.mb == 32 * 1024

    def test_requirements_validation_negative_gpu(self):
        with pytest.raises(ValueError, match="gpu_count must be non-negative"):
            ResourceRequirements(gpu=-1)

    def test_requirements_satisfies(self):
        req = ResourceRequirements(gpu=2, memory="16GB")
        available = [GPU(0, memory="24GB"), GPU(1, memory="24GB")]
        assert req.satisfies(available)

    def test_requirements_not_satisfies_count(self):
        req = ResourceRequirements(gpu=3)
        available = [GPU(0), GPU(1)]
        assert not req.satisfies(available)
```

**Step 2: Run tests to verify they fail**

```bash
cd gpudispatch
pytest tests/unit/core/test_resources.py -v
```

Expected: FAIL with import errors

**Step 3: Implement Resource classes**

```python
# src/gpudispatch/core/__init__.py
"""Core gpudispatch components."""

from gpudispatch.core.resources import GPU, Memory, Resource, ResourceRequirements

__all__ = ["GPU", "Memory", "Resource", "ResourceRequirements"]
```

```python
# src/gpudispatch/core/resources.py
"""Resource abstractions for GPU and memory management."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union


@dataclass(frozen=True)
class Memory:
    """Memory specification in megabytes."""

    mb: int

    @classmethod
    def from_string(cls, value: str) -> Memory:
        """Parse memory string like '16GB' or '4096MB'."""
        match = re.match(r"^(\d+(?:\.\d+)?)\s*(GB|MB|gb|mb)$", value.strip())
        if not match:
            raise ValueError(f"Invalid memory format: {value}. Use '16GB' or '4096MB'.")

        amount = float(match.group(1))
        unit = match.group(2).upper()

        if unit == "GB":
            return cls(mb=int(amount * 1024))
        return cls(mb=int(amount))

    def __str__(self) -> str:
        if self.mb >= 1024:
            return f"{self.mb / 1024:.1f}GB"
        return f"{self.mb}MB"

    def __lt__(self, other: Memory) -> bool:
        return self.mb < other.mb

    def __le__(self, other: Memory) -> bool:
        return self.mb <= other.mb

    def __gt__(self, other: Memory) -> bool:
        return self.mb > other.mb

    def __ge__(self, other: Memory) -> bool:
        return self.mb >= other.mb


@dataclass
class Resource:
    """Base class for compute resources."""
    pass


@dataclass(frozen=True)
class GPU(Resource):
    """GPU resource specification."""

    index: int
    memory: Optional[int] = None  # Memory in MB

    def __init__(self, index: int, memory: Optional[Union[str, int]] = None):
        object.__setattr__(self, 'index', index)

        if memory is None:
            object.__setattr__(self, 'memory', None)
        elif isinstance(memory, str):
            parsed = Memory.from_string(memory)
            object.__setattr__(self, 'memory', parsed.mb)
        else:
            object.__setattr__(self, 'memory', memory)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GPU):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)

    def __str__(self) -> str:
        if self.memory:
            return f"GPU({self.index}, {Memory(self.memory)})"
        return f"GPU({self.index})"


@dataclass
class ResourceRequirements:
    """Resource requirements for a job."""

    gpu_count: int = 0
    memory: Optional[Memory] = None

    def __init__(
        self,
        gpu: int = 0,
        memory: Optional[Union[str, Memory]] = None,
    ):
        if gpu < 0:
            raise ValueError("gpu_count must be non-negative")

        self.gpu_count = gpu

        if memory is None:
            self.memory = None
        elif isinstance(memory, str):
            self.memory = Memory.from_string(memory)
        else:
            self.memory = memory

    def satisfies(self, available: Sequence[GPU]) -> bool:
        """Check if available resources satisfy these requirements."""
        if len(available) < self.gpu_count:
            return False

        if self.memory is not None:
            for gpu in available[:self.gpu_count]:
                if gpu.memory is not None and gpu.memory < self.memory.mb:
                    return False

        return True
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/core/test_resources.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gpudispatch/core/ tests/unit/core/
git commit -m "feat(core): add resource abstractions (GPU, Memory, ResourceRequirements)"
```

---

### Task 1.3: Job and JobStatus

**Files:**
- Create: `gpudispatch/src/gpudispatch/core/job.py`
- Create: `gpudispatch/tests/unit/core/test_job.py`

**Step 1: Write failing tests for Job**

```python
# tests/unit/core/test_job.py
"""Tests for Job and JobStatus."""

import pytest
from datetime import datetime, timedelta
from gpudispatch.core.job import Job, JobStatus, JobResult


class TestJobStatus:
    def test_status_values(self):
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_status_is_terminal(self):
        assert not JobStatus.PENDING.is_terminal
        assert not JobStatus.QUEUED.is_terminal
        assert not JobStatus.RUNNING.is_terminal
        assert JobStatus.COMPLETED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELLED.is_terminal


class TestJob:
    def test_job_creation_minimal(self):
        def my_func():
            return 42

        job = Job(fn=my_func)
        assert job.fn == my_func
        assert job.status == JobStatus.PENDING
        assert job.id is not None
        assert job.gpu_count == 1  # Default

    def test_job_creation_with_args(self):
        def add(a, b):
            return a + b

        job = Job(fn=add, args=(1, 2))
        assert job.args == (1, 2)
        assert job.kwargs == {}

    def test_job_creation_with_kwargs(self):
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        job = Job(fn=greet, kwargs={"name": "World", "greeting": "Hi"})
        assert job.kwargs == {"name": "World", "greeting": "Hi"}

    def test_job_creation_with_resources(self):
        job = Job(fn=lambda: None, gpu=2, memory="16GB", priority=10)
        assert job.gpu_count == 2
        assert job.memory.mb == 16 * 1024
        assert job.priority == 10

    def test_job_id_unique(self):
        job1 = Job(fn=lambda: None)
        job2 = Job(fn=lambda: None)
        assert job1.id != job2.id

    def test_job_name_auto_generated(self):
        def my_training_function():
            pass

        job = Job(fn=my_training_function)
        assert job.name == "my_training_function"

    def test_job_name_explicit(self):
        job = Job(fn=lambda: None, name="custom_name")
        assert job.name == "custom_name"

    def test_job_status_transitions(self):
        job = Job(fn=lambda: None)
        assert job.status == JobStatus.PENDING

        job.status = JobStatus.QUEUED
        assert job.status == JobStatus.QUEUED

        job.status = JobStatus.RUNNING
        assert job.status == JobStatus.RUNNING

    def test_job_dependencies(self):
        job1 = Job(fn=lambda: 1, name="job1")
        job2 = Job(fn=lambda: 2, name="job2")
        job3 = Job(fn=lambda: 3, name="job3", after=[job1, job2])

        assert len(job3.dependencies) == 2
        assert job1.id in job3.dependencies
        assert job2.id in job3.dependencies

    def test_job_can_run_no_deps(self):
        job = Job(fn=lambda: None)
        completed_jobs: set[str] = set()
        assert job.can_run(completed_jobs)

    def test_job_can_run_deps_met(self):
        job1 = Job(fn=lambda: 1)
        job2 = Job(fn=lambda: 2, after=[job1])

        completed_jobs = {job1.id}
        assert job2.can_run(completed_jobs)

    def test_job_cannot_run_deps_not_met(self):
        job1 = Job(fn=lambda: 1)
        job2 = Job(fn=lambda: 2, after=[job1])

        completed_jobs: set[str] = set()
        assert not job2.can_run(completed_jobs)


class TestJobResult:
    def test_result_success(self):
        result = JobResult(value=42, status=JobStatus.COMPLETED)
        assert result.value == 42
        assert result.is_success
        assert result.error is None

    def test_result_failure(self):
        result = JobResult(
            value=None,
            status=JobStatus.FAILED,
            error="Something went wrong"
        )
        assert not result.is_success
        assert result.error == "Something went wrong"

    def test_result_runtime(self):
        start = datetime.now()
        end = start + timedelta(seconds=60)
        result = JobResult(
            value=42,
            status=JobStatus.COMPLETED,
            start_time=start,
            end_time=end
        )
        assert result.runtime_seconds == 60.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/core/test_job.py -v
```

Expected: FAIL with import errors

**Step 3: Implement Job classes**

```python
# src/gpudispatch/core/job.py
"""Job representation and lifecycle management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Sequence

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
        memory: Optional[str | Memory] = None,
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/core/test_job.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gpudispatch/core/job.py tests/unit/core/test_job.py
git commit -m "feat(core): add Job and JobStatus with dependency support"
```

---

### Task 1.4: Job Queue Implementation

**Files:**
- Create: `gpudispatch/src/gpudispatch/core/queue.py`
- Create: `gpudispatch/tests/unit/core/test_queue.py`

**Step 1: Write failing tests for JobQueue**

```python
# tests/unit/core/test_queue.py
"""Tests for job queue implementations."""

import pytest
from gpudispatch.core.job import Job, JobStatus
from gpudispatch.core.queue import FIFOQueue, PriorityQueue, JobQueue


class TestFIFOQueue:
    def test_empty_queue(self):
        queue = FIFOQueue()
        assert len(queue) == 0
        assert queue.empty()

    def test_put_and_get(self):
        queue = FIFOQueue()
        job = Job(fn=lambda: 42)

        queue.put(job)
        assert len(queue) == 1

        retrieved = queue.get()
        assert retrieved.id == job.id
        assert len(queue) == 0

    def test_fifo_order(self):
        queue = FIFOQueue()
        job1 = Job(fn=lambda: 1, name="first")
        job2 = Job(fn=lambda: 2, name="second")
        job3 = Job(fn=lambda: 3, name="third")

        queue.put(job1)
        queue.put(job2)
        queue.put(job3)

        assert queue.get().name == "first"
        assert queue.get().name == "second"
        assert queue.get().name == "third"

    def test_peek(self):
        queue = FIFOQueue()
        job = Job(fn=lambda: 42)
        queue.put(job)

        peeked = queue.peek()
        assert peeked.id == job.id
        assert len(queue) == 1  # Not removed

    def test_peek_empty(self):
        queue = FIFOQueue()
        assert queue.peek() is None

    def test_remove(self):
        queue = FIFOQueue()
        job1 = Job(fn=lambda: 1)
        job2 = Job(fn=lambda: 2)

        queue.put(job1)
        queue.put(job2)

        removed = queue.remove(job1.id)
        assert removed is True
        assert len(queue) == 1
        assert queue.get().id == job2.id

    def test_remove_not_found(self):
        queue = FIFOQueue()
        removed = queue.remove("nonexistent")
        assert removed is False

    def test_iter(self):
        queue = FIFOQueue()
        jobs = [Job(fn=lambda: i) for i in range(3)]
        for job in jobs:
            queue.put(job)

        queue_jobs = list(queue)
        assert len(queue_jobs) == 3


class TestPriorityQueue:
    def test_priority_order(self):
        queue = PriorityQueue()
        low = Job(fn=lambda: 1, name="low", priority=1)
        high = Job(fn=lambda: 2, name="high", priority=10)
        medium = Job(fn=lambda: 3, name="medium", priority=5)

        queue.put(low)
        queue.put(high)
        queue.put(medium)

        # Higher priority first
        assert queue.get().name == "high"
        assert queue.get().name == "medium"
        assert queue.get().name == "low"

    def test_same_priority_fifo(self):
        queue = PriorityQueue()
        job1 = Job(fn=lambda: 1, name="first", priority=5)
        job2 = Job(fn=lambda: 2, name="second", priority=5)

        queue.put(job1)
        queue.put(job2)

        # Same priority: FIFO order
        assert queue.get().name == "first"
        assert queue.get().name == "second"

    def test_update_priority(self):
        queue = PriorityQueue()
        job = Job(fn=lambda: 1, priority=1)
        queue.put(job)

        queue.update_priority(job.id, 100)

        peeked = queue.peek()
        assert peeked.priority == 100


class TestJobQueueInterface:
    @pytest.mark.parametrize("QueueClass", [FIFOQueue, PriorityQueue])
    def test_queue_interface(self, QueueClass):
        """Test that all queue implementations follow the interface."""
        queue: JobQueue = QueueClass()

        # All queues should have these methods
        assert hasattr(queue, 'put')
        assert hasattr(queue, 'get')
        assert hasattr(queue, 'peek')
        assert hasattr(queue, 'remove')
        assert hasattr(queue, 'empty')
        assert hasattr(queue, '__len__')
        assert hasattr(queue, '__iter__')
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/core/test_queue.py -v
```

Expected: FAIL with import errors

**Step 3: Implement queue classes**

```python
# src/gpudispatch/core/queue.py
"""Job queue implementations."""

from __future__ import annotations

import heapq
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Optional

from gpudispatch.core.job import Job


class JobQueue(ABC):
    """Abstract base class for job queues."""

    @abstractmethod
    def put(self, job: Job) -> None:
        """Add a job to the queue."""
        pass

    @abstractmethod
    def get(self) -> Optional[Job]:
        """Remove and return the next job, or None if empty."""
        pass

    @abstractmethod
    def peek(self) -> Optional[Job]:
        """Return the next job without removing it, or None if empty."""
        pass

    @abstractmethod
    def remove(self, job_id: str) -> bool:
        """Remove a job by ID. Returns True if found and removed."""
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Check if queue is empty."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of jobs in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Job]:
        """Iterate over jobs in queue order."""
        pass


class FIFOQueue(JobQueue):
    """First-in-first-out job queue."""

    def __init__(self) -> None:
        self._queue: deque[Job] = deque()
        self._lock = threading.Lock()

    def put(self, job: Job) -> None:
        with self._lock:
            self._queue.append(job)

    def get(self) -> Optional[Job]:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def peek(self) -> Optional[Job]:
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def remove(self, job_id: str) -> bool:
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    del self._queue[i]
                    return True
            return False

    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def __iter__(self) -> Iterator[Job]:
        with self._lock:
            return iter(list(self._queue))


@dataclass(order=True)
class _PriorityItem:
    """Wrapper for heap queue ordering."""
    priority: int
    sequence: int
    job: Job = field(compare=False)


class PriorityQueue(JobQueue):
    """Priority-based job queue. Higher priority values are dequeued first."""

    def __init__(self) -> None:
        self._heap: list[_PriorityItem] = []
        self._sequence = 0
        self._lock = threading.Lock()
        self._job_map: dict[str, _PriorityItem] = {}

    def put(self, job: Job) -> None:
        with self._lock:
            # Negate priority for max-heap behavior (heapq is min-heap)
            item = _PriorityItem(
                priority=-job.priority,
                sequence=self._sequence,
                job=job
            )
            self._sequence += 1
            heapq.heappush(self._heap, item)
            self._job_map[job.id] = item

    def get(self) -> Optional[Job]:
        with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                if item.job.id in self._job_map:
                    del self._job_map[item.job.id]
                    return item.job
            return None

    def peek(self) -> Optional[Job]:
        with self._lock:
            while self._heap:
                if self._heap[0].job.id in self._job_map:
                    return self._heap[0].job
                heapq.heappop(self._heap)
            return None

    def remove(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._job_map:
                del self._job_map[job_id]
                return True
            return False

    def update_priority(self, job_id: str, new_priority: int) -> bool:
        """Update priority of a job in the queue."""
        with self._lock:
            if job_id not in self._job_map:
                return False

            old_item = self._job_map[job_id]
            job = old_item.job
            job.priority = new_priority

            # Remove old entry (lazy deletion)
            del self._job_map[job_id]

            # Add with new priority
            new_item = _PriorityItem(
                priority=-new_priority,
                sequence=self._sequence,
                job=job
            )
            self._sequence += 1
            heapq.heappush(self._heap, new_item)
            self._job_map[job_id] = new_item
            return True

    def empty(self) -> bool:
        with self._lock:
            return len(self._job_map) == 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._job_map)

    def __iter__(self) -> Iterator[Job]:
        with self._lock:
            # Return jobs sorted by priority
            items = sorted(
                [item for item in self._heap if item.job.id in self._job_map],
                key=lambda x: (x.priority, x.sequence)
            )
            return iter([item.job for item in items])
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/core/test_queue.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gpudispatch/core/queue.py tests/unit/core/test_queue.py
git commit -m "feat(core): add FIFO and Priority job queue implementations"
```

---

### Task 1.5: GPU Detection Utility

**Files:**
- Create: `gpudispatch/src/gpudispatch/utils/__init__.py`
- Create: `gpudispatch/src/gpudispatch/utils/gpu.py`
- Create: `gpudispatch/tests/unit/utils/__init__.py`
- Create: `gpudispatch/tests/unit/utils/test_gpu.py`

**Step 1: Write failing tests for GPU detection**

```python
# tests/unit/utils/test_gpu.py
"""Tests for GPU detection utilities."""

import pytest
from unittest.mock import patch, MagicMock
from gpudispatch.utils.gpu import (
    detect_gpus,
    get_gpu_memory_usage,
    is_gpu_available,
    GPUInfo,
)


class TestGPUInfo:
    def test_gpu_info_creation(self):
        info = GPUInfo(
            index=0,
            name="NVIDIA A100",
            memory_total_mb=81920,
            memory_used_mb=1024,
            utilization_percent=45,
        )
        assert info.index == 0
        assert info.name == "NVIDIA A100"
        assert info.memory_free_mb == 81920 - 1024

    def test_gpu_info_memory_free(self):
        info = GPUInfo(
            index=0,
            name="Test GPU",
            memory_total_mb=16000,
            memory_used_mb=4000,
            utilization_percent=0,
        )
        assert info.memory_free_mb == 12000


class TestDetectGPUs:
    @patch('gpudispatch.utils.gpu.gpustat')
    def test_detect_gpus_success(self, mock_gpustat):
        # Mock gpustat response
        mock_gpu = MagicMock()
        mock_gpu.index = 0
        mock_gpu.name = "NVIDIA A100"
        mock_gpu.memory_total = 81920
        mock_gpu.memory_used = 1024
        mock_gpu.utilization = 45

        mock_collection = MagicMock()
        mock_collection.gpus = [mock_gpu]
        mock_gpustat.GPUStatCollection.new_query.return_value = mock_collection

        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA A100"

    @patch('gpudispatch.utils.gpu.gpustat')
    def test_detect_gpus_empty(self, mock_gpustat):
        mock_collection = MagicMock()
        mock_collection.gpus = []
        mock_gpustat.GPUStatCollection.new_query.return_value = mock_collection

        gpus = detect_gpus()
        assert len(gpus) == 0

    @patch('gpudispatch.utils.gpu.gpustat')
    def test_detect_gpus_error(self, mock_gpustat):
        mock_gpustat.GPUStatCollection.new_query.side_effect = Exception("No GPU")

        gpus = detect_gpus()
        assert len(gpus) == 0


class TestIsGPUAvailable:
    @patch('gpudispatch.utils.gpu.detect_gpus')
    def test_gpu_available_below_threshold(self, mock_detect):
        mock_detect.return_value = [
            GPUInfo(0, "GPU0", 16000, 400, 0),  # 400MB used < 500MB threshold
        ]

        assert is_gpu_available(0, memory_threshold_mb=500)

    @patch('gpudispatch.utils.gpu.detect_gpus')
    def test_gpu_not_available_above_threshold(self, mock_detect):
        mock_detect.return_value = [
            GPUInfo(0, "GPU0", 16000, 8000, 50),  # 8000MB used > 500MB threshold
        ]

        assert not is_gpu_available(0, memory_threshold_mb=500)

    @patch('gpudispatch.utils.gpu.detect_gpus')
    def test_gpu_not_found(self, mock_detect):
        mock_detect.return_value = [
            GPUInfo(0, "GPU0", 16000, 0, 0),
        ]

        assert not is_gpu_available(1)  # GPU 1 doesn't exist
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/utils/test_gpu.py -v
```

Expected: FAIL with import errors

**Step 3: Implement GPU utilities**

```python
# src/gpudispatch/utils/__init__.py
"""Utility modules for gpudispatch."""

from gpudispatch.utils.gpu import detect_gpus, get_gpu_memory_usage, is_gpu_available, GPUInfo

__all__ = ["detect_gpus", "get_gpu_memory_usage", "is_gpu_available", "GPUInfo"]
```

```python
# src/gpudispatch/utils/gpu.py
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


def wait_for_gpu(
    gpu_indices: list[int],
    memory_threshold_mb: int = 500,
    check_interval_seconds: float = 5.0,
    max_checks: int = 10,
) -> list[int]:
    """Wait for GPUs to become available.

    Args:
        gpu_indices: List of GPU indices to check.
        memory_threshold_mb: Memory threshold for availability.
        check_interval_seconds: Time between checks.
        max_checks: Maximum number of checks before returning.

    Returns:
        List of available GPU indices.
    """
    import time

    availability_count: dict[int, int] = {idx: 0 for idx in gpu_indices}

    for _ in range(max_checks):
        gpus = detect_gpus()
        gpu_map = {g.index: g for g in gpus}

        available = []
        for idx in gpu_indices:
            if idx in gpu_map and gpu_map[idx].memory_used_mb < memory_threshold_mb:
                availability_count[idx] += 1
                if availability_count[idx] >= max_checks:
                    available.append(idx)
            else:
                availability_count[idx] = 0

        if available:
            return available

        time.sleep(check_interval_seconds)

    return []
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/utils/test_gpu.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gpudispatch/utils/ tests/unit/utils/
git commit -m "feat(utils): add GPU detection and monitoring utilities"
```

---

### Task 1.6: Core Dispatcher

**Files:**
- Create: `gpudispatch/src/gpudispatch/core/dispatcher.py`
- Create: `gpudispatch/tests/unit/core/test_dispatcher.py`

**Step 1: Write failing tests for Dispatcher**

```python
# tests/unit/core/test_dispatcher.py
"""Tests for the core Dispatcher class."""

import pytest
from unittest.mock import patch, MagicMock
import time

from gpudispatch.core.dispatcher import Dispatcher
from gpudispatch.core.job import Job, JobStatus
from gpudispatch.core.resources import GPU


class TestDispatcherCreation:
    def test_dispatcher_default_creation(self):
        dispatcher = Dispatcher()
        assert dispatcher is not None
        assert dispatcher.is_running is False

    def test_dispatcher_with_gpus(self):
        dispatcher = Dispatcher(gpus=[0, 1, 2])
        assert len(dispatcher.available_gpus) == 3

    def test_dispatcher_with_memory_threshold(self):
        dispatcher = Dispatcher(memory_threshold="1GB")
        assert dispatcher.memory_threshold_mb == 1024


class TestDispatcherSubmit:
    def test_submit_function(self):
        dispatcher = Dispatcher(gpus=[0])

        def my_func():
            return 42

        job = dispatcher.submit(my_func)
        assert isinstance(job, Job)
        assert job.status == JobStatus.QUEUED

    def test_submit_with_args(self):
        dispatcher = Dispatcher(gpus=[0])

        def add(a, b):
            return a + b

        job = dispatcher.submit(add, args=(1, 2))
        assert job.args == (1, 2)

    def test_submit_with_resources(self):
        dispatcher = Dispatcher(gpus=[0, 1])

        job = dispatcher.submit(lambda: None, gpu=2, memory="16GB", priority=10)
        assert job.gpu_count == 2
        assert job.memory.mb == 16 * 1024
        assert job.priority == 10

    def test_submit_with_dependencies(self):
        dispatcher = Dispatcher(gpus=[0])

        job1 = dispatcher.submit(lambda: 1)
        job2 = dispatcher.submit(lambda: 2, after=[job1])

        assert job1.id in job2.dependencies


class TestDispatcherCancel:
    def test_cancel_queued_job(self):
        dispatcher = Dispatcher(gpus=[0])
        job = dispatcher.submit(lambda: 42)

        result = dispatcher.cancel(job.id)
        assert result is True
        assert job.status == JobStatus.CANCELLED

    def test_cancel_nonexistent_job(self):
        dispatcher = Dispatcher(gpus=[0])
        result = dispatcher.cancel("nonexistent")
        assert result is False


class TestDispatcherStats:
    def test_stats_empty(self):
        dispatcher = Dispatcher(gpus=[0, 1])
        stats = dispatcher.stats()

        assert stats.jobs_queued == 0
        assert stats.jobs_running == 0
        assert stats.jobs_completed == 0

    def test_stats_with_jobs(self):
        dispatcher = Dispatcher(gpus=[0])
        dispatcher.submit(lambda: 1)
        dispatcher.submit(lambda: 2)

        stats = dispatcher.stats()
        assert stats.jobs_queued == 2


class TestDispatcherLifecycle:
    def test_start_and_shutdown(self):
        dispatcher = Dispatcher(gpus=[0])

        dispatcher.start()
        assert dispatcher.is_running is True

        dispatcher.shutdown()
        assert dispatcher.is_running is False

    def test_context_manager(self):
        with Dispatcher(gpus=[0]) as dispatcher:
            assert dispatcher.is_running is True

        assert dispatcher.is_running is False
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/core/test_dispatcher.py -v
```

Expected: FAIL with import errors

**Step 3: Implement Dispatcher**

```python
# src/gpudispatch/core/dispatcher.py
"""Core Dispatcher for GPU job orchestration."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

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

    Manages job queue, GPU resources, and job execution.

    Example:
        >>> dispatcher = Dispatcher(gpus=[0, 1, 2, 3])
        >>> job = dispatcher.submit(train_model, gpu=2, memory="16GB")
        >>> result = dispatcher.wait(job)
    """

    def __init__(
        self,
        gpus: Optional[list[int]] = None,
        memory_threshold: str | int = "500MB",
        queue: Optional[JobQueue] = None,
        polling_interval: float = 5.0,
    ):
        """Initialize dispatcher.

        Args:
            gpus: List of GPU indices to use. Auto-detects if None.
            memory_threshold: Consider GPU free if memory below this.
            queue: Job queue implementation. Defaults to PriorityQueue.
            polling_interval: Seconds between GPU availability checks.
        """
        # Parse memory threshold
        if isinstance(memory_threshold, str):
            self.memory_threshold_mb = Memory.from_string(memory_threshold).mb
        else:
            self.memory_threshold_mb = memory_threshold

        # Initialize GPU list
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

        # Shutdown events
        self._shutdown_event = threading.Event()
        self._drain_event = threading.Event()

    @property
    def is_running(self) -> bool:
        """Check if dispatcher is running."""
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
        """Submit a job for execution.

        Args:
            fn: Function to execute.
            args: Positional arguments for fn.
            kwargs: Keyword arguments for fn.
            gpu: Number of GPUs required.
            memory: Memory requirement (e.g., "16GB").
            priority: Job priority (higher = sooner).
            name: Optional job name.
            after: Jobs that must complete before this one.

        Returns:
            Submitted Job object.
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
            self._queue.put(job)

        logger.info(f"Job {job.id} ({job.name}) submitted to queue")
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: ID of job to cancel.

        Returns:
            True if job was found and cancelled.
        """
        with self._lock:
            if job_id not in self._jobs:
                return False

            job = self._jobs[job_id]

            if job.status.is_terminal:
                return False

            self._queue.remove(job_id)
            job.status = JobStatus.CANCELLED

            logger.info(f"Job {job_id} cancelled")
            return True

    def stats(self) -> DispatcherStats:
        """Get current dispatcher statistics."""
        with self._lock:
            queued = len(self._queue)
            running = len(self._running_jobs)
            completed = sum(
                1 for j in self._jobs.values()
                if j.status == JobStatus.COMPLETED
            )
            failed = sum(
                1 for j in self._jobs.values()
                if j.status == JobStatus.FAILED
            )

            available = sum(
                1 for idx in self._gpu_indices
                if idx not in self._occupied_gpus
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
        """Start the dispatcher."""
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
            wait: If True, wait for current jobs to complete.
        """
        if not self._running:
            return

        self._shutdown_event.set()

        if wait and self._dispatch_thread:
            self._dispatch_thread.join(timeout=30)

        self._running = False
        logger.info("Dispatcher shutdown")

    def drain(self) -> None:
        """Stop accepting new jobs, finish current ones."""
        self._drain_event.set()
        logger.info("Dispatcher entering drain mode")

    def _dispatch_loop(self) -> None:
        """Main dispatch loop."""
        while not self._shutdown_event.is_set():
            try:
                self._process_queue()
            except Exception as e:
                logger.exception(f"Error in dispatch loop: {e}")

            time.sleep(self._polling_interval)

    def _process_queue(self) -> None:
        """Process jobs from queue."""
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

            # Dequeue and start job
            self._queue.get()
            self._start_job(job, available)

    def _find_available_gpus(self, count: int) -> list[int]:
        """Find available GPUs."""
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
        """Start executing a job."""
        job.status = JobStatus.RUNNING
        self._running_jobs[job.id] = job

        for gpu in gpus:
            self._occupied_gpus.add(gpu)

        logger.info(f"Starting job {job.id} on GPUs {gpus}")

        # Execute in thread
        thread = threading.Thread(
            target=self._execute_job,
            args=(job, gpus),
            daemon=True,
        )
        thread.start()

    def _execute_job(self, job: Job, gpus: list[int]) -> None:
        """Execute a job."""
        import os
        from datetime import datetime

        job.started_at = datetime.now()

        # Set CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

        try:
            result = job.fn(*job.args, **job.kwargs)
            job.result = result
            job.status = JobStatus.COMPLETED
            logger.info(f"Job {job.id} completed successfully")

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
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/core/test_dispatcher.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gpudispatch/core/dispatcher.py tests/unit/core/test_dispatcher.py
git commit -m "feat(core): add Dispatcher with job submission and GPU management"
```

---

### Task 1.7: @gpu Decorator

**Files:**
- Create: `gpudispatch/src/gpudispatch/decorators.py`
- Create: `gpudispatch/tests/unit/test_decorators.py`

**Step 1: Write failing tests for @gpu decorator**

```python
# tests/unit/test_decorators.py
"""Tests for the @gpu decorator."""

import pytest
from unittest.mock import patch, MagicMock

from gpudispatch.decorators import gpu


class TestGPUDecorator:
    def test_decorator_basic(self):
        @gpu(count=1)
        def my_func():
            return 42

        # Function should still be callable
        assert callable(my_func)
        assert my_func.__name__ == "my_func"

    def test_decorator_preserves_docstring(self):
        @gpu(1)
        def documented_func():
            """This is my docstring."""
            return 42

        assert documented_func.__doc__ == "This is my docstring."

    def test_decorator_with_args(self):
        @gpu(2, memory="16GB")
        def train(lr, batch_size):
            return lr * batch_size

        # Check decorator metadata
        assert train._gpu_count == 2
        assert train._gpu_memory == "16GB"

    def test_decorator_shorthand(self):
        # Just @gpu(1) should work
        @gpu(1)
        def simple_func():
            return 1

        assert simple_func._gpu_count == 1

    @patch('gpudispatch.decorators._default_dispatcher')
    def test_decorator_execution(self, mock_dispatcher):
        mock_dispatcher.return_value = MagicMock()
        mock_dispatcher.return_value.submit_and_wait.return_value = 42

        @gpu(1)
        def compute():
            return 42

        # When called, should dispatch
        result = compute()
        # The actual dispatch happens only when a global dispatcher is set
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_decorators.py -v
```

Expected: FAIL with import errors

**Step 3: Implement @gpu decorator**

```python
# src/gpudispatch/decorators.py
"""Decorators for GPU job dispatch."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

# Global default dispatcher (set by user or auto-detected)
_default_dispatcher: Optional[Any] = None


def set_default_dispatcher(dispatcher: Any) -> None:
    """Set the default dispatcher for @gpu decorated functions."""
    global _default_dispatcher
    _default_dispatcher = dispatcher


def get_default_dispatcher() -> Optional[Any]:
    """Get the current default dispatcher."""
    return _default_dispatcher


@overload
def gpu(fn: F) -> F: ...


@overload
def gpu(
    count: int = 1,
    *,
    memory: Optional[str] = None,
    priority: int = 0,
) -> Callable[[F], F]: ...


def gpu(
    fn: Optional[F] = None,
    count: int = 1,
    *,
    memory: Optional[str] = None,
    priority: int = 0,
) -> F | Callable[[F], F]:
    """Decorator to mark a function for GPU execution.

    Can be used as:
        @gpu
        def func(): ...

        @gpu(2)
        def func(): ...

        @gpu(count=2, memory="16GB")
        def func(): ...

    Args:
        fn: Function to decorate (when used without parentheses).
        count: Number of GPUs required.
        memory: Memory requirement (e.g., "16GB").
        priority: Job priority (higher = sooner).

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            dispatcher = get_default_dispatcher()

            if dispatcher is None:
                # No dispatcher set, run locally
                import warnings
                warnings.warn(
                    "No dispatcher set. Running function locally. "
                    "Use set_default_dispatcher() or Dispatcher context manager.",
                    RuntimeWarning,
                )
                return func(*args, **kwargs)

            # Submit to dispatcher
            job = dispatcher.submit(
                func,
                args=args,
                kwargs=kwargs,
                gpu=count,
                memory=memory,
                priority=priority,
            )

            # Wait for completion
            return dispatcher.wait(job)

        # Attach metadata for introspection
        wrapper._gpu_count = count  # type: ignore
        wrapper._gpu_memory = memory  # type: ignore
        wrapper._gpu_priority = priority  # type: ignore
        wrapper._original_fn = func  # type: ignore

        return wrapper  # type: ignore

    # Handle @gpu vs @gpu() vs @gpu(2)
    if fn is not None:
        # Called as @gpu without parentheses
        return decorator(fn)

    if isinstance(count, int):
        # Called as @gpu(2) or @gpu(count=2, ...)
        return decorator

    # Called as @gpu() - count defaults to 1
    return decorator
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_decorators.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gpudispatch/decorators.py tests/unit/test_decorators.py
git commit -m "feat: add @gpu decorator for easy function dispatch"
```

---

## Phase 1 Complete Checkpoint

At this point, the core foundation is complete:

- [x] Package structure with pyproject.toml
- [x] Resource abstractions (GPU, Memory)
- [x] Job and JobStatus
- [x] Job queues (FIFO, Priority)
- [x] GPU detection utilities
- [x] Core Dispatcher
- [x] @gpu decorator

**Verify everything works:**

```bash
cd gpudispatch
pytest tests/ -v --cov=src/gpudispatch --cov-report=term-missing
mypy src/gpudispatch --strict
```

**Commit phase completion:**

```bash
git add -A
git commit -m "milestone: complete Phase 1 - Core Foundation"
git tag v0.1.0-alpha
```

---

## Phase 2: Local Backend & Signal Handling

### Task 2.1: Backend Interface

**Files:**
- Create: `gpudispatch/src/gpudispatch/backends/__init__.py`
- Create: `gpudispatch/src/gpudispatch/backends/base.py`
- Create: `gpudispatch/tests/unit/backends/__init__.py`
- Create: `gpudispatch/tests/unit/backends/test_base.py`

[Continue with backend implementation...]

---

## Phase 3-8: [Subsequent phases follow same pattern]

Each subsequent phase follows the same TDD pattern:
1. Write failing tests
2. Verify tests fail
3. Implement minimal code
4. Verify tests pass
5. Commit

Phases include:
- Phase 2: Local Backend & Signal Handling
- Phase 3: Experiment Primitives (Grid, Sweep)
- Phase 4: SLURM Backend
- Phase 5: Observability (Prometheus, logging)
- Phase 6: CLI Interface
- Phase 7: Advanced Features (MIG, Memory estimation)
- Phase 8: Documentation & Polish

---

## Integration with Model-Zoo

After Phase 1-2, integrate back into Model-Zoo:

### Task: Update Model-Zoo to use gpudispatch

**Files:**
- Modify: `esd_experiment/src/run_experiment.py`
- Modify: `esd_experiment/gputracker/` → deprecated, use gpudispatch

```python
# esd_experiment/src/run_experiment.py (refactored)

from gpudispatch import Dispatcher, Experiment

def run_esd_experiment(model_list: str, output_dir: str, gpus: list[int], **esd_params):
    models = load_model_list(model_list)

    with Dispatcher(gpus=gpus) as dispatcher:
        jobs = []
        for model in models:
            job = dispatcher.submit(
                analyze_model,
                kwargs={"model_id": model.model_id, "output_dir": output_dir, **esd_params},
                gpu=1,
                memory="auto",
            )
            jobs.append(job)

        results = dispatcher.gather(jobs)

    return aggregate_results(results, output_dir)
```

---

*Plan complete. Ready for execution.*
