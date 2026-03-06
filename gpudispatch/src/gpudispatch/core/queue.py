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
        """Remove and return the next job from the queue."""
        pass

    @abstractmethod
    def peek(self) -> Optional[Job]:
        """Return the next job without removing it."""
        pass

    @abstractmethod
    def remove(self, job_id: str) -> bool:
        """Remove a specific job by ID. Returns True if found and removed."""
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Check if the queue is empty."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of jobs in the queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Job]:
        """Iterate over jobs in the queue."""
        pass


class FIFOQueue(JobQueue):
    """First-in-first-out job queue.

    Jobs are retrieved in the order they were added.
    Thread-safe implementation.
    """

    def __init__(self) -> None:
        self._queue: deque[Job] = deque()
        self._lock = threading.Lock()

    def put(self, job: Job) -> None:
        """Add a job to the end of the queue."""
        with self._lock:
            self._queue.append(job)

    def get(self) -> Optional[Job]:
        """Remove and return the first job in the queue."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def peek(self) -> Optional[Job]:
        """Return the first job without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None

    def remove(self, job_id: str) -> bool:
        """Remove a specific job by ID."""
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    del self._queue[i]
                    return True
            return False

    def empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def __len__(self) -> int:
        """Return the number of jobs in the queue."""
        with self._lock:
            return len(self._queue)

    def __iter__(self) -> Iterator[Job]:
        """Iterate over jobs in FIFO order."""
        with self._lock:
            return iter(list(self._queue))


@dataclass(order=True)
class _PriorityItem:
    """Internal wrapper for heap items with priority ordering."""
    priority: int
    sequence: int
    job: Job = field(compare=False)


class PriorityQueue(JobQueue):
    """Priority-based job queue.

    Jobs with higher priority values are retrieved first.
    For jobs with equal priority, FIFO order is maintained.
    Thread-safe implementation.
    """

    def __init__(self) -> None:
        self._heap: list[_PriorityItem] = []
        self._sequence = 0
        self._lock = threading.Lock()
        self._job_map: dict[str, _PriorityItem] = {}

    def put(self, job: Job) -> None:
        """Add a job to the queue based on its priority."""
        with self._lock:
            item = _PriorityItem(
                priority=-job.priority,  # Negate for max-heap behavior
                sequence=self._sequence,
                job=job
            )
            self._sequence += 1
            heapq.heappush(self._heap, item)
            self._job_map[job.id] = item

    def get(self) -> Optional[Job]:
        """Remove and return the highest priority job."""
        with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                if item.job.id in self._job_map:
                    del self._job_map[item.job.id]
                    return item.job
            return None

    def peek(self) -> Optional[Job]:
        """Return the highest priority job without removing it."""
        with self._lock:
            while self._heap:
                if self._heap[0].job.id in self._job_map:
                    return self._heap[0].job
                heapq.heappop(self._heap)
            return None

    def remove(self, job_id: str) -> bool:
        """Remove a specific job by ID (lazy deletion)."""
        with self._lock:
            if job_id in self._job_map:
                del self._job_map[job_id]
                return True
            return False

    def update_priority(self, job_id: str, new_priority: int) -> bool:
        """Update the priority of an existing job.

        Args:
            job_id: The ID of the job to update.
            new_priority: The new priority value.

        Returns:
            True if the job was found and updated, False otherwise.
        """
        with self._lock:
            if job_id not in self._job_map:
                return False
            old_item = self._job_map[job_id]
            job = old_item.job
            job.priority = new_priority
            del self._job_map[job_id]
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
        """Check if the queue is empty."""
        with self._lock:
            return len(self._job_map) == 0

    def __len__(self) -> int:
        """Return the number of jobs in the queue."""
        with self._lock:
            return len(self._job_map)

    def __iter__(self) -> Iterator[Job]:
        """Iterate over jobs in priority order (highest first)."""
        with self._lock:
            items = sorted(
                [item for item in self._heap if item.job.id in self._job_map],
                key=lambda x: (x.priority, x.sequence)
            )
            return iter([item.job for item in items])
