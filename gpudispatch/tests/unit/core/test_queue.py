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
        assert len(queue) == 1

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

    def test_get_empty(self):
        queue = FIFOQueue()
        assert queue.get() is None


class TestPriorityQueue:
    def test_priority_order(self):
        queue = PriorityQueue()
        low = Job(fn=lambda: 1, name="low", priority=1)
        high = Job(fn=lambda: 2, name="high", priority=10)
        medium = Job(fn=lambda: 3, name="medium", priority=5)
        queue.put(low)
        queue.put(high)
        queue.put(medium)
        assert queue.get().name == "high"
        assert queue.get().name == "medium"
        assert queue.get().name == "low"

    def test_same_priority_fifo(self):
        queue = PriorityQueue()
        job1 = Job(fn=lambda: 1, name="first", priority=5)
        job2 = Job(fn=lambda: 2, name="second", priority=5)
        queue.put(job1)
        queue.put(job2)
        assert queue.get().name == "first"
        assert queue.get().name == "second"

    def test_update_priority(self):
        queue = PriorityQueue()
        job = Job(fn=lambda: 1, priority=1)
        queue.put(job)
        queue.update_priority(job.id, 100)
        peeked = queue.peek()
        assert peeked.priority == 100

    def test_empty_queue(self):
        queue = PriorityQueue()
        assert len(queue) == 0
        assert queue.empty()

    def test_put_and_get(self):
        queue = PriorityQueue()
        job = Job(fn=lambda: 42)
        queue.put(job)
        assert len(queue) == 1
        retrieved = queue.get()
        assert retrieved.id == job.id
        assert len(queue) == 0

    def test_peek_empty(self):
        queue = PriorityQueue()
        assert queue.peek() is None

    def test_get_empty(self):
        queue = PriorityQueue()
        assert queue.get() is None

    def test_remove(self):
        queue = PriorityQueue()
        job1 = Job(fn=lambda: 1)
        job2 = Job(fn=lambda: 2)
        queue.put(job1)
        queue.put(job2)
        removed = queue.remove(job1.id)
        assert removed is True
        assert len(queue) == 1

    def test_remove_not_found(self):
        queue = PriorityQueue()
        removed = queue.remove("nonexistent")
        assert removed is False

    def test_update_priority_not_found(self):
        queue = PriorityQueue()
        result = queue.update_priority("nonexistent", 100)
        assert result is False

    def test_iter(self):
        queue = PriorityQueue()
        jobs = [Job(fn=lambda: i, priority=i) for i in range(3)]
        for job in jobs:
            queue.put(job)
        queue_jobs = list(queue)
        assert len(queue_jobs) == 3
        # Should be in priority order (highest first)
        assert queue_jobs[0].priority == 2
        assert queue_jobs[1].priority == 1
        assert queue_jobs[2].priority == 0


class TestJobQueueInterface:
    @pytest.mark.parametrize("QueueClass", [FIFOQueue, PriorityQueue])
    def test_queue_interface(self, QueueClass):
        queue: JobQueue = QueueClass()
        assert hasattr(queue, 'put')
        assert hasattr(queue, 'get')
        assert hasattr(queue, 'peek')
        assert hasattr(queue, 'remove')
        assert hasattr(queue, 'empty')
        assert hasattr(queue, '__len__')
        assert hasattr(queue, '__iter__')

    @pytest.mark.parametrize("QueueClass", [FIFOQueue, PriorityQueue])
    def test_queue_is_instance_of_jobqueue(self, QueueClass):
        queue = QueueClass()
        assert isinstance(queue, JobQueue)
