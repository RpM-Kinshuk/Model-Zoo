"""Tests for the core Dispatcher class."""

import pytest
from unittest.mock import patch, MagicMock
import time

from gpudispatch.core.dispatcher import Dispatcher, DispatcherStats
from gpudispatch.core.job import Job, JobStatus
from gpudispatch.core.resources import GPU


class TestDispatcherCreation:
    def test_dispatcher_default_creation(self):
        with patch('gpudispatch.core.dispatcher.detect_gpus') as mock:
            mock.return_value = []
            dispatcher = Dispatcher()
            assert dispatcher is not None
            assert dispatcher.is_running is False

    def test_dispatcher_with_gpus(self):
        dispatcher = Dispatcher(gpus=[0, 1, 2])
        assert len(dispatcher.available_gpus) == 3

    def test_dispatcher_with_memory_threshold(self):
        dispatcher = Dispatcher(gpus=[0], memory_threshold="1GB")
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
