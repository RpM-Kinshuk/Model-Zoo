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
