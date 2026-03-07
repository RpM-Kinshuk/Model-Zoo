"""Tests for the core Dispatcher class."""

import subprocess
import pytest
from unittest.mock import patch

from gpudispatch.core.dispatcher import Dispatcher
from gpudispatch.core.job import CommandResult, Job, JobStatus
from gpudispatch.observability.hooks import EventHook, hooks


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

    def test_submit_command(self):
        dispatcher = Dispatcher(gpus=[0])

        job = dispatcher.submit_command(
            ["python", "-c", "print('hello')"],
            gpu=1,
            memory="8GB",
            priority=5,
            env={"EXPERIMENT_MODE": "smoke"},
        )

        assert job.is_command
        assert job.command == ("python", "-c", "print('hello')")
        assert job.env["EXPERIMENT_MODE"] == "smoke"
        assert job.memory is not None and job.memory.mb == 8 * 1024

    def test_submit_script_python_defaults_to_current_interpreter(self):
        dispatcher = Dispatcher(gpus=[0])

        job = dispatcher.submit_script("train.py", script_args=["--epochs", "5"])

        assert job.is_command
        assert isinstance(job.command, tuple)
        assert job.command[0].endswith("python") or "python" in job.command[0]
        assert job.command[1:] == ("train.py", "--epochs", "5")

    def test_submit_command_merges_dispatcher_defaults(self):
        dispatcher = Dispatcher(
            gpus=[0],
            default_command_timeout=300,
            default_command_env={"GLOBAL": "1"},
        )

        merged = dispatcher.submit_command(["echo", "hello"], env={"LOCAL": "2"})
        override_timeout = dispatcher.submit_command(
            ["echo", "world"],
            timeout=30,
        )

        assert merged.timeout == 300
        assert merged.env == {"GLOBAL": "1", "LOCAL": "2"}
        assert override_timeout.timeout == 30


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


class TestDispatcherWait:
    def test_wait_returns_result_and_autostarts(self):
        dispatcher = Dispatcher(gpus=[0], polling_interval=0.01)

        with patch("gpudispatch.core.dispatcher.is_gpu_available", return_value=True):
            job = dispatcher.submit(lambda: 42)
            result = dispatcher.wait(job, timeout=1.0, poll_interval=0.01)

        assert result == 42
        assert job.status == JobStatus.COMPLETED
        dispatcher.shutdown()

    def test_wait_raises_for_failed_job(self):
        dispatcher = Dispatcher(gpus=[0], polling_interval=0.01)

        def fail_job():
            raise ValueError("boom")

        with patch("gpudispatch.core.dispatcher.is_gpu_available", return_value=True):
            job = dispatcher.submit(fail_job)
            with pytest.raises(RuntimeError, match="boom"):
                dispatcher.wait(job, timeout=1.0, poll_interval=0.01)

        dispatcher.shutdown()

    def test_wait_raises_for_cancelled_job(self):
        dispatcher = Dispatcher(gpus=[0])
        job = dispatcher.submit(lambda: 42)
        dispatcher.cancel(job.id)

        with pytest.raises(RuntimeError, match="cancelled"):
            dispatcher.wait(job, timeout=1.0, poll_interval=0.01)

    def test_wait_times_out_when_job_cannot_start(self):
        dispatcher = Dispatcher(gpus=[0], polling_interval=0.01)

        with patch("gpudispatch.core.dispatcher.is_gpu_available", return_value=False):
            job = dispatcher.submit(lambda: 42)
            with pytest.raises(TimeoutError):
                dispatcher.wait(job, timeout=0.05, poll_interval=0.01)

        dispatcher.shutdown()

    def test_wait_returns_command_result_for_subprocess_job(self):
        dispatcher = Dispatcher(gpus=[0], polling_interval=0.01)

        completed = subprocess.CompletedProcess(
            args=["echo", "ok"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

        with (
            patch("gpudispatch.core.dispatcher.is_gpu_available", return_value=True),
            patch("gpudispatch.core.dispatcher.subprocess.run", return_value=completed),
        ):
            job = dispatcher.submit_command(["echo", "ok"])
            result = dispatcher.wait(job, timeout=1.0, poll_interval=0.01)

        assert isinstance(result, CommandResult)
        assert result.stdout == "ok\n"
        assert result.returncode == 0
        dispatcher.shutdown()

    def test_wait_raises_for_failed_command_job(self):
        dispatcher = Dispatcher(gpus=[0], polling_interval=0.01)

        completed = subprocess.CompletedProcess(
            args=["bash", "-lc", "exit 2"],
            returncode=2,
            stdout="",
            stderr="boom",
        )

        with (
            patch("gpudispatch.core.dispatcher.is_gpu_available", return_value=True),
            patch("gpudispatch.core.dispatcher.subprocess.run", return_value=completed),
        ):
            job = dispatcher.submit_command(["bash", "-lc", "exit 2"])
            with pytest.raises(RuntimeError, match="exit=2"):
                dispatcher.wait(job, timeout=1.0, poll_interval=0.01)

        dispatcher.shutdown()


class TestDispatcherObservability:
    def setup_method(self):
        hooks.clear()

    def teardown_method(self):
        hooks.clear()

    def test_dispatcher_emits_lifecycle_hooks(self):
        dispatcher = Dispatcher(gpus=[0], polling_interval=0.01)
        events = []

        hook = EventHook(
            on_job_start=lambda **kw: events.append(("start", kw)),
            on_job_complete=lambda **kw: events.append(("complete", kw)),
        )
        hooks.register(hook)

        with patch("gpudispatch.core.dispatcher.is_gpu_available", return_value=True):
            job = dispatcher.submit(lambda: "ok", name="hooked-job")
            assert dispatcher.wait(job, timeout=1.0, poll_interval=0.01) == "ok"

        names = [event for event, _ in events]
        assert names == ["start", "complete"]
        assert events[0][1]["job_name"] == "hooked-job"
        assert events[1][1]["runtime_seconds"] >= 0
        dispatcher.shutdown()
