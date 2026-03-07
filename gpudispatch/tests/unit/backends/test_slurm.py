"""Tests for SLURMBackend implementation."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

from gpudispatch.backends.base import Backend
from gpudispatch.backends.slurm import SLURMBackend
from gpudispatch.core.resources import GPU, Memory


def _completed(
    args: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestSLURMBackendInstantiation:
    def test_inherits_from_backend(self):
        assert issubclass(SLURMBackend, Backend)

    def test_default_instantiation(self):
        backend = SLURMBackend()
        assert backend is not None

    def test_properties_are_exposed(self):
        backend = SLURMBackend(
            partition="a100",
            account="ml-team",
            time_limit="08:00:00",
            nodes=2,
            gpus_per_node=4,
            polling_interval=9,
            qos="high",
        )
        assert backend.name == "slurm"
        assert backend.partition == "a100"
        assert backend.account == "ml-team"
        assert backend.time_limit == "08:00:00"
        assert backend.nodes == 2
        assert backend.gpus_per_node == 4
        assert backend._polling_interval == 9
        assert backend._extra_config["qos"] == "high"


class TestSLURMBackendDiscoveryAndLifecycle:
    def test_start_discovers_gpus_from_slurm_env(self):
        backend = SLURMBackend()

        with (
            patch.dict(os.environ, {"SLURM_JOB_GPUS": "gpu[0-2]"}, clear=True),
            patch.object(backend, "_run_command", return_value=None),
        ):
            backend.start()

        assert backend.is_running is True
        assert [gpu.index for gpu in backend.list_available()] == [0, 1, 2]
        backend.shutdown()

    def test_start_discovers_gpus_from_cuda_visible_devices(self):
        backend = SLURMBackend()

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}, clear=True),
            patch.object(backend, "_run_command", return_value=None),
        ):
            backend.start()

        assert [gpu.index for gpu in backend.list_available()] == [1, 3]
        backend.shutdown()

    def test_start_discovers_gpu_count_from_scontrol_job_output(self):
        backend = SLURMBackend()

        def fake_run(command, timeout=15):  # noqa: ANN001
            if command[:3] == ["scontrol", "show", "job"]:
                return _completed(
                    command,
                    stdout="JobId=123 AllocTRES=cpu=8,gres/gpu=2",
                )
            return None

        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "123"}, clear=True),
            patch.object(backend, "_run_command", side_effect=fake_run),
        ):
            backend.start()

        assert [gpu.index for gpu in backend.list_available()] == [0, 1]
        backend.shutdown()

    def test_start_falls_back_to_nodes_times_gpus_per_node(self):
        backend = SLURMBackend(nodes=2, gpus_per_node=2)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(backend, "_run_command", return_value=None),
        ):
            backend.start()

        assert [gpu.index for gpu in backend.list_available()] == [0, 1, 2, 3]
        backend.shutdown()


class TestSLURMBackendAllocation:
    def test_allocate_release_and_list_available(self):
        backend = SLURMBackend()

        with (
            patch.dict(os.environ, {"SLURM_JOB_GPUS": "0,1"}, clear=True),
            patch.object(backend, "_run_command", return_value=None),
        ):
            backend.start()

        first = backend.allocate_gpus(1)
        assert len(first) == 1
        assert len(backend.list_available()) == 1

        second = backend.allocate_gpus(1)
        assert len(second) == 1
        assert first[0].index != second[0].index
        assert backend.allocate_gpus(1) == []

        backend.release_gpus(first)
        assert len(backend.list_available()) == 1
        backend.shutdown()

    def test_allocate_respects_memory_requirement_when_known(self):
        backend = SLURMBackend()
        backend._running = True
        backend._gpu_pool = [GPU(index=0, memory=1024), GPU(index=1, memory=16384)]

        selected = backend.allocate_gpus(1, memory=Memory.from_string("8GB"))
        assert len(selected) == 1
        assert selected[0].index == 1

    def test_allocate_non_positive_count_returns_empty(self):
        backend = SLURMBackend()
        backend._running = True
        backend._gpu_pool = [GPU(index=0)]

        assert backend.allocate_gpus(0) == []
        assert backend.allocate_gpus(-1) == []


class TestSLURMBackendHealthAndSchedulerHelpers:
    def test_health_check_requires_running(self):
        backend = SLURMBackend()
        assert backend.health_check() is False

    def test_health_check_uses_scontrol_ping_when_available(self):
        backend = SLURMBackend()
        backend._running = True
        with patch.object(
            backend,
            "_run_command",
            return_value=_completed(["scontrol", "ping"], returncode=0),
        ):
            assert backend.health_check() is True

    def test_health_check_falls_back_to_true_when_scontrol_unavailable(self):
        backend = SLURMBackend()
        backend._running = True
        with patch.object(backend, "_run_command", return_value=None):
            assert backend.health_check() is True

    def test_submit_job_returns_sbatch_job_id(self):
        backend = SLURMBackend()

        with patch.object(
            backend,
            "_run_required_command",
            return_value=_completed(["sbatch"], stdout="12345;cluster"),
        ) as mocked:
            job_id = backend._submit_job("echo hello")

        assert job_id == "12345"
        assert mocked.call_count == 1
        submitted_args = mocked.call_args.args[0]
        assert submitted_args[:2] == ["sbatch", "--parsable"]

    def test_check_job_status_prefers_squeue_then_falls_back_to_sacct(self):
        backend = SLURMBackend()

        with patch.object(
            backend,
            "_run_command",
            side_effect=[
                _completed(["squeue"], stdout="RUNNING\n"),
            ],
        ):
            assert backend._check_job_status("123") == "RUNNING"

        with patch.object(
            backend,
            "_run_command",
            side_effect=[
                _completed(["squeue"], stdout=""),
                _completed(["sacct"], stdout="COMPLETED\n"),
            ],
        ):
            assert backend._check_job_status("123") == "COMPLETED"

    def test_check_job_status_returns_unknown_when_scheduler_unavailable(self):
        backend = SLURMBackend()
        with patch.object(backend, "_run_command", return_value=None):
            assert backend._check_job_status("123") == "UNKNOWN"

    def test_cancel_job_calls_scancel(self):
        backend = SLURMBackend()
        with patch.object(backend, "_run_required_command") as mocked:
            backend._cancel_job("123")
        mocked.assert_called_once_with(["scancel", "123"])
