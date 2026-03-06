"""Tests for SLURMBackend stub implementation."""

import pytest

from gpudispatch.backends.slurm import SLURMBackend
from gpudispatch.backends.base import Backend
from gpudispatch.core.resources import GPU, Memory


class TestSLURMBackendInstantiation:
    """Tests for SLURMBackend creation and configuration storage."""

    def test_inherits_from_backend(self):
        """SLURMBackend inherits from Backend ABC."""
        assert issubclass(SLURMBackend, Backend)

    def test_default_instantiation(self):
        """SLURMBackend can be instantiated with defaults."""
        backend = SLURMBackend()
        assert backend is not None

    def test_name_property(self):
        """SLURMBackend has name 'slurm'."""
        backend = SLURMBackend()
        assert backend.name == "slurm"

    def test_default_partition(self):
        """Default partition is 'gpu'."""
        backend = SLURMBackend()
        assert backend.partition == "gpu"

    def test_custom_partition(self):
        """Custom partition is stored correctly."""
        backend = SLURMBackend(partition="high-priority")
        assert backend.partition == "high-priority"

    def test_default_account_is_none(self):
        """Default account is None."""
        backend = SLURMBackend()
        assert backend.account is None

    def test_custom_account(self):
        """Custom account is stored correctly."""
        backend = SLURMBackend(account="my-project-account")
        assert backend.account == "my-project-account"

    def test_default_time_limit(self):
        """Default time limit is '24:00:00'."""
        backend = SLURMBackend()
        assert backend.time_limit == "24:00:00"

    def test_custom_time_limit(self):
        """Custom time limit is stored correctly."""
        backend = SLURMBackend(time_limit="48:00:00")
        assert backend.time_limit == "48:00:00"

    def test_default_nodes(self):
        """Default nodes is 1."""
        backend = SLURMBackend()
        assert backend.nodes == 1

    def test_custom_nodes(self):
        """Custom nodes value is stored correctly."""
        backend = SLURMBackend(nodes=4)
        assert backend.nodes == 4

    def test_default_gpus_per_node(self):
        """Default gpus_per_node is 1."""
        backend = SLURMBackend()
        assert backend.gpus_per_node == 1

    def test_custom_gpus_per_node(self):
        """Custom gpus_per_node is stored correctly."""
        backend = SLURMBackend(gpus_per_node=8)
        assert backend.gpus_per_node == 8

    def test_extra_kwargs_stored(self):
        """Extra kwargs are stored for extensibility."""
        backend = SLURMBackend(
            constraint="a100",
            qos="high",
            mem_per_cpu="4G",
        )
        assert backend._extra_config["constraint"] == "a100"
        assert backend._extra_config["qos"] == "high"
        assert backend._extra_config["mem_per_cpu"] == "4G"


class TestSLURMBackendLifecycle:
    """Tests for start/shutdown lifecycle."""

    def test_is_running_false_initially(self):
        """Backend is not running before start."""
        backend = SLURMBackend()
        assert backend.is_running is False

    def test_start_sets_running_true(self):
        """start() sets is_running to True."""
        backend = SLURMBackend()
        backend.start()
        assert backend.is_running is True

    def test_shutdown_sets_running_false(self):
        """shutdown() sets is_running to False."""
        backend = SLURMBackend()
        backend.start()
        backend.shutdown()
        assert backend.is_running is False

    def test_context_manager(self):
        """Backend works as context manager."""
        with SLURMBackend() as backend:
            assert backend.is_running is True
        assert backend.is_running is False

    def test_health_check_returns_running_state(self):
        """health_check() returns is_running state."""
        backend = SLURMBackend()
        assert backend.health_check() is False
        backend.start()
        assert backend.health_check() is True
        backend.shutdown()
        assert backend.health_check() is False


class TestSLURMBackendStubMethods:
    """Tests for stub methods that raise NotImplementedError."""

    def test_allocate_gpus_raises_not_implemented(self):
        """allocate_gpus() raises NotImplementedError with helpful message."""
        backend = SLURMBackend()
        backend.start()
        with pytest.raises(NotImplementedError) as exc_info:
            backend.allocate_gpus(2)
        assert "_submit_job" in str(exc_info.value)

    def test_release_gpus_raises_not_implemented(self):
        """release_gpus() raises NotImplementedError with helpful message."""
        backend = SLURMBackend()
        backend.start()
        with pytest.raises(NotImplementedError) as exc_info:
            backend.release_gpus([GPU(index=0)])
        assert "_cancel_job" in str(exc_info.value)

    def test_list_available_raises_not_implemented(self):
        """list_available() raises NotImplementedError with helpful message."""
        backend = SLURMBackend()
        backend.start()
        with pytest.raises(NotImplementedError) as exc_info:
            backend.list_available()
        assert "sinfo" in str(exc_info.value) or "squeue" in str(exc_info.value)

    def test_submit_job_raises_not_implemented(self):
        """_submit_job() raises NotImplementedError."""
        backend = SLURMBackend()
        with pytest.raises(NotImplementedError) as exc_info:
            backend._submit_job("#!/bin/bash\necho hello")
        assert "_submit_job" in str(exc_info.value)

    def test_check_job_status_raises_not_implemented(self):
        """_check_job_status() raises NotImplementedError."""
        backend = SLURMBackend()
        with pytest.raises(NotImplementedError) as exc_info:
            backend._check_job_status("12345")
        assert "_check_job_status" in str(exc_info.value)

    def test_cancel_job_raises_not_implemented(self):
        """_cancel_job() raises NotImplementedError."""
        backend = SLURMBackend()
        with pytest.raises(NotImplementedError) as exc_info:
            backend._cancel_job("12345")
        assert "_cancel_job" in str(exc_info.value)


class TestSLURMBackendExtensibility:
    """Tests demonstrating extensibility patterns."""

    def test_subclass_can_override_submit_job(self):
        """Subclass can override _submit_job()."""

        class CustomSLURMBackend(SLURMBackend):
            def _submit_job(self, script: str) -> str:
                return "12345"

        backend = CustomSLURMBackend()
        job_id = backend._submit_job("#!/bin/bash\necho hello")
        assert job_id == "12345"

    def test_subclass_can_override_check_job_status(self):
        """Subclass can override _check_job_status()."""

        class CustomSLURMBackend(SLURMBackend):
            def _check_job_status(self, job_id: str) -> str:
                return "RUNNING"

        backend = CustomSLURMBackend()
        status = backend._check_job_status("12345")
        assert status == "RUNNING"
