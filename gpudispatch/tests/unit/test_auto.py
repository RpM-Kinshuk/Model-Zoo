"""Tests for auto_dispatcher environment detection and backend selection."""

import os
import pytest
from unittest.mock import patch, MagicMock

from gpudispatch.auto import (
    auto_dispatcher,
    detect_environment,
    EnvironmentType,
)
from gpudispatch.backends.local import LocalBackend
from gpudispatch.backends.base import Backend


class TestEnvironmentType:
    """Tests for EnvironmentType enum."""

    def test_environment_type_values(self):
        """EnvironmentType has expected values."""
        assert EnvironmentType.LOCAL.value == "local"
        assert EnvironmentType.SLURM.value == "slurm"
        assert EnvironmentType.KUBERNETES.value == "kubernetes"
        assert EnvironmentType.AWS.value == "aws"
        assert EnvironmentType.GCP.value == "gcp"


class TestDetectEnvironment:
    """Tests for detect_environment function."""

    def test_detect_slurm_by_job_id(self):
        """SLURM detected via SLURM_JOB_ID env var."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=False):
            env_type = detect_environment()
            assert env_type == EnvironmentType.SLURM

    def test_detect_slurm_by_nodelist(self):
        """SLURM detected via SLURM_NODELIST env var."""
        with patch.dict(os.environ, {"SLURM_NODELIST": "node[001-004]"}, clear=False):
            env_type = detect_environment()
            assert env_type == EnvironmentType.SLURM

    def test_detect_kubernetes_by_service_host(self):
        """Kubernetes detected via KUBERNETES_SERVICE_HOST env var."""
        env_vars = {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}
        with patch.dict(os.environ, env_vars, clear=False):
            # Clear SLURM vars if present
            with patch.dict(os.environ, {}, clear=False):
                for key in ["SLURM_JOB_ID", "SLURM_NODELIST"]:
                    os.environ.pop(key, None)
                env_type = detect_environment()
                assert env_type == EnvironmentType.KUBERNETES

    def test_detect_kubernetes_by_port(self):
        """Kubernetes detected via KUBERNETES_SERVICE_PORT env var."""
        env_vars = {"KUBERNETES_SERVICE_PORT": "443"}
        with patch.dict(os.environ, env_vars, clear=False):
            for key in ["SLURM_JOB_ID", "SLURM_NODELIST", "KUBERNETES_SERVICE_HOST"]:
                os.environ.pop(key, None)
            env_type = detect_environment()
            assert env_type == EnvironmentType.KUBERNETES

    def test_detect_local_is_default(self):
        """Local is the default when no cloud/cluster env detected."""
        # Clear all known environment vars
        env_vars_to_clear = [
            "SLURM_JOB_ID",
            "SLURM_NODELIST",
            "KUBERNETES_SERVICE_HOST",
            "KUBERNETES_SERVICE_PORT",
        ]
        with patch.dict(os.environ, {}, clear=False):
            for key in env_vars_to_clear:
                os.environ.pop(key, None)
            env_type = detect_environment()
            assert env_type == EnvironmentType.LOCAL

    def test_slurm_takes_precedence_over_kubernetes(self):
        """SLURM detection takes precedence over Kubernetes."""
        env_vars = {
            "SLURM_JOB_ID": "12345",
            "KUBERNETES_SERVICE_HOST": "10.0.0.1",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            env_type = detect_environment()
            assert env_type == EnvironmentType.SLURM


class TestAutoDispatcherLocalBackend:
    """Tests for auto_dispatcher returning LocalBackend."""

    def test_returns_backend_instance(self):
        """auto_dispatcher returns a Backend instance."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.LOCAL
            dispatcher = auto_dispatcher()
            assert isinstance(dispatcher, Backend)

    def test_local_environment_returns_local_backend(self):
        """Local environment returns LocalBackend."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.LOCAL
            dispatcher = auto_dispatcher()
            assert isinstance(dispatcher, LocalBackend)

    def test_local_backend_with_default_config(self):
        """LocalBackend created with sensible defaults."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.LOCAL
            dispatcher = auto_dispatcher()
            # Should use auto GPU detection by default
            assert dispatcher._gpus_config == "auto"

    def test_passes_kwargs_to_local_backend(self):
        """Keyword arguments are passed to LocalBackend."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.LOCAL
            dispatcher = auto_dispatcher(
                gpus=[0, 1],
                memory_threshold="1GB",
                polling_interval=10,
            )
            assert dispatcher._gpus_config == [0, 1]
            assert dispatcher._memory_threshold_mb == 1024
            assert dispatcher._polling_interval == 10


class TestAutoDispatcherUnsupportedEnvironments:
    """Tests for unsupported environments returning NotImplementedError."""

    def test_slurm_raises_not_implemented(self):
        """SLURM environment raises NotImplementedError with helpful message."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.SLURM
            with pytest.raises(NotImplementedError) as exc_info:
                auto_dispatcher()
            error_msg = str(exc_info.value).lower()
            assert "slurm" in error_msg
            assert "not yet" in error_msg or "not implemented" in error_msg or "future" in error_msg

    def test_kubernetes_raises_not_implemented(self):
        """Kubernetes environment raises NotImplementedError with helpful message."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.KUBERNETES
            with pytest.raises(NotImplementedError) as exc_info:
                auto_dispatcher()
            error_msg = str(exc_info.value).lower()
            assert "kubernetes" in error_msg or "k8s" in error_msg

    def test_aws_raises_not_implemented(self):
        """AWS environment raises NotImplementedError with helpful message."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.AWS
            with pytest.raises(NotImplementedError) as exc_info:
                auto_dispatcher()
            error_msg = str(exc_info.value).lower()
            assert "aws" in error_msg

    def test_gcp_raises_not_implemented(self):
        """GCP environment raises NotImplementedError with helpful message."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.GCP
            with pytest.raises(NotImplementedError) as exc_info:
                auto_dispatcher()
            error_msg = str(exc_info.value).lower()
            assert "gcp" in error_msg


class TestAutoDispatcherFallback:
    """Tests for graceful fallback behavior."""

    def test_unknown_environment_falls_back_to_local(self):
        """Unknown environment type falls back to LocalBackend."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            # Simulate an unknown environment type
            mock_detect.return_value = EnvironmentType.LOCAL
            dispatcher = auto_dispatcher()
            assert isinstance(dispatcher, LocalBackend)

    def test_detection_error_falls_back_to_local(self):
        """Detection error falls back to LocalBackend with warning."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.side_effect = Exception("Detection failed")
            with patch("gpudispatch.auto.logger") as mock_logger:
                dispatcher = auto_dispatcher()
                assert isinstance(dispatcher, LocalBackend)
                # Should log a warning
                mock_logger.warning.assert_called()


class TestAutoDispatcherIntegration:
    """Integration tests for auto_dispatcher."""

    def test_dispatcher_can_be_started(self):
        """Returned dispatcher can be started."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.LOCAL
            dispatcher = auto_dispatcher(gpus=[0])
            dispatcher.start()
            assert dispatcher.is_running
            dispatcher.shutdown()

    def test_dispatcher_works_as_context_manager(self):
        """Returned dispatcher works as context manager."""
        with patch("gpudispatch.auto.detect_environment") as mock_detect:
            mock_detect.return_value = EnvironmentType.LOCAL
            dispatcher = auto_dispatcher(gpus=[0])
            with dispatcher:
                assert dispatcher.is_running
            assert not dispatcher.is_running


class TestForceBackend:
    """Tests for forcing a specific backend."""

    def test_force_local_backend(self):
        """Can force LocalBackend regardless of environment."""
        # Even with SLURM env vars, forcing local should work
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=False):
            dispatcher = auto_dispatcher(force_backend="local")
            assert isinstance(dispatcher, LocalBackend)

    def test_force_slurm_raises_not_implemented(self):
        """Forcing SLURM backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            auto_dispatcher(force_backend="slurm")

    def test_force_kubernetes_raises_not_implemented(self):
        """Forcing Kubernetes backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            auto_dispatcher(force_backend="kubernetes")

    def test_force_invalid_backend_raises_value_error(self):
        """Forcing invalid backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            auto_dispatcher(force_backend="invalid_backend")
        assert "invalid_backend" in str(exc_info.value).lower() or "unknown" in str(exc_info.value).lower()
