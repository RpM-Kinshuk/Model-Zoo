"""Unit tests for CLI module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gpudispatch.cli.main import main
from gpudispatch.core import CommandResult
from gpudispatch.experiments import Experiment
from gpudispatch.experiments.registry import _reset_storage
from gpudispatch.experiments.storage import MemoryStorage
from gpudispatch.utils.gpu import GPUInfo


@dataclass
class _FakeJob:
    id: str
    name: str


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_gpus() -> list[GPUInfo]:
    """Create mock GPU info."""
    return [
        GPUInfo(
            index=0,
            name="NVIDIA RTX 4090",
            memory_total_mb=24576,
            memory_used_mb=8192,
            utilization_percent=45,
        ),
        GPUInfo(
            index=1,
            name="NVIDIA RTX 4090",
            memory_total_mb=24576,
            memory_used_mb=1024,
            utilization_percent=10,
        ),
    ]


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_main_help(self, runner: CliRunner) -> None:
        """Test main help output."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "gpudispatch" in result.output
        assert "Universal GPU orchestration" in result.output

    def test_main_version(self, runner: CliRunner) -> None:
        """Test version flag."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0." in result.output


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_with_gpus(
        self, runner: CliRunner, mock_gpus: list[GPUInfo]
    ) -> None:
        """Test status command with available GPUs."""
        with patch(
            "gpudispatch.utils.gpu.detect_gpus", return_value=mock_gpus
        ):
            result = runner.invoke(main, ["status"])
            assert result.exit_code == 0
            assert "GPU 0" in result.output
            assert "GPU 1" in result.output
            assert "NVIDIA RTX 4090" in result.output
            assert "8192MB" in result.output
            assert "24576MB" in result.output
            assert "45%" in result.output

    def test_status_no_gpus(self, runner: CliRunner) -> None:
        """Test status command when no GPUs are available."""
        with patch("gpudispatch.utils.gpu.detect_gpus", return_value=[]):
            result = runner.invoke(main, ["status"])
            assert result.exit_code == 0
            assert "No GPUs detected" in result.output


class TestShowCommand:
    """Tests for the show command."""

    def test_show_existing_experiment(self, runner: CliRunner) -> None:
        """Test showing an existing experiment."""
        # Reset storage and create a test experiment
        _reset_storage()
        storage = MemoryStorage()

        def dummy_fn(params: Dict[str, Any]) -> Dict[str, Any]:
            return {"loss": 0.5}

        exp = Experiment(
            fn=dummy_fn,
            name="test_exp",
            storage=storage,
        )

        # Mock load to return our experiment
        with patch("gpudispatch.experiments.load", return_value=exp):
            result = runner.invoke(main, ["show", "test_exp"])
            assert result.exit_code == 0
            assert "Experiment: test_exp" in result.output
            assert "Metric: loss" in result.output
            assert "Trials:" in result.output

    def test_show_nonexistent_experiment(self, runner: CliRunner) -> None:
        """Test showing a non-existent experiment."""
        with patch("gpudispatch.experiments.load", return_value=None):
            result = runner.invoke(main, ["show", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_show_requires_name(self, runner: CliRunner) -> None:
        """Test that show requires a name argument."""
        result = runner.invoke(main, ["show"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "NAME" in result.output


class TestListCommand:
    """Tests for the list command."""

    def test_list_experiments(self, runner: CliRunner) -> None:
        """Test listing experiments."""
        mock_experiments = ["exp1", "exp2", "exp3"]
        with patch(
            "gpudispatch.experiments.list_experiments",
            return_value=mock_experiments,
        ):
            result = runner.invoke(main, ["list"])
            assert result.exit_code == 0
            assert "exp1" in result.output
            assert "exp2" in result.output
            assert "exp3" in result.output

    def test_list_no_experiments(self, runner: CliRunner) -> None:
        """Test listing when no experiments exist."""
        with patch(
            "gpudispatch.experiments.list_experiments", return_value=[]
        ):
            result = runner.invoke(main, ["list"])
            assert result.exit_code == 0
            assert "No experiments found" in result.output


class TestProfilesCommand:
    def test_profiles_lists_presets(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["profiles"])
        assert result.exit_code == 0
        assert "quickstart" in result.output
        assert "batch" in result.output
        assert "high_reliability" in result.output


class TestRunScriptCommand:
    def _mock_dispatcher(self) -> MagicMock:
        dispatcher = MagicMock()
        dispatcher.__enter__.return_value = dispatcher
        dispatcher.__exit__.return_value = None
        return dispatcher

    def test_run_script_submits_script_with_profile_defaults(
        self,
        runner: CliRunner,
    ) -> None:
        dispatcher = self._mock_dispatcher()
        job = _FakeJob(id="job123", name="train")
        dispatcher.submit_script.return_value = job
        dispatcher.wait.return_value = CommandResult(
            command=["python", "train.py"],
            returncode=0,
            stdout="done\n",
            stderr="",
        )

        with patch("gpudispatch.cli.main.dispatcher_from_profile", return_value=dispatcher):
            result = runner.invoke(
                main,
                [
                    "run-script",
                    "--profile",
                    "quickstart",
                    "--gpu",
                    "2",
                    "--memory",
                    "8GB",
                    "--priority",
                    "3",
                    "--env",
                    "A=1",
                    "--env",
                    "B=2",
                    "train.py",
                    "--",
                    "--epochs",
                    "5",
                ],
            )

        assert result.exit_code == 0
        assert "done" in result.output
        assert "Job completed: train (job123)" in result.output

        dispatcher.submit_script.assert_called_once()
        kwargs = dispatcher.submit_script.call_args.kwargs
        assert kwargs["script_path"] == "train.py"
        assert kwargs["script_args"] == ("--epochs", "5")
        assert kwargs["gpu"] == 2
        assert kwargs["memory"] == "8GB"
        assert kwargs["priority"] == 3
        assert kwargs["env"] == {"A": "1", "B": "2"}

    def test_run_script_rejects_invalid_env_format(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["run-script", "--env", "BROKEN", "train.py"])
        assert result.exit_code != 0
        assert "Invalid --env value" in result.output

    def test_run_script_surfaces_job_failure(self, runner: CliRunner) -> None:
        dispatcher = self._mock_dispatcher()
        dispatcher.submit_script.return_value = _FakeJob(id="job999", name="failing-job")
        dispatcher.wait.side_effect = RuntimeError("boom")

        with patch("gpudispatch.cli.main.dispatcher_from_profile", return_value=dispatcher):
            result = runner.invoke(main, ["run-script", "train.py"])

        assert result.exit_code == 1
        assert "Job failed: boom" in result.output
