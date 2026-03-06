"""Tests for Trial dataclass and TrialStatus enum."""

from datetime import datetime, timedelta

import pytest

from gpudispatch.experiments.trial import Trial, TrialStatus


class TestTrialStatus:
    """Tests for TrialStatus enum."""

    def test_status_values(self):
        """TrialStatus has expected values."""
        assert TrialStatus.PENDING.value == "pending"
        assert TrialStatus.RUNNING.value == "running"
        assert TrialStatus.COMPLETED.value == "completed"
        assert TrialStatus.FAILED.value == "failed"
        assert TrialStatus.PRUNED.value == "pruned"

    def test_is_terminal_pending(self):
        """PENDING is not terminal."""
        assert not TrialStatus.PENDING.is_terminal

    def test_is_terminal_running(self):
        """RUNNING is not terminal."""
        assert not TrialStatus.RUNNING.is_terminal

    def test_is_terminal_completed(self):
        """COMPLETED is terminal."""
        assert TrialStatus.COMPLETED.is_terminal

    def test_is_terminal_failed(self):
        """FAILED is terminal."""
        assert TrialStatus.FAILED.is_terminal

    def test_is_terminal_pruned(self):
        """PRUNED is terminal."""
        assert TrialStatus.PRUNED.is_terminal


class TestTrialCreation:
    """Tests for Trial creation and defaults."""

    def test_create_trial_minimal(self):
        """Create trial with minimal arguments."""
        trial = Trial(id=1, params={"lr": 0.01})
        assert trial.id == 1
        assert trial.params == {"lr": 0.01}
        assert trial.metrics == {}
        assert trial.status == TrialStatus.PENDING
        assert trial.error is None
        assert trial.started_at is None
        assert trial.completed_at is None

    def test_create_trial_with_all_fields(self):
        """Create trial with all fields specified."""
        started = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 5, 30)
        trial = Trial(
            id=42,
            params={"lr": 0.001, "batch_size": 32},
            metrics={"loss": 0.5, "accuracy": 0.95},
            status=TrialStatus.COMPLETED,
            error=None,
            started_at=started,
            completed_at=completed,
        )
        assert trial.id == 42
        assert trial.params == {"lr": 0.001, "batch_size": 32}
        assert trial.metrics == {"loss": 0.5, "accuracy": 0.95}
        assert trial.status == TrialStatus.COMPLETED
        assert trial.started_at == started
        assert trial.completed_at == completed

    def test_create_failed_trial_with_error(self):
        """Create a failed trial with error message."""
        trial = Trial(
            id=5,
            params={"lr": 100},
            status=TrialStatus.FAILED,
            error="Learning rate too high, NaN loss",
        )
        assert trial.status == TrialStatus.FAILED
        assert trial.error == "Learning rate too high, NaN loss"


class TestTrialDuration:
    """Tests for Trial duration calculation."""

    def test_duration_seconds_completed_trial(self):
        """duration_seconds returns correct value for completed trial."""
        started = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 5, 30)  # 5 minutes 30 seconds later
        trial = Trial(
            id=1,
            params={},
            started_at=started,
            completed_at=completed,
        )
        assert trial.duration_seconds == 330.0  # 5*60 + 30

    def test_duration_seconds_no_start_time(self):
        """duration_seconds returns None when started_at is None."""
        trial = Trial(id=1, params={}, completed_at=datetime.now())
        assert trial.duration_seconds is None

    def test_duration_seconds_no_end_time(self):
        """duration_seconds returns None when completed_at is None."""
        trial = Trial(id=1, params={}, started_at=datetime.now())
        assert trial.duration_seconds is None

    def test_duration_seconds_both_none(self):
        """duration_seconds returns None when both times are None."""
        trial = Trial(id=1, params={})
        assert trial.duration_seconds is None


class TestTrialSerialization:
    """Tests for Trial to_dict and from_dict methods."""

    def test_to_dict_basic(self):
        """to_dict returns correct dictionary."""
        trial = Trial(
            id=1,
            params={"lr": 0.01},
            metrics={"loss": 0.5},
            status=TrialStatus.COMPLETED,
        )
        d = trial.to_dict()
        assert d["id"] == 1
        assert d["params"] == {"lr": 0.01}
        assert d["metrics"] == {"loss": 0.5}
        assert d["status"] == "completed"
        assert d["error"] is None
        assert d["started_at"] is None
        assert d["completed_at"] is None

    def test_to_dict_with_timestamps(self):
        """to_dict correctly serializes timestamps."""
        started = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 5, 30)
        trial = Trial(
            id=1,
            params={},
            started_at=started,
            completed_at=completed,
        )
        d = trial.to_dict()
        assert d["started_at"] == started.isoformat()
        assert d["completed_at"] == completed.isoformat()

    def test_from_dict_basic(self):
        """from_dict reconstructs trial correctly."""
        data = {
            "id": 42,
            "params": {"lr": 0.001},
            "metrics": {"loss": 0.25},
            "status": "completed",
            "error": None,
            "started_at": None,
            "completed_at": None,
        }
        trial = Trial.from_dict(data)
        assert trial.id == 42
        assert trial.params == {"lr": 0.001}
        assert trial.metrics == {"loss": 0.25}
        assert trial.status == TrialStatus.COMPLETED
        assert trial.error is None

    def test_from_dict_with_timestamps(self):
        """from_dict correctly parses timestamps."""
        started = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 5, 30)
        data = {
            "id": 1,
            "params": {},
            "metrics": {},
            "status": "running",
            "error": None,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
        }
        trial = Trial.from_dict(data)
        assert trial.started_at == started
        assert trial.completed_at == completed

    def test_roundtrip_serialization(self):
        """Trial survives to_dict -> from_dict roundtrip."""
        original = Trial(
            id=99,
            params={"lr": 0.01, "batch_size": 64},
            metrics={"loss": 0.1, "accuracy": 0.99},
            status=TrialStatus.COMPLETED,
            error=None,
            started_at=datetime(2024, 6, 15, 10, 30, 0),
            completed_at=datetime(2024, 6, 15, 11, 45, 30),
        )
        restored = Trial.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.params == original.params
        assert restored.metrics == original.metrics
        assert restored.status == original.status
        assert restored.started_at == original.started_at
        assert restored.completed_at == original.completed_at
