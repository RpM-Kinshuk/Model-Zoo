"""Tests for storage backends."""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Type

import pytest

from gpudispatch.experiments.storage import (
    FileStorage,
    MemoryStorage,
    SQLiteStorage,
    Storage,
)
from gpudispatch.experiments.trial import Trial, TrialStatus


class StorageContractTests:
    """Base test class with contract tests all Storage implementations must pass.

    Subclass this and set `storage_class` to run contract tests for each backend.
    """

    storage_class: Type[Storage]

    @pytest.fixture
    def storage(self, tmp_path: Path) -> Storage:
        """Create storage instance for testing."""
        raise NotImplementedError("Subclasses must implement this fixture")

    @pytest.fixture
    def sample_trial(self) -> Trial:
        """Create a sample trial for testing."""
        return Trial(
            id=1,
            params={"lr": 0.01, "batch_size": 32},
            metrics={"loss": 0.5, "accuracy": 0.95},
            status=TrialStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 5, 30),
        )

    @pytest.fixture
    def sample_config(self) -> dict:
        """Create a sample config for testing."""
        return {
            "search_space": {"lr": {"type": "log", "low": 1e-5, "high": 1e-1}},
            "max_trials": 100,
            "metric": "accuracy",
            "direction": "maximize",
        }

    # --- Trial tests ---

    def test_save_and_load_trial(self, storage: Storage, sample_trial: Trial):
        """save_trial and load_trial work correctly."""
        storage.save_trial("exp1", sample_trial)
        loaded = storage.load_trial("exp1", sample_trial.id)

        assert loaded is not None
        assert loaded.id == sample_trial.id
        assert loaded.params == sample_trial.params
        assert loaded.metrics == sample_trial.metrics
        assert loaded.status == sample_trial.status
        assert loaded.started_at == sample_trial.started_at
        assert loaded.completed_at == sample_trial.completed_at

    def test_load_trial_nonexistent(self, storage: Storage):
        """load_trial returns None for nonexistent trial."""
        result = storage.load_trial("nonexistent_exp", 999)
        assert result is None

    def test_load_trial_wrong_experiment(self, storage: Storage, sample_trial: Trial):
        """load_trial returns None when trial exists in different experiment."""
        storage.save_trial("exp1", sample_trial)
        result = storage.load_trial("exp2", sample_trial.id)
        assert result is None

    def test_save_trial_overwrites_existing(self, storage: Storage):
        """Saving a trial with same ID overwrites previous version."""
        trial1 = Trial(id=1, params={"lr": 0.01}, status=TrialStatus.RUNNING)
        trial2 = Trial(
            id=1,
            params={"lr": 0.01},
            metrics={"loss": 0.3},
            status=TrialStatus.COMPLETED,
        )

        storage.save_trial("exp1", trial1)
        storage.save_trial("exp1", trial2)

        loaded = storage.load_trial("exp1", 1)
        assert loaded is not None
        assert loaded.status == TrialStatus.COMPLETED
        assert loaded.metrics == {"loss": 0.3}

    def test_load_trials_empty(self, storage: Storage):
        """load_trials returns empty list for nonexistent experiment."""
        trials = storage.load_trials("nonexistent")
        assert trials == []

    def test_load_trials_multiple(self, storage: Storage):
        """load_trials returns all trials for experiment."""
        trial1 = Trial(id=1, params={"lr": 0.01})
        trial2 = Trial(id=2, params={"lr": 0.001})
        trial3 = Trial(id=3, params={"lr": 0.0001})

        storage.save_trial("exp1", trial1)
        storage.save_trial("exp1", trial2)
        storage.save_trial("exp1", trial3)

        trials = storage.load_trials("exp1")
        assert len(trials) == 3
        trial_ids = {t.id for t in trials}
        assert trial_ids == {1, 2, 3}

    def test_load_trials_isolated_by_experiment(self, storage: Storage):
        """load_trials only returns trials from specified experiment."""
        trial1 = Trial(id=1, params={"lr": 0.01})
        trial2 = Trial(id=2, params={"lr": 0.001})

        storage.save_trial("exp_a", trial1)
        storage.save_trial("exp_b", trial2)

        trials_a = storage.load_trials("exp_a")
        trials_b = storage.load_trials("exp_b")

        assert len(trials_a) == 1
        assert trials_a[0].id == 1
        assert len(trials_b) == 1
        assert trials_b[0].id == 2

    # --- Config tests ---

    def test_save_and_load_config(self, storage: Storage, sample_config: dict):
        """save_config and load_config work correctly."""
        storage.save_config("exp1", sample_config)
        loaded = storage.load_config("exp1")

        assert loaded == sample_config

    def test_load_config_nonexistent(self, storage: Storage):
        """load_config returns None for nonexistent experiment."""
        result = storage.load_config("nonexistent")
        assert result is None

    def test_save_config_overwrites(self, storage: Storage):
        """Saving config overwrites previous config."""
        config1 = {"max_trials": 10}
        config2 = {"max_trials": 100, "metric": "loss"}

        storage.save_config("exp1", config1)
        storage.save_config("exp1", config2)

        loaded = storage.load_config("exp1")
        assert loaded == config2

    # --- list_experiments tests ---

    def test_list_experiments_empty(self, storage: Storage):
        """list_experiments returns empty list when no experiments."""
        experiments = storage.list_experiments()
        assert experiments == []

    def test_list_experiments_with_trials(self, storage: Storage):
        """list_experiments returns experiments that have trials."""
        trial = Trial(id=1, params={"x": 1})
        storage.save_trial("exp1", trial)
        storage.save_trial("exp2", trial)

        experiments = storage.list_experiments()
        assert set(experiments) == {"exp1", "exp2"}

    def test_list_experiments_with_config_only(self, storage: Storage):
        """list_experiments includes experiments with only configs."""
        storage.save_config("config_only_exp", {"key": "value"})

        experiments = storage.list_experiments()
        assert "config_only_exp" in experiments


class TestMemoryStorage(StorageContractTests):
    """Tests for MemoryStorage."""

    storage_class = MemoryStorage

    @pytest.fixture
    def storage(self, tmp_path: Path) -> Storage:
        """Create MemoryStorage instance."""
        return MemoryStorage()

    def test_memory_storage_non_persistent(self):
        """MemoryStorage data is not shared between instances."""
        storage1 = MemoryStorage()
        storage1.save_trial("exp1", Trial(id=1, params={"x": 1}))

        storage2 = MemoryStorage()
        trials = storage2.load_trials("exp1")

        assert trials == []


class TestFileStorage(StorageContractTests):
    """Tests for FileStorage."""

    storage_class = FileStorage

    @pytest.fixture
    def storage(self, tmp_path: Path) -> Storage:
        """Create FileStorage instance."""
        return FileStorage(base_dir=tmp_path)

    def test_file_storage_creates_directory_structure(self, tmp_path: Path):
        """FileStorage creates expected directory structure."""
        storage = FileStorage(base_dir=tmp_path)
        trial = Trial(id=1, params={"x": 1})

        storage.save_trial("my-experiment", trial)

        exp_dir = tmp_path / "my-experiment"
        assert exp_dir.exists()
        assert (exp_dir / "trials.csv").exists()

    def test_file_storage_config_json(self, tmp_path: Path):
        """FileStorage saves config as JSON."""
        storage = FileStorage(base_dir=tmp_path)
        config = {"max_trials": 100, "nested": {"key": "value"}}

        storage.save_config("exp1", config)

        config_path = tmp_path / "exp1" / "config.json"
        assert config_path.exists()
        with open(config_path, "r") as f:
            loaded = json.load(f)
        assert loaded == config

    def test_file_storage_persistence(self, tmp_path: Path):
        """FileStorage persists data across instances."""
        storage1 = FileStorage(base_dir=tmp_path)
        storage1.save_trial("exp1", Trial(id=1, params={"x": 1}))
        storage1.save_config("exp1", {"key": "value"})

        # Create new instance pointing to same directory
        storage2 = FileStorage(base_dir=tmp_path)

        trials = storage2.load_trials("exp1")
        config = storage2.load_config("exp1")

        assert len(trials) == 1
        assert trials[0].id == 1
        assert config == {"key": "value"}


class TestSQLiteStorage(StorageContractTests):
    """Tests for SQLiteStorage."""

    storage_class = SQLiteStorage

    @pytest.fixture
    def storage(self, tmp_path: Path) -> Storage:
        """Create SQLiteStorage instance."""
        db_path = tmp_path / "experiments.db"
        return SQLiteStorage(db_path=db_path)

    def test_sqlite_creates_database(self, tmp_path: Path):
        """SQLiteStorage creates database file."""
        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(db_path=db_path)

        # Force table creation by saving something
        storage.save_trial("exp1", Trial(id=1, params={}))

        assert db_path.exists()

    def test_sqlite_creates_tables(self, tmp_path: Path):
        """SQLiteStorage creates expected tables."""
        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(db_path=db_path)

        # Force initialization
        storage.save_trial("exp1", Trial(id=1, params={}))

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check trials table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trials'"
        )
        assert cursor.fetchone() is not None

        # Check configs table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='configs'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_sqlite_persistence(self, tmp_path: Path):
        """SQLiteStorage persists data across instances."""
        db_path = tmp_path / "test.db"

        storage1 = SQLiteStorage(db_path=db_path)
        storage1.save_trial("exp1", Trial(id=1, params={"x": 1}))
        storage1.save_config("exp1", {"key": "value"})

        # Create new instance pointing to same database
        storage2 = SQLiteStorage(db_path=db_path)

        trials = storage2.load_trials("exp1")
        config = storage2.load_config("exp1")

        assert len(trials) == 1
        assert trials[0].id == 1
        assert config == {"key": "value"}

    def test_sqlite_in_memory_mode(self):
        """SQLiteStorage works with in-memory database."""
        storage = SQLiteStorage(db_path=":memory:")

        trial = Trial(id=1, params={"x": 1}, metrics={"y": 2})
        storage.save_trial("exp1", trial)

        loaded = storage.load_trial("exp1", 1)
        assert loaded is not None
        assert loaded.params == {"x": 1}
        assert loaded.metrics == {"y": 2}
