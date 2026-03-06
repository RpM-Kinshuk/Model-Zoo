"""Tests for the experiment registry module.

This module tests:
- Setting and getting the default experiment directory
- Listing experiments from the registry
- Loading experiments by name
- Default directory creation
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from gpudispatch.experiments import Experiment
from gpudispatch.experiments.registry import (
    get_storage,
    list_experiments,
    load,
    set_experiment_dir,
    _default_storage,
    _reset_storage,
)
from gpudispatch.experiments.storage import FileStorage


class TestSetExperimentDir:
    """Tests for set_experiment_dir function."""

    def teardown_method(self) -> None:
        """Reset global storage after each test."""
        _reset_storage()

    def test_sets_storage_to_file_storage(self) -> None:
        """Test that set_experiment_dir creates a FileStorage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_experiment_dir(tmpdir)
            storage = get_storage()

            assert isinstance(storage, FileStorage)

    def test_uses_provided_path(self) -> None:
        """Test that the provided path is used for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "my_experiments"
            set_experiment_dir(str(exp_dir))

            # Storage should be configured with this path
            storage = get_storage()
            assert isinstance(storage, FileStorage)
            # Directory should be created
            assert exp_dir.exists()


class TestGetStorage:
    """Tests for get_storage function."""

    def teardown_method(self) -> None:
        """Reset global storage after each test."""
        _reset_storage()

    def test_returns_file_storage(self) -> None:
        """Test that get_storage returns a FileStorage instance."""
        storage = get_storage()

        assert isinstance(storage, FileStorage)

    def test_creates_default_directory(self) -> None:
        """Test that get_storage creates default ~/.gpudispatch/experiments."""
        storage = get_storage()

        default_dir = Path.home() / ".gpudispatch" / "experiments"
        assert default_dir.exists()

    def test_returns_same_storage_on_repeated_calls(self) -> None:
        """Test that get_storage returns the same instance."""
        storage1 = get_storage()
        storage2 = get_storage()

        assert storage1 is storage2


class TestListExperiments:
    """Tests for list_experiments function."""

    def teardown_method(self) -> None:
        """Reset global storage after each test."""
        _reset_storage()

    def test_returns_empty_list_when_no_experiments(self) -> None:
        """Test that list_experiments returns empty list with no experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_experiment_dir(tmpdir)

            result = list_experiments()

            assert result == []

    def test_lists_experiment_names(self) -> None:
        """Test that list_experiments returns experiment names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_experiment_dir(tmpdir)

            # Create a simple experiment
            def dummy_fn(params: Dict[str, Any]) -> Dict[str, Any]:
                return {"loss": 0.5}

            exp = Experiment(
                fn=dummy_fn,
                name="test_exp",
                storage=get_storage(),
            )
            exp.run(trials=1)

            result = list_experiments()

            assert "test_exp" in result


class TestLoad:
    """Tests for load function."""

    def teardown_method(self) -> None:
        """Reset global storage after each test."""
        _reset_storage()

    def test_returns_none_for_nonexistent(self) -> None:
        """Test that load returns None for non-existent experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_experiment_dir(tmpdir)

            result = load("nonexistent")

            assert result is None

    def test_loads_existing_experiment(self) -> None:
        """Test that load returns Experiment for existing experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_experiment_dir(tmpdir)

            # Create experiment
            def dummy_fn(params: Dict[str, Any]) -> Dict[str, Any]:
                return {"loss": 0.5}

            exp = Experiment(
                fn=dummy_fn,
                name="loadable_exp",
                storage=get_storage(),
            )
            exp.run(trials=1)

            # Load it back
            loaded = load("loadable_exp")

            assert loaded is not None
            assert loaded.name == "loadable_exp"

    def test_loaded_experiment_has_trials(self) -> None:
        """Test that loaded experiment has access to stored trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_experiment_dir(tmpdir)

            # Create experiment with trials
            def dummy_fn(params: Dict[str, Any]) -> Dict[str, Any]:
                return {"loss": 0.5}

            storage = get_storage()
            exp = Experiment(
                fn=dummy_fn,
                name="trial_exp",
                storage=storage,
            )
            exp.run(trials=3)

            # Load and check trials are accessible
            loaded = load("trial_exp")
            assert loaded is not None
            trials = storage.load_trials("trial_exp")
            assert len(trials) == 3
