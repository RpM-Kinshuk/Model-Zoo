"""Tests for the Experiment class.

This module tests:
- Experiment initialization with various configurations
- Auto-generation of experiment names
- Strategy selection based on search space type
- Trial execution and result capture
- Exception handling for failed trials
- Loading experiments from storage
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict

import pytest

from gpudispatch.experiments.experiment import Experiment
from gpudispatch.experiments.results import Results
from gpudispatch.experiments.search_space import Distribution, Log, SearchSpace, Uniform
from gpudispatch.experiments.storage import MemoryStorage
from gpudispatch.experiments.strategies import GridStrategy, RandomStrategy
from gpudispatch.experiments.trial import Trial, TrialStatus


# Test fixtures and helper functions
def simple_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """Simple function that returns loss based on params."""
    return {"loss": params.get("lr", 0.1) * 10}


def multi_metric_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """Function that returns multiple metrics."""
    lr = params.get("lr", 0.1)
    return {
        "loss": lr * 10,
        "accuracy": 1.0 - lr,
        "f1": 0.9 - lr / 2,
    }


def failing_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """Function that always raises an exception."""
    raise ValueError("Intentional failure")


def sometimes_failing_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """Function that fails for certain param values."""
    if params.get("lr", 0) < 0.01:
        raise ValueError(f"lr too small: {params.get('lr')}")
    return {"loss": params.get("lr", 0.1) * 10}


# === Test Initialization ===


class TestExperimentInit:
    """Tests for Experiment initialization."""

    def test_init_with_minimal_args(self) -> None:
        """Test initialization with just a function."""
        exp = Experiment(fn=simple_function)

        assert exp.fn is simple_function
        assert exp.name is not None
        assert "simple_function" in exp.name

    def test_init_auto_generates_name_from_function(self) -> None:
        """Test that name is auto-generated from function name + timestamp."""
        exp = Experiment(fn=simple_function)

        # Name should contain function name
        assert "simple_function" in exp.name
        # Name should have timestamp-like suffix (YYYYMMDD_HHMMSS format)
        assert re.search(r"\d{8}_\d{6}$", exp.name) is not None

    def test_init_with_explicit_name(self) -> None:
        """Test initialization with an explicit name."""
        exp = Experiment(fn=simple_function, name="my_experiment")

        assert exp.name == "my_experiment"

    def test_init_with_search_space(self) -> None:
        """Test initialization with a search space."""
        space = SearchSpace.from_dict({"lr": [0.1, 0.01]})
        exp = Experiment(fn=simple_function, search_space=space)

        assert exp.search_space is space

    def test_init_with_strategy(self) -> None:
        """Test initialization with an explicit strategy."""
        strategy = GridStrategy()
        exp = Experiment(fn=simple_function, strategy=strategy)

        assert exp.strategy is strategy

    def test_init_with_storage(self) -> None:
        """Test initialization with an explicit storage."""
        storage = MemoryStorage()
        exp = Experiment(fn=simple_function, storage=storage)

        assert exp.storage is storage

    def test_init_default_storage_is_memory(self) -> None:
        """Test that default storage is MemoryStorage."""
        exp = Experiment(fn=simple_function)

        assert isinstance(exp.storage, MemoryStorage)

    def test_init_metric_defaults_to_loss(self) -> None:
        """Test that default metric is 'loss'."""
        exp = Experiment(fn=simple_function)

        assert exp.metric == "loss"

    def test_init_maximize_defaults_to_false(self) -> None:
        """Test that maximize defaults to False (minimizing)."""
        exp = Experiment(fn=simple_function)

        assert exp.maximize is False

    def test_init_gpu_defaults_to_zero(self) -> None:
        """Test that GPU defaults to 0."""
        exp = Experiment(fn=simple_function)

        assert exp.gpu == 0

    def test_init_with_maximize_true(self) -> None:
        """Test initialization with maximize=True."""
        exp = Experiment(fn=multi_metric_function, metric="accuracy", maximize=True)

        assert exp.metric == "accuracy"
        assert exp.maximize is True


# === Test Default Strategy Selection ===


class TestDefaultStrategySelection:
    """Tests for automatic strategy selection based on search space."""

    def test_grid_strategy_for_list_params(self) -> None:
        """Test that GridStrategy is used for list-only search spaces."""
        space = SearchSpace.from_dict({"lr": [0.1, 0.01], "batch": [16, 32]})
        exp = Experiment(fn=simple_function, search_space=space)

        assert isinstance(exp.strategy, GridStrategy)

    def test_random_strategy_for_distribution_params(self) -> None:
        """Test that RandomStrategy is used for distribution search spaces."""
        space = SearchSpace.from_dict({"lr": Log(1e-5, 1e-1)})
        exp = Experiment(fn=simple_function, search_space=space)

        assert isinstance(exp.strategy, RandomStrategy)

    def test_random_strategy_for_mixed_params(self) -> None:
        """Test that RandomStrategy is used for mixed grid+sweep spaces."""
        space = SearchSpace.from_dict({
            "model": ["small", "large"],  # grid
            "lr": Log(1e-5, 1e-1),  # sweep
        })
        exp = Experiment(fn=simple_function, search_space=space)

        assert isinstance(exp.strategy, RandomStrategy)


# === Test Run Method ===


class TestExperimentRun:
    """Tests for the Experiment.run method."""

    def test_run_executes_all_grid_combinations(self) -> None:
        """Test that run executes all grid combinations."""
        space = SearchSpace.from_dict({"lr": [0.1, 0.01]})
        exp = Experiment(fn=simple_function, search_space=space)

        results = exp.run()

        assert len(results.trials) == 2
        assert all(t.status == TrialStatus.COMPLETED for t in results.trials)

    def test_run_with_trials_limit(self) -> None:
        """Test that run respects the trials parameter."""
        space = SearchSpace.from_dict({"lr": Log(1e-5, 1e-1)})
        exp = Experiment(fn=simple_function, search_space=space)

        results = exp.run(trials=5)

        assert len(results.trials) == 5

    def test_run_captures_metrics_from_fn(self) -> None:
        """Test that metrics from fn are captured in trials."""
        space = SearchSpace.from_dict({"lr": [0.1]})
        exp = Experiment(fn=multi_metric_function, search_space=space)

        results = exp.run()

        assert len(results.trials) == 1
        trial = results.trials[0]
        assert "loss" in trial.metrics
        assert "accuracy" in trial.metrics
        assert "f1" in trial.metrics

    def test_run_returns_results_object(self) -> None:
        """Test that run returns a Results object."""
        space = SearchSpace.from_dict({"lr": [0.1]})
        exp = Experiment(fn=simple_function, search_space=space)

        results = exp.run()

        assert isinstance(results, Results)
        assert results.metric == exp.metric
        assert results.maximize == exp.maximize
        assert results.experiment_name == exp.name

    def test_run_saves_trials_to_storage(self) -> None:
        """Test that trials are saved to storage during run."""
        storage = MemoryStorage()
        space = SearchSpace.from_dict({"lr": [0.1, 0.01]})
        exp = Experiment(fn=simple_function, search_space=space, storage=storage)

        exp.run()

        stored_trials = storage.load_trials(exp.name)
        assert len(stored_trials) == 2

    def test_run_with_no_search_space(self) -> None:
        """Test that run works without a search space (single trial)."""
        exp = Experiment(fn=simple_function)

        results = exp.run(trials=1)

        assert len(results.trials) == 1
        # Should be called with empty params
        assert results.trials[0].params == {}

    def test_run_sets_trial_timestamps(self) -> None:
        """Test that trials have started_at and completed_at timestamps."""
        space = SearchSpace.from_dict({"lr": [0.1]})
        exp = Experiment(fn=simple_function, search_space=space)

        results = exp.run()

        trial = results.trials[0]
        assert trial.started_at is not None
        assert trial.completed_at is not None
        assert trial.completed_at >= trial.started_at


# === Test Exception Handling ===


class TestExceptionHandling:
    """Tests for exception handling during trial execution."""

    def test_run_marks_failed_trials_on_exception(self) -> None:
        """Test that exceptions mark trials as FAILED."""
        space = SearchSpace.from_dict({"lr": [0.1]})
        exp = Experiment(fn=failing_function, search_space=space)

        results = exp.run()

        assert len(results.trials) == 1
        assert results.trials[0].status == TrialStatus.FAILED
        assert results.trials[0].error is not None
        assert "Intentional failure" in results.trials[0].error

    def test_run_continues_after_failure(self) -> None:
        """Test that run continues executing after a trial fails."""
        space = SearchSpace.from_dict({"lr": [0.001, 0.1]})  # 0.001 will fail
        exp = Experiment(fn=sometimes_failing_function, search_space=space)

        results = exp.run()

        assert len(results.trials) == 2
        # One should fail, one should succeed
        failed = [t for t in results.trials if t.status == TrialStatus.FAILED]
        completed = [t for t in results.trials if t.status == TrialStatus.COMPLETED]
        assert len(failed) == 1
        assert len(completed) == 1

    def test_failed_trials_have_no_metrics(self) -> None:
        """Test that failed trials have empty metrics."""
        space = SearchSpace.from_dict({"lr": [0.1]})
        exp = Experiment(fn=failing_function, search_space=space)

        results = exp.run()

        assert results.trials[0].metrics == {}


# === Test Load Method ===


class TestExperimentLoad:
    """Tests for loading experiments from storage."""

    def test_load_restores_trials(self) -> None:
        """Test that load restores trials from storage."""
        storage = MemoryStorage()
        space = SearchSpace.from_dict({"lr": [0.1, 0.01]})
        exp = Experiment(
            fn=simple_function,
            name="test_exp",
            search_space=space,
            storage=storage,
        )
        exp.run()

        # Load from storage
        loaded = Experiment.load("test_exp", storage=storage)

        assert loaded.name == "test_exp"
        stored_trials = storage.load_trials("test_exp")
        assert len(stored_trials) == 2

    def test_load_returns_none_for_nonexistent(self) -> None:
        """Test that load returns None for non-existent experiment."""
        storage = MemoryStorage()

        loaded = Experiment.load("nonexistent", storage=storage)

        assert loaded is None

    def test_load_uses_stored_config(self) -> None:
        """Test that load uses the stored config."""
        storage = MemoryStorage()
        space = SearchSpace.from_dict({"lr": [0.1]})
        exp = Experiment(
            fn=simple_function,
            name="config_exp",
            search_space=space,
            storage=storage,
            metric="accuracy",
            maximize=True,
        )
        exp.run()

        loaded = Experiment.load("config_exp", storage=storage)

        assert loaded is not None
        assert loaded.metric == "accuracy"
        assert loaded.maximize is True


# === Test Results Integration ===


class TestResultsIntegration:
    """Tests for integration between Experiment and Results."""

    def test_results_best_returns_best_trial(self) -> None:
        """Test that results.best returns the best trial."""
        space = SearchSpace.from_dict({"lr": [0.1, 0.01, 0.001]})
        exp = Experiment(fn=simple_function, search_space=space, maximize=False)

        results = exp.run()

        best = results.best
        assert best is not None
        # lower lr = lower loss = better when minimize
        assert best.params["lr"] == 0.001

    def test_results_df_contains_all_trials(self) -> None:
        """Test that results.df contains all trials."""
        space = SearchSpace.from_dict({"lr": [0.1, 0.01]})
        exp = Experiment(fn=simple_function, search_space=space)

        results = exp.run()
        df = results.df

        assert len(df) == 2
        assert "lr" in df.columns
        assert "loss" in df.columns


# === Test Edge Cases ===


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_search_space_with_trials(self) -> None:
        """Test running with no search space but specifying trials."""
        exp = Experiment(fn=simple_function)

        results = exp.run(trials=3)

        assert len(results.trials) == 3
        # All trials should have empty params
        assert all(t.params == {} for t in results.trials)

    def test_large_grid_search(self) -> None:
        """Test that large grids are handled correctly."""
        space = SearchSpace.from_dict({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })  # 27 combinations
        exp = Experiment(fn=simple_function, search_space=space)

        results = exp.run()

        assert len(results.trials) == 27

    def test_run_with_zero_trials(self) -> None:
        """Test running with trials=0."""
        space = SearchSpace.from_dict({"lr": Log(1e-5, 1e-1)})
        exp = Experiment(fn=simple_function, search_space=space)

        results = exp.run(trials=0)

        assert len(results.trials) == 0

    def test_function_returning_non_dict_raises(self) -> None:
        """Test that a function returning non-dict is handled gracefully."""
        def bad_function(params: Dict[str, Any]) -> int:
            return 42  # type: ignore

        space = SearchSpace.from_dict({"lr": [0.1]})
        exp = Experiment(fn=bad_function, search_space=space)

        results = exp.run()

        # Should be marked as failed
        assert len(results.trials) == 1
        assert results.trials[0].status == TrialStatus.FAILED
