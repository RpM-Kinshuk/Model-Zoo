"""Tests for the @experiment decorator.

This module tests:
- Basic decorator usage with grid parameters
- Decorator usage with distributions (random search)
- Access to underlying Experiment object
- Access to original function
- Trials parameter handling
- Metric and maximize options
- Multiple parameter combinations
- Edge cases
"""

from __future__ import annotations

import functools
from typing import Any, Dict

import pytest

from gpudispatch.experiments.decorator import experiment
from gpudispatch.experiments.experiment import Experiment
from gpudispatch.experiments.results import Results
from gpudispatch.experiments.search_space import Log, Uniform, Int
from gpudispatch.experiments.strategies import GridStrategy, RandomStrategy
from gpudispatch.experiments.trial import TrialStatus


# === Test Basic Decorator Usage ===


class TestBasicDecoratorUsage:
    """Tests for basic @experiment decorator functionality."""

    def test_decorator_with_grid_params(self) -> None:
        """Test decorator converts lists to grid search."""
        @experiment(lr=[0.1, 0.01])
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr * 10}

        results = train()

        assert isinstance(results, Results)
        assert len(results.trials) == 2

    def test_decorator_with_multiple_grid_params(self) -> None:
        """Test decorator handles multiple grid parameters."""
        @experiment(lr=[0.1, 0.01], batch_size=[16, 32])
        def train(lr: float, batch_size: int) -> Dict[str, Any]:
            return {"loss": lr * batch_size}

        results = train()

        # 2 x 2 = 4 combinations
        assert len(results.trials) == 4

    def test_decorated_function_returns_results(self) -> None:
        """Test that decorated function returns Results object."""
        @experiment(x=[1, 2, 3])
        def my_fn(x: int) -> Dict[str, Any]:
            return {"loss": x}

        results = my_fn()

        assert isinstance(results, Results)
        assert results.metric == "loss"

    def test_decorated_function_preserves_name(self) -> None:
        """Test that decorated function preserves original name via functools.wraps."""
        @experiment(lr=[0.1])
        def train_model(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        assert train_model.__name__ == "train_model"

    def test_decorated_function_preserves_docstring(self) -> None:
        """Test that decorated function preserves docstring."""
        @experiment(lr=[0.1])
        def train_model(lr: float) -> Dict[str, Any]:
            """Train the model with given learning rate."""
            return {"loss": lr}

        assert "Train the model" in (train_model.__doc__ or "")


# === Test Distribution Parameters ===


class TestDistributionParameters:
    """Tests for decorator with distribution parameters (sweep/random search)."""

    def test_decorator_with_log_distribution(self) -> None:
        """Test decorator with Log distribution uses random strategy."""
        @experiment(lr=Log(1e-5, 1e-1), trials=5)
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        # Access the underlying experiment
        assert isinstance(train.experiment.strategy, RandomStrategy)

    def test_decorator_with_trials_for_distributions(self) -> None:
        """Test trials parameter controls number of random samples."""
        @experiment(lr=Log(1e-5, 1e-1), trials=10)
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        results = train()

        assert len(results.trials) == 10

    def test_decorator_with_mixed_params(self) -> None:
        """Test decorator with both grid and distribution params."""
        @experiment(model=["small", "large"], lr=Log(1e-5, 1e-1), trials=4)
        def train(model: str, lr: float) -> Dict[str, Any]:
            return {"loss": lr if model == "small" else lr * 2}

        results = train()

        assert len(results.trials) == 4
        # Should use RandomStrategy for mixed spaces
        assert isinstance(train.experiment.strategy, RandomStrategy)


# === Test Experiment Access ===


class TestExperimentAccess:
    """Tests for accessing the underlying Experiment object."""

    def test_experiment_attribute_exists(self) -> None:
        """Test that decorated function has .experiment attribute."""
        @experiment(lr=[0.1])
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        assert hasattr(train, "experiment")
        assert isinstance(train.experiment, Experiment)

    def test_experiment_name_auto_generated(self) -> None:
        """Test that experiment name is auto-generated from function."""
        @experiment(lr=[0.1])
        def my_training_fn(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        assert "my_training_fn" in my_training_fn.experiment.name

    def test_original_function_accessible(self) -> None:
        """Test that original function is accessible via .original."""
        def original_train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        decorated = experiment(lr=[0.1])(original_train)

        assert hasattr(decorated, "original")
        assert decorated.original is original_train


# === Test Metric and Maximize Options ===


class TestMetricOptions:
    """Tests for metric and maximize options."""

    def test_custom_metric(self) -> None:
        """Test decorator with custom metric name."""
        @experiment(lr=[0.1, 0.01], metric="accuracy")
        def train(lr: float) -> Dict[str, Any]:
            return {"accuracy": 1 - lr, "loss": lr}

        results = train()

        assert results.metric == "accuracy"

    def test_maximize_option(self) -> None:
        """Test decorator with maximize=True."""
        @experiment(lr=[0.1, 0.01], metric="accuracy", maximize=True)
        def train(lr: float) -> Dict[str, Any]:
            return {"accuracy": 1 - lr}

        results = train()

        assert results.maximize is True
        # Best should be the one with lowest lr (highest accuracy)
        assert results.best.params["lr"] == 0.01

    def test_default_metric_is_loss(self) -> None:
        """Test that default metric is 'loss'."""
        @experiment(lr=[0.1])
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        assert train.experiment.metric == "loss"

    def test_default_maximize_is_false(self) -> None:
        """Test that default maximize is False (minimize)."""
        @experiment(lr=[0.1])
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        assert train.experiment.maximize is False


# === Test Function Signature ===


class TestFunctionSignature:
    """Tests for function receiving correct parameters."""

    def test_function_receives_params_dict(self) -> None:
        """Test that function receives params as dict."""
        received_params = []

        @experiment(lr=[0.1], batch=[32])
        def train(lr: float, batch: int) -> Dict[str, Any]:
            received_params.append({"lr": lr, "batch": batch})
            return {"loss": lr}

        train()

        assert len(received_params) == 1
        assert received_params[0] == {"lr": 0.1, "batch": 32}

    def test_function_uses_kwargs_unpacking(self) -> None:
        """Test that parameters are unpacked as kwargs."""
        @experiment(a=[1], b=[2], c=[3])
        def my_fn(a: int, b: int, c: int) -> Dict[str, Any]:
            return {"loss": a + b + c}

        results = my_fn()

        assert results.trials[0].metrics["loss"] == 6


# === Test Edge Cases ===


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_decorator_without_params(self) -> None:
        """Test decorator can be used without search params."""
        @experiment()
        def train() -> Dict[str, Any]:
            return {"loss": 0.5}

        results = train(trials=1)

        assert len(results.trials) == 1

    def test_empty_grid_param(self) -> None:
        """Test that empty list results in failed trials (missing arg)."""
        # Empty list creates an empty search space - but function still expects arg
        @experiment(lr=[])
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        # With empty grid, it runs trials with empty params, which fails
        # because the function expects 'lr'
        results = train(trials=1)
        assert len(results.trials) == 1
        assert results.trials[0].status == TrialStatus.FAILED

    def test_trials_parameter_override(self) -> None:
        """Test that trials parameter passed to __call__ works."""
        @experiment(lr=Log(1e-5, 1e-1))
        def train(lr: float) -> Dict[str, Any]:
            return {"loss": lr}

        # Default behavior without specifying trials
        results = train(trials=3)
        assert len(results.trials) == 3

    def test_decorator_handles_exception_in_fn(self) -> None:
        """Test decorator handles exceptions in decorated function."""
        @experiment(lr=[0.1, 0.01])
        def failing_train(lr: float) -> Dict[str, Any]:
            if lr < 0.05:
                raise ValueError("lr too small")
            return {"loss": lr}

        results = failing_train()

        # Should have 2 trials, one failed
        assert len(results.trials) == 2
        failed = [t for t in results.trials if t.status == TrialStatus.FAILED]
        assert len(failed) == 1


# === Test Decorator Call Styles ===


class TestDecoratorCallStyles:
    """Tests for different decorator call styles."""

    def test_decorator_with_parens_no_args(self) -> None:
        """Test @experiment() with empty parens."""
        @experiment()
        def train() -> Dict[str, Any]:
            return {"loss": 0.5}

        assert hasattr(train, "experiment")

    def test_decorator_with_only_options(self) -> None:
        """Test @experiment with only metric/maximize options."""
        @experiment(metric="accuracy", maximize=True)
        def train() -> Dict[str, Any]:
            return {"accuracy": 0.9}

        results = train(trials=1)

        assert results.metric == "accuracy"
        assert results.maximize is True
