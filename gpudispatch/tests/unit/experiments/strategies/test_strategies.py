"""Tests for search strategy implementations."""

import random
from typing import Any, Dict, List, Optional

import pytest

from gpudispatch.experiments.search_space import (
    Choice,
    Log,
    SearchSpace,
    Uniform,
)
from gpudispatch.experiments.strategies import (
    GridStrategy,
    RandomStrategy,
    Strategy,
)
from gpudispatch.experiments.trial import Trial, TrialStatus


class TestStrategyBase:
    """Test the Strategy ABC."""

    def test_strategy_is_abstract(self):
        """Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Strategy()  # type: ignore

    def test_strategy_subclass_must_implement_suggest(self):
        """Subclass without suggest raises TypeError."""

        class IncompleteStrategy(Strategy):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_strategy_subclass_must_implement_name(self):
        """Subclass without name raises TypeError."""

        class IncompleteStrategy(Strategy):
            def suggest(
                self,
                search_space: SearchSpace,
                completed_trials: List[Trial],
            ) -> Optional[Dict[str, Any]]:
                return None

        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_strategy_subclass_complete(self):
        """Complete subclass can be instantiated."""

        class CompleteStrategy(Strategy):
            @property
            def name(self) -> str:
                return "complete"

            def suggest(
                self,
                search_space: SearchSpace,
                completed_trials: List[Trial],
            ) -> Optional[Dict[str, Any]]:
                return search_space.sample()

        strategy = CompleteStrategy()
        assert strategy.name == "complete"


class TestGridStrategy:
    """Tests for GridStrategy."""

    def test_grid_strategy_name(self):
        """GridStrategy has correct name."""
        strategy = GridStrategy()
        assert strategy.name == "grid"

    def test_grid_strategy_repr(self):
        """GridStrategy has proper repr."""
        strategy = GridStrategy()
        assert repr(strategy) == "GridStrategy()"

    def test_grid_strategy_suggests_first_combo(self):
        """GridStrategy suggests first combination when no trials."""
        space = SearchSpace.from_dict({
            "lr": [0.01, 0.001],
            "batch": [16, 32],
        })
        strategy = GridStrategy()

        suggestion = strategy.suggest(space, [])

        assert suggestion is not None
        assert suggestion == {"lr": 0.01, "batch": 16}

    def test_grid_strategy_iterates_all_combinations(self):
        """GridStrategy iterates through all grid combinations."""
        space = SearchSpace.from_dict({
            "lr": [0.01, 0.001],
            "batch": [16, 32],
        })
        strategy = GridStrategy()
        trials: List[Trial] = []

        suggestions = []
        for i in range(10):  # More than 4 to ensure None is returned
            suggestion = strategy.suggest(space, trials)
            if suggestion is None:
                break
            suggestions.append(suggestion)
            trials.append(Trial(id=i, params=suggestion, status=TrialStatus.COMPLETED))

        assert len(suggestions) == 4
        assert {"lr": 0.01, "batch": 16} in suggestions
        assert {"lr": 0.01, "batch": 32} in suggestions
        assert {"lr": 0.001, "batch": 16} in suggestions
        assert {"lr": 0.001, "batch": 32} in suggestions

    def test_grid_strategy_returns_none_when_complete(self):
        """GridStrategy returns None when all combinations tried."""
        space = SearchSpace.from_dict({"x": [1, 2]})
        strategy = GridStrategy()
        trials = [
            Trial(id=0, params={"x": 1}, status=TrialStatus.COMPLETED),
            Trial(id=1, params={"x": 2}, status=TrialStatus.COMPLETED),
        ]

        suggestion = strategy.suggest(space, trials)

        assert suggestion is None

    def test_grid_strategy_skips_completed_trials(self):
        """GridStrategy skips already-completed combinations."""
        space = SearchSpace.from_dict({"x": [1, 2, 3]})
        strategy = GridStrategy()
        trials = [
            Trial(id=0, params={"x": 1}, status=TrialStatus.COMPLETED),
        ]

        suggestion = strategy.suggest(space, trials)

        assert suggestion == {"x": 2}

    def test_grid_strategy_handles_empty_grid(self):
        """GridStrategy handles empty search space."""
        space = SearchSpace.from_dict({})
        strategy = GridStrategy()

        suggestion = strategy.suggest(space, [])

        # Empty grid yields one empty combination {}
        # but iter_grid for empty space returns [{}], so first call gets {}
        # then next call should return None
        assert suggestion == {}

    def test_grid_strategy_ignores_sweep_params_in_trials(self):
        """GridStrategy matches on grid params, ignoring sweep params in trials."""
        space = SearchSpace.from_dict({
            "model": ["small", "large"],
            "lr": Log(1e-5, 1e-1),  # sweep param
        })
        strategy = GridStrategy()

        # Trial has both grid and sweep params
        trials = [
            Trial(
                id=0,
                params={"model": "small", "lr": 0.001},
                status=TrialStatus.COMPLETED,
            ),
        ]

        suggestion = strategy.suggest(space, trials)

        # Should suggest next grid point "large", ignoring sweep
        assert suggestion == {"model": "large"}

    def test_grid_strategy_single_param(self):
        """GridStrategy works with single parameter."""
        space = SearchSpace.from_dict({"opt": ["adam", "sgd", "rmsprop"]})
        strategy = GridStrategy()

        suggestions = []
        trials: List[Trial] = []
        for i in range(5):
            s = strategy.suggest(space, trials)
            if s is None:
                break
            suggestions.append(s)
            trials.append(Trial(id=i, params=s, status=TrialStatus.COMPLETED))

        assert len(suggestions) == 3
        assert suggestions == [{"opt": "adam"}, {"opt": "sgd"}, {"opt": "rmsprop"}]


class TestRandomStrategy:
    """Tests for RandomStrategy."""

    def test_random_strategy_name(self):
        """RandomStrategy has correct name."""
        strategy = RandomStrategy(n_trials=10)
        assert strategy.name == "random"

    def test_random_strategy_repr(self):
        """RandomStrategy has proper repr."""
        strategy = RandomStrategy(n_trials=5)
        assert repr(strategy) == "RandomStrategy(n_trials=5)"

    def test_random_strategy_n_trials_property(self):
        """RandomStrategy exposes n_trials property."""
        strategy = RandomStrategy(n_trials=42)
        assert strategy.n_trials == 42

    def test_random_strategy_invalid_n_trials(self):
        """RandomStrategy raises ValueError for invalid n_trials."""
        with pytest.raises(ValueError, match="must be positive"):
            RandomStrategy(n_trials=0)

        with pytest.raises(ValueError, match="must be positive"):
            RandomStrategy(n_trials=-5)

    def test_random_strategy_suggests_samples(self):
        """RandomStrategy suggests random samples from search space."""
        random.seed(42)
        space = SearchSpace.from_dict({
            "lr": Log(1e-5, 1e-1),
            "dropout": Uniform(0.0, 0.5),
        })
        strategy = RandomStrategy(n_trials=10)

        suggestion = strategy.suggest(space, [])

        assert suggestion is not None
        assert "lr" in suggestion
        assert "dropout" in suggestion
        assert 1e-5 <= suggestion["lr"] <= 1e-1
        assert 0.0 <= suggestion["dropout"] <= 0.5

    def test_random_strategy_respects_n_trials_limit(self):
        """RandomStrategy returns None after n_trials."""
        space = SearchSpace.from_dict({"x": Uniform(0, 1)})
        strategy = RandomStrategy(n_trials=3)

        trials = [
            Trial(id=i, params={"x": i * 0.1}, status=TrialStatus.COMPLETED)
            for i in range(3)
        ]

        suggestion = strategy.suggest(space, trials)

        assert suggestion is None

    def test_random_strategy_counts_all_trials(self):
        """RandomStrategy counts trials regardless of status."""
        space = SearchSpace.from_dict({"x": Uniform(0, 1)})
        strategy = RandomStrategy(n_trials=2)

        trials = [
            Trial(id=0, params={"x": 0.1}, status=TrialStatus.COMPLETED),
            Trial(id=1, params={"x": 0.2}, status=TrialStatus.FAILED),
        ]

        suggestion = strategy.suggest(space, trials)

        assert suggestion is None

    def test_random_strategy_suggests_until_limit(self):
        """RandomStrategy suggests until n_trials reached."""
        random.seed(123)
        space = SearchSpace.from_dict({"x": Choice([1, 2, 3, 4, 5])})
        strategy = RandomStrategy(n_trials=5)
        trials: List[Trial] = []

        suggestions = []
        for i in range(10):  # Try more than n_trials
            s = strategy.suggest(space, trials)
            if s is None:
                break
            suggestions.append(s)
            trials.append(Trial(id=i, params=s, status=TrialStatus.COMPLETED))

        assert len(suggestions) == 5

    def test_random_strategy_with_grid_space(self):
        """RandomStrategy works with grid-only search space."""
        random.seed(42)
        space = SearchSpace.from_dict({
            "model": ["small", "large"],
            "batch": [16, 32],
        })
        strategy = RandomStrategy(n_trials=3)

        suggestion = strategy.suggest(space, [])

        assert suggestion is not None
        assert suggestion["model"] in ["small", "large"]
        assert suggestion["batch"] in [16, 32]

    def test_random_strategy_with_mixed_space(self):
        """RandomStrategy works with mixed grid+sweep search space."""
        random.seed(42)
        space = SearchSpace.from_dict({
            "model": ["small", "large"],  # grid
            "lr": Log(1e-5, 1e-1),  # sweep
        })
        strategy = RandomStrategy(n_trials=5)

        suggestion = strategy.suggest(space, [])

        assert suggestion is not None
        assert suggestion["model"] in ["small", "large"]
        assert 1e-5 <= suggestion["lr"] <= 1e-1


class TestStrategyIntegration:
    """Integration tests for strategies."""

    def test_grid_then_random_workflow(self):
        """Test typical workflow: grid search followed by random refinement."""
        # Phase 1: Grid search over models
        grid_space = SearchSpace.from_dict({
            "model": ["small", "medium", "large"],
        })
        grid_strategy = GridStrategy()
        grid_trials: List[Trial] = []

        for i in range(5):
            s = grid_strategy.suggest(grid_space, grid_trials)
            if s is None:
                break
            grid_trials.append(Trial(id=i, params=s, status=TrialStatus.COMPLETED))

        assert len(grid_trials) == 3

        # Phase 2: Random search over hyperparameters
        random.seed(42)
        random_space = SearchSpace.from_dict({
            "lr": Log(1e-5, 1e-1),
            "dropout": Uniform(0.0, 0.5),
        })
        random_strategy = RandomStrategy(n_trials=5)
        random_trials: List[Trial] = []

        for i in range(10):
            s = random_strategy.suggest(random_space, random_trials)
            if s is None:
                break
            random_trials.append(
                Trial(id=len(grid_trials) + i, params=s, status=TrialStatus.COMPLETED)
            )

        assert len(random_trials) == 5
