"""Core Experiment class for orchestrating hyperparameter search trials.

This module provides the Experiment class which:
- Orchestrates the execution of trials
- Manages the search space, strategy, and storage
- Captures metrics from trial function calls
- Handles exceptions gracefully
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from gpudispatch.experiments.results import Results
from gpudispatch.experiments.search_space import SearchSpace
from gpudispatch.experiments.storage import MemoryStorage, Storage
from gpudispatch.experiments.strategies import GridStrategy, RandomStrategy, Strategy
from gpudispatch.experiments.trial import Trial, TrialStatus
from gpudispatch.observability.hooks import hooks


class Experiment:
    """Core class for running hyperparameter search experiments.

    An Experiment orchestrates the execution of trials over a search space,
    using a strategy to suggest parameter configurations and storing results.

    Example:
        >>> def train(params):
        ...     return {"loss": params["lr"] * 10}
        ...
        >>> space = SearchSpace.from_dict({"lr": [0.1, 0.01]})
        >>> exp = Experiment(fn=train, search_space=space)
        >>> results = exp.run()
        >>> print(results.best_params)
    """

    def __init__(
        self,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        name: Optional[str] = None,
        search_space: Optional[SearchSpace] = None,
        strategy: Optional[Strategy] = None,
        storage: Optional[Storage] = None,
        metric: str = "loss",
        maximize: bool = False,
        gpu: int = 0,
    ) -> None:
        """Initialize an Experiment.

        Args:
            fn: The function to optimize. Should take a params dict and return
                a dict of metrics.
            name: Name for the experiment. Auto-generated from fn if not provided.
            search_space: Search space defining parameters to optimize.
            strategy: Strategy for suggesting configurations. Auto-selected based
                on search space if not provided.
            storage: Storage backend for persisting trials. Defaults to MemoryStorage.
            metric: Name of the metric to optimize. Defaults to "loss".
            maximize: Whether to maximize the metric. Defaults to False (minimize).
            gpu: GPU device to use. Defaults to 0.
        """
        self._fn = fn
        self._name = name or self._generate_name(fn)
        self._search_space = search_space or SearchSpace()
        self._strategy = strategy or self._default_strategy(self._search_space)
        self._storage = storage or MemoryStorage()
        self._metric = metric
        self._maximize = maximize
        self._gpu = gpu
        self._trials: List[Trial] = []
        self._next_trial_id = 1

        # Save initial config
        self._save_config()

    @property
    def fn(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Return the objective function."""
        return self._fn

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return self._name

    @property
    def search_space(self) -> SearchSpace:
        """Return the search space."""
        return self._search_space

    @property
    def strategy(self) -> Strategy:
        """Return the search strategy."""
        return self._strategy

    @property
    def storage(self) -> Storage:
        """Return the storage backend."""
        return self._storage

    @property
    def metric(self) -> str:
        """Return the metric to optimize."""
        return self._metric

    @property
    def maximize(self) -> bool:
        """Return whether to maximize the metric."""
        return self._maximize

    @property
    def gpu(self) -> int:
        """Return the GPU device index."""
        return self._gpu

    def run(self, trials: Optional[int] = None) -> Results:
        """Run the experiment.

        Executes trials suggested by the strategy, captures metrics from the
        objective function, and saves results to storage.

        Args:
            trials: Maximum number of trials to run. If None and using GridStrategy,
                runs all grid combinations. If None and using RandomStrategy,
                uses the strategy's n_trials.

        Returns:
            Results object containing all executed trials.
        """
        run_started_at = datetime.now()
        hooks.emit(
            "on_experiment_start",
            experiment_id=self._name,
            metric=self._metric,
            maximize=self._maximize,
            requested_trials=trials,
        )

        try:
            # Handle trials=0 case
            if trials == 0:
                results = Results(
                    trials=[],
                    metric=self._metric,
                    maximize=self._maximize,
                    experiment_name=self._name,
                )
                hooks.emit(
                    "on_experiment_complete",
                    experiment_id=self._name,
                    total_jobs=0,
                    successful_trials=0,
                    failed_trials=0,
                    runtime_seconds=(datetime.now() - run_started_at).total_seconds(),
                )
                return results

            # Determine trial count based on strategy type
            trial_count = 0
            max_trials = trials

            # For grid strategies without explicit trial count, run all combinations
            if max_trials is None and isinstance(self._strategy, GridStrategy):
                max_trials = self._search_space.grid_size or 1

            # For random strategies without explicit trial count, use strategy's n_trials
            if max_trials is None and isinstance(self._strategy, RandomStrategy):
                max_trials = self._strategy.n_trials

            # Default to 1 trial if nothing else specified
            if max_trials is None:
                max_trials = 1

            while trial_count < max_trials:
                # Get next suggested params
                params = self._strategy.suggest(self._search_space, self._trials)

                # If strategy returns None, it's exhausted
                if params is None:
                    break

                # Execute trial
                trial = self._execute_trial(params)
                self._trials.append(trial)

                # Save to storage
                self._storage.save_trial(self._name, trial)

                trial_count += 1

            results = Results(
                trials=list(self._trials),
                metric=self._metric,
                maximize=self._maximize,
                experiment_name=self._name,
            )

            successful_trials = len(
                [trial for trial in self._trials if trial.status == TrialStatus.COMPLETED]
            )
            failed_trials = len(
                [trial for trial in self._trials if trial.status == TrialStatus.FAILED]
            )

            hooks.emit(
                "on_experiment_complete",
                experiment_id=self._name,
                total_jobs=len(self._trials),
                successful_trials=successful_trials,
                failed_trials=failed_trials,
                runtime_seconds=(datetime.now() - run_started_at).total_seconds(),
                metric=self._metric,
                maximize=self._maximize,
            )
            return results

        except Exception as exc:
            hooks.emit(
                "on_experiment_failed",
                experiment_id=self._name,
                error=str(exc),
                runtime_seconds=(datetime.now() - run_started_at).total_seconds(),
                metric=self._metric,
                maximize=self._maximize,
            )
            raise

    @classmethod
    def load(cls, name: str, storage: Optional[Storage] = None) -> Optional["Experiment"]:
        """Load an experiment from storage.

        Args:
            name: Name of the experiment to load.
            storage: Storage backend to load from. Defaults to MemoryStorage.

        Returns:
            Experiment instance if found, None otherwise.
        """
        storage = storage or MemoryStorage()

        # Check if experiment exists
        config = storage.load_config(name)
        if config is None:
            return None

        # Create experiment with stored config
        # Note: fn is not stored, so we use a placeholder
        def placeholder_fn(params: Dict[str, Any]) -> Dict[str, Any]:
            raise RuntimeError("Loaded experiment has no function - set fn manually")

        exp = cls(
            fn=placeholder_fn,
            name=name,
            storage=storage,
            metric=config.get("metric", "loss"),
            maximize=config.get("maximize", False),
            gpu=config.get("gpu", 0),
        )

        # Load existing trials
        exp._trials = storage.load_trials(name)
        if exp._trials:
            exp._next_trial_id = max(t.id for t in exp._trials) + 1

        return exp

    def _generate_name(self, fn: Callable) -> str:
        """Generate experiment name from function name and timestamp.

        Args:
            fn: The objective function.

        Returns:
            Generated name in format "{fn_name}_{timestamp}".
        """
        fn_name = getattr(fn, "__name__", "experiment")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{fn_name}_{timestamp}"

    def _default_strategy(self, search_space: SearchSpace) -> Strategy:
        """Select default strategy based on search space type.

        Args:
            search_space: The search space to analyze.

        Returns:
            GridStrategy for pure grid spaces, RandomStrategy otherwise.
        """
        # Use GridStrategy for pure grid search spaces
        if search_space.is_grid:
            return GridStrategy()

        # Use RandomStrategy for sweep or mixed spaces
        # Default to 10 trials for random search
        return RandomStrategy(n_trials=10)

    def _execute_trial(self, params: Dict[str, Any]) -> Trial:
        """Execute a single trial with the given parameters.

        Args:
            params: Parameter configuration for this trial.

        Returns:
            Trial object with results or error.
        """
        trial = Trial(
            id=self._next_trial_id,
            params=params,
            status=TrialStatus.RUNNING,
            started_at=datetime.now(),
        )
        self._next_trial_id += 1

        hooks.emit(
            "on_experiment_trial_start",
            experiment_id=self._name,
            trial_id=trial.id,
            params=params,
        )

        try:
            result = self._fn(params)

            # Validate result is a dict
            if not isinstance(result, dict):
                raise TypeError(
                    f"Function must return a dict of metrics, got {type(result).__name__}"
                )

            trial.metrics = result
            trial.status = TrialStatus.COMPLETED

            hooks.emit(
                "on_experiment_trial_complete",
                experiment_id=self._name,
                trial_id=trial.id,
                params=params,
                metrics=trial.metrics,
                runtime_seconds=(datetime.now() - trial.started_at).total_seconds(),
            )

        except Exception as e:
            trial.status = TrialStatus.FAILED
            trial.error = str(e)

            hooks.emit(
                "on_experiment_trial_failed",
                experiment_id=self._name,
                trial_id=trial.id,
                params=params,
                error=trial.error,
                runtime_seconds=(datetime.now() - trial.started_at).total_seconds(),
            )

        trial.completed_at = datetime.now()
        return trial

    def _save_config(self) -> None:
        """Save experiment configuration to storage."""
        config = {
            "name": self._name,
            "metric": self._metric,
            "maximize": self._maximize,
            "gpu": self._gpu,
            "strategy": self._strategy.name,
        }
        self._storage.save_config(self._name, config)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Experiment(name={self._name!r}, "
            f"metric={self._metric!r}, "
            f"maximize={self._maximize}, "
            f"strategy={self._strategy.name!r})"
        )
