"""Experiment decorator for easy function-based experiment definition.

This module provides a decorator that turns a function into an experiment,
with support for both grid search (lists) and random search (distributions).

Example:
    >>> @experiment(lr=[1e-4, 1e-3], batch_size=[16, 32])
    ... def train(lr, batch_size):
    ...     return {"loss": train_model(lr, batch_size)}
    ...
    >>> results = train()  # Runs all 4 combinations
    >>> train.experiment.name  # Auto-generated name
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from gpudispatch.experiments.experiment import Experiment
from gpudispatch.experiments.results import Results
from gpudispatch.experiments.search_space import Distribution, SearchSpace

F = TypeVar("F", bound=Callable[..., Dict[str, Any]])


def experiment(
    _fn: Optional[F] = None,
    *,
    trials: Optional[int] = None,
    metric: str = "loss",
    maximize: bool = False,
    **search_params: Union[list, Distribution],
) -> Union[F, Callable[[F], F]]:
    """Decorator to turn a function into an experiment.

    Lists become grid search parameters, Distribution instances become
    random search parameters.

    Args:
        _fn: The function to decorate (set automatically when used without parens).
        trials: Maximum number of trials to run. For grid search, defaults to
            all combinations. For random search, required or uses strategy default.
        metric: Name of the metric to optimize. Defaults to "loss".
        maximize: Whether to maximize the metric. Defaults to False (minimize).
        **search_params: Parameter name to list or Distribution mapping.

    Returns:
        Decorated function that runs the experiment when called.

    Example:
        >>> # Grid search
        >>> @experiment(lr=[1e-4, 1e-3], batch_size=[16, 32])
        ... def train(lr, batch_size):
        ...     return {"loss": train_model(lr, batch_size)}
        ...
        >>> results = train()

        >>> # Random search with distributions
        >>> @experiment(lr=Log(1e-5, 1e-1), trials=20)
        ... def train(lr):
        ...     return {"loss": ...}
        ...
        >>> results = train()

        >>> # Access underlying experiment
        >>> train.experiment.name
    """
    def decorator(fn: F) -> F:
        # Build SearchSpace from search_params
        space = SearchSpace.from_dict(search_params) if search_params else SearchSpace()

        # Wrap the function to accept kwargs and return a dict
        def objective(params: Dict[str, Any]) -> Dict[str, Any]:
            return fn(**params)

        # Create Experiment
        exp = Experiment(
            fn=objective,
            name=None,  # Auto-generate from fn name
            search_space=space,
            metric=metric,
            maximize=maximize,
        )
        # Override the auto-generated name to use the original function name
        exp._name = exp._generate_name(fn)

        @functools.wraps(fn)
        def wrapper(*, trials: Optional[int] = trials) -> Results:
            return exp.run(trials=trials)

        # Attach experiment and original function as attributes
        wrapper.experiment = exp  # type: ignore[attr-defined]
        wrapper.original = fn  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    # Handle both @experiment and @experiment() syntax
    if _fn is not None:
        return decorator(_fn)
    return decorator
