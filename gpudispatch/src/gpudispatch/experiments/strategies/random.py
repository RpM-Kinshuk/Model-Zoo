"""Random search strategy implementation.

This module provides RandomStrategy which samples random parameter
configurations from the search space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gpudispatch.experiments.strategies.base import Strategy

if TYPE_CHECKING:
    from gpudispatch.experiments.search_space import SearchSpace
    from gpudispatch.experiments.trial import Trial


class RandomStrategy(Strategy):
    """Random sampling strategy.

    Samples random configurations from the search space for a fixed number
    of trials. Returns None when the specified number of trials is reached.

    Example:
        >>> from gpudispatch.experiments import SearchSpace, Log
        >>> space = SearchSpace.from_dict({"lr": Log(1e-5, 1e-1)})
        >>> strategy = RandomStrategy(n_trials=10)
        >>> strategy.suggest(space, [])  # Random sample
        {'lr': 0.00123...}
    """

    def __init__(self, n_trials: int) -> None:
        """Initialize random strategy.

        Args:
            n_trials: Maximum number of trials to run.

        Raises:
            ValueError: If n_trials is not positive.
        """
        if n_trials <= 0:
            raise ValueError("n_trials must be positive")
        self._n_trials = n_trials

    @property
    def n_trials(self) -> int:
        """Return the maximum number of trials."""
        return self._n_trials

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "random"

    def suggest(
        self,
        search_space: "SearchSpace",
        completed_trials: List["Trial"],
    ) -> Optional[Dict[str, Any]]:
        """Suggest a random configuration from the search space.

        Args:
            search_space: The search space to sample from.
            completed_trials: List of completed trials.

        Returns:
            Random parameter configuration, or None if n_trials reached.
        """
        if len(completed_trials) >= self._n_trials:
            return None

        return search_space.sample()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RandomStrategy(n_trials={self._n_trials})"
