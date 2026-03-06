"""Grid search strategy implementation.

This module provides GridStrategy which exhaustively iterates through all
parameter combinations defined in the search space's grid component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gpudispatch.experiments.strategies.base import Strategy

if TYPE_CHECKING:
    from gpudispatch.experiments.search_space import SearchSpace
    from gpudispatch.experiments.trial import Trial


class GridStrategy(Strategy):
    """Exhaustive grid search strategy.

    Iterates through all combinations of grid parameters in the search space.
    Returns None when all combinations have been tried.

    For search spaces with sweep (random) parameters, the grid strategy only
    considers the grid component and ignores sweep parameters.

    Example:
        >>> from gpudispatch.experiments import SearchSpace
        >>> space = SearchSpace.from_dict({"lr": [0.01, 0.001], "batch": [16, 32]})
        >>> strategy = GridStrategy()
        >>> strategy.suggest(space, [])  # First combination
        {'lr': 0.01, 'batch': 16}
    """

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return "grid"

    def suggest(
        self,
        search_space: "SearchSpace",
        completed_trials: List["Trial"],
    ) -> Optional[Dict[str, Any]]:
        """Suggest the next grid combination to try.

        Args:
            search_space: The search space with grid parameters.
            completed_trials: List of completed trials.

        Returns:
            Next untried grid combination, or None if all tried.
        """
        # Get all grid points
        all_combinations = list(search_space.iter_grid())

        if not all_combinations:
            return None

        # Get params from completed trials
        completed_params = [trial.params for trial in completed_trials]

        # Find first combination not yet tried
        for combo in all_combinations:
            if not self._params_match_any(combo, completed_params):
                return combo

        # All combinations have been tried
        return None

    def _params_match_any(
        self,
        params: Dict[str, Any],
        completed_params: List[Dict[str, Any]],
    ) -> bool:
        """Check if params match any of the completed params.

        Only compares keys present in params (grid parameters).

        Args:
            params: Parameters to check.
            completed_params: List of completed parameter sets.

        Returns:
            True if params match any completed set.
        """
        for completed in completed_params:
            if self._params_equal(params, completed):
                return True
        return False

    def _params_equal(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
    ) -> bool:
        """Check if two param dicts have equal values for shared keys.

        Only compares keys present in params1.

        Args:
            params1: First parameter dict (grid params).
            params2: Second parameter dict (may have additional sweep params).

        Returns:
            True if all keys in params1 have equal values in params2.
        """
        for key, value in params1.items():
            if key not in params2:
                return False
            if params2[key] != value:
                return False
        return True

    def __repr__(self) -> str:
        """Return string representation."""
        return "GridStrategy()"
