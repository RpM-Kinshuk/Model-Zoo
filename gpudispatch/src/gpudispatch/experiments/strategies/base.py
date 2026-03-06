"""Abstract base class for search strategies.

This module provides the Strategy ABC that all search strategies must implement.
Strategies are responsible for suggesting parameter configurations to try,
based on the search space and completed trials.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from gpudispatch.experiments.search_space import SearchSpace
    from gpudispatch.experiments.trial import Trial


class Strategy(ABC):
    """Abstract base class for hyperparameter search strategies.

    A strategy is responsible for suggesting the next set of parameters to try
    based on the search space definition and the history of completed trials.

    Subclasses must implement:
    - suggest(): Propose the next parameter configuration
    - name: Return the strategy name

    Example:
        >>> class MyStrategy(Strategy):
        ...     def suggest(self, search_space, completed_trials):
        ...         return search_space.sample()
        ...     @property
        ...     def name(self):
        ...         return "my_strategy"
    """

    @abstractmethod
    def suggest(
        self,
        search_space: "SearchSpace",
        completed_trials: List["Trial"],
    ) -> Optional[Dict[str, Any]]:
        """Suggest the next parameters to try.

        Args:
            search_space: The search space defining valid parameter ranges.
            completed_trials: List of trials that have already been run.

        Returns:
            Dictionary of parameter values for the next trial, or None if
            the search is complete (e.g., all grid combinations tried).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the strategy.

        Returns:
            A string identifier for this strategy (e.g., "grid", "random").
        """
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}()"
