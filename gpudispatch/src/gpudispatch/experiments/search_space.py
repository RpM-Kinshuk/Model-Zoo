"""Search space primitives for experiment configuration.

This module provides building blocks for defining hyperparameter search spaces:

- Distribution: Abstract base class for all distributions
- Choice: Categorical choice from a list
- Log: Log-uniform distribution (for learning rates)
- Uniform: Uniform distribution
- Int: Integer range [low, high] inclusive
- Range: Stepped range (like arange but inclusive)
- Grid: Exhaustive combinations
- Sweep: Sampled distributions
- SearchSpace: Combined grid + sweep
"""

from __future__ import annotations

import itertools
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union


class Distribution(ABC):
    """Abstract base class for all distributions.

    All distributions must implement:
    - sample(): Return a random sample from the distribution
    - contains(value): Check if a value is within the distribution's range
    """

    @abstractmethod
    def sample(self) -> Any:
        """Return a random sample from the distribution."""
        pass

    @abstractmethod
    def contains(self, value: Any) -> bool:
        """Check if value is within the distribution's range."""
        pass


class Choice(Distribution):
    """Categorical choice from a list of values.

    Example:
        >>> c = Choice([16, 32, 64])
        >>> c.sample()  # Returns one of 16, 32, 64
    """

    def __init__(self, values: List[Any]) -> None:
        """Initialize with list of possible values.

        Args:
            values: List of values to choose from.
        """
        self._values = list(values)

    @property
    def values(self) -> List[Any]:
        """Return the list of possible values."""
        return self._values

    def sample(self) -> Any:
        """Return a random choice from the values."""
        return random.choice(self._values)

    def contains(self, value: Any) -> bool:
        """Check if value is one of the choices."""
        return value in self._values

    def __len__(self) -> int:
        """Return the number of choices."""
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all choices."""
        return iter(self._values)


class Log(Distribution):
    """Log-uniform distribution (for learning rates and other scale parameters).

    Samples are drawn uniformly in log-space, meaning each order of magnitude
    has roughly equal probability.

    Example:
        >>> log_dist = Log(1e-5, 1e-1)
        >>> log_dist.sample()  # Value in [1e-5, 1e-1] with log scale
    """

    def __init__(self, low: float, high: float) -> None:
        """Initialize log-uniform distribution.

        Args:
            low: Lower bound (must be positive).
            high: Upper bound (must be greater than low).
        """
        if low <= 0:
            raise ValueError("low must be positive for log distribution")
        if high <= low:
            raise ValueError("high must be greater than low")
        self._low = low
        self._high = high
        self._log_low = math.log(low)
        self._log_high = math.log(high)

    @property
    def low(self) -> float:
        """Return the lower bound."""
        return self._low

    @property
    def high(self) -> float:
        """Return the upper bound."""
        return self._high

    def sample(self) -> float:
        """Return a log-uniform sample in [low, high]."""
        log_sample = random.uniform(self._log_low, self._log_high)
        return math.exp(log_sample)

    def contains(self, value: Any) -> bool:
        """Check if value is in [low, high]."""
        if not isinstance(value, (int, float)):
            return False
        return self._low <= value <= self._high


class Uniform(Distribution):
    """Uniform distribution over a continuous range.

    Example:
        >>> u = Uniform(0.0, 0.5)
        >>> u.sample()  # Value in [0.0, 0.5]
    """

    def __init__(self, low: float, high: float) -> None:
        """Initialize uniform distribution.

        Args:
            low: Lower bound.
            high: Upper bound (must be >= low).
        """
        if high < low:
            raise ValueError("high must be >= low")
        self._low = low
        self._high = high

    @property
    def low(self) -> float:
        """Return the lower bound."""
        return self._low

    @property
    def high(self) -> float:
        """Return the upper bound."""
        return self._high

    def sample(self) -> float:
        """Return a uniform sample in [low, high]."""
        return random.uniform(self._low, self._high)

    def contains(self, value: Any) -> bool:
        """Check if value is in [low, high]."""
        if not isinstance(value, (int, float)):
            return False
        return self._low <= value <= self._high


class Int(Distribution):
    """Integer distribution over a range [low, high] inclusive.

    Example:
        >>> i = Int(4, 32)
        >>> i.sample()  # Integer in [4, 32]
    """

    def __init__(self, low: int, high: int) -> None:
        """Initialize integer distribution.

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (inclusive, must be >= low).
        """
        if high < low:
            raise ValueError("high must be >= low")
        self._low = low
        self._high = high

    @property
    def low(self) -> int:
        """Return the lower bound."""
        return self._low

    @property
    def high(self) -> int:
        """Return the upper bound."""
        return self._high

    def sample(self) -> int:
        """Return a random integer in [low, high]."""
        return random.randint(self._low, self._high)

    def contains(self, value: Any) -> bool:
        """Check if value is an integer in [low, high]."""
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        return self._low <= value <= self._high


class Range(Distribution):
    """Stepped range distribution (like numpy.arange but inclusive of endpoint).

    Creates a discrete set of values from start to stop (inclusive) with step.

    Example:
        >>> r = Range(0.1, 1.0, 0.1)  # [0.1, 0.2, ..., 1.0]
        >>> r.sample()  # One of the values
    """

    def __init__(
        self, start: Union[int, float], stop: Union[int, float], step: Union[int, float]
    ) -> None:
        """Initialize stepped range.

        Args:
            start: Starting value.
            stop: Ending value (inclusive).
            step: Step size.
        """
        self._start = start
        self._stop = stop
        self._step = step
        self._values = self._generate_values()

    def _generate_values(self) -> List[Union[int, float]]:
        """Generate all values in the range."""
        values: List[Union[int, float]] = []
        # Use a tolerance-based approach to handle floating point issues
        num_steps = int(round((self._stop - self._start) / self._step)) + 1
        for i in range(num_steps):
            value = self._start + i * self._step
            values.append(value)
        return values

    @property
    def values(self) -> List[Union[int, float]]:
        """Return the list of values in the range."""
        return self._values

    def sample(self) -> Union[int, float]:
        """Return a random value from the range."""
        return random.choice(self._values)

    def contains(self, value: Any) -> bool:
        """Check if value is in the range (with floating point tolerance)."""
        if not isinstance(value, (int, float)):
            return False
        return any(math.isclose(value, v, rel_tol=1e-9) for v in self._values)

    def __len__(self) -> int:
        """Return the number of values in the range."""
        return len(self._values)

    def __iter__(self) -> Iterator[Union[int, float]]:
        """Iterate over all values in the range."""
        return iter(self._values)


class Grid:
    """Exhaustive grid of parameter combinations.

    Example:
        >>> g = Grid(lr=[1e-4, 1e-3], batch=[16, 32])
        >>> list(g)  # All 4 combinations
        >>> g.size  # 4
    """

    def __init__(self, **kwargs: List[Any]) -> None:
        """Initialize grid with parameter lists.

        Args:
            **kwargs: Parameter names mapped to lists of values.
        """
        self._params: Dict[str, List[Any]] = dict(kwargs)

    @property
    def parameters(self) -> List[str]:
        """Return list of parameter names."""
        return list(self._params.keys())

    @property
    def size(self) -> int:
        """Return total number of combinations."""
        if not self._params:
            return 0
        result = 1
        for values in self._params.values():
            result *= len(values)
        return result

    def __len__(self) -> int:
        """Return total number of combinations."""
        return self.size

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all parameter combinations."""
        if not self._params:
            return iter([])

        keys = list(self._params.keys())
        value_lists = [self._params[k] for k in keys]

        for combo in itertools.product(*value_lists):
            yield dict(zip(keys, combo))


class Sweep:
    """Sampled parameter distributions.

    Example:
        >>> s = Sweep(lr=Log(1e-5, 1e-1), dropout=Uniform(0.0, 0.5))
        >>> s.sample()  # {"lr": 0.001, "dropout": 0.3}
    """

    def __init__(self, **kwargs: Distribution) -> None:
        """Initialize sweep with parameter distributions.

        Args:
            **kwargs: Parameter names mapped to Distribution instances.
        """
        self._distributions: Dict[str, Distribution] = dict(kwargs)

    @property
    def parameters(self) -> List[str]:
        """Return list of parameter names."""
        return list(self._distributions.keys())

    @property
    def distributions(self) -> Dict[str, Distribution]:
        """Return the parameter distributions."""
        return self._distributions

    def sample(self) -> Dict[str, Any]:
        """Sample from all distributions.

        Returns:
            Dict mapping parameter names to sampled values.
        """
        return {name: dist.sample() for name, dist in self._distributions.items()}


class SearchSpace:
    """Combined grid + sweep search space.

    SearchSpace separates parameters into:
    - Grid parameters: Lists of values for exhaustive search
    - Sweep parameters: Distributions for random sampling

    Example:
        >>> space = SearchSpace.from_dict({
        ...     "model": ["small", "large"],  # Grid
        ...     "lr": Log(1e-5, 1e-1),         # Sweep
        ... })
        >>> space.has_grid  # True
        >>> space.has_sweep  # True
        >>> space.is_grid  # False (has both)
    """

    def __init__(
        self,
        grid: Optional[Grid] = None,
        sweep: Optional[Sweep] = None,
    ) -> None:
        """Initialize search space with grid and/or sweep components.

        Args:
            grid: Grid component for exhaustive search.
            sweep: Sweep component for random sampling.
        """
        self._grid = grid
        self._sweep = sweep

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "SearchSpace":
        """Create SearchSpace from a dictionary of parameters.

        Lists become grid parameters, Distribution instances become sweep parameters.

        Args:
            params: Dict mapping parameter names to lists or Distribution instances.

        Returns:
            SearchSpace with grid and sweep components.
        """
        grid_params: Dict[str, List[Any]] = {}
        sweep_params: Dict[str, Distribution] = {}

        for name, value in params.items():
            if isinstance(value, Distribution):
                sweep_params[name] = value
            elif isinstance(value, list):
                grid_params[name] = value
            else:
                raise ValueError(
                    f"Parameter {name} must be a list or Distribution, got {type(value)}"
                )

        grid = Grid(**grid_params) if grid_params else None
        sweep = Sweep(**sweep_params) if sweep_params else None

        return cls(grid=grid, sweep=sweep)

    @property
    def has_grid(self) -> bool:
        """Check if search space has a grid component."""
        return self._grid is not None and self._grid.size > 0

    @property
    def has_sweep(self) -> bool:
        """Check if search space has a sweep component."""
        return self._sweep is not None and len(self._sweep.parameters) > 0

    @property
    def is_grid(self) -> bool:
        """Check if search space is pure grid (no sweep component)."""
        return self.has_grid and not self.has_sweep

    @property
    def grid_size(self) -> int:
        """Return the size of the grid component."""
        if self._grid is None:
            return 0
        return self._grid.size

    @property
    def parameters(self) -> List[str]:
        """Return all parameter names."""
        params: List[str] = []
        if self._grid is not None:
            params.extend(self._grid.parameters)
        if self._sweep is not None:
            params.extend(self._sweep.parameters)
        return params

    def iter_grid(self) -> Iterator[Dict[str, Any]]:
        """Iterate over grid points.

        Yields:
            Dict of grid parameter values for each combination.
        """
        if self._grid is not None:
            yield from self._grid
        else:
            yield {}

    def sample(self) -> Dict[str, Any]:
        """Sample a complete configuration.

        If grid exists, picks a random grid point.
        If sweep exists, samples from all distributions.
        Combines both if both exist.

        Returns:
            Dict mapping all parameter names to values.
        """
        result: Dict[str, Any] = {}

        # Sample from grid (pick a random point)
        if self._grid is not None and self._grid.size > 0:
            grid_points = list(self._grid)
            result.update(random.choice(grid_points))

        # Sample from sweep
        if self._sweep is not None:
            result.update(self._sweep.sample())

        return result
