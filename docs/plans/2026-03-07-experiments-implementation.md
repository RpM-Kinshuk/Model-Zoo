# Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a hybrid experiment API with `@experiment` decorator for beginners and `Experiment` class for experts.

**Architecture:** Search space primitives define what to explore, strategies decide how to explore, storage persists results. The decorator is syntactic sugar over the class. All experiments auto-save and capture reproducibility info.

**Tech Stack:** Python 3.9+, pandas, optuna (for bayesian), sqlite3, pytest

---

## Phase 3 Task Overview

| Task | Component | Estimated Tests |
|------|-----------|-----------------|
| 3.1 | Search Space Primitives | 20 |
| 3.2 | Trial and Results | 15 |
| 3.3 | Storage Backends | 18 |
| 3.4 | Strategy Interface + Grid/Random | 16 |
| 3.5 | Bayesian Strategy (Optuna) | 10 |
| 3.6 | Experiment Class | 20 |
| 3.7 | @experiment Decorator | 12 |
| 3.8 | Reproducibility | 10 |
| 3.9 | Experiment Registry (load/list/compare) | 12 |

---

## Task 3.1: Search Space Primitives

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/__init__.py`
- Create: `gpudispatch/src/gpudispatch/experiments/search_space.py`
- Create: `gpudispatch/tests/unit/experiments/__init__.py`
- Create: `gpudispatch/tests/unit/experiments/test_search_space.py`

**Step 1: Write failing tests for distribution types**

```python
# tests/unit/experiments/test_search_space.py
"""Tests for search space primitives."""

import pytest
from gpudispatch.experiments.search_space import (
    Grid, Sweep, Choice, Log, Uniform, Int, Range,
    SearchSpace, Distribution,
)


class TestChoice:
    def test_choice_creation(self):
        c = Choice([16, 32, 64])
        assert c.values == [16, 32, 64]

    def test_choice_sample(self):
        c = Choice(["a", "b", "c"])
        sample = c.sample()
        assert sample in ["a", "b", "c"]

    def test_choice_contains(self):
        c = Choice([1, 2, 3])
        assert c.contains(2)
        assert not c.contains(4)


class TestLog:
    def test_log_creation(self):
        l = Log(1e-5, 1e-1)
        assert l.low == 1e-5
        assert l.high == 1e-1

    def test_log_sample_in_range(self):
        l = Log(1e-5, 1e-1)
        for _ in range(100):
            sample = l.sample()
            assert 1e-5 <= sample <= 1e-1

    def test_log_contains(self):
        l = Log(1e-5, 1e-1)
        assert l.contains(1e-3)
        assert not l.contains(1.0)


class TestUniform:
    def test_uniform_creation(self):
        u = Uniform(0.0, 1.0)
        assert u.low == 0.0
        assert u.high == 1.0

    def test_uniform_sample_in_range(self):
        u = Uniform(0.0, 0.5)
        for _ in range(100):
            sample = u.sample()
            assert 0.0 <= sample <= 0.5


class TestInt:
    def test_int_creation(self):
        i = Int(1, 10)
        assert i.low == 1
        assert i.high == 10

    def test_int_sample_is_integer(self):
        i = Int(1, 100)
        for _ in range(100):
            sample = i.sample()
            assert isinstance(sample, int)
            assert 1 <= sample <= 100


class TestRange:
    def test_range_creation(self):
        r = Range(0.1, 1.0, 0.1)
        assert r.start == 0.1
        assert r.stop == 1.0
        assert r.step == 0.1

    def test_range_values(self):
        r = Range(0.0, 0.3, 0.1)
        values = r.to_list()
        assert len(values) == 4  # 0.0, 0.1, 0.2, 0.3
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(0.3)


class TestGrid:
    def test_grid_creation(self):
        g = Grid(lr=[1e-4, 1e-3], batch=[16, 32])
        assert g.params == {"lr": [1e-4, 1e-3], "batch": [16, 32]}

    def test_grid_size(self):
        g = Grid(a=[1, 2], b=[3, 4, 5])
        assert g.size == 6  # 2 * 3

    def test_grid_iter(self):
        g = Grid(a=[1, 2], b=["x", "y"])
        combos = list(g)
        assert len(combos) == 4
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 2, "b": "y"} in combos


class TestSweep:
    def test_sweep_creation(self):
        s = Sweep(lr=Log(1e-5, 1e-1), dropout=Uniform(0.0, 0.5))
        assert "lr" in s.params
        assert "dropout" in s.params

    def test_sweep_sample(self):
        s = Sweep(lr=Log(1e-5, 1e-1), batch=Choice([16, 32]))
        sample = s.sample()
        assert "lr" in sample
        assert "batch" in sample
        assert 1e-5 <= sample["lr"] <= 1e-1
        assert sample["batch"] in [16, 32]


class TestSearchSpace:
    def test_from_dict_lists_become_grid(self):
        space = SearchSpace.from_dict({"lr": [1e-4, 1e-3]})
        assert space.is_grid

    def test_from_dict_distributions_become_sweep(self):
        space = SearchSpace.from_dict({"lr": Log(1e-5, 1e-1)})
        assert space.is_sweep

    def test_from_dict_mixed(self):
        space = SearchSpace.from_dict({
            "model": ["small", "large"],  # Grid
            "lr": Log(1e-5, 1e-1),         # Sweep
        })
        assert space.has_grid
        assert space.has_sweep
```

**Step 2: Run tests to verify they fail**

```bash
cd gpudispatch && PYTHONPATH=src pytest tests/unit/experiments/test_search_space.py -v
```

Expected: FAIL with import errors

**Step 3: Implement search space primitives**

```python
# src/gpudispatch/experiments/__init__.py
"""Experiment primitives for hyperparameter optimization."""

from gpudispatch.experiments.search_space import (
    Grid,
    Sweep,
    Choice,
    Log,
    Uniform,
    Int,
    Range,
    SearchSpace,
    Distribution,
)

__all__ = [
    "Grid",
    "Sweep",
    "Choice",
    "Log",
    "Uniform",
    "Int",
    "Range",
    "SearchSpace",
    "Distribution",
]
```

```python
# src/gpudispatch/experiments/search_space.py
"""Search space primitives for experiment configuration."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterator, List, Optional, Union


class Distribution(ABC):
    """Base class for parameter distributions."""

    @abstractmethod
    def sample(self) -> Any:
        """Sample a value from this distribution."""
        pass

    @abstractmethod
    def contains(self, value: Any) -> bool:
        """Check if value is within distribution bounds."""
        pass


@dataclass
class Choice(Distribution):
    """Categorical choice from a list of values.

    Example:
        >>> c = Choice([16, 32, 64])
        >>> c.sample()  # Returns one of 16, 32, or 64
    """
    values: List[Any]

    def sample(self) -> Any:
        return random.choice(self.values)

    def contains(self, value: Any) -> bool:
        return value in self.values

    def __iter__(self):
        return iter(self.values)


@dataclass
class Log(Distribution):
    """Log-uniform distribution (good for learning rates).

    Example:
        >>> l = Log(1e-5, 1e-1)
        >>> l.sample()  # Returns value in [1e-5, 1e-1] with log scale
    """
    low: float
    high: float

    def __post_init__(self):
        if self.low <= 0 or self.high <= 0:
            raise ValueError("Log distribution requires positive bounds")
        if self.low >= self.high:
            raise ValueError("low must be less than high")

    def sample(self) -> float:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(random.uniform(log_low, log_high))

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass
class Uniform(Distribution):
    """Uniform distribution over continuous range.

    Example:
        >>> u = Uniform(0.0, 0.5)
        >>> u.sample()  # Returns value in [0.0, 0.5]
    """
    low: float
    high: float

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError("low must be less than high")

    def sample(self) -> float:
        return random.uniform(self.low, self.high)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass
class Int(Distribution):
    """Integer distribution over range [low, high] inclusive.

    Example:
        >>> i = Int(4, 32)
        >>> i.sample()  # Returns integer in [4, 32]
    """
    low: int
    high: int

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError("low must be less than high")

    def sample(self) -> int:
        return random.randint(self.low, self.high)

    def contains(self, value: int) -> bool:
        return isinstance(value, int) and self.low <= value <= self.high


@dataclass
class Range(Distribution):
    """Stepped range of values (like numpy.arange but inclusive).

    Example:
        >>> r = Range(0.1, 1.0, 0.1)
        >>> r.to_list()  # [0.1, 0.2, ..., 1.0]
    """
    start: float
    stop: float
    step: float

    def to_list(self) -> List[float]:
        """Convert range to list of values."""
        values = []
        current = self.start
        while current <= self.stop + self.step / 2:  # Handle float precision
            values.append(round(current, 10))
            current += self.step
        return values

    def sample(self) -> float:
        return random.choice(self.to_list())

    def contains(self, value: float) -> bool:
        return value in self.to_list()


@dataclass
class Grid:
    """Grid search space - exhaustive combinations.

    Example:
        >>> g = Grid(lr=[1e-4, 1e-3], batch_size=[16, 32])
        >>> list(g)  # All 4 combinations
    """
    params: Dict[str, List[Any]] = field(default_factory=dict)

    def __init__(self, **kwargs: List[Any]):
        self.params = kwargs

    @property
    def size(self) -> int:
        """Total number of combinations."""
        if not self.params:
            return 0
        total = 1
        for values in self.params.values():
            total *= len(values)
        return total

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all parameter combinations."""
        if not self.params:
            return
        keys = list(self.params.keys())
        for values in product(*self.params.values()):
            yield dict(zip(keys, values))

    def __len__(self) -> int:
        return self.size


@dataclass
class Sweep:
    """Sweep search space - sampled distributions.

    Example:
        >>> s = Sweep(lr=Log(1e-5, 1e-1), dropout=Uniform(0.0, 0.5))
        >>> s.sample()  # {"lr": 0.001, "dropout": 0.3}
    """
    params: Dict[str, Distribution] = field(default_factory=dict)

    def __init__(self, **kwargs: Distribution):
        self.params = kwargs

    def sample(self) -> Dict[str, Any]:
        """Sample one configuration from the sweep."""
        return {name: dist.sample() for name, dist in self.params.items()}


@dataclass
class SearchSpace:
    """Combined search space that can hold both grid and sweep parameters.

    Example:
        >>> space = SearchSpace.from_dict({
        ...     "model": ["small", "large"],  # Grid
        ...     "lr": Log(1e-5, 1e-1),         # Sweep
        ... })
    """
    grid_params: Dict[str, List[Any]] = field(default_factory=dict)
    sweep_params: Dict[str, Distribution] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> SearchSpace:
        """Create SearchSpace from mixed dict of lists and distributions."""
        grid_params = {}
        sweep_params = {}

        for name, value in params.items():
            if isinstance(value, list):
                grid_params[name] = value
            elif isinstance(value, Distribution):
                sweep_params[name] = value
            elif isinstance(value, (Grid, Sweep)):
                raise ValueError(
                    f"Use Grid/Sweep contents directly, not nested: {name}"
                )
            else:
                # Single value treated as single-item grid
                grid_params[name] = [value]

        return cls(grid_params=grid_params, sweep_params=sweep_params)

    @property
    def is_grid(self) -> bool:
        """True if only grid parameters (no distributions)."""
        return bool(self.grid_params) and not self.sweep_params

    @property
    def is_sweep(self) -> bool:
        """True if only sweep parameters (no lists)."""
        return bool(self.sweep_params) and not self.grid_params

    @property
    def has_grid(self) -> bool:
        """True if any grid parameters."""
        return bool(self.grid_params)

    @property
    def has_sweep(self) -> bool:
        """True if any sweep parameters."""
        return bool(self.sweep_params)

    @property
    def param_names(self) -> List[str]:
        """All parameter names."""
        return list(self.grid_params.keys()) + list(self.sweep_params.keys())
```

**Step 4: Run tests to verify they pass**

```bash
cd gpudispatch && PYTHONPATH=src pytest tests/unit/experiments/test_search_space.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(experiments): add search space primitives

- Grid, Sweep for search space types
- Choice, Log, Uniform, Int, Range distributions
- SearchSpace for mixed grid+sweep
"
```

---

## Task 3.2: Trial and Results

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/trial.py`
- Create: `gpudispatch/src/gpudispatch/experiments/results.py`
- Create: `gpudispatch/tests/unit/experiments/test_trial.py`
- Create: `gpudispatch/tests/unit/experiments/test_results.py`

**Step 1: Write failing tests for Trial**

```python
# tests/unit/experiments/test_trial.py
"""Tests for Trial dataclass."""

import pytest
from datetime import datetime
from gpudispatch.experiments.trial import Trial, TrialStatus


class TestTrialStatus:
    def test_status_values(self):
        assert TrialStatus.PENDING.value == "pending"
        assert TrialStatus.RUNNING.value == "running"
        assert TrialStatus.COMPLETED.value == "completed"
        assert TrialStatus.FAILED.value == "failed"

    def test_is_terminal(self):
        assert not TrialStatus.PENDING.is_terminal
        assert not TrialStatus.RUNNING.is_terminal
        assert TrialStatus.COMPLETED.is_terminal
        assert TrialStatus.FAILED.is_terminal


class TestTrial:
    def test_trial_creation(self):
        t = Trial(id=1, params={"lr": 0.001})
        assert t.id == 1
        assert t.params == {"lr": 0.001}
        assert t.status == TrialStatus.PENDING

    def test_trial_with_metrics(self):
        t = Trial(id=1, params={"lr": 0.001}, metrics={"accuracy": 0.95})
        assert t.metrics["accuracy"] == 0.95

    def test_trial_duration(self):
        t = Trial(id=1, params={})
        t.started_at = datetime(2026, 1, 1, 12, 0, 0)
        t.completed_at = datetime(2026, 1, 1, 12, 5, 30)
        assert t.duration_seconds == 330  # 5 min 30 sec

    def test_trial_to_dict(self):
        t = Trial(id=1, params={"lr": 0.001}, metrics={"acc": 0.9})
        d = t.to_dict()
        assert d["id"] == 1
        assert d["params"]["lr"] == 0.001
        assert d["metrics"]["acc"] == 0.9

    def test_trial_from_dict(self):
        d = {"id": 1, "params": {"lr": 0.001}, "metrics": {"acc": 0.9}}
        t = Trial.from_dict(d)
        assert t.id == 1
        assert t.params["lr"] == 0.001
```

**Step 2: Write failing tests for Results**

```python
# tests/unit/experiments/test_results.py
"""Tests for Results container."""

import pytest
import pandas as pd
from gpudispatch.experiments.trial import Trial, TrialStatus
from gpudispatch.experiments.results import Results


class TestResults:
    @pytest.fixture
    def sample_trials(self):
        return [
            Trial(id=1, params={"lr": 0.001}, metrics={"accuracy": 0.90}),
            Trial(id=2, params={"lr": 0.01}, metrics={"accuracy": 0.95}),
            Trial(id=3, params={"lr": 0.1}, metrics={"accuracy": 0.80}),
        ]

    def test_results_creation(self, sample_trials):
        r = Results(trials=sample_trials, metric="accuracy")
        assert len(r.trials) == 3
        assert r.metric == "accuracy"

    def test_best_returns_highest(self, sample_trials):
        r = Results(trials=sample_trials, metric="accuracy", maximize=True)
        assert r.best.params["lr"] == 0.01
        assert r.best.metrics["accuracy"] == 0.95

    def test_best_returns_lowest(self, sample_trials):
        r = Results(trials=sample_trials, metric="accuracy", maximize=False)
        assert r.best.params["lr"] == 0.1

    def test_best_params(self, sample_trials):
        r = Results(trials=sample_trials, metric="accuracy")
        assert r.best_params == {"lr": 0.01}

    def test_best_metrics(self, sample_trials):
        r = Results(trials=sample_trials, metric="accuracy")
        assert r.best_metrics == {"accuracy": 0.95}

    def test_top_n(self, sample_trials):
        r = Results(trials=sample_trials, metric="accuracy")
        top2 = r.top(2)
        assert len(top2) == 2
        assert top2[0].metrics["accuracy"] == 0.95
        assert top2[1].metrics["accuracy"] == 0.90

    def test_to_dataframe(self, sample_trials):
        r = Results(trials=sample_trials, metric="accuracy")
        df = r.df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "lr" in df.columns
        assert "accuracy" in df.columns

    def test_failed_and_successful(self):
        trials = [
            Trial(id=1, params={}, metrics={"acc": 0.9}, status=TrialStatus.COMPLETED),
            Trial(id=2, params={}, status=TrialStatus.FAILED, error="OOM"),
            Trial(id=3, params={}, metrics={"acc": 0.8}, status=TrialStatus.COMPLETED),
        ]
        r = Results(trials=trials, metric="acc")
        assert len(r.successful) == 2
        assert len(r.failed) == 1
        assert r.failed[0].error == "OOM"

    def test_empty_results(self):
        r = Results(trials=[], metric="accuracy")
        assert r.best is None
        assert r.best_params == {}
        assert len(r.df) == 0
```

**Step 3: Implement Trial and Results**

```python
# src/gpudispatch/experiments/trial.py
"""Trial representation for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TrialStatus(Enum):
    """Status of an experiment trial."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"

    @property
    def is_terminal(self) -> bool:
        return self in (TrialStatus.COMPLETED, TrialStatus.FAILED, TrialStatus.PRUNED)


@dataclass
class Trial:
    """A single trial in an experiment.

    Attributes:
        id: Unique trial identifier
        params: Parameter values for this trial
        metrics: Result metrics (populated after completion)
        status: Current trial status
        error: Error message if failed
        started_at: When trial started
        completed_at: When trial finished
    """
    id: int
    params: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: TrialStatus = TrialStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration in seconds, or None if not completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "params": self.params,
            "metrics": self.metrics,
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trial:
        """Create Trial from dictionary."""
        return cls(
            id=data["id"],
            params=data.get("params", {}),
            metrics=data.get("metrics", {}),
            status=TrialStatus(data.get("status", "pending")),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )
```

```python
# src/gpudispatch/experiments/results.py
"""Results container for experiment trials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from gpudispatch.experiments.trial import Trial, TrialStatus


@dataclass
class Results:
    """Container for experiment results with analysis helpers.

    Example:
        >>> results.best  # Best trial
        >>> results.df    # pandas DataFrame
        >>> results.top(5)  # Top 5 trials
    """
    trials: List[Trial] = field(default_factory=list)
    metric: str = "loss"
    maximize: bool = True
    experiment_name: Optional[str] = None

    @property
    def best(self) -> Optional[Trial]:
        """Best trial by primary metric."""
        successful = self.successful
        if not successful:
            return None

        return max(
            successful,
            key=lambda t: t.metrics.get(self.metric, float("-inf") if self.maximize else float("inf")),
        ) if self.maximize else min(
            successful,
            key=lambda t: t.metrics.get(self.metric, float("inf")),
        )

    @property
    def best_params(self) -> Dict[str, Any]:
        """Parameters of best trial."""
        return self.best.params if self.best else {}

    @property
    def best_metrics(self) -> Dict[str, Any]:
        """Metrics of best trial."""
        return self.best.metrics if self.best else {}

    @property
    def successful(self) -> List[Trial]:
        """Trials that completed successfully."""
        return [t for t in self.trials if t.status == TrialStatus.COMPLETED]

    @property
    def failed(self) -> List[Trial]:
        """Trials that failed."""
        return [t for t in self.trials if t.status == TrialStatus.FAILED]

    def top(self, n: int) -> List[Trial]:
        """Top N trials by primary metric."""
        successful = self.successful
        sorted_trials = sorted(
            successful,
            key=lambda t: t.metrics.get(self.metric, float("-inf") if self.maximize else float("inf")),
            reverse=self.maximize,
        )
        return sorted_trials[:n]

    @property
    def df(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if not self.trials:
            return pd.DataFrame()

        rows = []
        for t in self.trials:
            row = {"trial_id": t.id, "status": t.status.value}
            row.update(t.params)
            row.update(t.metrics)
            if t.duration_seconds:
                row["duration_seconds"] = t.duration_seconds
            rows.append(row)

        return pd.DataFrame(rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Alias for df property."""
        return self.df

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Experiment: {self.experiment_name or 'unnamed'}",
            f"Total trials: {len(self.trials)}",
            f"Successful: {len(self.successful)}",
            f"Failed: {len(self.failed)}",
        ]
        if self.best:
            lines.append(f"Best {self.metric}: {self.best_metrics.get(self.metric)}")
            lines.append(f"Best params: {self.best_params}")
        return "\n".join(lines)
```

**Step 4: Update experiments __init__.py**

```python
# Add to src/gpudispatch/experiments/__init__.py
from gpudispatch.experiments.trial import Trial, TrialStatus
from gpudispatch.experiments.results import Results

# Update __all__
__all__ = [
    # ... existing exports ...
    "Trial",
    "TrialStatus",
    "Results",
]
```

**Step 5: Run tests and commit**

```bash
cd gpudispatch && PYTHONPATH=src pytest tests/unit/experiments/ -v
git add -A && git commit -m "feat(experiments): add Trial and Results classes"
```

---

## Task 3.3: Storage Backends

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/storage/__init__.py`
- Create: `gpudispatch/src/gpudispatch/experiments/storage/base.py`
- Create: `gpudispatch/src/gpudispatch/experiments/storage/memory.py`
- Create: `gpudispatch/src/gpudispatch/experiments/storage/file.py`
- Create: `gpudispatch/src/gpudispatch/experiments/storage/sqlite.py`
- Create: `gpudispatch/tests/unit/experiments/storage/__init__.py`
- Create: `gpudispatch/tests/unit/experiments/storage/test_storage.py`

**Step 1: Write failing tests**

```python
# tests/unit/experiments/storage/test_storage.py
"""Tests for storage backends."""

import pytest
import tempfile
import os
from pathlib import Path

from gpudispatch.experiments.trial import Trial, TrialStatus
from gpudispatch.experiments.storage import (
    Storage, MemoryStorage, FileStorage, SQLiteStorage
)


class TestMemoryStorage:
    def test_save_and_load_trial(self):
        storage = MemoryStorage()
        trial = Trial(id=1, params={"lr": 0.001}, metrics={"acc": 0.9})
        storage.save_trial("exp1", trial)
        loaded = storage.load_trial("exp1", 1)
        assert loaded.params == {"lr": 0.001}

    def test_load_all_trials(self):
        storage = MemoryStorage()
        storage.save_trial("exp1", Trial(id=1, params={"a": 1}))
        storage.save_trial("exp1", Trial(id=2, params={"a": 2}))
        trials = storage.load_trials("exp1")
        assert len(trials) == 2

    def test_save_config(self):
        storage = MemoryStorage()
        storage.save_config("exp1", {"search_space": {"lr": [0.001, 0.01]}})
        config = storage.load_config("exp1")
        assert config["search_space"]["lr"] == [0.001, 0.01]


class TestFileStorage:
    def test_save_and_load_trial(self, tmp_path):
        storage = FileStorage(str(tmp_path))
        trial = Trial(id=1, params={"lr": 0.001}, metrics={"acc": 0.9})
        trial.status = TrialStatus.COMPLETED
        storage.save_trial("exp1", trial)
        loaded = storage.load_trial("exp1", 1)
        assert loaded.params["lr"] == 0.001

    def test_creates_directory_structure(self, tmp_path):
        storage = FileStorage(str(tmp_path))
        storage.save_trial("my-exp", Trial(id=1, params={}))
        assert (tmp_path / "my-exp").exists()
        assert (tmp_path / "my-exp" / "trials.csv").exists()

    def test_list_experiments(self, tmp_path):
        storage = FileStorage(str(tmp_path))
        storage.save_trial("exp1", Trial(id=1, params={}))
        storage.save_trial("exp2", Trial(id=1, params={}))
        exps = storage.list_experiments()
        assert set(exps) == {"exp1", "exp2"}


class TestSQLiteStorage:
    def test_save_and_load_trial(self, tmp_path):
        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(str(db_path))
        trial = Trial(id=1, params={"lr": 0.001}, metrics={"acc": 0.9})
        trial.status = TrialStatus.COMPLETED
        storage.save_trial("exp1", trial)
        loaded = storage.load_trial("exp1", 1)
        assert loaded.params["lr"] == 0.001

    def test_query(self, tmp_path):
        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(str(db_path))
        storage.save_trial("exp1", Trial(id=1, params={"lr": 0.001}, metrics={"acc": 0.9}, status=TrialStatus.COMPLETED))
        storage.save_trial("exp1", Trial(id=2, params={"lr": 0.01}, metrics={"acc": 0.95}, status=TrialStatus.COMPLETED))
        results = storage.query("SELECT * FROM trials WHERE experiment_name = ?", ("exp1",))
        assert len(results) == 2

    def test_list_experiments(self, tmp_path):
        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(str(db_path))
        storage.save_trial("exp1", Trial(id=1, params={}))
        storage.save_trial("exp2", Trial(id=1, params={}))
        exps = storage.list_experiments()
        assert set(exps) == {"exp1", "exp2"}


class TestStorageInterface:
    @pytest.mark.parametrize("storage_cls", [MemoryStorage])
    def test_is_storage_instance(self, storage_cls):
        storage = storage_cls()
        assert isinstance(storage, Storage)
```

**Step 2: Implement storage backends**

```python
# src/gpudispatch/experiments/storage/__init__.py
"""Storage backends for experiment persistence."""

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.storage.memory import MemoryStorage
from gpudispatch.experiments.storage.file import FileStorage
from gpudispatch.experiments.storage.sqlite import SQLiteStorage

__all__ = ["Storage", "MemoryStorage", "FileStorage", "SQLiteStorage"]
```

```python
# src/gpudispatch/experiments/storage/base.py
"""Base storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from gpudispatch.experiments.trial import Trial


class Storage(ABC):
    """Abstract base class for experiment storage backends."""

    @abstractmethod
    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        """Save a trial to storage."""
        pass

    @abstractmethod
    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        """Load a specific trial."""
        pass

    @abstractmethod
    def load_trials(self, experiment_name: str) -> List[Trial]:
        """Load all trials for an experiment."""
        pass

    @abstractmethod
    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        """Save experiment configuration."""
        pass

    @abstractmethod
    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment configuration."""
        pass

    @abstractmethod
    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        pass
```

```python
# src/gpudispatch/experiments/storage/memory.py
"""In-memory storage for quick experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.trial import Trial


class MemoryStorage(Storage):
    """In-memory storage (non-persistent).

    Good for quick tests and small experiments.
    """

    def __init__(self):
        self._trials: Dict[str, Dict[int, Trial]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        if experiment_name not in self._trials:
            self._trials[experiment_name] = {}
        self._trials[experiment_name][trial.id] = trial

    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        return self._trials.get(experiment_name, {}).get(trial_id)

    def load_trials(self, experiment_name: str) -> List[Trial]:
        return list(self._trials.get(experiment_name, {}).values())

    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        self._configs[experiment_name] = config

    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        return self._configs.get(experiment_name)

    def list_experiments(self) -> List[str]:
        return list(set(self._trials.keys()) | set(self._configs.keys()))
```

```python
# src/gpudispatch/experiments/storage/file.py
"""File-based storage for experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.trial import Trial, TrialStatus


class FileStorage(Storage):
    """File-based storage using JSON and CSV.

    Directory structure:
        base_dir/
        ├── experiment-name/
        │   ├── config.json
        │   ├── trials.csv
        │   └── best.json
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _exp_dir(self, experiment_name: str) -> Path:
        return self.base_dir / experiment_name

    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        exp_dir = self._exp_dir(experiment_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        trials_file = exp_dir / "trials.csv"
        file_exists = trials_file.exists()

        # Flatten trial to row
        row = {"trial_id": trial.id, "status": trial.status.value}
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        row.update({f"metric_{k}": v for k, v in trial.metrics.items()})
        if trial.error:
            row["error"] = trial.error

        # Append to CSV
        with open(trials_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        trials = self.load_trials(experiment_name)
        for t in trials:
            if t.id == trial_id:
                return t
        return None

    def load_trials(self, experiment_name: str) -> List[Trial]:
        trials_file = self._exp_dir(experiment_name) / "trials.csv"
        if not trials_file.exists():
            return []

        trials = []
        with open(trials_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                params = {k[6:]: _parse_value(v) for k, v in row.items() if k.startswith("param_")}
                metrics = {k[7:]: _parse_value(v) for k, v in row.items() if k.startswith("metric_")}
                trial = Trial(
                    id=int(row["trial_id"]),
                    params=params,
                    metrics=metrics,
                    status=TrialStatus(row.get("status", "completed")),
                    error=row.get("error"),
                )
                trials.append(trial)
        return trials

    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        exp_dir = self._exp_dir(experiment_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_file = exp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        config_file = self._exp_dir(experiment_name) / "config.json"
        if not config_file.exists():
            return None
        with open(config_file) as f:
            return json.load(f)

    def list_experiments(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]


def _parse_value(v: str) -> Any:
    """Parse CSV string value to Python type."""
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    return v
```

```python
# src/gpudispatch/experiments/storage/sqlite.py
"""SQLite-based storage for experiments."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.trial import Trial, TrialStatus


class SQLiteStorage(Storage):
    """SQLite database storage for experiments.

    Good for large experiments and querying across experiments.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    trial_id INTEGER NOT NULL,
                    params TEXT NOT NULL,
                    metrics TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(experiment_name, trial_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS configs (
                    experiment_name TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trials_experiment
                ON trials(experiment_name)
            """)

    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trials
                (experiment_name, trial_id, params, metrics, status, error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experiment_name,
                trial.id,
                json.dumps(trial.params),
                json.dumps(trial.metrics),
                trial.status.value,
                trial.error,
            ))

    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT trial_id, params, metrics, status, error
                FROM trials WHERE experiment_name = ? AND trial_id = ?
            """, (experiment_name, trial_id)).fetchone()
            if row:
                return Trial(
                    id=row[0],
                    params=json.loads(row[1]),
                    metrics=json.loads(row[2]) if row[2] else {},
                    status=TrialStatus(row[3]),
                    error=row[4],
                )
        return None

    def load_trials(self, experiment_name: str) -> List[Trial]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT trial_id, params, metrics, status, error
                FROM trials WHERE experiment_name = ?
                ORDER BY trial_id
            """, (experiment_name,)).fetchall()
            return [
                Trial(
                    id=row[0],
                    params=json.loads(row[1]),
                    metrics=json.loads(row[2]) if row[2] else {},
                    status=TrialStatus(row[3]),
                    error=row[4],
                )
                for row in rows
            ]

    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO configs (experiment_name, config)
                VALUES (?, ?)
            """, (experiment_name, json.dumps(config, default=str)))

    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT config FROM configs WHERE experiment_name = ?
            """, (experiment_name,)).fetchone()
            if row:
                return json.loads(row[0])
        return None

    def list_experiments(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT DISTINCT experiment_name FROM trials
                UNION
                SELECT experiment_name FROM configs
            """).fetchall()
            return [row[0] for row in rows]

    def query(self, sql: str, params: tuple = ()) -> List[Any]:
        """Run custom SQL query."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()
```

**Step 3: Run tests and commit**

```bash
cd gpudispatch && PYTHONPATH=src pytest tests/unit/experiments/storage/ -v
git add -A && git commit -m "feat(experiments): add storage backends (Memory, File, SQLite)"
```

---

## Task 3.4: Strategy Interface + Grid/Random

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/strategies/__init__.py`
- Create: `gpudispatch/src/gpudispatch/experiments/strategies/base.py`
- Create: `gpudispatch/src/gpudispatch/experiments/strategies/grid.py`
- Create: `gpudispatch/src/gpudispatch/experiments/strategies/random.py`
- Create: `gpudispatch/tests/unit/experiments/strategies/__init__.py`
- Create: `gpudispatch/tests/unit/experiments/strategies/test_strategies.py`

[Tests and implementation follow same pattern - define Strategy ABC, GridStrategy iterates all combos, RandomStrategy samples from distributions]

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(experiments): add Strategy interface with Grid and Random strategies"
```

---

## Task 3.5: Bayesian Strategy (Optuna Integration)

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/strategies/bayesian.py`
- Create: `gpudispatch/tests/unit/experiments/strategies/test_bayesian.py`

[BayesianStrategy wraps Optuna's TPESampler for smart hyperparameter search]

---

## Task 3.6: Experiment Class

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/experiment.py`
- Create: `gpudispatch/tests/unit/experiments/test_experiment.py`

Core orchestration - takes function, search space, strategy, storage and runs trials.

---

## Task 3.7: @experiment Decorator

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/decorator.py`
- Create: `gpudispatch/tests/unit/experiments/test_decorator.py`

Syntactic sugar over Experiment class.

---

## Task 3.8: Reproducibility

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/reproducibility.py`
- Create: `gpudispatch/tests/unit/experiments/test_reproducibility.py`

Auto-capture seeds, git commit, environment info.

---

## Task 3.9: Experiment Registry

**Files:**
- Create: `gpudispatch/src/gpudispatch/experiments/registry.py`
- Create: `gpudispatch/tests/unit/experiments/test_registry.py`

Global `experiments.list()`, `experiments.load()`, `experiments.compare()`.

---

## Final Integration

After all tasks, update main `gpudispatch/__init__.py` to export experiment API:

```python
from gpudispatch.experiments import (
    experiment, Experiment, Grid, Sweep,
    Choice, Log, Uniform, Int, Range,
)
```

---

*Plan complete. Ready for execution.*
