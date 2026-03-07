# Contributor Guide: Strategy Authors

This guide is for contributors implementing new hyperparameter search strategies.

## Strategy Contract

All strategies must implement `gpudispatch.experiments.strategies.base.Strategy`:

- `suggest(search_space, completed_trials) -> Optional[dict[str, Any]]`
- `name` property

`Experiment.run()` repeatedly calls `suggest(...)` until it returns `None`.

## Behavioral Rules

1. **Return `None` when exhausted**
   - For finite strategies (for example, grid-like), return `None` when no suggestions remain.
2. **Do not mutate trial history**
   - Treat `completed_trials` as read-only input.
3. **Respect search space constraints**
   - Suggestions must be valid for the provided search space.
4. **Reproducibility matters**
   - If randomness is involved, expose/configure deterministic seeding where possible.

## Implementation Steps

1. Add strategy module under `src/gpudispatch/experiments/strategies/`.
2. Export it from `src/gpudispatch/experiments/strategies/__init__.py`.
3. Add tests under `tests/unit/experiments/`.
4. Update docs and README examples if the strategy is public-facing.

## Testing Checklist

- Suggestion validity for the given search space
- Exhaustion behavior (`None` when done)
- Deterministic behavior under fixed seeds (for stochastic strategies)
- Duplicate handling policy (if any)
- Performance sanity for larger search spaces

Use existing implementations for reference:

- `GridStrategy` (`src/gpudispatch/experiments/strategies/grid.py`)
- `RandomStrategy` (`src/gpudispatch/experiments/strategies/random.py`)

## Minimal Skeleton

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gpudispatch.experiments.strategies.base import Strategy
from gpudispatch.experiments.search_space import SearchSpace
from gpudispatch.experiments.trial import Trial


class MyStrategy(Strategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    def suggest(
        self,
        search_space: SearchSpace,
        completed_trials: List[Trial],
    ) -> Optional[Dict[str, Any]]:
        # Return dict for next trial, or None when search is done.
        ...
```
