# Contributor Guide: Backend Authors

This guide is for contributors adding a new backend (for example, Kubernetes or cloud schedulers).

## Backend Contract

Every backend must implement `gpudispatch.backends.base.Backend`:

- `name`
- `is_running`
- `start()`
- `shutdown()`
- `allocate_gpus(count, memory=None)`
- `release_gpus(gpus)`
- `list_available()`
- `health_check()`

## Behavioral Rules

1. **Idempotent lifecycle**
   - Repeated `start()`/`shutdown()` calls should be safe.
2. **Thread safety**
   - Allocation/release operations must be protected against races.
3. **Non-blocking allocation contract**
   - If allocation cannot be satisfied, return `[]` instead of blocking indefinitely.
4. **Release is best-effort and safe**
   - Releasing unknown/already released resources should not crash.
5. **Fast health checks**
   - `health_check()` should be cheap and suitable for repeated calls.

## Recommended Implementation Steps

1. Add backend module under `src/gpudispatch/backends/`.
2. Export the backend in `src/gpudispatch/backends/__init__.py`.
3. Wire auto-detection/creation in `src/gpudispatch/auto.py` if applicable.
4. Add unit tests under `tests/unit/backends/`.
5. Update architecture and README docs.

## Testing Checklist

Use `tests/unit/backends/test_local.py` and `tests/unit/backends/test_slurm.py` as templates.

Required test areas:

- Instantiation and property validation
- Lifecycle behavior (`start`, `shutdown`, context manager)
- Allocate/release/list cycle correctness
- Edge cases (insufficient resources, invalid input, repeated operations)
- Health check behavior when dependencies are unavailable

## Minimal Skeleton

```python
from __future__ import annotations

from typing import List, Optional

from gpudispatch.backends.base import Backend
from gpudispatch.core.resources import GPU, Memory


class MyBackend(Backend):
    @property
    def name(self) -> str:
        return "mybackend"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        ...

    def shutdown(self) -> None:
        ...

    def allocate_gpus(self, count: int, memory: Optional[Memory] = None) -> List[GPU]:
        ...

    def release_gpus(self, gpus: List[GPU]) -> None:
        ...

    def list_available(self) -> List[GPU]:
        ...

    def health_check(self) -> bool:
        ...
```
