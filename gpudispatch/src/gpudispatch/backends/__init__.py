"""Backend implementations for GPU orchestration."""

from gpudispatch.backends.base import Backend
from gpudispatch.backends.local import LocalBackend

__all__ = [
    "Backend",
    "LocalBackend",
]
