"""gpudispatch: Universal GPU orchestration - from laptop to supercomputer."""

from gpudispatch.decorators import gpu, set_default_dispatcher, get_default_dispatcher

__version__: str = "0.1.0"

__all__ = [
    "__version__",
    "gpu",
    "set_default_dispatcher",
    "get_default_dispatcher",
]
