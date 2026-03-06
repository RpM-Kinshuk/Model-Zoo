"""gpudispatch: Universal GPU orchestration - from laptop to supercomputer."""

from gpudispatch.decorators import gpu, set_default_dispatcher, get_default_dispatcher
from gpudispatch.auto import auto_dispatcher
from gpudispatch.experiments import experiment

__version__: str = "0.1.0"

__all__ = [
    "__version__",
    "experiment",
    "gpu",
    "set_default_dispatcher",
    "get_default_dispatcher",
    "auto_dispatcher",
]
