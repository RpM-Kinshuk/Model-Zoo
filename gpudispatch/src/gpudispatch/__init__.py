"""gpudispatch: Universal GPU orchestration - from laptop to supercomputer."""

from gpudispatch.decorators import gpu, set_default_dispatcher, get_default_dispatcher
from gpudispatch.auto import auto_dispatcher
from gpudispatch.core import CommandResult, Dispatcher
from gpudispatch.experiments import experiment
from gpudispatch.profiles import (
    DispatcherProfile,
    dispatcher_from_profile,
    get_profile,
    list_profiles,
)

__version__: str = "0.1.0"

__all__ = [
    "__version__",
    "CommandResult",
    "DispatcherProfile",
    "Dispatcher",
    "experiment",
    "gpu",
    "dispatcher_from_profile",
    "get_profile",
    "list_profiles",
    "set_default_dispatcher",
    "get_default_dispatcher",
    "auto_dispatcher",
]
