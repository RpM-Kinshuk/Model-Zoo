"""Opinionated dispatcher profiles for quick, powerful workflow setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gpudispatch.core import Dispatcher


@dataclass(frozen=True)
class DispatcherProfile:
    """Preset configuration for a Dispatcher instance."""

    name: str
    description: str
    dispatcher_kwargs: dict[str, Any]
    default_command_timeout: Optional[float] = None
    default_command_env: dict[str, str] = field(default_factory=dict)


_PROFILES: dict[str, DispatcherProfile] = {
    "quickstart": DispatcherProfile(
        name="quickstart",
        description="Fast feedback for local experimentation and iterative debugging.",
        dispatcher_kwargs={
            "memory_threshold": "500MB",
            "polling_interval": 0.5,
        },
        default_command_timeout=None,
        default_command_env={"PYTHONUNBUFFERED": "1"},
    ),
    "batch": DispatcherProfile(
        name="batch",
        description="Throughput-oriented defaults for long-running batch workloads.",
        dispatcher_kwargs={
            "memory_threshold": "1GB",
            "polling_interval": 3.0,
        },
        default_command_timeout=6 * 60 * 60,
        default_command_env={"PYTHONUNBUFFERED": "1"},
    ),
    "high_reliability": DispatcherProfile(
        name="high_reliability",
        description="Conservative scheduling defaults for stable production-style runs.",
        dispatcher_kwargs={
            "memory_threshold": "2GB",
            "polling_interval": 1.0,
        },
        default_command_timeout=24 * 60 * 60,
        default_command_env={"PYTHONUNBUFFERED": "1"},
    ),
}


def list_profiles() -> Dict[str, str]:
    """Return available profile names mapped to their description."""
    return {name: profile.description for name, profile in _PROFILES.items()}


def get_profile(name: str) -> DispatcherProfile:
    """Resolve a profile by name (case-insensitive)."""
    key = name.strip().lower()
    if key not in _PROFILES:
        available = ", ".join(sorted(_PROFILES.keys()))
        raise ValueError(f"Unknown profile '{name}'. Available profiles: {available}")
    return _PROFILES[key]


def dispatcher_from_profile(profile: str = "quickstart", **overrides: Any) -> Dispatcher:
    """Create a Dispatcher configured with an opinionated profile.

    Args:
        profile: Profile name. One of: quickstart, batch, high_reliability.
        **overrides: Any Dispatcher init overrides (e.g., gpus=[0, 1]).

    Returns:
        Configured Dispatcher instance.
    """
    preset = get_profile(profile)
    dispatcher_kwargs: dict[str, Any] = dict(preset.dispatcher_kwargs)
    dispatcher_kwargs["default_command_timeout"] = preset.default_command_timeout
    dispatcher_kwargs["default_command_env"] = dict(preset.default_command_env)
    dispatcher_kwargs.update(overrides)
    return Dispatcher(**dispatcher_kwargs)
