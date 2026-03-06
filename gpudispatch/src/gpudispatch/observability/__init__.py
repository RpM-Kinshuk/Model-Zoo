"""Observability module for gpudispatch."""

from gpudispatch.observability.hooks import EventHook, HookRegistry, LoggingHook, hooks

__all__ = [
    "EventHook",
    "HookRegistry",
    "LoggingHook",
    "hooks",
]
