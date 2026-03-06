"""Extensible observability hooks for gpudispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class EventHook:
    """Hook for observability events.

    Subclass this to create custom hooks for Prometheus, OpenTelemetry, etc.
    """

    on_job_start: Optional[Callable[..., None]] = None
    on_job_complete: Optional[Callable[..., None]] = None
    on_job_failed: Optional[Callable[..., None]] = None
    on_experiment_start: Optional[Callable[..., None]] = None
    on_experiment_complete: Optional[Callable[..., None]] = None


class HookRegistry:
    """Registry for observability hooks."""

    def __init__(self) -> None:
        self._hooks: list[EventHook] = []

    def register(self, hook: EventHook) -> None:
        """Register a hook for events."""
        self._hooks.append(hook)

    def unregister(self, hook: EventHook) -> None:
        """Unregister a hook."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def clear(self) -> None:
        """Clear all registered hooks."""
        self._hooks.clear()

    def emit(self, event: str, **kwargs: Any) -> None:
        """Emit an event to all registered hooks."""
        for hook in self._hooks:
            callback = getattr(hook, event, None)
            if callback is not None:
                try:
                    callback(**kwargs)
                except Exception as e:
                    logger.warning(f"Hook {hook} failed on {event}: {e}")

    @property
    def hooks(self) -> list[EventHook]:
        """Return list of registered hooks."""
        return list(self._hooks)


# Global registry
hooks = HookRegistry()


class LoggingHook(EventHook):
    """Logs events to Python logger."""

    def __init__(self, logger_name: str = "gpudispatch.events") -> None:
        self._logger = logging.getLogger(logger_name)
        super().__init__(
            on_job_start=self._on_job_start,
            on_job_complete=self._on_job_complete,
            on_job_failed=self._on_job_failed,
            on_experiment_start=self._on_experiment_start,
            on_experiment_complete=self._on_experiment_complete,
        )

    def _on_job_start(self, job_id: str, job_name: str, **kwargs: Any) -> None:
        self._logger.info(f"Job started: {job_name} ({job_id})")

    def _on_job_complete(
        self, job_id: str, job_name: str, runtime_seconds: float, **kwargs: Any
    ) -> None:
        self._logger.info(f"Job completed: {job_name} ({job_id}) in {runtime_seconds:.2f}s")

    def _on_job_failed(
        self, job_id: str, job_name: str, error: str, **kwargs: Any
    ) -> None:
        self._logger.error(f"Job failed: {job_name} ({job_id}): {error}")

    def _on_experiment_start(self, experiment_id: str, **kwargs: Any) -> None:
        self._logger.info(f"Experiment started: {experiment_id}")

    def _on_experiment_complete(
        self, experiment_id: str, total_jobs: int, **kwargs: Any
    ) -> None:
        self._logger.info(f"Experiment completed: {experiment_id} ({total_jobs} jobs)")
