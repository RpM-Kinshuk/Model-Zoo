"""Extensible observability hooks for gpudispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    on_experiment_failed: Optional[Callable[..., None]] = None
    on_experiment_trial_start: Optional[Callable[..., None]] = None
    on_experiment_trial_complete: Optional[Callable[..., None]] = None
    on_experiment_trial_failed: Optional[Callable[..., None]] = None


class MetricsHook(EventHook):
    """Collect counters and runtime aggregates from hook events."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {
            "job_started": 0,
            "job_completed": 0,
            "job_failed": 0,
            "experiment_started": 0,
            "experiment_completed": 0,
            "experiment_failed": 0,
            "trial_started": 0,
            "trial_completed": 0,
            "trial_failed": 0,
        }
        self._runtime_seconds: dict[str, float] = {
            "job_runtime_seconds": 0.0,
            "experiment_runtime_seconds": 0.0,
            "trial_runtime_seconds": 0.0,
        }
        self._latest_trial_metrics: dict[str, dict[str, Any]] = {}

        super().__init__(
            on_job_start=self._on_job_start,
            on_job_complete=self._on_job_complete,
            on_job_failed=self._on_job_failed,
            on_experiment_start=self._on_experiment_start,
            on_experiment_complete=self._on_experiment_complete,
            on_experiment_failed=self._on_experiment_failed,
            on_experiment_trial_start=self._on_experiment_trial_start,
            on_experiment_trial_complete=self._on_experiment_trial_complete,
            on_experiment_trial_failed=self._on_experiment_trial_failed,
        )

    @property
    def counters(self) -> dict[str, int]:
        """Return event counters."""
        return dict(self._counters)

    @property
    def runtime_seconds(self) -> dict[str, float]:
        """Return accumulated runtime metrics."""
        return dict(self._runtime_seconds)

    @property
    def latest_trial_metrics(self) -> dict[str, dict[str, Any]]:
        """Return latest completed trial metrics by experiment/trial key."""
        return {k: dict(v) for k, v in self._latest_trial_metrics.items()}

    def snapshot(self) -> dict[str, Any]:
        """Return a full metrics snapshot."""
        return {
            "counters": self.counters,
            "runtime_seconds": self.runtime_seconds,
            "latest_trial_metrics": self.latest_trial_metrics,
        }

    def _on_job_start(self, **kwargs: Any) -> None:
        self._counters["job_started"] += 1

    def _on_job_complete(self, runtime_seconds: float = 0.0, **kwargs: Any) -> None:
        self._counters["job_completed"] += 1
        self._runtime_seconds["job_runtime_seconds"] += max(0.0, runtime_seconds)

    def _on_job_failed(self, **kwargs: Any) -> None:
        self._counters["job_failed"] += 1

    def _on_experiment_start(self, **kwargs: Any) -> None:
        self._counters["experiment_started"] += 1

    def _on_experiment_complete(
        self,
        runtime_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._counters["experiment_completed"] += 1
        self._runtime_seconds["experiment_runtime_seconds"] += max(0.0, runtime_seconds)

    def _on_experiment_failed(self, **kwargs: Any) -> None:
        self._counters["experiment_failed"] += 1

    def _on_experiment_trial_start(self, **kwargs: Any) -> None:
        self._counters["trial_started"] += 1

    def _on_experiment_trial_complete(
        self,
        experiment_id: str,
        trial_id: int,
        metrics: Optional[dict[str, Any]] = None,
        runtime_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._counters["trial_completed"] += 1
        self._runtime_seconds["trial_runtime_seconds"] += max(0.0, runtime_seconds)
        key = f"{experiment_id}:{trial_id}"
        self._latest_trial_metrics[key] = dict(metrics or {})

    def _on_experiment_trial_failed(self, **kwargs: Any) -> None:
        self._counters["trial_failed"] += 1


@dataclass(frozen=True)
class TraceSpan:
    """In-memory trace span generated from hook events."""

    name: str
    status: str
    started_at: datetime
    ended_at: datetime
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Span duration in seconds."""
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


class TraceHook(EventHook):
    """Collect simple in-memory spans from lifecycle events."""

    def __init__(self) -> None:
        self._open_spans: dict[str, tuple[str, datetime, dict[str, Any]]] = {}
        self._spans: list[TraceSpan] = []

        super().__init__(
            on_job_start=self._on_job_start,
            on_job_complete=self._on_job_complete,
            on_job_failed=self._on_job_failed,
            on_experiment_start=self._on_experiment_start,
            on_experiment_complete=self._on_experiment_complete,
            on_experiment_failed=self._on_experiment_failed,
            on_experiment_trial_start=self._on_experiment_trial_start,
            on_experiment_trial_complete=self._on_experiment_trial_complete,
            on_experiment_trial_failed=self._on_experiment_trial_failed,
        )

    @property
    def spans(self) -> list[TraceSpan]:
        """Return recorded spans."""
        return list(self._spans)

    def clear(self) -> None:
        """Clear recorded/open spans."""
        self._open_spans.clear()
        self._spans.clear()

    def _open(self, key: str, name: str, **attributes: Any) -> None:
        self._open_spans[key] = (name, datetime.now(), dict(attributes))

    def _close(
        self,
        key: str,
        *,
        status: str,
        runtime_seconds: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        now = datetime.now()
        if key in self._open_spans:
            name, started_at, base_attributes = self._open_spans.pop(key)
        else:
            name = key
            started_at = now
            base_attributes = {}

        if runtime_seconds is not None:
            ended_at = started_at + timedelta(seconds=max(0.0, runtime_seconds))
            if ended_at < started_at:
                ended_at = started_at
        else:
            ended_at = now

        merged_attributes = dict(base_attributes)
        merged_attributes.update(attributes)

        self._spans.append(
            TraceSpan(
                name=name,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                attributes=merged_attributes,
            )
        )

    def _on_job_start(self, job_id: str, job_name: str, **kwargs: Any) -> None:
        self._open(f"job:{job_id}", "job", job_id=job_id, job_name=job_name)

    def _on_job_complete(
        self,
        job_id: str,
        runtime_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._close(
            f"job:{job_id}",
            status="ok",
            runtime_seconds=runtime_seconds,
            **kwargs,
        )

    def _on_job_failed(
        self,
        job_id: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        self._close(f"job:{job_id}", status="error", error=error, **kwargs)

    def _on_experiment_start(self, experiment_id: str, **kwargs: Any) -> None:
        self._open(
            f"experiment:{experiment_id}",
            "experiment",
            experiment_id=experiment_id,
            **kwargs,
        )

    def _on_experiment_complete(
        self,
        experiment_id: str,
        runtime_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._close(
            f"experiment:{experiment_id}",
            status="ok",
            runtime_seconds=runtime_seconds,
            **kwargs,
        )

    def _on_experiment_failed(
        self,
        experiment_id: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        self._close(
            f"experiment:{experiment_id}",
            status="error",
            error=error,
            **kwargs,
        )

    def _on_experiment_trial_start(
        self,
        experiment_id: str,
        trial_id: int,
        **kwargs: Any,
    ) -> None:
        self._open(
            f"trial:{experiment_id}:{trial_id}",
            "experiment_trial",
            experiment_id=experiment_id,
            trial_id=trial_id,
            **kwargs,
        )

    def _on_experiment_trial_complete(
        self,
        experiment_id: str,
        trial_id: int,
        runtime_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._close(
            f"trial:{experiment_id}:{trial_id}",
            status="ok",
            runtime_seconds=runtime_seconds,
            **kwargs,
        )

    def _on_experiment_trial_failed(
        self,
        experiment_id: str,
        trial_id: int,
        error: str,
        **kwargs: Any,
    ) -> None:
        self._close(
            f"trial:{experiment_id}:{trial_id}",
            status="error",
            error=error,
            **kwargs,
        )


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
            on_experiment_failed=self._on_experiment_failed,
            on_experiment_trial_start=self._on_experiment_trial_start,
            on_experiment_trial_complete=self._on_experiment_trial_complete,
            on_experiment_trial_failed=self._on_experiment_trial_failed,
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

    def _on_experiment_failed(
        self,
        experiment_id: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        self._logger.error(f"Experiment failed: {experiment_id}: {error}")

    def _on_experiment_trial_start(
        self,
        experiment_id: str,
        trial_id: int,
        **kwargs: Any,
    ) -> None:
        self._logger.info(f"Experiment trial started: {experiment_id}#{trial_id}")

    def _on_experiment_trial_complete(
        self,
        experiment_id: str,
        trial_id: int,
        runtime_seconds: float,
        **kwargs: Any,
    ) -> None:
        self._logger.info(
            f"Experiment trial completed: {experiment_id}#{trial_id} in "
            f"{runtime_seconds:.2f}s"
        )

    def _on_experiment_trial_failed(
        self,
        experiment_id: str,
        trial_id: int,
        error: str,
        **kwargs: Any,
    ) -> None:
        self._logger.error(f"Experiment trial failed: {experiment_id}#{trial_id}: {error}")
