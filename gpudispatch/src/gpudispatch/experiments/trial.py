"""Trial dataclass and status enum for experiment tracking.

This module provides:
- TrialStatus: Enum representing trial lifecycle states
- Trial: Dataclass representing a single experiment trial with parameters and results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TrialStatus(Enum):
    """Trial lifecycle status.

    A trial progresses through these states:
    - PENDING: Created but not yet started
    - RUNNING: Currently executing
    - COMPLETED: Finished successfully
    - FAILED: Finished with an error
    - PRUNED: Stopped early (e.g., by pruning in hyperparameter optimization)
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) status.

        Terminal statuses are: COMPLETED, FAILED, PRUNED.

        Returns:
            True if the trial cannot transition to another state.
        """
        return self in (TrialStatus.COMPLETED, TrialStatus.FAILED, TrialStatus.PRUNED)


@dataclass
class Trial:
    """A single trial in an experiment.

    A trial represents one execution of an experiment with a specific set of
    hyperparameters. It tracks the parameters used, metrics computed, status,
    timing information, and any errors.

    Example:
        >>> trial = Trial(id=1, params={"lr": 0.01, "batch_size": 32})
        >>> trial.status
        <TrialStatus.PENDING: 'pending'>
        >>> trial.metrics["loss"] = 0.5
        >>> trial.status = TrialStatus.COMPLETED
    """

    id: int
    params: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: TrialStatus = TrialStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get trial duration in seconds.

        Returns:
            Duration in seconds if both started_at and completed_at are set,
            None otherwise.
        """
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trial to a dictionary.

        Timestamps are serialized to ISO format strings.
        Status is serialized to its string value.

        Returns:
            Dictionary representation of the trial.
        """
        return {
            "id": self.id,
            "params": self.params,
            "metrics": self.metrics,
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trial":
        """Deserialize trial from a dictionary.

        Timestamps are parsed from ISO format strings.
        Status is parsed from its string value.

        Args:
            data: Dictionary containing trial data.

        Returns:
            Reconstructed Trial instance.
        """
        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])

        completed_at = None
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"])

        return cls(
            id=data["id"],
            params=data["params"],
            metrics=data.get("metrics", {}),
            status=TrialStatus(data["status"]),
            error=data.get("error"),
            started_at=started_at,
            completed_at=completed_at,
        )
