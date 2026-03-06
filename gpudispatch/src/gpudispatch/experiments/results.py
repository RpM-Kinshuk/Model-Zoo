"""Results container for experiment trials with DataFrame analysis.

This module provides:
- Results: Container class for managing and analyzing experiment trials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from gpudispatch.experiments.trial import Trial, TrialStatus

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Results:
    """Container for experiment results with analysis capabilities.

    Results holds a collection of trials and provides methods for:
    - Finding the best trial based on a metric
    - Filtering successful/failed trials
    - Converting to pandas DataFrame for analysis
    - Generating summary statistics

    Example:
        >>> results = Results(trials, metric="loss", maximize=False)
        >>> best = results.best  # Trial with lowest loss
        >>> print(results.best_params)  # {"lr": 0.001, "batch_size": 32}
        >>> df = results.df  # pandas DataFrame for analysis
    """

    trials: List[Trial]
    metric: str = "loss"
    maximize: bool = True
    experiment_name: Optional[str] = None

    @property
    def best(self) -> Optional[Trial]:
        """Get the best trial based on the metric.

        Only considers completed trials that have the specified metric.

        Returns:
            Best trial, or None if no valid trials exist.
        """
        valid_trials = [
            t for t in self.trials
            if t.status == TrialStatus.COMPLETED and self.metric in t.metrics
        ]
        if not valid_trials:
            return None

        if self.maximize:
            return max(valid_trials, key=lambda t: t.metrics[self.metric])
        else:
            return min(valid_trials, key=lambda t: t.metrics[self.metric])

    @property
    def best_params(self) -> Dict[str, Any]:
        """Get the parameters of the best trial.

        Returns:
            Parameters dict, or empty dict if no best trial.
        """
        best = self.best
        if best is None:
            return {}
        return best.params

    @property
    def best_metrics(self) -> Dict[str, Any]:
        """Get the metrics of the best trial.

        Returns:
            Metrics dict, or empty dict if no best trial.
        """
        best = self.best
        if best is None:
            return {}
        return best.metrics

    @property
    def successful(self) -> List[Trial]:
        """Get all completed trials.

        Returns:
            List of trials with COMPLETED status.
        """
        return [t for t in self.trials if t.status == TrialStatus.COMPLETED]

    @property
    def failed(self) -> List[Trial]:
        """Get all failed trials.

        Returns:
            List of trials with FAILED status.
        """
        return [t for t in self.trials if t.status == TrialStatus.FAILED]

    def top(self, n: int) -> List[Trial]:
        """Get the top N trials by metric.

        Only considers completed trials with the specified metric.
        Trials are sorted by metric value (descending if maximize=True,
        ascending if maximize=False).

        Args:
            n: Number of trials to return.

        Returns:
            List of up to n best trials, sorted by metric.
        """
        valid_trials = [
            t for t in self.trials
            if t.status == TrialStatus.COMPLETED and self.metric in t.metrics
        ]
        sorted_trials = sorted(
            valid_trials,
            key=lambda t: t.metrics[self.metric],
            reverse=self.maximize,
        )
        return sorted_trials[:n]

    @property
    def df(self) -> "pd.DataFrame":
        """Convert results to pandas DataFrame.

        Each row is a trial. Columns include:
        - id: Trial ID
        - status: Trial status as string
        - All parameter names
        - All metric names

        Returns:
            DataFrame with one row per trial.

        Raises:
            ImportError: If pandas is not installed.
        """
        import pandas as pd

        if not self.trials:
            return pd.DataFrame()

        rows = []
        for trial in self.trials:
            row: Dict[str, Any] = {
                "id": trial.id,
                "status": trial.status.value,
            }
            row.update(trial.params)
            row.update(trial.metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Generate a summary of the experiment results.

        Returns:
            String summary including total trials, success/failure counts,
            and best metric value.
        """
        total = len(self.trials)
        successful = len(self.successful)
        failed_count = len(self.failed)

        lines = [
            f"Experiment Results Summary",
            f"{'='*40}",
            f"Total trials: {total}",
            f"Successful: {successful}",
            f"Failed: {failed_count}",
        ]

        best = self.best
        if best is not None:
            best_value = best.metrics.get(self.metric)
            lines.append(f"Best {self.metric}: {best_value}")
            lines.append(f"Best params: {best.params}")

        if self.experiment_name:
            lines.insert(0, f"Experiment: {self.experiment_name}")

        return "\n".join(lines)
