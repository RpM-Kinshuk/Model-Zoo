"""Tests for Results container class."""

from datetime import datetime

import pytest

from gpudispatch.experiments.results import Results
from gpudispatch.experiments.trial import Trial, TrialStatus


# Helper function to create test trials
def make_trial(
    id: int,
    params: dict,
    metrics: dict = None,
    status: TrialStatus = TrialStatus.COMPLETED,
    error: str = None,
) -> Trial:
    """Create a trial for testing."""
    return Trial(
        id=id,
        params=params,
        metrics=metrics or {},
        status=status,
        error=error,
    )


class TestResultsCreation:
    """Tests for Results creation."""

    def test_create_results_minimal(self):
        """Create Results with minimal arguments."""
        trials = [make_trial(1, {"lr": 0.01})]
        results = Results(trials=trials)
        assert results.trials == trials
        assert results.metric == "loss"  # default
        assert results.maximize is True  # default
        assert results.experiment_name is None

    def test_create_results_with_all_fields(self):
        """Create Results with all fields specified."""
        trials = [make_trial(1, {"lr": 0.01})]
        results = Results(
            trials=trials,
            metric="accuracy",
            maximize=True,
            experiment_name="my_experiment",
        )
        assert results.metric == "accuracy"
        assert results.maximize is True
        assert results.experiment_name == "my_experiment"

    def test_create_results_empty_trials(self):
        """Create Results with empty trials list."""
        results = Results(trials=[])
        assert results.trials == []


class TestResultsBest:
    """Tests for Results.best property."""

    def test_best_maximize_true(self):
        """best returns trial with highest metric when maximize=True."""
        trials = [
            make_trial(1, {"lr": 0.01}, metrics={"loss": 0.5}),
            make_trial(2, {"lr": 0.001}, metrics={"loss": 0.9}),
            make_trial(3, {"lr": 0.1}, metrics={"loss": 0.3}),
        ]
        results = Results(trials=trials, metric="loss", maximize=True)
        assert results.best.id == 2  # 0.9 is highest

    def test_best_maximize_false(self):
        """best returns trial with lowest metric when maximize=False."""
        trials = [
            make_trial(1, {"lr": 0.01}, metrics={"loss": 0.5}),
            make_trial(2, {"lr": 0.001}, metrics={"loss": 0.9}),
            make_trial(3, {"lr": 0.1}, metrics={"loss": 0.3}),
        ]
        results = Results(trials=trials, metric="loss", maximize=False)
        assert results.best.id == 3  # 0.3 is lowest

    def test_best_empty_trials(self):
        """best returns None when no trials."""
        results = Results(trials=[])
        assert results.best is None

    def test_best_no_successful_trials(self):
        """best returns None when no successful trials."""
        trials = [
            make_trial(1, {"lr": 0.01}, status=TrialStatus.FAILED),
            make_trial(2, {"lr": 0.001}, status=TrialStatus.PRUNED),
        ]
        results = Results(trials=trials)
        assert results.best is None

    def test_best_missing_metric(self):
        """best skips trials without the specified metric."""
        trials = [
            make_trial(1, {"lr": 0.01}, metrics={}),  # No loss metric
            make_trial(2, {"lr": 0.001}, metrics={"loss": 0.5}),
        ]
        results = Results(trials=trials, metric="loss", maximize=False)
        assert results.best.id == 2


class TestResultsBestParams:
    """Tests for Results.best_params and best_metrics properties."""

    def test_best_params(self):
        """best_params returns params of best trial."""
        trials = [
            make_trial(1, {"lr": 0.01}, metrics={"loss": 0.5}),
            make_trial(2, {"lr": 0.001, "batch": 32}, metrics={"loss": 0.3}),
        ]
        results = Results(trials=trials, metric="loss", maximize=False)
        assert results.best_params == {"lr": 0.001, "batch": 32}

    def test_best_params_empty(self):
        """best_params returns empty dict when no best trial."""
        results = Results(trials=[])
        assert results.best_params == {}

    def test_best_metrics(self):
        """best_metrics returns metrics of best trial."""
        trials = [
            make_trial(1, {"lr": 0.01}, metrics={"loss": 0.5, "acc": 0.8}),
            make_trial(2, {"lr": 0.001}, metrics={"loss": 0.3, "acc": 0.95}),
        ]
        results = Results(trials=trials, metric="loss", maximize=False)
        assert results.best_metrics == {"loss": 0.3, "acc": 0.95}

    def test_best_metrics_empty(self):
        """best_metrics returns empty dict when no best trial."""
        results = Results(trials=[])
        assert results.best_metrics == {}


class TestResultsFiltering:
    """Tests for Results.successful and failed properties."""

    def test_successful_trials(self):
        """successful returns only completed trials."""
        trials = [
            make_trial(1, {}, status=TrialStatus.COMPLETED),
            make_trial(2, {}, status=TrialStatus.FAILED),
            make_trial(3, {}, status=TrialStatus.COMPLETED),
            make_trial(4, {}, status=TrialStatus.PRUNED),
            make_trial(5, {}, status=TrialStatus.RUNNING),
        ]
        results = Results(trials=trials)
        successful = results.successful
        assert len(successful) == 2
        assert all(t.status == TrialStatus.COMPLETED for t in successful)

    def test_failed_trials(self):
        """failed returns only failed trials."""
        trials = [
            make_trial(1, {}, status=TrialStatus.COMPLETED),
            make_trial(2, {}, status=TrialStatus.FAILED, error="Error 1"),
            make_trial(3, {}, status=TrialStatus.FAILED, error="Error 2"),
            make_trial(4, {}, status=TrialStatus.PRUNED),
        ]
        results = Results(trials=trials)
        failed = results.failed
        assert len(failed) == 2
        assert all(t.status == TrialStatus.FAILED for t in failed)


class TestResultsTop:
    """Tests for Results.top method."""

    def test_top_n(self):
        """top(n) returns n best trials."""
        trials = [
            make_trial(1, {}, metrics={"loss": 0.5}),
            make_trial(2, {}, metrics={"loss": 0.9}),
            make_trial(3, {}, metrics={"loss": 0.3}),
            make_trial(4, {}, metrics={"loss": 0.7}),
        ]
        results = Results(trials=trials, metric="loss", maximize=True)
        top3 = results.top(3)
        assert len(top3) == 3
        # Should be ordered: 0.9, 0.7, 0.5
        assert top3[0].id == 2
        assert top3[1].id == 4
        assert top3[2].id == 1

    def test_top_n_minimize(self):
        """top(n) respects maximize=False."""
        trials = [
            make_trial(1, {}, metrics={"loss": 0.5}),
            make_trial(2, {}, metrics={"loss": 0.1}),
            make_trial(3, {}, metrics={"loss": 0.3}),
        ]
        results = Results(trials=trials, metric="loss", maximize=False)
        top2 = results.top(2)
        # Should be ordered: 0.1, 0.3
        assert top2[0].id == 2
        assert top2[1].id == 3

    def test_top_more_than_available(self):
        """top(n) returns all trials when n > available."""
        trials = [
            make_trial(1, {}, metrics={"loss": 0.5}),
            make_trial(2, {}, metrics={"loss": 0.3}),
        ]
        results = Results(trials=trials, metric="loss", maximize=False)
        top5 = results.top(5)
        assert len(top5) == 2


class TestResultsDataFrame:
    """Tests for Results.df property."""

    def test_df_basic(self):
        """df property returns a DataFrame with correct data."""
        pytest.importorskip("pandas")
        trials = [
            make_trial(1, {"lr": 0.01}, metrics={"loss": 0.5}),
            make_trial(2, {"lr": 0.001}, metrics={"loss": 0.3}),
        ]
        results = Results(trials=trials)
        df = results.df

        assert len(df) == 2
        assert "id" in df.columns
        assert "status" in df.columns
        assert "lr" in df.columns
        assert "loss" in df.columns
        assert list(df["id"]) == [1, 2]
        assert list(df["lr"]) == [0.01, 0.001]
        assert list(df["loss"]) == [0.5, 0.3]

    def test_df_empty_trials(self):
        """df returns empty DataFrame for no trials."""
        pytest.importorskip("pandas")
        results = Results(trials=[])
        df = results.df
        assert len(df) == 0

    def test_df_preserves_status(self):
        """df includes status as string."""
        pytest.importorskip("pandas")
        trials = [
            make_trial(1, {}, status=TrialStatus.COMPLETED),
            make_trial(2, {}, status=TrialStatus.FAILED),
        ]
        results = Results(trials=trials)
        df = results.df
        assert list(df["status"]) == ["completed", "failed"]


class TestResultsSummary:
    """Tests for Results.summary method."""

    def test_summary_basic(self):
        """summary returns a string with key statistics."""
        trials = [
            make_trial(1, {"lr": 0.01}, metrics={"loss": 0.5}, status=TrialStatus.COMPLETED),
            make_trial(2, {"lr": 0.001}, metrics={"loss": 0.3}, status=TrialStatus.COMPLETED),
            make_trial(3, {"lr": 0.1}, metrics={"loss": 0.9}, status=TrialStatus.FAILED),
        ]
        results = Results(trials=trials, metric="loss", maximize=False)
        summary = results.summary()

        assert "3" in summary  # total trials
        assert "2" in summary  # successful trials
        assert "1" in summary  # failed trials
        assert "0.3" in summary  # best metric

    def test_summary_empty_trials(self):
        """summary handles empty trials."""
        results = Results(trials=[])
        summary = results.summary()
        assert "0" in summary  # total trials
