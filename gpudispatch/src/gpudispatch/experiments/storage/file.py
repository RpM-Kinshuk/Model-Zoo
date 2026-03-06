"""File-based storage backend for experiments.

This module provides a file-based storage implementation that persists
experiment data to JSON and CSV files in a directory structure.

Directory structure:
    base_dir/
        experiment-name/
            config.json    - Experiment configuration
            trials.csv     - All trials in CSV format
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.trial import Trial


class FileStorage(Storage):
    """File-based storage backend.

    Stores experiment data in a directory structure with:
    - config.json: Experiment configuration as JSON
    - trials.csv: Trial data in CSV format (one row per trial)

    This backend is useful for:
    - Human-readable experiment logs
    - Easy integration with spreadsheet tools
    - Sharing results via file transfer

    Example:
        >>> storage = FileStorage(base_dir=Path("/tmp/experiments"))
        >>> storage.save_trial("exp1", Trial(id=1, params={"lr": 0.01}))
        >>> # Data saved to /tmp/experiments/exp1/trials.csv
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize file storage.

        Args:
            base_dir: Base directory for storing experiment data.
                     Will be created if it doesn't exist.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _experiment_dir(self, experiment_name: str) -> Path:
        """Get directory path for an experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Path to the experiment directory.
        """
        return self._base_dir / experiment_name

    def _trials_path(self, experiment_name: str) -> Path:
        """Get path to trials CSV file.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Path to trials.csv file.
        """
        return self._experiment_dir(experiment_name) / "trials.csv"

    def _config_path(self, experiment_name: str) -> Path:
        """Get path to config JSON file.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Path to config.json file.
        """
        return self._experiment_dir(experiment_name) / "config.json"

    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        """Save a trial to CSV file.

        Loads all existing trials, updates/adds the trial, and rewrites the file.

        Args:
            experiment_name: Name of the experiment.
            trial: Trial to save.
        """
        exp_dir = self._experiment_dir(experiment_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Load existing trials
        existing_trials = {t.id: t for t in self.load_trials(experiment_name)}

        # Update or add the trial
        existing_trials[trial.id] = trial

        # Write all trials back to CSV
        self._write_trials_csv(experiment_name, list(existing_trials.values()))

    def _write_trials_csv(self, experiment_name: str, trials: List[Trial]) -> None:
        """Write trials to CSV file.

        Args:
            experiment_name: Name of the experiment.
            trials: List of trials to write.
        """
        trials_path = self._trials_path(experiment_name)

        if not trials:
            return

        # Convert trials to dicts for CSV writing
        rows = [trial.to_dict() for trial in trials]

        # Serialize nested dicts as JSON strings
        for row in rows:
            row["params"] = json.dumps(row["params"])
            row["metrics"] = json.dumps(row["metrics"])

        # Write CSV
        fieldnames = ["id", "params", "metrics", "status", "error", "started_at", "completed_at"]
        with open(trials_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        """Load a single trial by ID.

        Args:
            experiment_name: Name of the experiment.
            trial_id: ID of the trial to load.

        Returns:
            The trial if found, None otherwise.
        """
        trials = self.load_trials(experiment_name)
        for trial in trials:
            if trial.id == trial_id:
                return trial
        return None

    def load_trials(self, experiment_name: str) -> List[Trial]:
        """Load all trials from CSV file.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            List of all trials for the experiment.
        """
        trials_path = self._trials_path(experiment_name)

        if not trials_path.exists():
            return []

        trials = []
        with open(trials_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse JSON fields
                row["id"] = int(row["id"])
                row["params"] = json.loads(row["params"])
                row["metrics"] = json.loads(row["metrics"])
                # Handle empty error field
                if row["error"] == "":
                    row["error"] = None
                # Handle empty timestamp fields
                if row["started_at"] == "":
                    row["started_at"] = None
                if row["completed_at"] == "":
                    row["completed_at"] = None

                trial = Trial.from_dict(row)
                trials.append(trial)

        return trials

    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        """Save experiment configuration to JSON file.

        Args:
            experiment_name: Name of the experiment.
            config: Configuration dictionary to save.
        """
        exp_dir = self._experiment_dir(experiment_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        config_path = self._config_path(experiment_name)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment configuration from JSON file.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Configuration dictionary if found, None otherwise.
        """
        config_path = self._config_path(experiment_name)

        if not config_path.exists():
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_experiments(self) -> List[str]:
        """List all experiment names in storage.

        Returns:
            List of experiment names (subdirectories that contain
            either trials.csv or config.json).
        """
        experiments = []

        if not self._base_dir.exists():
            return experiments

        for path in self._base_dir.iterdir():
            if path.is_dir():
                # Check if directory contains experiment data
                has_trials = (path / "trials.csv").exists()
                has_config = (path / "config.json").exists()
                if has_trials or has_config:
                    experiments.append(path.name)

        return experiments
