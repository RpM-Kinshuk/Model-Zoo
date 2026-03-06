"""SQLite storage backend for experiments.

This module provides an SQLite-based storage implementation that persists
experiment data to a database file with query support.

Tables:
    trials: Stores trial data with experiment_name, trial_id, and JSON data
    configs: Stores experiment configuration as JSON
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.trial import Trial


class SQLiteStorage(Storage):
    """SQLite-based storage backend.

    Stores experiment data in an SQLite database with two tables:
    - trials: (experiment_name, trial_id, data) - Trial data as JSON
    - configs: (experiment_name, data) - Config data as JSON

    This backend is useful for:
    - Query support for filtering trials
    - Atomic operations and transactions
    - Efficient storage for large numbers of trials

    Example:
        >>> storage = SQLiteStorage(db_path=Path("/tmp/experiments.db"))
        >>> storage.save_trial("exp1", Trial(id=1, params={"lr": 0.01}))
        >>> trial = storage.load_trial("exp1", 1)

        # In-memory database for testing
        >>> storage = SQLiteStorage(db_path=":memory:")
    """

    def __init__(self, db_path: Union[Path, str]) -> None:
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for
                    in-memory database (useful for testing).
        """
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        # For in-memory databases, keep a persistent connection
        # since each connection to :memory: is a separate database
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        For file-based databases, creates a new connection.
        For in-memory databases, returns the persistent connection.

        Returns:
            SQLite connection object.
        """
        if self._is_memory:
            if self._conn is None:
                self._conn = sqlite3.connect(self._db_path)
            return self._conn
        return sqlite3.connect(self._db_path)

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close a connection if not using in-memory mode.

        Args:
            conn: Connection to close.
        """
        if not self._is_memory:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema if not exists."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create trials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                experiment_name TEXT NOT NULL,
                trial_id INTEGER NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (experiment_name, trial_id)
            )
        """)

        # Create configs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configs (
                experiment_name TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)

        conn.commit()
        self._close_connection(conn)

    def save_trial(self, experiment_name: str, trial: Trial) -> None:
        """Save a trial to the database.

        Uses INSERT OR REPLACE to handle updates.

        Args:
            experiment_name: Name of the experiment.
            trial: Trial to save.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        data = json.dumps(trial.to_dict())
        cursor.execute(
            """
            INSERT OR REPLACE INTO trials (experiment_name, trial_id, data)
            VALUES (?, ?, ?)
            """,
            (experiment_name, trial.id, data),
        )

        conn.commit()
        self._close_connection(conn)

    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]:
        """Load a single trial by ID.

        Args:
            experiment_name: Name of the experiment.
            trial_id: ID of the trial to load.

        Returns:
            The trial if found, None otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT data FROM trials
            WHERE experiment_name = ? AND trial_id = ?
            """,
            (experiment_name, trial_id),
        )

        row = cursor.fetchone()
        self._close_connection(conn)

        if row is None:
            return None

        data = json.loads(row[0])
        return Trial.from_dict(data)

    def load_trials(self, experiment_name: str) -> List[Trial]:
        """Load all trials for an experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            List of all trials for the experiment.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT data FROM trials
            WHERE experiment_name = ?
            ORDER BY trial_id
            """,
            (experiment_name,),
        )

        rows = cursor.fetchall()
        self._close_connection(conn)

        trials = []
        for row in rows:
            data = json.loads(row[0])
            trials.append(Trial.from_dict(data))

        return trials

    def save_config(self, experiment_name: str, config: Dict[str, Any]) -> None:
        """Save experiment configuration to the database.

        Uses INSERT OR REPLACE to handle updates.

        Args:
            experiment_name: Name of the experiment.
            config: Configuration dictionary to save.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        data = json.dumps(config)
        cursor.execute(
            """
            INSERT OR REPLACE INTO configs (experiment_name, data)
            VALUES (?, ?)
            """,
            (experiment_name, data),
        )

        conn.commit()
        self._close_connection(conn)

    def load_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment configuration from the database.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Configuration dictionary if found, None otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT data FROM configs
            WHERE experiment_name = ?
            """,
            (experiment_name,),
        )

        row = cursor.fetchone()
        self._close_connection(conn)

        if row is None:
            return None

        return json.loads(row[0])

    def list_experiments(self) -> List[str]:
        """List all experiment names in the database.

        Returns:
            List of unique experiment names from both trials and configs.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get experiments from trials
        cursor.execute("SELECT DISTINCT experiment_name FROM trials")
        trial_experiments = {row[0] for row in cursor.fetchall()}

        # Get experiments from configs
        cursor.execute("SELECT experiment_name FROM configs")
        config_experiments = {row[0] for row in cursor.fetchall()}

        self._close_connection(conn)

        return list(trial_experiments | config_experiments)
