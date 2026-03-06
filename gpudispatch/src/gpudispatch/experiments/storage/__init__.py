"""Storage backends for experiment persistence.

This module provides multiple storage backends for persisting experiment
trials and configuration:

- MemoryStorage: In-memory storage (non-persistent, useful for testing)
- FileStorage: JSON/CSV file-based storage
- SQLiteStorage: SQLite database storage with query support

All backends implement the Storage ABC and can be used interchangeably.

Example:
    >>> from gpudispatch.experiments.storage import MemoryStorage, FileStorage
    >>> from gpudispatch.experiments.trial import Trial
    >>>
    >>> # Use memory storage for testing
    >>> storage = MemoryStorage()
    >>>
    >>> # Use file storage for persistence
    >>> storage = FileStorage(base_dir=Path("/tmp/experiments"))
    >>>
    >>> # Save and load trials
    >>> storage.save_trial("exp1", Trial(id=1, params={"lr": 0.01}))
    >>> trial = storage.load_trial("exp1", 1)
"""

from gpudispatch.experiments.storage.base import Storage
from gpudispatch.experiments.storage.file import FileStorage
from gpudispatch.experiments.storage.memory import MemoryStorage
from gpudispatch.experiments.storage.sqlite import SQLiteStorage

__all__ = [
    "FileStorage",
    "MemoryStorage",
    "SQLiteStorage",
    "Storage",
]
