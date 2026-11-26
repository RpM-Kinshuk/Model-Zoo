"""Exclusive locks on persistent lock files used for atomic artifact updates."""

from contextlib import contextmanager
import os
from pathlib import Path


_WINDOWS = os.name == "nt"


@contextmanager
def exclusive_file_lock(path: Path):
    with path.open("a+b") as stream:
        fd = stream.fileno()
        if _WINDOWS:
            import msvcrt

            # Byte-range locking also works when the lock file is still empty.
            stream.seek(0)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _WINDOWS:
                stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
