"""Platform lock dispatch without requiring a Windows test host."""

from types import SimpleNamespace
import sys

from evidence import file_lock


def test_windows_always_unlocks_the_same_byte(tmp_path, monkeypatch):
    import pytest

    calls = []

    def locking(fd, mode, count):
        import os

        calls.append((mode, count, os.lseek(fd, 0, os.SEEK_CUR)))

    monkeypatch.setattr(file_lock, "_WINDOWS", True)
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_LOCK=1, LK_UNLCK=2))
    with pytest.raises(RuntimeError):
        with file_lock.exclusive_file_lock(tmp_path / ".lock"):
            raise RuntimeError("failed update")
    assert calls == [(1, 1, 0), (2, 1, 0)]
