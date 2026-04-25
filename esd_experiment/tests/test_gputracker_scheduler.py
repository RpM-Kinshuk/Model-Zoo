import sys
import threading
from pathlib import Path
from types import SimpleNamespace

_UNSET = object()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from gputracker.gputracker import DispatchThread


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _Child:
    def __init__(self, alive):
        self._alive = alive

    def is_alive(self):
        return self._alive


def _dispatch_thread(max_concurrent_jobs=None, config_max_concurrent_jobs=_UNSET):
    config = {}
    if config_max_concurrent_jobs is not _UNSET:
        config["max_concurrent_jobs"] = config_max_concurrent_jobs
    dispatcher = SimpleNamespace(
        config=config,
        lock=threading.Lock(),
    )
    return DispatchThread(
        name="test",
        bash_command_list=[],
        logger=_Logger(),
        dispatcher=dispatcher,
        max_concurrent_jobs=max_concurrent_jobs,
    )


def test_dispatch_thread_has_slot_when_concurrent_job_limit_is_unset():
    thread = _dispatch_thread()

    assert thread._has_job_slot([_Child(True), _Child(True)])


def test_dispatch_thread_enforces_max_concurrent_job_slot():
    thread = _dispatch_thread(max_concurrent_jobs=1)

    assert not thread._has_job_slot([_Child(True)])
    assert thread._has_job_slot([_Child(False)])


def test_dispatch_thread_uses_live_configured_max_concurrent_jobs():
    thread = _dispatch_thread(max_concurrent_jobs=1, config_max_concurrent_jobs=2)

    assert thread._has_job_slot([_Child(True)])
    assert not thread._has_job_slot([_Child(True), _Child(True)])


def test_dispatch_thread_allows_config_to_disable_cli_job_limit():
    thread = _dispatch_thread(max_concurrent_jobs=1, config_max_concurrent_jobs=None)

    assert thread._has_job_slot([_Child(True)])
