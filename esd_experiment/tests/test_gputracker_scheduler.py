import sys
import threading
import json
import time
import os
import shlex
import signal
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

_UNSET = object()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from gputracker.gputracker import ChildThread, DispatchThread, GPUDispatcher, WorkerJob, WorkerStateTracker, heartbeat_is_stale, heartbeat_stage_is_stale
import gputracker.gputracker as gputracker_module
import gputracker.supervision as supervision_module


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


def _dispatcher(tmp_path, monkeypatch):
    monkeypatch.setattr(GPUDispatcher, "_instance", None)
    config_path = tmp_path / "gpu_config.json"
    config_path.write_text(json.dumps({"available_gpus": [0], "termination_grace_seconds": 1}))
    return GPUDispatcher(config_path=str(config_path))


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


def test_gputracker_reexports_supervision_symbols():
    assert gputracker_module.WorkerJob is supervision_module.WorkerJob
    assert gputracker_module.WorkerStateTracker is supervision_module.WorkerStateTracker
    assert gputracker_module.heartbeat_is_stale is supervision_module.heartbeat_is_stale
    assert gputracker_module.heartbeat_stage_is_stale is supervision_module.heartbeat_stage_is_stale


def test_worker_state_tracker_creates_and_deletes_worker_cache(tmp_path: Path):
    tracker = WorkerStateTracker(
        log_dir=tmp_path / "logs",
        cache_root=tmp_path / "cache",
        run_id="run-1",
        runner_pid=42,
    )
    job = WorkerJob(
        command="python worker.py",
        worker_id="000001-org--model",
        label="org/model",
        model_id="org/model",
        terminal_status_path=str(tmp_path / "logs" / "terminal_status" / "org--model.json"),
    )

    record = tracker.start_worker(job, cuda_devices=[0], pid=111, pgid=111)
    assert record.cache_path == tmp_path / "cache" / "run-1" / "000001-org--model"
    (record.cache_path / "hub").mkdir(parents=True)
    (record.cache_path / "hub" / "blob").write_text("cached")

    tracker.finish_worker(job.worker_id, returncode=0)

    assert not record.cache_path.exists()


def test_child_thread_sets_worker_cache_environment(tmp_path: Path):
    thread = ChildThread(
        name="test",
        counter=1,
        cuda_devices=[0],
        job=WorkerJob(command="true"),
        logger=_Logger(),
        dispatcher=SimpleNamespace(),
    )
    env = {}
    record = SimpleNamespace(cache_path=tmp_path / "worker-cache")

    thread._apply_worker_cache_env(env, record)

    assert env["HF_HOME"] == str(record.cache_path)
    assert env["HF_HUB_CACHE"] == str(record.cache_path / "hub")
    assert env["TRANSFORMERS_CACHE"] == str(record.cache_path / "transformers")
    assert env["HF_DATASETS_CACHE"] == str(record.cache_path / "datasets")


def test_dispatcher_loads_minimal_stale_process_config(tmp_path: Path):
    config_path = tmp_path / "gpu_config.json"
    config_path.write_text(
        json.dumps(
            {
                "available_gpus": [0, 1],
                "max_checks": 1,
                "memory_threshold_mb": 500,
                "max_concurrent_jobs": 2,
                "stale_process_action": "terminate",
                "heartbeat_timeout_seconds": 123,
                "stage_timeout_seconds": {"load": 456, "default": 789},
                "termination_grace_seconds": 7,
            }
        )
    )
    GPUDispatcher._instance = None

    dispatcher = GPUDispatcher(config_path=str(config_path))

    assert dispatcher.config["stale_process_action"] == "terminate"
    assert dispatcher.config["heartbeat_timeout_seconds"] == 123
    assert dispatcher.config["stage_timeout_seconds"] == {"load": 456, "default": 789}
    assert dispatcher.config["termination_grace_seconds"] == 7


def test_dispatcher_invalid_stale_action_fails_closed_to_log(tmp_path: Path):
    config_path = tmp_path / "gpu_config.json"
    config_path.write_text(json.dumps({"available_gpus": [0], "stale_process_action": "delete_everything"}))
    GPUDispatcher._instance = None

    dispatcher = GPUDispatcher(config_path=str(config_path))

    assert dispatcher.config["stale_process_action"] == "log"


def test_heartbeat_stage_stale_uses_stage_entered_at_not_heartbeat_mtime(tmp_path: Path):
    heartbeat_path = tmp_path / "worker.heartbeat.json"
    heartbeat_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stage_entered_at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
                "state": "running",
                "stage": "load",
            }
        )
    )

    stale = heartbeat_stage_is_stale(
        heartbeat_path,
        {"load": 10, "default": 0},
        now=datetime.now(timezone.utc).timestamp(),
    )

    assert stale == ("stale_stage_timeout", "Stage 'load' has not advanced for 10 seconds")


def test_child_thread_prefers_stage_timeout_when_heartbeat_is_fresh(tmp_path: Path):
    dispatcher = SimpleNamespace(
        config={
            "stale_process_action": "log",
            "heartbeat_timeout_seconds": 3600,
            "stage_timeout_seconds": {"load": 10, "default": 0},
            "termination_grace_seconds": 1,
        },
        lock=threading.Lock(),
    )
    thread = ChildThread(
        name="test",
        counter=1,
        cuda_devices=[0],
        job=WorkerJob(command="true", worker_id="worker-1"),
        logger=_Logger(),
        dispatcher=dispatcher,
    )
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", run_id="run-1", runner_pid=42)
    record = tracker.start_worker(thread.job, cuda_devices=[0], pid=111, pgid=111)
    record.heartbeat_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stage_entered_at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
                "state": "running",
                "stage": "load",
            }
        )
    )

    reason, message = thread._stale_worker_reason(record)

    assert reason == "stale_stage_timeout"
    assert message == "Stage 'load' has not advanced for 10 seconds"


def test_worker_state_tracker_writes_current_state_and_deletes_active_files(tmp_path: Path):
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", run_id="run-1", runner_pid=42)
    job = WorkerJob(
        command="python worker.py",
        worker_id="000001-org--model",
        label="org/model",
        model_id="org/model",
        terminal_status_path=str(tmp_path / "logs" / "terminal_status" / "org--model.json"),
    )

    record = tracker.start_worker(job, cuda_devices=[0], pid=111, pgid=111)
    started_state = json.loads((tmp_path / "logs" / "current_state.json").read_text())
    assert started_state["active_count"] == 1
    assert started_state["active_workers"][0]["heartbeat"]["stage"] == "dispatch"
    record.heartbeat_path.write_text(json.dumps({"updated_at": "old"}))
    record.log_path.write_text("worker output\\n")
    tracker.finish_worker(job.worker_id, returncode=0)

    current_state = json.loads((tmp_path / "logs" / "current_state.json").read_text())
    assert current_state["active_workers"] == []
    assert not record.heartbeat_path.exists()
    assert not record.log_path.exists()


def test_worker_state_tracker_writes_fallback_status_for_crash(tmp_path: Path):
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", run_id="run-1", runner_pid=42)
    terminal_status_path = tmp_path / "logs" / "terminal_status" / "org--model.json"
    job = WorkerJob(
        command="python worker.py",
        worker_id="000001-org--model",
        label="org/model",
        model_id="org/model",
        terminal_status_path=str(terminal_status_path),
    )

    record = tracker.start_worker(job, cuda_devices=[0], pid=111, pgid=111)
    record.log_path.write_text("last useful line\\n")
    tracker.finish_worker(job.worker_id, returncode=1)

    payload = json.loads(terminal_status_path.read_text())
    assert payload["status"] == "failed"
    assert payload["stage"] == "supervisor"
    assert payload["reason"] == "process_exit_1"
    assert payload["origin"] == "dispatcher"
    assert "last useful line" in payload["log_tail"]
    assert not record.heartbeat_path.exists()
    assert not record.log_path.exists()


def test_worker_state_tracker_periodically_refreshes_current_state(tmp_path: Path):
    tracker = WorkerStateTracker(
        log_dir=tmp_path / "logs",
        run_id="run-1",
        runner_pid=42,
        refresh_interval_seconds=0.01,
    )
    job = WorkerJob(
        command="python worker.py",
        worker_id="000001-org--model",
        label="org/model",
        model_id="org/model",
        terminal_status_path=str(tmp_path / "logs" / "terminal_status" / "org--model.json"),
    )

    record = tracker.start_worker(job, cuda_devices=[0], pid=111, pgid=111)
    record.heartbeat_path.write_text(json.dumps({"stage": "load", "state": "running"}))
    time.sleep(0.05)
    tracker.close()

    current_state = json.loads((tmp_path / "logs" / "current_state.json").read_text())
    assert current_state["active_workers"][0]["heartbeat"]["stage"] == "load"


def test_worker_state_tracker_replaces_stale_prior_terminal_status_on_crash(tmp_path: Path):
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", run_id="run-1", runner_pid=42)
    terminal_status_path = tmp_path / "logs" / "terminal_status" / "org--model.json"
    terminal_status_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_status_path.write_text(json.dumps({"status": "failed", "origin": "old-run"}))
    job = WorkerJob(
        command="python worker.py",
        worker_id="000001-org--model",
        label="org/model",
        model_id="org/model",
        terminal_status_path=str(terminal_status_path),
    )

    tracker.start_worker(job, cuda_devices=[0], pid=111, pgid=111)
    tracker.finish_worker(job.worker_id, returncode=1)

    payload = json.loads(terminal_status_path.read_text())
    assert payload["origin"] == "dispatcher"
    assert payload["reason"] == "process_exit_1"


def test_worker_state_tracker_does_not_overwrite_worker_terminal_status(tmp_path: Path):
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", run_id="run-1", runner_pid=42)
    terminal_status_path = tmp_path / "logs" / "terminal_status" / "org--model.json"
    terminal_status_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_status_path.write_text(json.dumps({"status": "failed", "origin": "worker"}))
    job = WorkerJob(
        command="python worker.py",
        worker_id="000001-org--model",
        label="org/model",
        model_id="org/model",
        terminal_status_path=str(terminal_status_path),
    )

    tracker.start_worker(job, cuda_devices=[0], pid=111, pgid=111)
    time.sleep(0.01)
    terminal_status_path.write_text(json.dumps({"status": "failed", "origin": "worker"}))
    tracker.finish_worker(job.worker_id, returncode=1)

    payload = json.loads(terminal_status_path.read_text())
    assert payload["origin"] == "worker"


def test_heartbeat_is_stale_uses_file_mtime(tmp_path: Path):
    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text("{}")

    assert heartbeat_is_stale(heartbeat_path, timeout_seconds=10, now=heartbeat_path.stat().st_mtime + 11)
    assert not heartbeat_is_stale(heartbeat_path, timeout_seconds=10, now=heartbeat_path.stat().st_mtime + 9)


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("ignore_term", [False, True])
def test_stop_waits_for_worker_group_cleanup(tmp_path, monkeypatch, stop_signal, ignore_term):
    """Exercise real children, including a child surviving its shell's SIGTERM."""
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    dispatcher.occupied_gpus.add(0)
    ready = tmp_path / "ready.json"
    code = (
        "import json, os, pathlib, signal, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "cache = pathlib.Path(os.environ['HF_HOME']); (cache / 'blob').write_text('cached'); "
        + f"pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid(), 'cache': str(cache)}})); "
        + "time.sleep(60)"
    )
    terminal = tmp_path / "logs/terminal_status/org--model.json"
    job = WorkerJob(command=shlex.join([sys.executable, "-c", code]) + " & wait",
                    worker_id="worker-1", model_id="org/model", terminal_status_path=str(terminal))
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", cache_root=tmp_path / "cache", run_id="run-1")
    thread = ChildThread("test", 1, [0], job, _Logger(), dispatcher, tracker)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    thread.start()
    worker_pid = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        payload = json.loads(ready.read_text())
        worker_pid = payload["pid"]
        dispatcher.handle_hard_stop(stop_signal, None)
        dispatcher.handle_hard_stop(stop_signal, None)  # Repeated signals must not bypass cleanup.
        thread.join(timeout=12)
        assert not thread.is_alive()
        assert not Path(payload["cache"]).exists()
        assert json.loads((tmp_path / "logs/current_state.json").read_text())["active_workers"] == []
        outcome = json.loads(terminal.read_text())
        assert outcome["status"] == "failed"
        assert outcome["reason"] in {"run_interrupted", "run_interrupted_killed"}
        assert not dispatcher.occupied_gpus
        if psutil.pid_exists(worker_pid):
            assert psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        dispatcher.shutdown_event.set()
        if worker_pid is not None:
            try:
                os.killpg(os.getpgid(worker_pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        thread.join(timeout=12)
        unrelated.terminate()
        unrelated.wait(timeout=5)
        tracker.close()


def test_stop_before_child_launch_releases_selected_gpu(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    dispatcher.shutdown_event.set()
    dispatcher.occupied_gpus.add(0)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Worker launched after stop"))
    thread = ChildThread("test", 1, [0], WorkerJob(command="true"), _Logger(), dispatcher)
    thread.run()
    assert not dispatcher.occupied_gpus


def test_supervisor_write_failure_terminates_worker_before_cleanup(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    dispatcher.occupied_gpus.add(0)
    ready = tmp_path / "ready"
    command = shlex.join([sys.executable, "-c", f"import pathlib, time; pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"])
    job = WorkerJob(command="exec " + command, worker_id="worker-1", model_id="org/model",
                    terminal_status_path=str(tmp_path / "logs/terminal_status/org--model.json"))
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", cache_root=tmp_path / "cache", run_id="run-1")
    thread = ChildThread("test", 1, [0], job, _Logger(), dispatcher, tracker)
    processes = []

    def fail_pid_write(*_args):
        processes.append(thread.process)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), "CPU test child did not start"
        raise OSError("injected PID-state write failure")

    monkeypatch.setattr(tracker, "update_worker_pid", fail_pid_write)
    thread.start()
    try:
        thread.join(timeout=12)
        assert not thread.is_alive()
        assert dispatcher.error and dispatcher.shutdown_event.is_set()
        assert processes and processes[0].poll() is not None
        assert not dispatcher.occupied_gpus
        assert not (tmp_path / "cache/run-1/worker-1").exists()
        assert json.loads(Path(job.terminal_status_path).read_text())["status"] == "failed"
    finally:
        dispatcher.shutdown_event.set()
        if thread.process is not None and thread.process.poll() is None:
            os.killpg(thread.process.pid, signal.SIGKILL)
            thread.process.wait(timeout=5)
        thread.join(timeout=12)
        tracker.close()


@pytest.mark.parametrize("failed_step", ["status", "cache"])
def test_finalization_failure_stops_dispatch_but_releases_dead_worker(tmp_path, monkeypatch, failed_step):
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    dispatcher.occupied_gpus.add(0)
    job = WorkerJob(command="exit 1", worker_id="worker-1", model_id="org/model",
                    terminal_status_path=str(tmp_path / "logs/terminal_status/org--model.json"))
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", cache_root=tmp_path / "cache", run_id="run-1")

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failed_step} failure")

    if failed_step == "status":
        monkeypatch.setattr(tracker, "_write_fallback_terminal_status", fail)
    else:
        monkeypatch.setattr(supervision_module.shutil, "rmtree", fail)
    thread = ChildThread("test", 1, [0], job, _Logger(), dispatcher, tracker)
    try:
        thread.run()
        assert dispatcher.error and dispatcher.shutdown_event.is_set()
        assert not dispatcher.occupied_gpus
        cache = tmp_path / "cache/run-1/worker-1"
        if failed_step == "status":
            assert not cache.exists()
            assert (tracker.active_dir / "worker-1.log").exists()
        else:
            assert cache.exists()
            assert Path(job.terminal_status_path).exists()
    finally:
        tracker.close()


def test_dispatch_start_failure_returns_unstarted_reservation(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    launched = []

    def reserve(_num_needed, progress=None):
        dispatcher.occupied_gpus.add(0)
        return [0]

    def fail_start(thread):
        launched.append(thread.job.worker_id)
        raise RuntimeError("injected thread-start failure")

    monkeypatch.setattr(dispatcher, "get_free_gpus", reserve)
    monkeypatch.setattr(ChildThread, "start", fail_start)
    thread = DispatchThread("test", [WorkerJob("true", worker_id="first"), WorkerJob("true", worker_id="second")],
                            _Logger(), dispatcher)
    thread.run()
    assert launched == ["first"]
    assert dispatcher.error and dispatcher.shutdown_event.is_set()
    assert not dispatcher.occupied_gpus


def test_unconfirmed_termination_retains_cache_and_gpu_reservation(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    dispatcher.occupied_gpus.add(0)
    tracker = WorkerStateTracker(log_dir=tmp_path / "logs", cache_root=tmp_path / "cache", run_id="run-1")
    thread = ChildThread("test", 1, [0], WorkerJob("unused", worker_id="worker-1"), _Logger(), dispatcher, tracker)
    process = SimpleNamespace(pid=987654321, poll=lambda: None)

    def start(*_args, **_kwargs):
        dispatcher.shutdown_event.set()
        return process

    def unconfirmed(*_args):
        raise TimeoutError("injected unconfirmed process-group termination")

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(thread, "_group_running", lambda _proc: True)
    monkeypatch.setattr(thread, "_terminate_process", unconfirmed)
    try:
        thread.run()
        assert dispatcher.error and dispatcher.shutdown_event.is_set()
        assert dispatcher.occupied_gpus == {0}
        assert thread.process is process
        assert (tmp_path / "cache/run-1/worker-1").exists()
        assert (tracker.active_dir / "worker-1.json").exists()
    finally:
        tracker.close()


@pytest.mark.parametrize("stuck_at", ["cleanup", "refresh"])
def test_shutdown_deadline_exits_without_waiting_for_stuck_cleanup(tmp_path, monkeypatch, stuck_at):
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    thread = DispatchThread("test", [], _Logger(), dispatcher)
    process = SimpleNamespace(pid=987654321)
    thread.workers.append(SimpleNamespace(process=process, job=WorkerJob("unused", worker_id="worker-1"),
                                           last_progress=0, is_alive=lambda: True))
    thread.last_progress = 0
    dispatcher.shutdown_event.set()
    dispatcher.stop_started_at = 0
    signals = []
    messages = []
    release_stderr = threading.Event()

    def emergency_exit(code):
        raise SystemExit(code)

    def write_stderr(fd, message):
        messages.append(message)
        if stuck_at == "refresh":
            release_stderr.wait()  # Diagnostics must not defeat the deadline either.

    if stuck_at == "refresh":
        thread.state_tracker = SimpleNamespace(refresh_started_at=0)
    monkeypatch.setattr(thread, "is_alive", lambda: stuck_at == "cleanup")
    monkeypatch.setattr(thread, "join", lambda timeout: None)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(time, "monotonic", lambda: 100)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(os, "write", write_stderr)
    monkeypatch.setattr(os, "_exit", emergency_exit)
    try:
        with pytest.raises(SystemExit) as result:
            thread.wait_for_completion()
    finally:
        release_stderr.set()
    assert result.value.code == 1
    assert signals == [(process.pid, signal.SIGKILL)]
    assert b"termination/cleanup may be incomplete" in messages[0]


def test_initial_config_failure_does_not_fall_back_to_default_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(GPUDispatcher, "_instance", None)
    config_path = tmp_path / "gpu_config.json"
    with pytest.raises(ValueError, match="initial GPU configuration"):
        GPUDispatcher(str(config_path))
    config_path.write_text(json.dumps({"available_gpus": [3]}))
    assert GPUDispatcher(str(config_path)).config["available_gpus"] == [3]


def test_gpu_selection_skips_small_active_jobs(tmp_path, monkeypatch):
    dispatcher = _dispatcher(tmp_path, monkeypatch)
    dispatcher.config.update(available_gpus=[0, 1], max_checks=1)
    stats = SimpleNamespace(gpus=[{"memory.used": 10, "processes": [{"pid": 123}]},
                                 {"memory.used": 0, "processes": []}])
    monkeypatch.setattr(gputracker_module.gpustat.GPUStatCollection, "new_query", lambda: stats)
    assert dispatcher.get_free_gpus(1) == [1]
