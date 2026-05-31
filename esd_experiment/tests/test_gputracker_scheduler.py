import sys
import threading
import json
import time
from pathlib import Path
from types import SimpleNamespace

_UNSET = object()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from gputracker.gputracker import ChildThread, DispatchThread, GPUDispatcher, WorkerJob, WorkerStateTracker, heartbeat_is_stale
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
                "termination_grace_seconds": 7,
            }
        )
    )
    GPUDispatcher._instance = None

    dispatcher = GPUDispatcher(config_path=str(config_path))

    assert dispatcher.config["stale_process_action"] == "terminate"
    assert dispatcher.config["heartbeat_timeout_seconds"] == 123
    assert dispatcher.config["termination_grace_seconds"] == 7


def test_dispatcher_invalid_stale_action_fails_closed_to_log(tmp_path: Path):
    config_path = tmp_path / "gpu_config.json"
    config_path.write_text(json.dumps({"available_gpus": [0], "stale_process_action": "delete_everything"}))
    GPUDispatcher._instance = None

    dispatcher = GPUDispatcher(config_path=str(config_path))

    assert dispatcher.config["stale_process_action"] == "log"


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
