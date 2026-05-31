"""Worker supervision primitives used by the GPU dispatcher."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import logging
import os
import re
import signal
import threading
import time
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)


def safe_worker_name(value: str, default: str = "worker") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    safe = safe.strip("-._")
    return safe or default


def _tail_text(path: Path, max_chars: int = 65536) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except Exception as exc:
        return f"<could not read log: {exc}>"
    return text[-max_chars:]


def _read_json_optional(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def heartbeat_is_stale(
    heartbeat_path: Path,
    timeout_seconds: int,
    now: Optional[float] = None,
    started_at_epoch: Optional[float] = None,
) -> bool:
    if timeout_seconds <= 0:
        return False
    now = time.time() if now is None else now
    try:
        last_update = heartbeat_path.stat().st_mtime
    except FileNotFoundError:
        if started_at_epoch is None:
            return False
        last_update = started_at_epoch
    return now - last_update > timeout_seconds


@dataclass(frozen=True)
class WorkerJob:
    command: str
    worker_id: str = ""
    label: str = ""
    model_id: str = ""
    terminal_status_path: str = ""


@dataclass
class WorkerRecord:
    job: WorkerJob
    cuda_devices: list[int]
    active_path: Path
    heartbeat_path: Path
    log_path: Path
    started_at: str = field(default_factory=_utc_now)
    started_at_epoch: float = field(default_factory=time.time)
    pid: Optional[int] = None
    pgid: Optional[int] = None
    state: str = "running"
    reason: str = ""
    terminal_status_mtime_ns: Optional[int] = None

    def to_state(self) -> dict:
        return {
            "worker_id": self.job.worker_id,
            "label": self.job.label,
            "model_id": self.job.model_id,
            "pid": self.pid,
            "pgid": self.pgid,
            "state": self.state,
            "reason": self.reason,
            "cuda_devices": self.cuda_devices,
            "started_at": self.started_at,
            "heartbeat_path": str(self.heartbeat_path),
            "log_path": str(self.log_path),
            "command": self.job.command,
        }


class WorkerStateTracker:
    def __init__(
        self,
        log_dir,
        run_id: Optional[str] = None,
        runner_pid: Optional[int] = None,
        logger=None,
        refresh_interval_seconds: int = 30,
    ):
        self.log_dir = Path(log_dir)
        self.run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        self.runner_pid = os.getpid() if runner_pid is None else runner_pid
        self.logger = logger or logging.getLogger(__name__)
        self.active_dir = self.log_dir / "active_workers" / self.run_id
        self.current_state_path = self.log_dir / "current_state.json"
        self.records: dict[str, WorkerRecord] = {}
        self.lock = threading.Lock()
        self.refresh_interval_seconds = refresh_interval_seconds
        self._stop_event = threading.Event()
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self._write_current_state_locked()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _write_current_state_locked(self) -> None:
        active_workers = []
        for record in self.records.values():
            state = record.to_state()
            heartbeat = _read_json_optional(record.heartbeat_path)
            if heartbeat is not None:
                state["heartbeat"] = heartbeat
            active_workers.append(state)
        payload = {
            "updated_at": _utc_now(),
            "run_id": self.run_id,
            "runner_pid": self.runner_pid,
            "active_count": len(active_workers),
            "active_workers": active_workers,
        }
        _write_json_atomic(self.current_state_path, payload)

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(self.refresh_interval_seconds):
            with self.lock:
                self._write_current_state_locked()

    def close(self) -> None:
        self._stop_event.set()
        self._refresh_thread.join(timeout=1)
        with self.lock:
            self._write_current_state_locked()

    def _worker_id_for(self, job: WorkerJob) -> str:
        return safe_worker_name(job.worker_id or job.label or job.model_id or job.command)

    def start_worker(self, job: WorkerJob, cuda_devices: list[int], pid: Optional[int] = None, pgid: Optional[int] = None) -> WorkerRecord:
        worker_id = self._worker_id_for(job)
        normalized_job = WorkerJob(
            command=job.command,
            worker_id=worker_id,
            label=job.label or worker_id,
            model_id=job.model_id,
            terminal_status_path=job.terminal_status_path,
        )
        record = WorkerRecord(
            job=normalized_job,
            cuda_devices=list(cuda_devices),
            active_path=self.active_dir / f"{worker_id}.json",
            heartbeat_path=self.active_dir / f"{worker_id}.heartbeat.json",
            log_path=self.active_dir / f"{worker_id}.log",
            pid=pid,
            pgid=pgid,
            terminal_status_mtime_ns=self._terminal_status_mtime_ns(normalized_job.terminal_status_path),
        )
        with self.lock:
            self.records[worker_id] = record
            _write_json_atomic(record.active_path, record.to_state())
            _write_json_atomic(
                record.heartbeat_path,
                {
                    "updated_at": _utc_now(),
                    "worker_id": worker_id,
                    "model_id": normalized_job.model_id,
                    "state": "starting",
                    "stage": "dispatch",
                    "origin": "dispatcher",
                    "pid": pid,
                    "pgid": pgid,
                },
            )
            record.log_path.parent.mkdir(parents=True, exist_ok=True)
            record.log_path.touch(exist_ok=True)
            self._write_current_state_locked()
        return record

    def update_worker_pid(self, worker_id: str, pid: Optional[int], pgid: Optional[int]) -> None:
        with self.lock:
            record = self.records.get(worker_id)
            if record is None:
                return
            record.pid = pid
            record.pgid = pgid
            _write_json_atomic(record.active_path, record.to_state())
            self._write_current_state_locked()

    def mark_worker(self, worker_id: str, state: str, reason: str = "") -> None:
        with self.lock:
            record = self.records.get(worker_id)
            if record is None:
                return
            record.state = state
            record.reason = reason
            _write_json_atomic(record.active_path, record.to_state())
            self._write_current_state_locked()

    def _terminal_status_mtime_ns(self, terminal_status_path: str) -> Optional[int]:
        if not terminal_status_path:
            return None
        try:
            return Path(terminal_status_path).stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _terminal_status_updated_for_current_worker(self, record: WorkerRecord, status_path: Path) -> bool:
        try:
            current_mtime_ns = status_path.stat().st_mtime_ns
        except FileNotFoundError:
            return False
        if record.terminal_status_mtime_ns is None:
            return True
        return current_mtime_ns != record.terminal_status_mtime_ns

    def _fallback_reason(self, returncode: Optional[int], explicit_reason: Optional[str]) -> str:
        if explicit_reason:
            return explicit_reason
        if returncode is None:
            return "worker_exit_unknown"
        if returncode < 0:
            signum = -returncode
            try:
                return f"signal_{signal.Signals(signum).name}"
            except ValueError:
                return f"signal_{signum}"
        return f"process_exit_{returncode}"

    def _write_fallback_terminal_status(
        self,
        record: WorkerRecord,
        returncode: Optional[int],
        reason: Optional[str],
        message: Optional[str],
    ) -> None:
        if not record.job.terminal_status_path:
            return
        status_path = Path(record.job.terminal_status_path)
        if self._terminal_status_updated_for_current_worker(record, status_path):
            return
        failure_reason = self._fallback_reason(returncode, reason)
        payload = {
            "ts": _utc_now(),
            "model_id": record.job.model_id,
            "status": "failed",
            "stage": "supervisor",
            "reason": failure_reason,
            "message": message or failure_reason,
            "attempt": 0,
            "origin": "dispatcher",
            "worker_id": record.job.worker_id,
            "pid": record.pid,
            "pgid": record.pgid,
            "returncode": returncode,
            "cuda_devices": record.cuda_devices,
            "command": record.job.command,
            "log_tail": _tail_text(record.log_path),
        }
        _write_json_atomic(status_path, payload)
        logs_dir = status_path.parent.parent
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / "failure_records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        with open(logs_dir / "failed_models.txt", "a", encoding="utf-8") as f:
            f.write(f"{record.job.model_id}\tsupervisor\t{failure_reason}\t{payload['message']}\n")

    def finish_worker(
        self,
        worker_id: str,
        returncode: Optional[int] = 0,
        reason: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        with self.lock:
            record = self.records.pop(worker_id, None)
        if record is None:
            return
        if returncode != 0 or reason:
            self._write_fallback_terminal_status(record, returncode, reason, message)
        for path in (record.active_path, record.heartbeat_path, record.log_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                self.logger.warning(f"Could not remove active worker file {path}: {exc}")
        with self.lock:
            self._write_current_state_locked()
