#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs. 
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.

from typing import Optional
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time

import gpustat
import psutil

from .supervision import (
    WorkerJob,
    WorkerRecord,
    WorkerStateTracker,
    heartbeat_is_stale,
    heartbeat_stage_is_stale,
    safe_worker_name,
)

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_NCHECK = 10            # number of checks to know if gpu free
GPU_MEMORY_THRESHOLD = 500 # MB?
STALE_PROCESS_ACTIONS = {"log", "terminate"}


class GPUDispatcher:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GPUDispatcher, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path="gpu_config.json"):
        if hasattr(self, "initialized") and self.initialized: return
        self.config_path = config_path
        self.lock = threading.Lock()

        # State Flags
        self.shutdown_event = threading.Event() # Hard stop
        self.drain_event = threading.Event()   # Graceful stop
        self.reload_event = threading.Event()
        self.error = None
        self.stop_started_at = None
        self.occupied_gpus = set()  # Track GPU usage

        # Default Config
        self.config = {
            "available_gpus": AVAILABLE_GPUS,
            "max_checks": MAX_NCHECK,
            "memory_threshold_mb": GPU_MEMORY_THRESHOLD,
            "max_concurrent_jobs": None,
            "stale_process_action": "log",
            "heartbeat_timeout_seconds": 3600,
            "stage_timeout_seconds": {
                "load": 7200,
                "analyze": 28800,
                "save": 1800,
                "default": 14400,
            },
            "termination_grace_seconds": 60,
        }
        if not self.load_config():
            raise ValueError(f"Cannot load initial GPU configuration: {self.config_path}")
        self.initialized = True

    def _normalize_stage_timeouts(self, value) -> dict[str, int]:
        if value is None:
            return {}
        if isinstance(value, (int, float, str)):
            try:
                return {"default": max(0, int(value))}
            except (TypeError, ValueError):
                logging.error("Invalid stage_timeout_seconds %r; disabling stage timeouts", value)
                return {}
        if not isinstance(value, dict):
            logging.error("Invalid stage_timeout_seconds %r; disabling stage timeouts", value)
            return {}
        normalized = {}
        for stage, seconds in value.items():
            stage_name = str(stage).strip()
            if not stage_name:
                continue
            try:
                normalized[stage_name] = max(0, int(seconds))
            except (TypeError, ValueError):
                logging.error("Invalid timeout for stage %r: %r; ignoring", stage, seconds)
        return normalized

    def _normalize_config(self, raw_config: dict) -> dict:
        merged_config = dict(self.config)
        merged_config.update(raw_config)
        merged_config["available_gpus"] = [int(x) for x in merged_config["available_gpus"]]
        gpus = merged_config["available_gpus"]
        if any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
            raise ValueError("available_gpus must contain distinct nonnegative GPU indices")
        merged_config["max_checks"] = int(merged_config["max_checks"])
        if merged_config["max_checks"] < 1:
            raise ValueError("max_checks must be >= 1")
        merged_config["memory_threshold_mb"] = int(merged_config["memory_threshold_mb"])
        if merged_config["memory_threshold_mb"] < 1:
            raise ValueError("memory_threshold_mb must be >= 1")
        if merged_config.get("max_concurrent_jobs") is not None:
            merged_config["max_concurrent_jobs"] = int(merged_config["max_concurrent_jobs"])
            if merged_config["max_concurrent_jobs"] < 1:
                raise ValueError("max_concurrent_jobs must be >= 1")
        action = str(merged_config.get("stale_process_action", "log")).strip().lower()
        if action not in STALE_PROCESS_ACTIONS:
            logging.error("Invalid stale_process_action %r; using 'log'", action)
            action = "log"
        merged_config["stale_process_action"] = action
        merged_config["heartbeat_timeout_seconds"] = max(0, int(merged_config.get("heartbeat_timeout_seconds", 3600)))
        merged_config["stage_timeout_seconds"] = self._normalize_stage_timeouts(merged_config.get("stage_timeout_seconds", {}))
        merged_config["termination_grace_seconds"] = max(1, int(merged_config.get("termination_grace_seconds", 60)))
        return merged_config

    def load_config(self):
        """Reloads configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    new_config = json.load(f)
                merged_config = self._normalize_config(new_config)
                with self.lock:
                    self.config = merged_config
                logging.info(f"Configuration reloaded: GPUs {self.config['available_gpus']}")
                return True
            else:
                logging.error(f"Config file {self.config_path} not found")
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
        return False

    def request_stop(self):
        if self.stop_started_at is None:
            self.stop_started_at = time.monotonic()
        self.shutdown_event.set()

    def fail(self, message):
        # No file/log I/O here: this is also called when storage itself fails.
        self.error = self.error or str(message)
        self.request_stop()

    # --- Signal Handlers ---
    def handle_hard_stop(self, signum, frame):
        """Stop dispatch; supervisors terminate owned groups and finish cleanup."""
        self.request_stop()

    def handle_drain(self, signum, frame):
        """SIGUSR1: Stop new jobs, wait for current ones"""
        self.drain_event.set()

    def handle_reload(self, signum, frame):
        """SIGHUP: Reload Configuration"""
        self.reload_event.set()  # Never perform filesystem I/O inside a signal handler.

    def setup_signals(self):
        signal.signal(signal.SIGINT, self.handle_hard_stop)
        signal.signal(signal.SIGTERM, self.handle_hard_stop)
        signal.signal(signal.SIGUSR1, self.handle_drain)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self.handle_reload)

    def get_free_gpus(self, num_needed, progress=None):
        """Blocking call to get free GPUs"""
        counter = {}

        while not self.shutdown_event.is_set():
            if self.drain_event.is_set(): return None # Signal to stop dispatching new jobs
            if progress:
                progress()
            if self.reload_event.is_set():
                self.reload_event.clear()
                self.load_config()  # Failed reloads retain the previous settings.
        
            try:
                #Always read latest config
                with self.lock:
                    allowed_gpus = self.config["available_gpus"]
                    threshold = self.config["memory_threshold_mb"]
                    max_checks = self.config["max_checks"]
                    current_occupied = list(self.occupied_gpus)

                stats = gpustat.GPUStatCollection.new_query()

                # Logic to find free GPUs
                candidates = []
                for i in allowed_gpus:
                    if i >= len(stats.gpus):
                        self.fail(f"Configured GPU {i} is not present; check gpu_config.json")
                        return None
                    
                    if (stats.gpus[i]['memory.used'] < threshold
                            and stats.gpus[i]['processes'] == [] and i not in current_occupied):
                        counter[i] = counter.get(i, 0) + 1
                        if counter[i] >= max_checks:
                            candidates.append(i)
                    else:
                        counter.update({i: 0})

                if len(candidates) >= num_needed:
                    selected = candidates[:num_needed]
                    with self.lock:
                        self.occupied_gpus.update(selected)
                    return selected

                time.sleep(5)

            except Exception as e:
                logging.error(f"Could not query GPU stats: {e}")
                time.sleep(5)

        return None

# --- Thread Classes ---

class DispatchThread(threading.Thread):
    def __init__(
        self,
        name,
        bash_command_list,
        logger,
        dispatcher,
        num_gpus_needed=1,
        config_path="gpu_config.json",
        max_concurrent_jobs=None,
        state_dir=None,
        run_id=None,
        cache_root=None,
    ):
        threading.Thread.__init__(self)
        self.name = name
        self.daemon = True
        self.bash_command_list = bash_command_list
        self.logger = logger
        self.dispatcher = dispatcher
        self.num_gpus_needed = num_gpus_needed
        self.max_concurrent_jobs = max_concurrent_jobs
        self.state_tracker = None
        self.state_options = dict(log_dir=state_dir, run_id=run_id, logger=logger, cache_root=cache_root) if state_dir else None
        self.workers = []
        self.last_progress = time.monotonic()

    def _progress(self):
        self.last_progress = time.monotonic()

    def wait_for_completion(self):
        """Existing main-thread wait, with a fail-closed escape for stuck control I/O.

        Worker computations have their own stage limits. Supervisor operations
        should return within one grace period (at least 30 seconds). After stop,
        allow TERM, KILL and cleanup; never wait forever for Python threads that
        cannot be cancelled while blocked in filesystem calls.
        """
        signaled = set()
        while self.is_alive() or (self.state_tracker and self.state_tracker.refresh_started_at is not None):
            if self.is_alive():
                self.join(timeout=0.5)
            else:
                time.sleep(0.5)  # A blocked refresh can outlive the dispatcher.
            now = time.monotonic()
            grace = self.dispatcher.config["termination_grace_seconds"]
            control_limit = max(30, grace)
            checks = [(self.name, self.last_progress)] if self.is_alive() else []
            checks.extend((worker.job.worker_id, worker.last_progress)
                          for worker in self.workers if worker.is_alive())
            refresh_started_at = self.state_tracker.refresh_started_at if self.state_tracker else None
            if refresh_started_at is not None:
                checks.append(("state refresh", refresh_started_at))
            for name, last_progress in checks:
                if now - last_progress > control_limit:
                    self.dispatcher.fail(f"Supervisor stalled at {name}; check storage and retained worker state")
            if not self.dispatcher.shutdown_event.is_set():
                continue
            self.dispatcher.request_stop()
            elapsed = now - self.dispatcher.stop_started_at
            stop_signal = signal.SIGTERM if elapsed < grace else signal.SIGKILL
            for worker in self.workers:
                proc = worker.process
                if proc is not None and (proc.pid, stop_signal) not in signaled:
                    try:
                        os.killpg(proc.pid, stop_signal)
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        self.dispatcher.fail(f"Could not signal owned worker group {proc.pid}: {exc}")
                    signaled.add((proc.pid, stop_signal))
            if elapsed > grace + control_limit + 10:
                # Logging/atexit can themselves wait for a blocked file handler.
                # Preserve on-disk evidence; do not call cleanup or release GPUs.
                message = (f"Shutdown deadline exceeded: {self.dispatcher.error or 'worker cleanup stalled'}. "
                           "SIGKILL attempted for owned groups; termination/cleanup may be incomplete. "
                           "Inspect worker PIDs and caches before resuming.\n")
                # Even stderr may block on shared storage or a full pipe.
                def report():
                    try:
                        os.write(2, message.encode())
                    except OSError:
                        pass
                try:
                    reporter = threading.Thread(target=report, daemon=True)
                    reporter.start()
                    reporter.join(timeout=0.2)
                finally:
                    os._exit(1)

    def _current_max_concurrent_jobs(self):
        if hasattr(self.dispatcher, "config"):
            lock = getattr(self.dispatcher, "lock", None)
            if lock is None:
                config = self.dispatcher.config
            else:
                with lock:
                    config = dict(self.dispatcher.config)
            if "max_concurrent_jobs" in config:
                return config["max_concurrent_jobs"]
        return self.max_concurrent_jobs

    def _has_job_slot(self, threads):
        max_concurrent_jobs = self._current_max_concurrent_jobs()
        if max_concurrent_jobs is None:
            return True
        return sum(1 for thread in threads if thread.is_alive()) < max_concurrent_jobs

    def _wait_for_job_slot(self, threads):
        while not self._has_job_slot(threads):
            self._progress()
            if self.dispatcher.reload_event.is_set():
                self.dispatcher.reload_event.clear()
                self.dispatcher.load_config()
            if self.dispatcher.shutdown_event.is_set() or self.dispatcher.drain_event.is_set():
                return False
            time.sleep(5)
        return True

    def _coerce_job(self, item, index: int) -> WorkerJob:
        if isinstance(item, WorkerJob):
            worker_id = item.worker_id or f"{index:06d}-{safe_worker_name(item.label or item.model_id or item.command)}"
            return WorkerJob(
                command=item.command,
                worker_id=worker_id,
                label=item.label or item.model_id or worker_id,
                model_id=item.model_id,
                terminal_status_path=item.terminal_status_path,
            )
        command = str(item)
        return WorkerJob(command=command, worker_id=f"{index:06d}-{safe_worker_name(command)[:80]}", label=command[:120])

    def run(self):
        try:
            self._progress()
            if self.state_options:
                self.state_tracker = WorkerStateTracker(**self.state_options, on_failure=self.dispatcher.fail)
            self.logger.info(f"Starting PID: {os.getpid()}: {self.name}")
            self.logger.info("Controls: SIGHUP=Reload Config, SIGUSR1=Drain, SIGINT/SIGTERM=Stop and clean up")
            for i, item in enumerate(self.bash_command_list):
                self._progress()
                if self.dispatcher.shutdown_event.is_set() or self.dispatcher.drain_event.is_set():
                    break
                if not self._wait_for_job_slot(self.workers):
                    break
                job = self._coerce_job(item, i)
                cuda_devices = self.dispatcher.get_free_gpus(self.num_gpus_needed, progress=self._progress)
                if cuda_devices is None:
                    break
                try:
                    worker = ChildThread(job.worker_id, 1, cuda_devices, job,
                                         self.logger, self.dispatcher, self.state_tracker)
                    worker.start()
                    self.workers.append(worker)
                except Exception:
                    with self.dispatcher.lock:
                        self.dispatcher.occupied_gpus.difference_update(cuda_devices)
                    raise
                time.sleep(2)
        except Exception as exc:
            self.dispatcher.fail(f"Dispatch failed: {exc}")
        finally:
            for worker in self.workers:
                while worker.is_alive():
                    self._progress()
                    if self.dispatcher.reload_event.is_set():
                        self.dispatcher.reload_event.clear()
                        self.dispatcher.load_config()
                    worker.join(timeout=0.5)
            self._progress()
            if self.state_tracker:
                try:
                    self.state_tracker.close()
                except Exception as exc:
                    self.dispatcher.fail(f"Could not close worker state: {exc}")


class ChildThread(threading.Thread):
    def __init__(self, name, counter, cuda_devices, job, logger, dispatcher, state_tracker=None):
        threading.Thread.__init__(self)
        self.name = name
        self.counter = counter
        self.cuda_devices = cuda_devices
        self.job = job if isinstance(job, WorkerJob) else WorkerJob(command=str(job))
        self.bash_command = self.job.command
        self.logger = logger
        self.dispatcher = dispatcher
        self.state_tracker = state_tracker
        self.daemon = True
        self._stale_logged = False
        self.process = None
        self.last_progress = time.monotonic()

    def _apply_worker_cache_env(self, env: dict, record: Optional[WorkerRecord]) -> None:
        if record is None or record.cache_path is None:
            return
        cache_path = record.cache_path
        env["HF_HOME"] = str(cache_path)
        env["HF_HUB_CACHE"] = str(cache_path / "hub")
        env["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
        env["HF_DATASETS_CACHE"] = str(cache_path / "datasets")

    def _current_stale_config(self) -> tuple[str, int, dict[str, int], int]:
        with self.dispatcher.lock:
            config = dict(self.dispatcher.config)
        return (
            config.get("stale_process_action", "log"),
            int(config.get("heartbeat_timeout_seconds", 3600)),
            dict(config.get("stage_timeout_seconds", {})),
            int(config.get("termination_grace_seconds", 60)),
        )

    def _group_running(self, proc):
        """Check our whole session group, not only the shell. Zombies hold no GPU."""
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return False
        found = False
        for member in psutil.process_iter():
            try:
                if os.getpgid(member.pid) == proc.pid:
                    found = True
                    if member.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}:
                        return True
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass
            except (PermissionError, psutil.AccessDenied):
                return True  # Unknown is not permission to reuse the GPU.
        if found:
            return False
        # The group may have exited during enumeration; otherwise it is unknown.
        try:
            os.killpg(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def _terminate_process(self, proc, grace_seconds: int, reason: str) -> tuple[Optional[int], str]:
        for stop_signal, timeout in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 5)):
            self.last_progress = time.monotonic()
            if not self._group_running(proc):
                return proc.poll(), reason
            try:
                os.killpg(proc.pid, stop_signal)
            except ProcessLookupError:
                pass
            if stop_signal == signal.SIGKILL:
                reason += "_killed"
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self.last_progress = time.monotonic()
                if not self._group_running(proc):
                    return proc.poll(), reason
                time.sleep(0.1)
        if self._group_running(proc):
            raise TimeoutError(f"Cannot confirm termination of worker group {proc.pid}; cache and GPU reservation retained")
        return proc.poll(), reason

    def _stale_worker_reason(
        self,
        record: WorkerRecord,
        heartbeat_timeout_seconds: Optional[int] = None,
        stage_timeout_seconds: Optional[dict[str, int]] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if heartbeat_timeout_seconds is None or stage_timeout_seconds is None:
            _, heartbeat_timeout_seconds, stage_timeout_seconds, _ = self._current_stale_config()
        if heartbeat_is_stale(
            record.heartbeat_path,
            heartbeat_timeout_seconds,
            started_at_epoch=record.started_at_epoch,
        ):
            return "stale_heartbeat_timeout", f"No heartbeat update for {heartbeat_timeout_seconds} seconds"
        stage_stale = heartbeat_stage_is_stale(record.heartbeat_path, stage_timeout_seconds)
        if stage_stale is not None:
            return stage_stale
        return None, None

    def _wait_for_process(self, proc, record: Optional[WorkerRecord]) -> tuple[Optional[int], Optional[str], Optional[str]]:
        while True:
            self.last_progress = time.monotonic()
            if self.dispatcher.shutdown_event.is_set():
                return None, "run_interrupted", "Run stopped by SIGINT/SIGTERM"
            try:
                returncode = proc.wait(timeout=5)
                if self.dispatcher.shutdown_event.is_set():
                    return returncode, "run_interrupted", "Run stopped by SIGINT/SIGTERM"
                return returncode, None, None
            except subprocess.TimeoutExpired:
                if record is None:
                    continue
                action, timeout_seconds, stage_timeouts, _ = self._current_stale_config()
                reason, message = self._stale_worker_reason(record, timeout_seconds, stage_timeouts)
                if reason is None:
                    continue
                if action == "terminate":
                    self.logger.warning(f"Terminating stale worker {record.job.worker_id}: {message}")
                    if self.state_tracker:
                        self.state_tracker.mark_worker(record.job.worker_id, "terminating", reason)
                    return None, reason, message
                if not self._stale_logged:
                    self.logger.warning(f"Stale worker observed {record.job.worker_id}: {message}")
                    if self.state_tracker:
                        self.state_tracker.mark_worker(record.job.worker_id, "stale", reason)
                    self._stale_logged = True

    def run(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cuda_devices))
        proc = None
        record = None
        log_handle = None
        returncode = None
        finish_reason = None
        finish_message = None

        try:
            self.last_progress = time.monotonic()
            if self.dispatcher.shutdown_event.is_set():
                return  # Signal arrived after GPU selection but before launch.
            self.logger.info(f"Executing on GPUs {self.cuda_devices}: {self.bash_command}")
            if self.state_tracker:
                record = self.state_tracker.start_worker(self.job, self.cuda_devices)
                env["WORKER_HEARTBEAT_FILE"] = str(record.heartbeat_path)
                self._apply_worker_cache_env(env, record)
                log_handle = open(record.log_path, "a", encoding="utf-8", buffering=1)

            if self.dispatcher.shutdown_event.is_set():
                finish_reason = "run_interrupted"  # Stop may arrive during setup I/O.
                return
            proc = subprocess.Popen(
                self.bash_command,
                shell=True,
                env=env,
                start_new_session=True,
                stdout=log_handle or None,
                stderr=subprocess.STDOUT if log_handle else None,
            )
            self.process = proc  # In-memory ownership precedes any status-file I/O.
            if self.state_tracker and record:
                try:
                    pgid = os.getpgid(proc.pid)
                except Exception:
                    pgid = None
                self.state_tracker.update_worker_pid(record.job.worker_id, proc.pid, pgid)
            returncode, finish_reason, finish_message = self._wait_for_process(proc, record)

        except Exception as e:
            finish_reason = finish_reason or "worker_supervisor_error"
            finish_message = str(e)
            returncode = 1 if returncode is None else returncode
            self.dispatcher.fail(f"Worker supervision failed for {self.job.worker_id}: {e}")
        finally:
            self.last_progress = time.monotonic()
            stopped = proc is None
            try:
                if proc is not None:
                    if self._group_running(proc):
                        _, _, _, grace = self._current_stale_config()
                        returncode, finish_reason = self._terminate_process(
                            proc, grace, finish_reason or "worker_descendants_remaining")
                    stopped = True
                    self.process = None
            except Exception as exc:
                self.dispatcher.fail(f"Worker {self.job.worker_id}: {exc}")
            try:
                if log_handle is not None:
                    log_handle.close()
            except Exception as exc:
                self.dispatcher.fail(f"Could not close log for {self.job.worker_id}: {exc}")
            try:
                self.last_progress = time.monotonic()
                if stopped and self.state_tracker and record:
                    self.state_tracker.finish_worker(record.job.worker_id, returncode=returncode,
                                                     reason=finish_reason, message=finish_message)
            except Exception as exc:
                self.dispatcher.fail(f"Could not finalize {self.job.worker_id}: {exc}")
            finally:
                if stopped:
                    with self.dispatcher.lock:
                        self.dispatcher.occupied_gpus.difference_update(self.cuda_devices)


def get_logger(path, fname):
    if not os.path.exists(path):
        os.mkdir(path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_log_handler = logging.FileHandler(os.path.join(path, fname))
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_log_handler)
    logger.addHandler(stderr_log_handler)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()

    return logger
