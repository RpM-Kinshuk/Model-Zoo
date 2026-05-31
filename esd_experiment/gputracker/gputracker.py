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

from .supervision import (
    WorkerJob,
    WorkerRecord,
    WorkerStateTracker,
    heartbeat_is_stale,
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
        self.initialized = True

        self.config_path = config_path
        self.lock = threading.Lock()

        # State Flags
        self.shutdown_event = threading.Event() # Hard stop
        self.drain_event = threading.Event()   # Graceful stop
        self.active_workers = [] # Track PIDs
        self.occupied_gpus = set()  # Track GPU usage

        # Default Config
        self.config = {
            "available_gpus": AVAILABLE_GPUS,
            "max_checks": MAX_NCHECK,
            "memory_threshold_mb": GPU_MEMORY_THRESHOLD,
            "max_concurrent_jobs": None,
            "stale_process_action": "log",
            "heartbeat_timeout_seconds": 3600,
            "termination_grace_seconds": 60,
        }
        self.load_config()

    def _normalize_config(self, raw_config: dict) -> dict:
        merged_config = dict(self.config)
        merged_config.update(raw_config)
        merged_config["available_gpus"] = [int(x) for x in merged_config["available_gpus"]]
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
            else:
                logging.warning(f"Config file {self.config_path} not found. Using defaults.")
        except Exception as e:
            logging.error(f"Failed to load config: {e}")

    # --- Signal Handlers ---
    def handle_hard_stop(self, signum, frame):
        """SIGINT/SIGTERM: Immediate shutdown"""
        logging.critical(f"\n🛑 Received Signal {signum}. Hard stopping...")
        self.shutdown_event.set()
        self.kill_all_workers()
        sys.exit(1)

    def handle_drain(self, signum, frame):
        """SIGUSR1: Stop new jobs, wait for current ones"""
        logging.warning(f"\n⚠️ Received Signal {signum}. Entering DRAIN MODE.")
        logging.warning("No new jobs will start. Waiting for active jobs to finish...")
        self.drain_event.set()

    def handle_reload(self, signum, frame):
        """SIGHUP: Reload Configuration"""
        logging.info(f"\n🔄 Received Signal {signum}. Reloading configuration...")
        self.load_config()

    def setup_signals(self):
        signal.signal(signal.SIGINT, self.handle_hard_stop)
        signal.signal(signal.SIGTERM, self.handle_hard_stop)
        signal.signal(signal.SIGUSR1, self.handle_drain)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self.handle_reload)

    # --- Worker Management ---
    def kill_all_workers(self):
        with self.lock:
            workers = list(self.active_workers)
        for p in workers:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception as e:
                    logging.error(f"Error killing process {p.pid}: {e}")

    def register_worker(self, proc):
        with self.lock:
            self.active_workers.append(proc)

    def unregister_worker(self, proc):
        with self.lock:
            if proc in self.active_workers:
                self.active_workers.remove(proc)

    def get_free_gpus(self, num_needed):
        """Blocking call to get free GPUs"""
        counter = {}

        while not self.shutdown_event.is_set():
            if self.drain_event.is_set(): return None # Signal to stop dispatching new jobs
        
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
                    if i >= len(stats.gpus): continue
                    
                    if stats.gpus[i]['memory.used'] < threshold and i not in current_occupied:
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
        self.bash_command_list = bash_command_list
        self.logger = logger
        self.dispatcher = dispatcher
        self.num_gpus_needed = num_gpus_needed
        self.max_concurrent_jobs = max_concurrent_jobs
        self.state_tracker = (
            WorkerStateTracker(state_dir, run_id=run_id, logger=logger, cache_root=cache_root)
            if state_dir
            else None
        )

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
        self.logger.info(f"Starting PID: {os.getpid()}: {self.name}")
        self.logger.info("Controls: SIGHUP=Reload Config, SIGUSR1=Drain/Graceful Exit, SIGINT=Kill")
        if self.max_concurrent_jobs is not None:
            self.logger.info(f"Max concurrent jobs: {self.max_concurrent_jobs}")

        threads = []
        for i, item in enumerate(self.bash_command_list):
            # Check for Hard Stop
            if self.dispatcher.shutdown_event.is_set(): break

            # Check for Drain Mode
            if self.dispatcher.drain_event.is_set():
                self.logger.warning("Drain mode active. Skipping remaining jobs.")
                break

            if not self._wait_for_job_slot(threads):
                break

            time.sleep(0.3)

            #if os.path.isfile(result_name):
            #    print("Result already exists! {0}".format(result_name))
            #    continue
            #    
            #else:
            #    print("Result not ready yet. Running it for a second time: {0}".format(result_name))
            
            # Get Resources (blocking call)
            cuda_devices = self.dispatcher.get_free_gpus(self.num_gpus_needed)
            if cuda_devices is None: break

            job = self._coerce_job(item, i)
            thread1 = ChildThread(
                f"{i}th + {job.command}",
                1,
                cuda_devices,
                job,
                self.logger,
                self.dispatcher,
                self.state_tracker,
            )
            thread1.start()
            threads.append(thread1)

            time.sleep(2)  # Slight delay to avoid race conditions

        # join all.
        for t in threads:
            t.join()

        if self.state_tracker:
            self.state_tracker.close()

        self.logger.info("Exiting " + self.name)


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

    def _apply_worker_cache_env(self, env: dict, record: Optional[WorkerRecord]) -> None:
        if record is None or record.cache_path is None:
            return
        cache_path = record.cache_path
        env["HF_HOME"] = str(cache_path)
        env["HF_HUB_CACHE"] = str(cache_path / "hub")
        env["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
        env["HF_DATASETS_CACHE"] = str(cache_path / "datasets")

    def _current_stale_config(self) -> tuple[str, int, int]:
        with self.dispatcher.lock:
            config = dict(self.dispatcher.config)
        return (
            config.get("stale_process_action", "log"),
            int(config.get("heartbeat_timeout_seconds", 3600)),
            int(config.get("termination_grace_seconds", 60)),
        )

    def _terminate_stale_process(self, proc, grace_seconds: int) -> tuple[Optional[int], str]:
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            pgid = None
        if pgid is None:
            proc.terminate()
        else:
            os.killpg(pgid, signal.SIGTERM)
        try:
            return proc.wait(timeout=grace_seconds), "stale_heartbeat_timeout"
        except subprocess.TimeoutExpired:
            if pgid is None:
                proc.kill()
            else:
                os.killpg(pgid, signal.SIGKILL)
            return proc.wait(), "stale_heartbeat_timeout_killed"

    def _wait_for_process(self, proc, record: Optional[WorkerRecord]) -> tuple[Optional[int], Optional[str], Optional[str]]:
        while True:
            try:
                return proc.wait(timeout=5), None, None
            except subprocess.TimeoutExpired:
                if record is None:
                    continue
                action, timeout_seconds, grace_seconds = self._current_stale_config()
                if not heartbeat_is_stale(record.heartbeat_path, timeout_seconds, started_at_epoch=record.started_at_epoch):
                    continue
                message = f"No heartbeat update for {timeout_seconds} seconds"
                if action == "terminate":
                    self.logger.warning(f"Terminating stale worker {record.job.worker_id}: {message}")
                    if self.state_tracker:
                        self.state_tracker.mark_worker(record.job.worker_id, "terminating", "stale_heartbeat_timeout")
                    returncode, reason = self._terminate_stale_process(proc, grace_seconds)
                    return returncode, reason, message
                if not self._stale_logged:
                    self.logger.warning(f"Stale worker observed {record.job.worker_id}: {message}")
                    if self.state_tracker:
                        self.state_tracker.mark_worker(record.job.worker_id, "stale", "stale_heartbeat_timeout")
                    self._stale_logged = True

    def run(self):
        # torch_cuda_init()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cuda_devices))

        self.logger.info(f"Executing on GPUs {self.cuda_devices}: {self.bash_command}")
        proc = None
        record = None
        log_handle = None
        returncode = None
        finish_reason = None
        finish_message = None

        # start_memory = 0
        # for gpu in self.cuda_devices:
        #     reset_peak_memory_allocated(device=gpu)
        #     start_memory += memory_allocated(device=gpu)
        try:
            if self.state_tracker:
                record = self.state_tracker.start_worker(self.job, self.cuda_devices)
                env["WORKER_HEARTBEAT_FILE"] = str(record.heartbeat_path)
                self._apply_worker_cache_env(env, record)
                log_handle = open(record.log_path, "a", encoding="utf-8", buffering=1)

            proc = subprocess.Popen(
                self.bash_command,
                shell=True,
                env=env,
                preexec_fn=os.setsid,
                stdout=log_handle or None,
                stderr=subprocess.STDOUT if log_handle else None,
            )
            if self.state_tracker and record:
                try:
                    pgid = os.getpgid(proc.pid)
                except Exception:
                    pgid = None
                self.state_tracker.update_worker_pid(record.job.worker_id, proc.pid, pgid)
            self.dispatcher.register_worker(proc)
            returncode, finish_reason, finish_message = self._wait_for_process(proc, record)

        except Exception as e:
            finish_reason = finish_reason or "worker_supervisor_error"
            finish_message = str(e)
            returncode = 1 if returncode is None else returncode
            self.logger.error(f"Error in {self.name}: {e}")
        finally:
            if proc is not None:
                self.dispatcher.unregister_worker(proc)
            if log_handle is not None:
                log_handle.close()
            if self.state_tracker and record:
                self.state_tracker.finish_worker(
                    record.job.worker_id,
                    returncode=returncode,
                    reason=finish_reason,
                    message=finish_message,
                )
            # Always free the GPUs
            with self.dispatcher.lock:
                for gpu in self.cuda_devices:
                    if gpu in self.dispatcher.occupied_gpus:
                        self.dispatcher.occupied_gpus.remove(gpu)

            self.logger.info(f"Finished {self.name} \n\n")


def cleanup_workers():
    # Only useful if instance exists
    if GPUDispatcher._instance:
        GPUDispatcher._instance.handle_hard_stop(signal.SIGTERM, None)


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
