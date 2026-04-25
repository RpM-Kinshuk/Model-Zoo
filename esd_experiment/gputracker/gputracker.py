#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs. 
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.

import threading
import time
import os
import sys
import gpustat
import logging
import subprocess
import signal
import json
# from torch.cuda import max_memory_allocated, reset_peak_memory_stats, reset_max_memory_allocated, memory_allocated
# from torch.cuda import init as torch_cuda_init

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_NCHECK=10              # number of checks to know if gpu free
GPU_MEMORY_THRESHOLD = 500 # MB?

class GPUDispatcher:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GPUDispatcher, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path="gpu_config.json"):
        if hasattr(self, 'initialized') and self.initialized: return
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
        }
        self.load_config()
    
    def load_config(self):
        """Reloads configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    new_config = json.load(f)

                merged_config = dict(self.config)
                merged_config.update(new_config)
                merged_config["available_gpus"] = [int(x) for x in merged_config["available_gpus"]]
                if merged_config.get("max_concurrent_jobs") is not None:
                    merged_config["max_concurrent_jobs"] = int(merged_config["max_concurrent_jobs"])
                    if merged_config["max_concurrent_jobs"] < 1:
                        raise ValueError("max_concurrent_jobs must be >= 1")

                with self.lock:
                    self.config = merged_config
                
                logging.info(f"✅ Configuration Reloaded: GPUs {self.config['available_gpus']}")
            else:
                logging.warning(f"⚠️ Config file {self.config_path} not found. Using defaults.")
        except Exception as e:
            logging.error(f"❌ Failed to load config: {e}")
    
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
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self.handle_reload)

    # --- Worker Management ---
    def kill_all_workers(self):
        with self.lock:
            for p in self.active_workers:
                if p.poll() is None:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    except Exception as e:
                        logging.error(f"Error killing process {p.pid}: {e}")
                        pass

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
                    allowed_gpus = self.config['available_gpus']
                    threshold = self.config['memory_threshold_mb']
                    max_checks = self.config['max_checks']
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
    def __init__(self, name, bash_command_list, logger, dispatcher, num_gpus_needed=1, config_path="gpu_config.json", max_concurrent_jobs=None):
        threading.Thread.__init__(self)
        self.name = name
        self.bash_command_list = bash_command_list
        self.logger = logger
        self.dispatcher = dispatcher
        self.num_gpus_needed = num_gpus_needed
        self.max_concurrent_jobs = max_concurrent_jobs

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

    def run(self):
        self.logger.info(f"Starting PID: {os.getpid()}: {self.name}")
        self.logger.info("Controls: SIGHUP=Reload Config, SIGUSR1=Drain/Graceful Exit, SIGINT=Kill")
        if self.max_concurrent_jobs is not None:
            self.logger.info(f"Max concurrent jobs: {self.max_concurrent_jobs}")
        
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):
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
            
            thread1 = ChildThread(f"{i}th + {bash_command}", 1, cuda_devices, bash_command, self.logger, self.dispatcher)
            thread1.start()
            threads.append(thread1)

            time.sleep(2)  # Slight delay to avoid race conditions

        # join all.
        for t in threads:
            t.join()
        
        self.logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, name, counter, cuda_devices, bash_command, logger, dispatcher):
        threading.Thread.__init__(self)
        self.name = name
        self.counter = counter
        self.cuda_devices = cuda_devices
        self.bash_command = bash_command
        self.logger = logger
        self.dispatcher = dispatcher
        self.daemon = True # Ensure thread exits if main program exits

    def run(self):
        # torch_cuda_init()
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.cuda_devices))

        self.logger.info(f'Executing on GPUs {self.cuda_devices}: {self.bash_command}')
        # start_memory = 0
        # for gpu in self.cuda_devices:
        #     reset_peak_memory_allocated(device=gpu)
        #     start_memory += memory_allocated(device=gpu)
        try:
            proc = subprocess.Popen(
                self.bash_command,
                shell=True,
                env=env,
                preexec_fn=os.setsid
            )
            self.dispatcher.register_worker(proc)
            proc.wait() # Wait for process to complete
            self.dispatcher.unregister_worker(proc)
        
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}")
        finally:
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
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()

    return logger
