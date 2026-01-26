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
# from torch.cuda import max_memory_allocated, reset_peak_memory_stats, reset_max_memory_allocated, memory_allocated
# from torch.cuda import init as torch_cuda_init

# Global registry to track active subprocesses for cleanup
active_workers = []
active_workers_lock = threading.Lock()
exitFlag = False  # Global flag for graceful shutdown

GPU_MEMORY_THRESHOLD = 500 # MB?
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_NCHECK=10              # number of checks to know if gpu free
occupied_gpus = []
occupied_lock = threading.Lock() # Thread safety for gpu list

## If we need to wait for the entire clean cluster to start, select False here

all_empty = {"ind": True}
#all_empty = {"ind": False}


# NEW
def mark_occupied(gpu_ids):
    global occupied_gpus
    with occupied_lock:
        occupied_gpus.extend(gpu_ids)


def remove_occupied(gpu_ids):
    global occupied_gpus
    with occupied_lock:
        for gpu in gpu_ids:
            if gpu in occupied_gpus:
                occupied_gpus.remove(gpu)


def cleanup_workers():
    """Kill all active subprocesses."""
    global exitFlag
    exitFlag = True
    print("\n[GPU_TRACKER] Cleaning up active workers...")
    with active_workers_lock:
        for p in active_workers:
            if p.poll() is None:  # If process is still running
                try:
                    # Send SIGTERM to the process group to ensure shell + python child die
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception as e:
                    print(f"Error killing process {p.pid}: {e}")
    print("[GPU_TRACKER] Cleanup complete.")


def num_available_GPUs(gpus):
    sum_i = 0
    for i, stat in enumerate(gpus):
        if stat['memory.used'] < 100:
            sum_i += 1
    return sum_i


def get_free_gpu_indices(logger, num_gpus_needed):
    '''
        Return a list of available GPU indices.
    '''
    counter = {}
    while not exitFlag:
        try:
            stats = gpustat.GPUStatCollection.new_query()
            # print('stats length: ', len(stats))
        except Exception as e:
            logger.warning(f"Could not query GPU stats: {e}")
            time.sleep(5)
            continue
        
        if num_available_GPUs(stats.gpus) >= num_gpus_needed:
            all_empty["ind"] = True
            
        if not all_empty["ind"]:
            logger.info("Other experiments not finished...")
            time.sleep(10)
            continue
        
        max_checks = 0
        max_gpu_id = -1
        available_gpus = []

        with occupied_lock:
            current_occupied = set(occupied_gpus)
        
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            
            if memory_used < GPU_MEMORY_THRESHOLD and i in AVAILABLE_GPUS and i not in current_occupied:
                if i not in counter:
                    counter.update({i: 0})
                else:
                    counter[i] = counter[i] + 1
                
                if counter[i] >= MAX_NCHECK:
                    available_gpus.append(i)
                    if len(available_gpus) == num_gpus_needed:
                        mark_occupied(available_gpus)
                        return available_gpus
            else:
                counter.update({i: 0})

            if counter[i] > max_checks:
                max_checks = counter[i]
                max_gpu_id = i
        
        if max_gpu_id != -1:
            logger.info(f"Waiting on GPUs, Checking {max_checks}/{MAX_NCHECK} at gpu {max_gpu_id}")
        
        for _ in range(10):
            if exitFlag: return []
            time.sleep(1)
        return []


class DispatchThread(threading.Thread):
    def __init__(self, name, bash_command_list, logger, gpu_m_th, gpu_list, maxcheck, num_gpus_needed=1):
        threading.Thread.__init__(self)
        self.name = name
        self.bash_command_list = bash_command_list
        self.logger = logger
        self.num_gpus_needed = num_gpus_needed
        
        global GPU_MEMORY_THRESHOLD, AVAILABLE_GPUS, MAX_NCHECK
        GPU_MEMORY_THRESHOLD = gpu_m_th
        AVAILABLE_GPUS = gpu_list
        MAX_NCHECK = maxcheck

    def run(self):
        self.logger.info("Starting " + self.name)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):
            if exitFlag: break
            time.sleep(0.3)

            #if os.path.isfile(result_name):
            #    print("Result already exists! {0}".format(result_name))
            #    continue
            #    
            #else:
            #    print("Result not ready yet. Running it for a second time: {0}".format(result_name))
            
            cuda_devices = get_free_gpu_indices(self.logger, self.num_gpus_needed)
            if not cuda_devices and exitFlag: break
            
            thread1 = ChildThread(f"{i}th + {bash_command}", 1, cuda_devices, bash_command, self.logger)
            thread1.start()
            threads.append(thread1)

            time.sleep(5)  # Slight delay to avoid race conditions

        # join all.
        for t in threads:
            t.join()
        self.logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, name, counter, cuda_devices, bash_command, logger):
        threading.Thread.__init__(self)
        self.name = name
        self.counter = counter
        self.cuda_devices = cuda_devices
        self.bash_command = bash_command
        self.logger = logger
        self.daemon = True # Ensure thread exits if main program exits

    def run(self):
        # torch_cuda_init()
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.cuda_devices))
        bash_command = self.bash_command

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

            with active_workers_lock:
                active_workers.append(proc)

            proc.wait() # Wait for process to complete

            with active_workers_lock:
                if proc in active_workers:
                    active_workers.remove(proc)
        
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}")
        finally:
            remove_occupied(self.cuda_devices)
            self.logger.info(f"Finished {self.name} \n\n")


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

