"""One bounded download-ahead lane using future workers' private Hub caches."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

METADATA_ALLOWANCE = 32 * 1024**2


def _download(job, cache, stop):
    cache.mkdir(parents=True, exist_ok=True)
    request = cache / "prefetch-request.json"
    request.write_text(json.dumps(job.prefetch_files))
    env = os.environ.copy()
    env.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"),
               TRANSFORMERS_CACHE=str(cache / "hub"), HF_XET_CACHE=str(cache / "xet"),
               HF_MODULES_CACHE=str(cache / "modules"), HF_DATASETS_CACHE=str(cache / "datasets"),
               HF_HUB_DISABLE_XET="1")
    # A subprocess is needed: Hub cache constants are resolved at import time.
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(request)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    timeout = (job.stage_timeout_seconds or {}).get("load", 7200)
    deadline = time.monotonic() + timeout if timeout > 0 else float("inf")
    budget = sum(item[3] for item in job.prefetch_files) + METADATA_ALLOWANCE
    try:
        while proc.poll() is None:
            used = sum(path.stat().st_size for path in cache.rglob("*") if not path.is_symlink() and path.is_file())
            if stop.is_set() or used > budget or time.monotonic() > deadline:
                raise OSError("Prefetch interrupted or exceeded its download budget")
            stop.wait(.25)
        if proc.returncode:
            raise OSError("Prefetch unavailable; worker will download")
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
            except ProcessLookupError:
                proc.wait(timeout=5)
        request.unlink(missing_ok=True)


class Prefetcher:
    """Reserve bytes until worker launch, then let normal worker cleanup own it."""
    def __init__(self, jobs, tracker, *, max_models, max_bytes, downloader=None):
        self.jobs, self.tracker = list(jobs), tracker
        self.max_models, self.max_bytes = max_models, max_bytes
        self.downloader = downloader or _download
        self.stop = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-prefetch")
        self.pending, self.transferred = {}, {}
        self.index, self.reserved = 0, 0
        self._fill()

    def _fill(self):
        if not self.tracker or not self.tracker.cache_root or self.stop.is_set():
            return
        while self.index < len(self.jobs) and len(self.pending) < self.max_models:
            job = self.jobs[self.index]
            size = sum(item[3] for item in job.prefetch_files) + METADATA_ALLOWANCE
            if not job.prefetch_files or size > self.max_bytes:
                self.index += 1
                continue
            if self.reserved + size > self.max_bytes:
                break
            path = self.tracker._cache_path_for(job.worker_id)
            future = self.pool.submit(self.downloader, job, path, self.stop)
            self.pending[job.worker_id] = (future, path, size)
            self.reserved += size
            self.index += 1

    def claim(self, job, progress, stopped=lambda: False):
        self._fill()
        entry = self.pending.get(job.worker_id)
        if entry is None:
            return job
        future, path, _ = entry
        while not self.stop.is_set():
            if stopped():
                self.stop.set()
                return job
            progress()
            try:
                future.result(timeout=.5)
                return replace(job, cache_path=str(path))
            except TimeoutError:
                continue
            except Exception:
                # Partial files remain private and can be reused by normal SDK retries.
                return job
        return job

    def launched(self, job):
        entry = self.pending.pop(job.worker_id, None)
        if entry:
            self.transferred[job.worker_id] = entry[1]
            self.reserved -= entry[2]
        self._fill()

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
        for future, path, size in self.pending.values():
            # The download helper exited before deleting its unlaunched cache.
            shutil.rmtree(path, ignore_errors=True)
        for worker_id, path in self.transferred.items():
            if worker_id not in self.tracker.adopted_cache_ids:
                shutil.rmtree(path, ignore_errors=True)
        self.pending.clear()


def _main(request):
    from huggingface_hub import hf_hub_download
    files = json.loads(Path(request).read_text())
    token = os.environ.get("HF_TOKEN") or False
    for repo, revision, filename, size in files:
        hf_hub_download(repo_id=repo, revision=revision, filename=filename,
                        token=token, cache_dir=os.environ["HF_HUB_CACHE"])


if __name__ == "__main__":
    _main(sys.argv[1])
