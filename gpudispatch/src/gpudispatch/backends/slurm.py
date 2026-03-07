"""SLURM cluster backend for GPU orchestration."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from typing import List, Optional, Sequence, Set

from gpudispatch.backends.base import Backend
from gpudispatch.core.resources import GPU, Memory

logger = logging.getLogger(__name__)

_GPU_COUNT_PATTERNS = (
    re.compile(r"AllocTRES=.*?gres/gpu=(\d+)"),
    re.compile(r"TRESPerNode=.*?gres/gpu=(\d+)"),
    re.compile(r"Gres=.*?gpu(?::[A-Za-z0-9_-]+)?:(\d+)"),
)


class SLURMBackend(Backend):
    """SLURM cluster backend.

    This backend is intended for execution inside SLURM allocations. It can:
    - Discover GPUs from SLURM/CUDA environment variables
    - Track in-process GPU allocations/releases safely
    - Query scheduler health and job status through SLURM commands

    Args:
        partition: SLURM partition to submit jobs to. Default: "gpu".
        account: SLURM account for job accounting. Default: None.
        time_limit: Maximum job runtime (HH:MM:SS format). Default: "24:00:00".
        nodes: Number of nodes to request. Default: 1.
        gpus_per_node: Number of GPUs per node. Default: 1.
        polling_interval: Polling interval for status checks. Default: 5.
    """

    def __init__(
        self,
        partition: str = "gpu",
        account: Optional[str] = None,
        time_limit: str = "24:00:00",
        nodes: int = 1,
        gpus_per_node: int = 1,
        polling_interval: int = 5,
        **kwargs,
    ):
        self._partition = partition
        self._account = account
        self._time_limit = time_limit
        self._nodes = nodes
        self._gpus_per_node = gpus_per_node
        self._polling_interval = polling_interval
        self._extra_config = kwargs
        self._running = False
        self._gpu_pool: List[GPU] = []
        self._occupied_gpus: Set[int] = set()
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "slurm"

    @property
    def is_running(self) -> bool:
        """Whether the backend is currently active."""
        return self._running

    @property
    def partition(self) -> str:
        """SLURM partition name."""
        return self._partition

    @property
    def account(self) -> Optional[str]:
        """SLURM account name."""
        return self._account

    @property
    def time_limit(self) -> str:
        """Job time limit in HH:MM:SS format."""
        return self._time_limit

    @property
    def nodes(self) -> int:
        """Number of nodes requested."""
        return self._nodes

    @property
    def gpus_per_node(self) -> int:
        """Number of GPUs per node."""
        return self._gpus_per_node

    def start(self) -> None:
        """Initialize and start the backend.

        Discovers available GPU indices from SLURM env/context.
        """
        with self._lock:
            if self._running:
                return

            self._gpu_pool = self._discover_gpu_pool()
            self._occupied_gpus.clear()
            self._running = True

    def shutdown(self) -> None:
        """Gracefully shutdown the backend.

        Releases in-process GPU allocations and marks backend as stopped.
        """
        with self._lock:
            self._occupied_gpus.clear()
            self._running = False

    def allocate_gpus(
        self, count: int, memory: Optional[Memory] = None
    ) -> List[GPU]:
        """Allocate GPUs for a job.

        Returns an empty list when resources are insufficient.
        """
        if count <= 0:
            return []

        with self._lock:
            available = [gpu for gpu in self._gpu_pool if gpu.index not in self._occupied_gpus]

            if memory is not None:
                available = [
                    gpu
                    for gpu in available
                    if gpu.memory is None or gpu.memory >= memory.mb
                ]

            if len(available) < count:
                return []

            selected = available[:count]
            self._occupied_gpus.update(gpu.index for gpu in selected)
            return [
                GPU(index=gpu.index, memory=memory.mb if memory is not None else gpu.memory)
                for gpu in selected
            ]

    def release_gpus(self, gpus: List[GPU]) -> None:
        """Release allocated GPUs.
        """
        with self._lock:
            for gpu in gpus:
                self._occupied_gpus.discard(gpu.index)

    def list_available(self) -> List[GPU]:
        """List currently available GPUs.
        """
        with self._lock:
            return [
                GPU(index=gpu.index, memory=gpu.memory)
                for gpu in self._gpu_pool
                if gpu.index not in self._occupied_gpus
            ]

    def health_check(self) -> bool:
        """Verify backend is operational.

        Returns:
            bool: True if backend is running and SLURM responds (when available).
        """
        if not self._running:
            return False

        result = self._run_command(["scontrol", "ping"])
        if result is None:
            return True
        return result.returncode == 0

    def _discover_gpu_pool(self) -> List[GPU]:
        """Discover GPU pool from SLURM context and local runtime."""
        indices = self._discover_gpu_indices_from_env()
        if not indices:
            indices = self._discover_gpu_indices_from_job()
        if not indices:
            indices = self._discover_gpu_indices_from_nvidia_smi()

        if not indices:
            fallback_count = self._nodes * self._gpus_per_node
            if fallback_count > 0:
                logger.warning(
                    "Could not discover SLURM GPU indices; "
                    "falling back to configured count nodes*gpus_per_node=%s",
                    fallback_count,
                )
                indices = list(range(fallback_count))

        memory_map = self._query_gpu_memory_map()
        return [GPU(index=idx, memory=memory_map.get(idx)) for idx in indices]

    def _discover_gpu_indices_from_env(self) -> List[int]:
        """Discover GPU indices from common SLURM/CUDA environment variables."""
        for env_var in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
            raw = os.environ.get(env_var)
            if not raw:
                continue
            parsed = self._parse_gpu_indices(raw)
            if parsed:
                return parsed
        return []

    def _discover_gpu_indices_from_job(self) -> List[int]:
        """Discover GPU count from `scontrol show job` output."""
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return []

        result = self._run_command(["scontrol", "show", "job", job_id])
        if result is None or result.returncode != 0:
            return []

        output = result.stdout

        # Example: Gres=gpu:4(S:0-3)
        step_match = re.search(r"S:([0-9,\-]+)", output)
        if step_match:
            parsed = self._parse_index_expression(step_match.group(1))
            if parsed:
                return self._normalize_indices(parsed)

        for pattern in _GPU_COUNT_PATTERNS:
            match = pattern.search(output)
            if match:
                count = int(match.group(1))
                if count > 0:
                    return list(range(count))

        return []

    def _discover_gpu_indices_from_nvidia_smi(self) -> List[int]:
        """Discover GPU indices from local nvidia-smi output."""
        result = self._run_command(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
        )
        if result is None or result.returncode != 0:
            return []

        indices: List[int] = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                indices.append(int(stripped))
        return self._normalize_indices(indices)

    def _query_gpu_memory_map(self) -> dict[int, int]:
        """Return mapping of GPU index -> total memory in MB."""
        result = self._run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if result is None or result.returncode != 0:
            return {}

        memory_map: dict[int, int] = {}
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            if parts[0].isdigit() and parts[1].isdigit():
                memory_map[int(parts[0])] = int(parts[1])
        return memory_map

    def _parse_gpu_indices(self, raw_value: str) -> List[int]:
        """Parse SLURM/CUDA GPU list strings into concrete indices."""
        value = raw_value.strip()
        if not value or value.lower() in {"n/a", "none", "(null)"}:
            return []

        bracket_ranges = re.findall(r"\[([0-9,\-]+)\]", value)
        if bracket_ranges:
            expanded: List[int] = []
            for expr in bracket_ranges:
                expanded.extend(self._parse_index_expression(expr))
            return self._normalize_indices(expanded)

        parsed: List[int] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                token = token.rsplit(":", 1)[-1]
            parsed.extend(self._parse_index_expression(token))

        if parsed:
            return self._normalize_indices(parsed)

        fallback_numbers = [int(match) for match in re.findall(r"\d+", value)]
        return self._normalize_indices(fallback_numbers)

    def _parse_index_expression(self, expression: str) -> List[int]:
        """Parse comma/range expressions like '0,2-4' into indices."""
        indices: List[int] = []
        for part in expression.split(","):
            part = part.strip()
            if not part:
                continue

            range_match = re.fullmatch(r"(\d+)-(\d+)", part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if end >= start:
                    indices.extend(range(start, end + 1))
                continue

            if part.isdigit():
                indices.append(int(part))

        return indices

    def _normalize_indices(self, indices: Sequence[int]) -> List[int]:
        """Deduplicate and sort GPU indices."""
        return sorted({idx for idx in indices if idx >= 0})

    def _run_command(
        self,
        command: Sequence[str],
        timeout: int = 15,
    ) -> Optional[subprocess.CompletedProcess[str]]:
        """Run command if available, returning None when binary is missing."""
        executable = command[0]
        if shutil.which(executable) is None:
            return None

        try:
            return subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            logger.debug("SLURM command failed (%s): %s", " ".join(command), exc)
            return None

    def _run_required_command(
        self,
        command: Sequence[str],
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        """Run command and raise a RuntimeError on failure."""
        result = self._run_command(command, timeout=timeout)
        if result is None:
            raise RuntimeError(
                f"Required command '{command[0]}' is not available in PATH"
            )
        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown error"
            raise RuntimeError(
                f"Command '{' '.join(command)}' failed (exit={result.returncode}): {stderr}"
            )
        return result

    def _submit_job(self, script: str) -> str:
        """Submit a job to SLURM scheduler.

        Args:
            script: SLURM job script content.

        Returns:
            str: Job ID returned by SLURM.

        Raises:
            RuntimeError: If sbatch is unavailable or submission fails.
        """
        script_body = script if script.startswith("#!") else f"#!/bin/bash\n{script}\n"

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".sh",
                prefix="gpudispatch-slurm-",
                delete=False,
            ) as tmp:
                tmp.write(script_body)
                temp_path = tmp.name

            result = self._run_required_command(["sbatch", "--parsable", temp_path])
            raw_job_id = result.stdout.strip().split(";", 1)[0].strip()
            if not raw_job_id:
                raise RuntimeError("sbatch returned an empty job id")
            return raw_job_id
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _check_job_status(self, job_id: str) -> str:
        """Check status of a SLURM job.

        Args:
            job_id: SLURM job ID.

        Returns:
            Status string, or "UNKNOWN" if status could not be resolved.
        """
        squeue = self._run_command(["squeue", "-h", "-j", job_id, "-o", "%T"])
        if squeue is not None and squeue.returncode == 0:
            for line in squeue.stdout.splitlines():
                status = line.strip()
                if status:
                    return status.upper()

        sacct = self._run_command(["sacct", "-n", "-j", job_id, "--format=State"])
        if sacct is not None and sacct.returncode == 0:
            for line in sacct.stdout.splitlines():
                status = line.strip()
                if status:
                    return status.split()[0].split("+", 1)[0].upper()

        return "UNKNOWN"

    def _cancel_job(self, job_id: str) -> None:
        """Cancel a SLURM job.

        Args:
            job_id: SLURM job ID to cancel.

        Raises:
            RuntimeError: If scancel is unavailable or cancellation fails.
        """
        self._run_required_command(["scancel", job_id])
