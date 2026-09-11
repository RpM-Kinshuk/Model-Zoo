"""
NetESD: Efficient Empirical Spectral Density estimator for neural networks.

Main entry point for ESD analysis with multi-GPU support.
"""
# import os, psutil
# NUM_WORKERS = 1  # Adjust this based on the workload and system capabilities
# physical_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
# per_worker = max(1, physical_cores // max(1, NUM_WORKERS))  # NUM_WORKERS = threads or processes to use

# os.environ["OMP_NUM_THREADS"] = str(per_worker)
# os.environ["MKL_NUM_THREADS"] = str(per_worker)
# os.environ["OPENBLAS_NUM_THREADS"] = str(per_worker)
# os.environ["NUMEXPR_NUM_THREADS"] = str(per_worker)
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(per_worker)  # macOS

import torch
# torch.set_num_threads(per_worker)
# torch.set_num_interop_threads(1)
import torch.nn as nn
import queue
import multiprocessing as mp
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any

from .constants import RESULT_KEYS
from .core import compute_esd_for_weight, mp_worker
from .utils import iter_eligible_layers, estimate_compute_cost

# Concurrency model:
# - Thread backend: one worker thread per GPU (no per-device locks needed).
# - Process backend: one subprocess per GPU (isolated CUDA contexts).

### -- Main NetESD Estimator Function -- ###
def net_esd_estimator(
        net: nn.Module,
        EVALS_THRESH: float = 1e-5,
        bins: int = 100,
        fix_fingers: Optional[str] = None,
        xmin_pos: int = 2,
        conv_norm: float = 0.5,
        filter_zeros: bool = False,
        use_svd: bool = True,
        filter_type: Optional[bool] = True,
        save_eigs: Optional[bool] = True,
        parallel: Optional[bool] = True,
        backend: Optional[str] = "thread",
        max_workers: Optional[int] = None,
        device_ids: Optional[List[int]] = None,
        compute_dtype: str = "float32",
        coverage: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Any]]:
    """
    Highly optimized estimator for Empirical Spectral Density (ESD) and Alpha parameter,
    designed for maximum GPU efficiency and corrected for multi-GPU models.

    Args:
        net (nn.Module): The neural network model to evaluate (can be parallelized).
        EVALS_THRESH (float, optional): Threshold to filter near-zero eigenvalues. Defaults to 1e-5.
        bins (int, optional): Number of bins for histogram when using 'xmin_peak'. Defaults to 100.
        fix_fingers (str, optional): Method to select xmin. Can be 'xmin_peak' or 'xmin_mid'.
        xmin_pos (int, optional): Position in eigenvalue spectrum to choose xmin for 'xmin_mid'. Defaults to 2.
        conv_norm (float, optional): Eigenvalue scale for pooled convolutional kernel slices. Defaults to 0.5.
        filter_zeros (bool, optional): Whether to filter eigenvalues below EVALS_THRESH. Defaults to False.
        use_svd (bool, optional): If True, use SVD for eigenvalue computation instead of Gram matrix method.
        filter_type (bool, optional): Restrict to supported Linear/Embedding/Conv layouts and apply the legacy Linear aspect-ratio filter.
        save_eigs (bool, optional): If True, save computed eigenvalues in results.
        parallel (bool, optional): If True, dispatch layer computations across multiple GPUs.
        backend (str, optional): 'thread' for one thread per GPU; 'process' for one subprocess per GPU.
        max_workers (int, optional): Max concurrent workers. Defaults to len(device_ids) or auto-selected pool size.
        device_ids (List[int], optional): GPU IDs to use for compute. If None, auto-select GPUs not used by the model; if none available, use all.
        compute_dtype (str, optional): Spectrum computation precision, 'float32' or 'float64'. Loading precision is separate.
        coverage (list, optional): Append per-module eligibility, skip reasons, emitted identities and fit statuses here.

    Returns:
        dict: Aligned metric lists. Full module_name and slice identify each
            measurement; legacy longname is retained for compatibility.
    """
    if compute_dtype not in ("float32", "float64"):
        raise ValueError("compute_dtype must be 'float32' or 'float64'")
    results = {key: [] for key in RESULT_KEYS}
    print("=================================")
    print(f"Running optimized multi-GPU ESD estimator with:")
    print(f"  fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, filter_zeros: {filter_zeros}")
    print(
        f"  use_svd: {use_svd}, filter_type: {filter_type}, save_eigs: {save_eigs}")
    print(
        f"  parallel: {parallel}, device_ids: {device_ids if device_ids else 'auto'}, max_workers: {max_workers if max_workers else 'auto'}, backend: {backend}")
    print("=================================")

    with torch.no_grad():
        coverage_records: List[Dict[str, Any]] = []
        eligible = list(iter_eligible_layers(net, filter_type, coverage_records))
        if coverage is not None:
            coverage.extend(coverage_records)
        if not parallel or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            ordered = [
                compute_esd_for_weight(
                    name, w, EVALS_THRESH, bins, fix_fingers, xmin_pos,
                    conv_norm, filter_zeros, use_svd, save_eigs, None, params,
                    compute_dtype,
                )
                for name, w, params in eligible
            ]
        else:
            used_cuda_ids = sorted(p.device.index for p in net.parameters() if getattr(p, 'is_cuda', False))
            all_ids = list(range(torch.cuda.device_count()))
            if device_ids is not None and len(device_ids) > 0:
                pool = [i for i in device_ids if i in all_ids]
            else:
                # Prefer GPUs not hosting the model; if none, fall back to all GPUs
                pool = [i for i in all_ids if i not in used_cuda_ids] or all_ids
            if not pool:
                raise ValueError("No valid CUDA devices selected for ESD computation")
            workers = min((max_workers or len(pool) or 1), len(pool))

            # Size- and cost-aware task ordering (largest compute first)
            tasks = []
            for idx, (name, w, params) in enumerate(eligible):
                cost = estimate_compute_cost(w, use_svd)
                tasks.append((idx, name, w, params, cost))
            tasks_sorted = sorted(tasks, key=lambda t: t[4], reverse=True)

            if backend == "thread":
                # One worker thread per GPU; each bound to a fixed device id; shared task queue.
                tasks_q: "queue.Queue[tuple]" = queue.Queue()
                for t in tasks_sorted:
                    tasks_q.put(t)
                ordered: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

                def _thread_worker_loop(dev_id: int) -> None:
                    while True:
                        try:
                            orig_idx, name, w, params, _ = tasks_q.get_nowait()
                        except queue.Empty:
                            break
                        res = compute_esd_for_weight(
                            name, w, EVALS_THRESH, bins, fix_fingers, xmin_pos,
                            conv_norm, filter_zeros, use_svd, save_eigs, dev_id, params,
                            compute_dtype,
                        )
                        ordered[orig_idx] = res

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_thread_worker_loop, dev_id) for dev_id in pool[:workers]]
                    for f in futs:
                        f.result()
            elif backend == "process":
                # One subprocess per GPU; shared task/result queues; spawn context for CUDA safety.
                ctx = get_context('spawn')
                task_q: mp.Queue = ctx.Queue(maxsize=max(16, 2 * workers))
                result_q: mp.Queue = ctx.Queue()
                procs: List[BaseProcess] = []

                def _put_task(item) -> None:
                    # A dead subprocess must not leave the producer blocked on
                    # a full queue before it reaches the result-reading loop.
                    while True:
                        try:
                            task_q.put(item, timeout=1)
                            return
                        except queue.Full:
                            if any(p.exitcode not in (None, 0) for p in procs) or not any(p.is_alive() for p in procs):
                                raise RuntimeError("An ESD subprocess exited while tasks were being queued")

                ordered: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
                try:
                    for dev_id in pool[:workers]:
                        p = ctx.Process(target=mp_worker, args=(
                            task_q, result_q, dev_id,
                            EVALS_THRESH, bins, fix_fingers, xmin_pos,
                            conv_norm, filter_zeros, use_svd, save_eigs, compute_dtype
                        ))
                        p.daemon = False
                        p.start()
                        procs.append(p)

                    for orig_idx, name, w, params, _ in tasks_sorted:
                        # Cast at the same precision used by the core. This
                        # also permits bfloat16 weights, unsupported by numpy.
                        cpu_weight = w.detach().to(device="cpu", dtype=getattr(torch, compute_dtype))
                        _put_task((orig_idx, name, cpu_weight.numpy(), params))

                    for _ in procs:
                        _put_task(None)

                    for _ in range(len(tasks)):
                        while True:
                            try:
                                idx, res, error = result_q.get(timeout=1)
                                break
                            except queue.Empty:
                                if any(p.exitcode not in (None, 0) for p in procs) or not any(p.is_alive() for p in procs):
                                    raise RuntimeError("An ESD subprocess exited before returning all layer results")
                        if error is not None:
                            raise RuntimeError(f"ESD computation failed for {eligible[idx][0]!r}: {error}")
                        ordered[idx] = res
                    for p in procs:
                        p.join(timeout=5)
                finally:
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                        p.join(timeout=5)
                    # Do not wait for unsent tasks after a worker failure.
                    task_q.cancel_join_thread()
                    task_q.close()
                    result_q.close()
            else:
                raise ValueError(f"Unknown backend: {backend}. Expected 'thread' or 'process'.")

        identity = {
            measurement_name: (record["module_name"], slice_name)
            for record in coverage_records
            for measurement_name, slice_name in zip(record["measurement_names"], record["measurement_slices"])
        }
        computed = {}
        for (name, weight, _), result in zip(eligible, ordered):
            if result is None:
                continue
            if result.get("longname") != name:
                raise ValueError(f"ESD result identity does not match requested layer {name!r}")
            result["module_name"], result["slice"] = identity[name]
            # Multiprocess transport can cast checkpoint-native bfloat16 to
            # numpy-compatible float32; record the actual input, not transport.
            result["source_dtype"] = str(weight.dtype).removeprefix("torch.")
            computed[name] = result
            for key in results:
                results[key].append(result.get(key, float("nan")))
        for record in coverage_records:
            if record["status"] != "eligible":
                continue
            names = record["measurement_names"]
            analyzed = [name for name in names if name in computed]
            record["fit_statuses"] = {
                name: computed[name].get("fit_status", "unknown") for name in analyzed
            }
            if len(analyzed) == len(names):
                record["status"] = "analyzed"
            elif analyzed:
                record["status"], record["reason"] = "partially_analyzed", "no_spectrum_for_some_slices"
            else:
                record["status"], record["reason"] = "skipped", "no_spectrum"
    return results


# Export main API
__all__ = ['net_esd_estimator', 'RESULT_KEYS']
