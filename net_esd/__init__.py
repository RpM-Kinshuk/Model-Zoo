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
        conv_norm (float, optional): Normalization for convolutional layers (currently unused). Defaults to 0.5.
        filter_zeros (bool, optional): Whether to filter eigenvalues below EVALS_THRESH. Defaults to False.
        use_svd (bool, optional): If True, use SVD for eigenvalue computation instead of Gram matrix method.
        filter_type (bool, optional): If True, only process Conv1d, Conv2d, and Linear layers.
        save_eigs (bool, optional): If True, save computed eigenvalues in results.
        parallel (bool, optional): If True, dispatch layer computations across multiple GPUs.
        backend (str, optional): 'thread' for one thread per GPU; 'process' for one subprocess per GPU.
        max_workers (int, optional): Max concurrent workers. Defaults to len(device_ids) or auto-selected pool size.
        device_ids (List[int], optional): GPU IDs to use for compute. If None, auto-select GPUs not used by the model; if none available, use all.

    Returns:
        dict: A dictionary containing the computed metrics for each layer.
    """
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
        eligible = list(iter_eligible_layers(net, filter_type))
        if not parallel or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            layer_results = [
                res for (name, w, params) in eligible
                if (res := compute_esd_for_weight(name, w, EVALS_THRESH, bins, fix_fingers, xmin_pos, conv_norm, filter_zeros, use_svd, save_eigs, None, params)) is not None
            ]
        else:
            used_cuda_ids = sorted(p.device.index for p in net.parameters() if getattr(p, 'is_cuda', False))
            all_ids = list(range(torch.cuda.device_count()))
            if device_ids is not None and len(device_ids) > 0:
                pool = [i for i in device_ids if i in all_ids]
            else:
                # Prefer GPUs not hosting the model; if none, fall back to all GPUs
                pool = [i for i in all_ids if i not in used_cuda_ids] or all_ids
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
                            conv_norm, filter_zeros, use_svd, save_eigs, dev_id, params
                        )
                        ordered[orig_idx] = res

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_thread_worker_loop, dev_id) for dev_id in pool[:workers]]
                    for f in futs:
                        f.result()
                layer_results = [r for r in ordered if r is not None]

            elif backend == "process":
                # One subprocess per GPU; shared task/result queues; spawn context for CUDA safety.
                ctx = get_context('spawn')
                task_q: mp.Queue = ctx.Queue(maxsize=max(16, 2 * workers))
                result_q: mp.Queue = ctx.Queue()
                procs: List[BaseProcess] = []

                for dev_id in pool[:workers]:
                    p = ctx.Process(target=mp_worker, args=(
                        task_q, result_q, dev_id,
                        EVALS_THRESH, bins, fix_fingers, xmin_pos,
                        conv_norm, filter_zeros, use_svd, save_eigs, params
                    ))
                    p.daemon = False
                    p.start()
                    procs.append(p)

                for orig_idx, name, w, params, _ in tasks_sorted:
                    # Send CPU numpy arrays to subprocesses (copy). For very large layers, consider shared memory.
                    task_q.put((orig_idx, name, w.detach().cpu().numpy(), params))

                # Send termination sentinels
                for _ in procs:
                    task_q.put(None)

                ordered: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
                for _ in range(len(tasks)):
                    idx, res = result_q.get()
                    ordered[idx] = res

                for p in procs:
                    p.join()

                layer_results = [r for r in ordered if r is not None]
            else:
                raise ValueError(f"Unknown backend: {backend}. Expected 'thread' or 'process'.")

        for r in layer_results:
            for key, value in r.items():
                if key in results:
                    results[key].append(value)
    return results


# Export main API
__all__ = ['net_esd_estimator', 'RESULT_KEYS']