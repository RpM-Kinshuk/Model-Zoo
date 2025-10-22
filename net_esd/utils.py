import torch
import torch.nn as nn
import numpy as np
from typing import Iterator, Tuple, Optional

from .constants import EPSILON


def matrix_rank_torch(svals: torch.Tensor, N: int, tol: Optional[float] = None) -> torch.Tensor:
    """Matrix rank, computed from singular values, using torch. (Source: WeightWatcher)"""
    if tol is None:
        tol = svals.max() * N * torch.finfo(svals.dtype).eps # type: ignore
    return torch.sum(svals > tol) # type: ignore


def matrix_entropy_torch(evals: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Matrix entropy, computed directly from eigenvalues using PyTorch. (Source: WeightWatcher)"""
    if rank <= 0: return torch.tensor(-1.0, device=evals.device)
    evals_sum = torch.sum(evals)
    if evals_sum <= 0: return torch.tensor(-1.0, device=evals.device)
    p = (evals / evals_sum) + EPSILON
    log_rank = torch.log(rank.float() + EPSILON)
    if torch.abs(log_rank) < 1e-9: return torch.tensor(-1.0, device=evals.device)
    return -torch.sum(p * torch.log(p)) / log_rank


def iter_eligible_layers(net: nn.Module, filter_type: Optional[bool] = True) -> Iterator[Tuple[str, torch.Tensor, int]]:
    """Yield (name, weight_tensor_or_slice, params_count) for eligible layers.

    Mirrors the behavior of get_module_names_shapes() and attn_split_qkv() from
    ESD-Independence/WW_LLMs-main/WW_LLMs-main/utils.py but implemented efficiently
    without deepcopy. Also applies the Linear high-aspect-ratio classifier skip
    used in their estimator (max/min >= 8).
    """
    for name, module in net.named_modules():
        if filter_type:
            if type(module).__name__.lower() not in ["conv2d", "conv1d", "linear"]:
                continue
        if not hasattr(module, "weight"):
            continue
        # weight = module.weight.data
        weight_param = getattr(module, "weight", None)
        if not isinstance(weight_param, torch.nn.Parameter):
            continue
        weight: torch.Tensor = weight_param.detach()
        if weight.ndim <= 1:
            continue

        if filter_type:
            if type(module).__name__.lower() == "linear":
                mx, mn = max(weight.shape), min(weight.shape)
                if mn > 0 and (mx / mn) >= 8:
                    continue

        name_l = name.lower()
        bias = getattr(module, "bias", None)
        bias_params = bias.numel() if (bias is not None and getattr(bias, "requires_grad", False)) else 0

        if ("attn" in name_l or "attention" in name_l) and weight.ndim == 2:
            m, n = weight.shape
            emitted = False
            if m == n // 3:
                dim = m
                slices = [
                    (f"{name}_q", weight[:, :dim]),
                    (f"{name}_k", weight[:, dim:2*dim]),
                    (f"{name}_v", weight[:, 2*dim:]),
                ]
                emitted = True
            elif n == m // 3:
                dim = n
                slices = [
                    (f"{name}_q", weight[:dim, :]),
                    (f"{name}_k", weight[dim:2*dim, :]),
                    (f"{name}_v", weight[2*dim:, :]),
                ]
                emitted = True
            if emitted:
                for sname, sw in slices:
                    yield sname, sw, sw.numel()
                continue

        yield name, weight, (weight.numel() + bias_params)


def estimate_compute_cost(weight: torch.Tensor, use_svd: bool) -> int:
    """Estimate dense eigenspectrum compute cost for scheduling.

    Using Gram-based path in `_squared_singular_values`:
      - For a 2D matrix A in R^{M x N}, we form G of size min(M,N) and run eigvalsh.
        Cost ~ O(min(M,N)^2 * max(M,N)) to build G + O(min(M,N)^3) for eigvalsh.
      - For Conv weights (O, I, kH, kW) we create B = kH*kW slices of (O x I) matrices.
        Total cost ~ B * [min(O,I)^2 * max(O,I) + min(O,I)^3].
    Using SVD path:
        - For a 2D matrix A in R^{M x N}, cost ~ O(M*N^2) if M <= N else O(N*M^2).
        - For Conv weights, total cost ~ B * [M*N^2 if M <= N else N*M^2].
    We return an integer proxy of the above expression.
    """
    if weight.ndim == 2:
        M, N = int(weight.shape[0]), int(weight.shape[1])
        m, n = (M, N) if M <= N else (N, M)
        if use_svd:
            return m*n*n
        return m*m*n + m*m*m
    elif weight.ndim > 2:
        O, I = int(weight.shape[0]), int(weight.shape[1])
        B = int(np.prod(weight.shape[2:]))
        m, n = (O, I) if O <= I else (I, O)
        if use_svd:
            return B * (m*n*n)
        return B * (m*m*n + m*m*m)
    return int(weight.numel())