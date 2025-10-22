"""Core computation functions for ESD analysis."""

import torch
import numpy as np
import math
import multiprocessing as mp
from typing import Optional, Dict, Any

from .constants import logger
from .utils import matrix_rank_torch, matrix_entropy_torch


def compute_esd_for_weight(
        name: str, weight_t: torch.Tensor, EVALS_THRESH: float, bins: int,
        fix_fingers: Optional[str], xmin_pos: int, conv_norm: float,
        filter_zeros: bool, use_svd: bool, save_eigs: Optional[bool],
        device_id: Optional[int], params: int
) -> Optional[Dict[str, Any]]:
    """Compute ESD-related metrics for a single layer's weight on the specified device.
    Returns a dict with metrics for that layer, or None if not enough eigenvalues.
    """
    with torch.no_grad():
        use_cuda = device_id is not None and torch.cuda.is_available()
        device = torch.device(f"cuda:{device_id}") if use_cuda else (
            weight_t.device if weight_t.device.type != 'cpu' else torch.device('cpu')
        )
        if device.type.startswith('cuda') and weight_t.device.type == 'cpu':
            # Pinned host memory can accelerate H2D transfers when non_blocking=True
            matrix = weight_t.pin_memory().to(device=device, non_blocking=True).float()
        else:
            matrix = weight_t.to(device=device, non_blocking=True).float()
        matrix = matrix.contiguous()
        if len(matrix.shape) < 2:
            logger.warning(f"Skipping layer {name} due to invalid dimensions: {matrix.shape}")
            return None

        if len(matrix.shape) > 2:
            matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
            matrix = matrix.transpose(1, 2).transpose(0, 1)

        # Reshaping for Conv layers. This creates a stack of (kernel_H * kernel_W) matrices,
        # each of size (out_channels, in_channels).
        # if len(matrix.shape) > 2:
        #     # The permute+reshape is a more direct and typically faster way to achieve
        #     # the same tensor shape as the original flatten and two transposes.
        #     out_channels, in_channels = matrix.shape[0], matrix.shape[1]
        #     matrix = matrix.permute(2, 3, 0, 1).reshape(-1, out_channels, in_channels)
        #     # Apply normalization in-place
        #     matrix.mul_(math.sqrt(conv_norm))

        # Single in-flight op per GPU is enforced by per-GPU workers.
        eigs = squared_singular_values(matrix, use_svd)

        nz_eigs = eigs[eigs > EVALS_THRESH] if filter_zeros else eigs
        if nz_eigs.numel() == 0: nz_eigs = eigs

        N = nz_eigs.numel()
        if N <= 1: return None

        nz_eigs, _ = torch.sort(nz_eigs)
        log_nz_eigs = torch.log(nz_eigs)
        spectral_norm = nz_eigs[-1].item()
        fnorm = torch.sum(nz_eigs).item()

        final_alpha, final_D = -1.0, 1.0
        if fix_fingers == 'xmin_mid':
            i = N // xmin_pos
            if i >= N - 1: i = N - 2
            xmin = nz_eigs[i]
            n = float(N - i)
            seq = torch.arange(n, device=device)
            alpha_tensor = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            if alpha_tensor.item() > 1:
                final_alpha = alpha_tensor.item()
                final_D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n)).item()
        else:
            # Vectorized calculation of alpha and D
            i_s = torch.arange(N - 1, device=device)
            n_s = (N - i_s).float()

            rev_log_eigs = torch.flip(log_nz_eigs, dims=[0])
            rev_cumsum_log_eigs = torch.cumsum(rev_log_eigs, dim=0)
            cumsum_log_eigs = torch.flip(rev_cumsum_log_eigs, dims=[0])
            sum_log_eigs_from_i = cumsum_log_eigs[:-1]

            alphas = 1 + n_s / (sum_log_eigs_from_i - n_s * log_nz_eigs[:-1])

            xmins_col = nz_eigs[:-1].view(-1, 1)
            eigs_row = nz_eigs.view(1, -1)
            mask = (torch.arange(N, device=device).view(1, -1) >= torch.arange(N - 1, device=device).view(-1, 1))

            ratios = eigs_row / xmins_col
            exponents = (-alphas.view(-1, 1) + 1)
            power_term = torch.pow(ratios, exponents)

            seq = torch.arange(N, device=device).view(1, -1) - torch.arange(N - 1, device=device).view(-1, 1)
            seq_term = seq.float() / n_s.view(-1, 1)

            d_matrix = torch.abs(1 - power_term - seq_term)
            d_matrix_masked = torch.where(mask, d_matrix, torch.tensor(0.0, device=device))

            Ds, _ = torch.max(d_matrix_masked, dim=1)

            valid_indices = (alphas > 1)
            if fix_fingers == 'xmin_peak':
                log10_nz_eigs = torch.log10(nz_eigs)
                min_e, max_e = log10_nz_eigs.min(), log10_nz_eigs.max()
                counts = torch.histc(log10_nz_eigs, bins, min=min_e.item(), max=max_e.item())
                boundaries = torch.linspace(min_e, max_e, bins + 1, device=device)
                ih = torch.argmax(counts)
                xmin2 = 10 ** boundaries[ih]
                xmin_min = 0.95 * xmin2
                xmin_max = 1.5 * xmin2
                peak_indices = (nz_eigs[:-1] >= xmin_min) & (nz_eigs[:-1] <= xmin_max)
                valid_indices &= peak_indices

            if torch.any(valid_indices):
                valid_Ds = Ds[valid_indices]
                min_D_index_in_valid = torch.argmin(valid_Ds)

                original_indices = torch.where(valid_indices)[0]
                min_D_index = original_indices[min_D_index_in_valid]

                final_alpha = alphas[min_D_index].item()
                final_D = Ds[min_D_index].item()

        svals = torch.sqrt(nz_eigs)
        hard_rank_tensor = matrix_rank_torch(svals, weight_t.shape[1])
        entropy_tensor = matrix_entropy_torch(nz_eigs, hard_rank_tensor)
        log_alpha_norm = torch.log10(torch.sum(nz_eigs ** final_alpha)) if final_alpha > 0 else torch.tensor(-1.0)

        return {
            'D': final_D, 'M': weight_t.shape[0], 'N': weight_t.shape[1],
            'alpha': final_alpha, 'longname': name,
            'alpha_weighted': final_alpha * math.log10(spectral_norm) if spectral_norm > 0 else 0.0,
            'entropy': entropy_tensor.item(),
            'log_alpha_norm': log_alpha_norm.item(),
            'log_norm': np.log10(fnorm) if fnorm > 0 else 0.0,
            'log_spectral_norm': np.log10(spectral_norm) if spectral_norm > 0 else 0.0,
            'matrix_rank': hard_rank_tensor.item(),
            'norm': fnorm, 'num_evals': N, 'spectral_norm': spectral_norm,
            'stable_rank': fnorm / spectral_norm if spectral_norm > 0 else 0.0,
            'xmax': nz_eigs[-1].item(), 'xmin': nz_eigs[0].item(),
            'params': params, 'eigs': nz_eigs.detach().cpu().numpy() if save_eigs else None,
        }


def squared_singular_values(matrix: torch.Tensor, use_svd: bool = True) -> torch.Tensor:
    """Compute squared singular values efficiently via Gram matrices or SVD.

    Accepts 2D (M,N) or batched 3D (B,M,N) tensors and returns a 1D tensor of
    squared singular values (flattened for batched input). 
    If use_svd is True, uses torch.linalg.svdvals directly.
    Otherwise,
    Uses eigvalsh on the smaller Gram matrix (A A^T if M<=N else A^T A) for
    speed and numerical stability, clamping tiny negative values to zero.
    
    Note: The Gram method may lose precision on eigenvalues smaller than ~1e-14
    for severely ill-conditioned matrices (condition number > 1e+7) due to 
    condition number squaring. For ESD analysis with EVALS_THRESH >= 1e-5, 
    this has negligible impact on all computed metrics.
    """
    if use_svd:
        svals = torch.linalg.svdvals(matrix)
        return torch.square(svals) if matrix.ndim == 2 else torch.square(svals).reshape(-1)
    
    # Ensure numeric symmetry before eigvalsh to avoid off-diagonal noise
    if matrix.ndim == 2:
        M, N = matrix.shape
        if M <= N:
            G = matrix @ matrix.transpose(-1, -2)
        else:
            G = matrix.transpose(-1, -2) @ matrix
        # Symmetrize and enforce contiguity
        G = 0.5 * (G + G.transpose(-1, -2))
        G = G.contiguous()
        try:
            evals = torch.linalg.eigvalsh(G)
        except RuntimeError:
            # Retry with a tiny diagonal jitter to fix potential numerical pathologies
            try:
                eps = torch.finfo(G.dtype).eps
                I = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
                evals = torch.linalg.eigvalsh(G + eps * I)
            except Exception:
                # Fallback: exact SVD path
                svals = torch.linalg.svdvals(matrix)
                return torch.square(svals)
        return torch.clamp(evals, min=0)
    elif matrix.ndim == 3:
        B, M, N = matrix.shape
        if M <= N:
            G = torch.matmul(matrix, matrix.transpose(-1, -2))  # [B, M, M]
        else:
            G = torch.matmul(matrix.transpose(-1, -2), matrix)  # [B, N, N]
        # Symmetrize and enforce contiguity
        G = 0.5 * (G + G.transpose(-1, -2))
        G = G.contiguous()
        try:
            evals = torch.linalg.eigvalsh(G)
            return torch.clamp(evals, min=0).reshape(-1)
        except RuntimeError:
            # Retry with jitter
            try:
                eps = torch.finfo(G.dtype).eps
                eye_n = G.shape[-1]
                I = torch.eye(eye_n, device=G.device, dtype=G.dtype).expand(B, eye_n, eye_n)
                evals = torch.linalg.eigvalsh(G + eps * I)
                return torch.clamp(evals, min=0).reshape(-1)
            except Exception:
                # Fallback: exact batched SVD path
                svals = torch.linalg.svdvals(matrix)
                return torch.square(svals).reshape(-1)
    else:
        logger.warning(f"_squared_singular_values: unsupported shape {matrix.shape}")
        return torch.empty(0, device=matrix.device, dtype=matrix.dtype)


def mp_worker(task_q: "mp.Queue", result_q: "mp.Queue", dev_id: int,
               EVALS_THRESH: float, bins: int, fix_fingers: Optional[str], xmin_pos: int,
               conv_norm: float, filter_zeros: bool, use_svd: bool, save_eigs: bool, params: int) -> None:
    """Multiprocessing worker: binds to a GPU device and processes tasks from a queue.

    Each task is (orig_idx, name, weight_numpy, params). Results are (orig_idx, result_dict_or_None).
    """
    # Bind CUDA device in subprocess (no-op if CUDA not available)
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(dev_id)
        except Exception as e:
            logger.warning(f"_mp_worker: failed to set CUDA device {dev_id}: {e}")

    while True:
        item = task_q.get()
        if item is None:
            break
        orig_idx, name, w_np, params = item
        try:
            # Reconstruct tensor on CPU and let _compute_esd_for_weight handle H2D with pinning
            weight_t = torch.from_numpy(w_np)
            res = compute_esd_for_weight(
                name, weight_t, EVALS_THRESH, bins, fix_fingers, xmin_pos,
                conv_norm, filter_zeros, use_svd, save_eigs, dev_id, params
            )
        except Exception as e:
            logger.exception(f"_mp_worker: error processing {name} on cuda:{dev_id}")
            res = None
        result_q.put((orig_idx, res))

