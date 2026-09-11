"""Core computation functions for ESD analysis."""

import torch
import math
import multiprocessing as mp
from typing import Optional, Dict, Any

from .constants import logger
from .utils import matrix_rank_torch, matrix_entropy_torch


# Exhaustive cutoff search is still quadratic in work, but only this many
# candidate tails are materialized together (O(chunk_size * spectrum_size)).
_KS_CHUNK_SIZE = 64


def _fit_tail_chunk(eigs: torch.Tensor, starts: torch.Tensor):
    """Pareto MLE and full KS for a small batch of cutoffs, in float64.

    ``eigs`` is sorted float64 and contains only positive eigenvalues.
    Each start must be the first occurrence of its cutoff, so tied observations
    are included in full. Near-cutoff log1p ratios avoid cancellation even for
    narrow float64 spectra far from unit scale. Log differences handle wider
    ratios without overflow; expm1 retains precision when evaluating the CDF.
    """
    positions = torch.arange(eigs.numel(), device=eigs.device)
    n = (eigs.numel() - starts).double()
    cutoffs = eigs[starts, None]
    log_ratios = (eigs[None, :] - cutoffs).clamp_min_(0)
    near_cutoff = log_ratios <= cutoffs
    # Limit the numerator *before* division, avoiding overflow for spectra
    # spanning most of float64's range. Wider ratios use log differences below.
    log_ratios = torch.minimum(log_ratios, cutoffs).div_(cutoffs).log1p_()
    log_eigs = torch.log(eigs)
    log_ratios = torch.where(
        near_cutoff, log_ratios, log_eigs[None, :] - log_eigs[starts, None],
    )
    alphas = 1 + n / log_ratios.sum(dim=1)
    cdf = -torch.expm1((1 - alphas[:, None]) * log_ratios)
    empirical_left = (positions[None, :] - starts[:, None]).double() / n[:, None]
    cdf_delta = cdf - empirical_left
    mask = positions[None, :] >= starts[:, None]
    D_minus = torch.where(mask, cdf_delta, -math.inf).amax(dim=1)
    D_plus = torch.where(mask, 1 / n[:, None] - cdf_delta, -math.inf).amax(dim=1)
    return alphas, torch.maximum(D_plus, D_minus)


def compute_esd_for_weight(
        name: str, weight_t: torch.Tensor, EVALS_THRESH: float, bins: int,
        fix_fingers: Optional[str], xmin_pos: int, conv_norm: float,
        filter_zeros: bool, use_svd: bool, save_eigs: Optional[bool],
        device_id: Optional[int], params: int, compute_dtype: str = 'float32'
) -> Optional[Dict[str, Any]]:
    """Compute ESD-related metrics for a single layer's weight on the specified device.
    Keep layer measurements even when its retained spectrum cannot be fitted.
    Returns None only for weights without a supported, computable spectrum.

    Existing norm/rank/entropy fields describe the retained spectrum; ``raw_*``
    fields describe the full computed spectrum before the absolute filter.
    Saved ``eigs`` always contains that full spectrum, including numerical zeros.
    For convolutions both spectra pool normalized spatial-kernel slices, not the
    singular values of the complete convolution operator.
    """
    if compute_dtype not in ('float32', 'float64'):
        raise ValueError("compute_dtype must be 'float32' or 'float64'")
    dtype = getattr(torch, compute_dtype)
    source_dtype = str(weight_t.dtype).removeprefix('torch.')
    layouts = {2: 'matrix', 3: 'conv1d_slices', 4: 'conv2d_slices', 5: 'conv3d_slices'}
    if weight_t.ndim not in layouts:
        logger.warning(f"Skipping layer {name} due to unsupported dimensions: {weight_t.shape}")
        return None
    with torch.no_grad():
        use_cuda = device_id is not None and torch.cuda.is_available()
        device = torch.device(f"cuda:{device_id}") if use_cuda else (
            weight_t.device if weight_t.device.type != 'cpu' else torch.device('cpu')
        )
        if device.type.startswith('cuda') and weight_t.device.type == 'cpu':
            # Pinned host memory can accelerate H2D transfers when non_blocking=True
            matrix = weight_t.pin_memory().to(device=device, dtype=dtype, non_blocking=True)
        else:
            matrix = weight_t.to(device=device, dtype=dtype, non_blocking=True)
        matrix = matrix.contiguous()

        # Standard Conv1d/2d/3d layouts are (out, in/groups, *spatial_kernel).
        # Pool one (out, in/groups) matrix per spatial location, retaining the
        # existing Conv2d normalization. Do not mutate a caller-owned tensor.
        if matrix.ndim in (3, 4, 5):
            out_channels, in_channels = matrix.shape[0], matrix.shape[1]
            spatial_axes = tuple(range(2, matrix.ndim))
            matrix = matrix.permute(*spatial_axes, 0, 1).contiguous().reshape(
                math.prod(weight_t.shape[2:]), out_channels, in_channels,
            )
            matrix = matrix * math.sqrt(conv_norm)

        # Single in-flight op per GPU is enforced by per-GPU workers.
        eigs = squared_singular_values(matrix, use_svd)
        if eigs.numel() == 0:
            return None
        if not torch.isfinite(eigs).all():
            raise ValueError(f"Layer {name} produced non-finite eigenvalues")

        eigs = torch.sort(eigs).values
        nz_eigs = eigs[eigs > EVALS_THRESH] if filter_zeros else eigs
        spectral_norm = nz_eigs[-1].item() if nz_eigs.numel() else 0.0
        fnorm = torch.sum(nz_eigs, dtype=torch.float64).item()

        # Fitting requires positive values, even when the saved spectrum includes
        # zeros. Double-precision log sums avoid cancellation for narrow tails.
        fit_eigs = nz_eigs[nz_eigs > 0]
        N = fit_eigs.numel()
        fit_eigs64 = fit_eigs.double()
        log_fit_eigs = torch.log(fit_eigs64)
        final_alpha, final_D = math.nan, math.nan
        fit_xmin, n_tail = math.nan, 0
        fit_status = 'no_valid_cutoff'
        if nz_eigs.numel() == 0:
            fit_status = 'no_retained_eigenvalues'
        elif N < 2:
            fit_status = 'insufficient_positive_eigenvalues'
        elif fit_eigs[0] == fit_eigs[-1]:
            fit_status = 'constant_spectrum'
        if N > 1 and fix_fingers == 'xmin_mid':
            i = N // xmin_pos
            if i >= N - 1: i = N - 2
            # A cutoff includes every occurrence of its value, including ties.
            i = torch.searchsorted(fit_eigs, fit_eigs[i]).item()
            xmin = fit_eigs[i]
            if xmin < fit_eigs[-1]:
                alphas, Ds = _fit_tail_chunk(fit_eigs64, torch.tensor([i], device=device))
                alpha, D = alphas.item(), Ds.item()
                if math.isfinite(alpha) and alpha > 1 and math.isfinite(D):
                    final_alpha, final_D = alpha, D
                    fit_xmin, n_tail = xmin.item(), N - i
        elif N > 1:
            # Never start inside a group of equal eigenvalues: the tail is x >= xmin.
            valid_indices = fit_eigs[:-1] < fit_eigs[-1]
            valid_indices[1:] &= fit_eigs[1:-1] > fit_eigs[:-2]
            if fix_fingers == 'xmin_peak':
                log10_nz_eigs = torch.log10(fit_eigs)
                min_e, max_e = log10_nz_eigs.min(), log10_nz_eigs.max()
                counts = torch.histc(log10_nz_eigs, bins, min=min_e.item(), max=max_e.item())
                boundaries = torch.linspace(min_e, max_e, bins + 1, device=device,
                                            dtype=log10_nz_eigs.dtype)
                ih = torch.argmax(counts)
                xmin2 = 10 ** boundaries[ih]
                xmin_min = 0.95 * xmin2
                xmin_max = 1.5 * xmin2
                peak_indices = (fit_eigs[:-1] >= xmin_min) & (fit_eigs[:-1] <= xmin_max)
                valid_indices &= peak_indices

            candidates = torch.where(valid_indices)[0]
            for starts in candidates.split(_KS_CHUNK_SIZE):
                if not starts.numel():
                    continue
                alphas, Ds = _fit_tail_chunk(fit_eigs64, starts)
                valid = torch.isfinite(alphas) & (alphas > 1) & torch.isfinite(Ds)
                if not torch.any(valid):
                    continue
                valid_Ds = torch.where(valid, Ds, math.inf)
                winner = torch.argmin(valid_Ds)
                D = valid_Ds[winner].item()
                # Strict comparison preserves the first cutoff when distances tie.
                if not math.isfinite(final_D) or D < final_D:
                    i = starts[winner].item()
                    final_alpha, final_D = alphas[winner].item(), D
                    fit_xmin, n_tail = fit_eigs[i].item(), N - i

        if math.isfinite(final_alpha):
            fit_status = 'fitted'

        svals = torch.sqrt(nz_eigs)
        hard_rank_tensor = matrix_rank_torch(svals, max(weight_t.shape[:2]))
        entropy_tensor = matrix_entropy_torch(nz_eigs, hard_rank_tensor)
        if nz_eigs.numel() == eigs.numel():
            raw_rank, raw_entropy, raw_norm = hard_rank_tensor, entropy_tensor, fnorm
        else:
            raw_rank = matrix_rank_torch(torch.sqrt(eigs), max(weight_t.shape[:2]))
            raw_entropy = matrix_entropy_torch(eigs, raw_rank)
            raw_norm = torch.sum(eigs, dtype=torch.float64).item()
        log_alpha_norm = math.nan
        if math.isfinite(final_alpha):
            log_alpha_norm = torch.logsumexp(final_alpha * log_fit_eigs, dim=0).item() / math.log(10)

        return {
            'D': final_D, 'M': weight_t.shape[0], 'N': weight_t.shape[1],
            'alpha': final_alpha, 'longname': name, 'compute_device': str(device),
            'alpha_weighted': final_alpha * math.log10(spectral_norm) if spectral_norm > 0 else math.nan,
            'entropy': entropy_tensor.item(),
            'log_alpha_norm': log_alpha_norm,
            'log_norm': math.log10(fnorm) if fnorm > 0 else math.nan,
            'log_spectral_norm': math.log10(spectral_norm) if spectral_norm > 0 else math.nan,
            'matrix_rank': hard_rank_tensor.item(),
            'norm': fnorm, 'num_evals': nz_eigs.numel(), 'spectral_norm': spectral_norm,
            'stable_rank': fnorm / spectral_norm if spectral_norm > 0 else math.nan,
            'xmax': nz_eigs[-1].item() if nz_eigs.numel() else math.nan,
            'xmin': nz_eigs[0].item() if nz_eigs.numel() else math.nan,
            'fit_xmin': fit_xmin, 'n_tail': n_tail,
            'fit_status': fit_status,
            'raw_num_evals': eigs.numel(),
            'raw_norm': raw_norm,
            'raw_spectral_norm': eigs[-1].item(),
            'raw_matrix_rank': raw_rank.item(), 'raw_entropy': raw_entropy.item(),
            'evals_thresh': EVALS_THRESH, 'filter_zeros': bool(filter_zeros),
            'source_dtype': source_dtype, 'compute_dtype': compute_dtype,
            'weight_layout': layouts[weight_t.ndim],
            'params': params, 'eigs': eigs.detach().cpu().numpy() if save_eigs else None,
        }


def squared_singular_values(matrix: torch.Tensor, use_svd: bool = True) -> torch.Tensor:
    """Compute squared singular values efficiently via Gram matrices or SVD.

    Accepts 2D (M,N) or batched 3D (B,M,N) tensors and returns a 1D tensor of
    squared singular values (flattened for batched input). 
    If use_svd is True, uses torch.linalg.svdvals directly.
    Otherwise,
    Uses eigvalsh on the smaller Gram matrix (A A^T if M<=N else A^T A) for
    speed and numerical stability, clamping tiny negative values to zero.
    
    Note: The Gram method may lose precision on eigenvalues for severely 
    ill-conditioned matrices due to condition number squaring.
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
               conv_norm: float, filter_zeros: bool, use_svd: bool, save_eigs: bool,
               compute_dtype: str = 'float32') -> None:
    """Multiprocessing worker: binds to a GPU device and processes tasks from a queue.

    Each task is (orig_idx, name, weight_numpy, params). Results include an error
    field: (orig_idx, result_dict_or_None, error_or_None), so numerical failures
    cannot silently become missing layers in the parent process.
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
                conv_norm, filter_zeros, use_svd, save_eigs, dev_id, params, compute_dtype
            )
            error = None
        except Exception as e:
            logger.exception(f"_mp_worker: error processing {name} on cuda:{dev_id}")
            res = None
            error = f"{type(e).__name__}: {e}"
        result_q.put((orig_idx, res, error))
