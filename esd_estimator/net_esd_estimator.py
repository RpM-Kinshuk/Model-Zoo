import torch
import torch.nn as nn
import numpy as np
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import List, Optional, Dict, Any, Iterator, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# EPSILON = 1e-10
EPSILON = 6e-05

RESULT_KEYS = [
    'D', 'M', 'N', 'alpha', 'alpha_weighted', 'entropy', 'log_alpha_norm',
    'log_norm', 'log_spectral_norm', 'longname', 'matrix_rank', 'norm',
    'num_evals', 'spectral_norm', 'stable_rank', 'xmax', 'xmin', 'params', 'eigs'
]

### -- Main NetESD Estimator Function -- ###
def net_esd_estimator(
        net: nn.Module,
        EVALS_THRESH: float = 1e-5,
        bins: int = 100,
        fix_fingers: Optional[str] = None,
        xmin_pos: int = 2,
        conv_norm: float = 0.5,
        filter_zeros: bool = False,
        parallel: bool = True,
        device_ids: Optional[List[int]] = None,
        max_workers: Optional[int] = None,
        filter_type: Optional[bool] = True,
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
        parallel (bool, optional): If True, dispatch layer computations across multiple GPUs.
        device_ids (List[int], optional): GPU IDs to use for compute. If None, auto-select GPUs not used by the model; if none available, use all.
        max_workers (int, optional): Max concurrent workers. Defaults to len(device_ids) or auto-selected pool size.
        filter_type (bool, optional): If True, only process Conv1d, Conv2d, and Linear layers with aspect ratio filtering.

    Returns:
        dict: A dictionary containing the computed metrics for each layer.
    """
    results = {key: [] for key in RESULT_KEYS}
    print("=================================")
    print(f"Running optimized multi-GPU ESD estimator with:")
    print(f"  fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, filter_zeros: {filter_zeros}")
    print(
        f"  parallel: {parallel}, device_ids: {device_ids if device_ids else 'auto'}, max_workers: {max_workers if max_workers else 'auto'}")
    print("=================================")

    with torch.no_grad():
        eligible = list(_iter_eligible_layers(net, filter_type))
        if not parallel or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            layer_results = [
                res for (name, w, params) in eligible
                if (res := _compute_esd_for_weight(name, w, EVALS_THRESH, bins, fix_fingers, xmin_pos, conv_norm, filter_zeros, params, None)) is not None
            ]
        else:
            used_cuda_ids = sorted(p.device.index for p in net.parameters() if getattr(p, 'is_cuda', False))
            all_ids = list(range(torch.cuda.device_count()))
            if device_ids is not None and len(device_ids) > 0:
                pool = [i for i in device_ids if i in all_ids]
            else:
                # Prefer GPUs not hosting the model; if none, fall back to all GPUs
                pool = [i for i in all_ids if i not in used_cuda_ids] or all_ids
            workers = max_workers or len(pool) or 1
            dev_cycle = cycle(pool)

            futures = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for idx, (name, w, params) in enumerate(eligible):
                    futures.append((idx, ex.submit(_compute_esd_for_weight, name, w, EVALS_THRESH, bins, fix_fingers, xmin_pos, conv_norm, filter_zeros, params, next(dev_cycle))))
                ordered = [None] * len(futures)
                for idx, fut in futures:
                    ordered[idx] = fut.result()
            layer_results = [r for r in ordered if r is not None]

        for r in layer_results:
            for key, value in r.items():
                if key in results:
                    results[key].append(value)
    return results


def _compute_esd_for_weight(
        name: str, weight_t: torch.Tensor, EVALS_THRESH: float, bins: int,
        fix_fingers: Optional[str], xmin_pos: int, conv_norm: float,
        filter_zeros: bool, params: int, device_id: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Compute ESD-related metrics for a single layer's weight on the specified device.
    Returns a dict with metrics for that layer, or None if not enough eigenvalues.
    """
    with torch.no_grad():
        use_cuda = device_id is not None and torch.cuda.is_available()
        device = torch.device(f"cuda:{device_id}") if use_cuda else (
            weight_t.device if weight_t.device.type != 'cpu' else torch.device('cpu')
        )
        matrix = weight_t.to(device=device, non_blocking=True).float()
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

        eigs = torch.square(torch.linalg.svdvals(matrix).flatten())

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
        hard_rank_tensor = _matrix_rank_torch(svals, weight_t.shape[1])
        entropy_tensor = _matrix_entropy_torch(nz_eigs, hard_rank_tensor)
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
            'params': params, 'eigs': nz_eigs.detach().cpu().numpy(),
        }


def _matrix_rank_torch(svals: torch.Tensor, N: int, tol: Optional[float] = None) -> torch.Tensor:
    """Matrix rank, computed from singular values, using torch."""
    if tol is None:
        tol = svals.max() * N * torch.finfo(svals.dtype).eps # type: ignore
    return torch.sum(svals > tol) # type: ignore


def _matrix_entropy_torch(evals: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Matrix entropy, computed directly from eigenvalues using PyTorch"""
    if rank <= 0: return torch.tensor(-1.0, device=evals.device)
    evals_sum = torch.sum(evals)
    if evals_sum <= 0: return torch.tensor(-1.0, device=evals.device)
    p = (evals / evals_sum) + EPSILON
    log_rank = torch.log(rank.float() + EPSILON)
    if torch.abs(log_rank) < 1e-9: return torch.tensor(-1.0, device=evals.device)
    return -torch.sum(p * torch.log(p)) / log_rank


def _iter_eligible_layers(net: nn.Module, filter_type: Optional[bool] = True) -> Iterator[Tuple[str, torch.Tensor, int]]:
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


def merge_lora_weights(model):
    """Merge LoRA weights into the base model layers in-place."""
    lora_layers = {name: module for name, module in model.named_modules() if 'lora_' in name}
    if not lora_layers:
        return

    logger.info("Merging LoRA weights...")
    merged_layers = set()

    for name, module in model.named_modules():
        if "lora_A" in name and name not in merged_layers:
            base_name = name.rsplit('.lora_A', 1)[0]
            lora_b_name = name.replace("lora_A", "lora_B")

            base_module = dict(model.named_modules()).get(base_name)
            lora_a_module = module
            lora_b_module = lora_layers.get(lora_b_name)

            if base_module and lora_a_module and lora_b_module:
                logger.info(f"Merging LoRA for {base_name}")
                if hasattr(base_module, 'weight'):
                    base_weights = base_module.weight.data
                    lora_a_weights = lora_a_module.weight.data
                    lora_b_weights = lora_b_module.weight.data
                    
                    # scaling is often stored in the peft config, but not directly in the layer.
                    # Common default is 1.0. Some models might have a scaling factor.
                    # Here we assume scaling is implicitly handled or is 1.
                    lora_update = lora_b_weights @ lora_a_weights
                    base_module.weight.data += lora_update

                    # Mark as merged to avoid double merging
                    merged_layers.add(name)
                    merged_layers.add(lora_b_name)