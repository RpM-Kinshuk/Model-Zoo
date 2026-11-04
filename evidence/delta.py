"""Block update statistics with abstention at the storage noise floor."""
import math
import os

import numpy as np
import torch

from net_esd.core import _svdvals_for_measurement

FLOOR_FRAC_IDENTICAL = 0.5

# Explicit mantissa bits and minimum normal exponent of each stored format.
_STORAGE_PRECISION = {
    torch.bfloat16: (7, -126),
    torch.float16: (10, -14),
    torch.float8_e4m3fn: (3, -6),
    torch.float8_e5m2: (2, -14),
}


def _compute_device(value):
    if value == "auto":
        assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        value = "cuda:0" if assigned and assigned != "-1" else "cpu"
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError):
        raise ValueError("Invalid spectrum compute device") from None
    if device.type == "cpu" and device.index is None:
        return device
    if device.type != "cuda":
        raise ValueError("Spectrum compute device must be cpu or cuda:<index>")
    if not torch.cuda.is_available():
        raise ValueError("Requested CUDA compute device is unavailable")
    index = device.index if device.index is not None else 0
    if index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA compute device index is unavailable")
    return torch.device("cuda", index)


def _delta_svdvals(matrix, device):
    if device.type == "cpu":
        return torch.linalg.svdvals(matrix).numpy()
    # Inputs are the original CPU-computed float64 delta/whitened matrices.
    # Use the collector's precision-tested CUDA QR driver, never Jacobi SVD.
    return _svdvals_for_measurement(matrix.to(device)).cpu().numpy()


def _delta_stored(tensor, storage_dtype, device):
    return tensor.detach().to(device=device, dtype=storage_dtype).contiguous()


def gavish_donoho_rank(singular_values: np.ndarray, rows: int, cols: int) -> int:
    if rows <= 0 or cols <= 0:
        raise ValueError('expected positive matrix dimensions')
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('expected finite nonnegative singular values')
    if not values.size:
        return 0
    beta = min(rows, cols) / max(rows, cols)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return int(np.count_nonzero(values > omega * np.median(values)))


def block_delta(child: torch.Tensor, parent: torch.Tensor,
                storage_dtype: torch.dtype, *, compute_device="cpu") -> dict:
    device = _compute_device(compute_device)
    if child.shape != parent.shape:
        raise ValueError('child and parent shapes must match')
    if child.ndim != 2 or min(child.shape) == 0:
        raise ValueError('expected nonempty matrix blocks')
    child_stored = _delta_stored(child, storage_dtype, device)
    parent_stored = _delta_stored(parent, storage_dtype, device)
    rows, cols = parent.shape
    identical = (child_stored.view(torch.uint8).reshape(rows, cols, -1)
                 == parent_stored.view(torch.uint8).reshape(rows, cols, -1)).all(dim=-1)
    frac_identical = identical.double().mean().item()
    child_float, parent_float = child_stored.float(), parent_stored.float()
    if not torch.isfinite(child_float).all() or not torch.isfinite(parent_float).all():
        raise ValueError('expected finite block values')
    delta = (child_float - parent_float).double()
    parent_norm = parent_float.double().norm().item()
    delta_norm = delta.norm().item()
    result = {
        'frac_identical': frac_identical,
        'rel_update_norm': delta_norm / parent_norm if parent_norm else (
            float('inf') if delta_norm else 0.0
        ),
        'rank': -1,
        'rank_confidence': 'low',
        'status': 'identical' if frac_identical == 1.0 else 'below_floor',
        'top_singular_values': np.full(16, np.nan, dtype=np.float32),
    }
    if frac_identical == 1.0 or frac_identical > FLOOR_FRAC_IDENTICAL:
        return result

    del child_stored, parent_stored, child_float, identical
    raw_values = _delta_svdvals(delta, device)
    count = min(16, raw_values.size)
    result['top_singular_values'][:count] = raw_values[:count]
    if storage_dtype in _STORAGE_PRECISION:
        mantissa_bits, min_exponent = _STORAGE_PRECISION[storage_dtype]
        magnitude = parent_float.abs().double()
        del parent_float
        _, exponent = torch.frexp(magnitude)
        del _
        exponent = torch.where(magnitude == 0, min_exponent + 1, exponent)
        del magnitude
        spacing = torch.ldexp(torch.ones_like(delta), torch.clamp(
            exponent - mantissa_bits - 1, min=min_exponent - mantissa_bits,
        ))
        del exponent
        variance = spacing.square() / 12
        del spacing
        u = torch.ones(rows, dtype=torch.float64, device=delta.device)
        v = torch.ones(cols, dtype=torch.float64, device=delta.device)
        for _ in range(50):
            u = 1 / ((variance @ v) / cols)
            v = 1 / ((variance.T @ u) / rows)
        whitened = u.sqrt()[:, None] * delta * v.sqrt()[None, :]
        del variance, delta, u, v
        values = _delta_svdvals(whitened, device)
        threshold = 1.1 * (math.sqrt(rows) + math.sqrt(cols))
        rank = int(np.count_nonzero(values > threshold))
    else:
        values = raw_values
        rank = gavish_donoho_rank(values, rows, cols)
    confidence = 'low'
    if rank > 0:
        if rank == values.size or values[rank] == 0 or values[rank - 1] / values[rank] >= 1.5:
            confidence = 'high'
    result.update(rank=rank, rank_confidence=confidence, status='ok')
    return result
