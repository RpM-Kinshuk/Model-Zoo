"""Block update statistics with abstention at the storage noise floor."""
import math

import numpy as np
import torch

FLOOR_FRAC_IDENTICAL = 0.5

# Explicit mantissa bits and minimum normal exponent of each stored format.
_STORAGE_PRECISION = {
    torch.bfloat16: (7, -126),
    torch.float16: (10, -14),
    torch.float8_e4m3fn: (3, -6),
    torch.float8_e5m2: (2, -14),
}


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
                storage_dtype: torch.dtype) -> dict:
    if child.shape != parent.shape:
        raise ValueError('child and parent shapes must match')
    if child.ndim != 2 or min(child.shape) == 0:
        raise ValueError('expected nonempty matrix blocks')
    child_stored = child.detach().to(device='cpu', dtype=storage_dtype).contiguous()
    parent_stored = parent.detach().to(device='cpu', dtype=storage_dtype).contiguous()
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

    raw_values = torch.linalg.svdvals(delta).numpy()
    count = min(16, raw_values.size)
    result['top_singular_values'][:count] = raw_values[:count]
    if storage_dtype in _STORAGE_PRECISION:
        mantissa_bits, min_exponent = _STORAGE_PRECISION[storage_dtype]
        magnitude = parent_float.abs().double()
        _, exponent = torch.frexp(magnitude)
        exponent = torch.where(magnitude == 0, min_exponent + 1, exponent)
        spacing = torch.ldexp(torch.ones_like(delta), torch.clamp(
            exponent - mantissa_bits - 1, min=min_exponent - mantissa_bits,
        ))
        variance = spacing.square() / 12
        u, v = torch.ones(rows, dtype=torch.float64), torch.ones(cols, dtype=torch.float64)
        for _ in range(50):
            u = 1 / ((variance @ v) / cols)
            v = 1 / ((variance.T @ u) / rows)
        whitened = u.sqrt()[:, None] * delta * v.sqrt()[None, :]
        values = torch.linalg.svdvals(whitened).numpy()
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
