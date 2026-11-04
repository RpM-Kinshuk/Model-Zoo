"""Deterministic nested sketches and contiguous row-block selection."""
from collections import OrderedDict

import numpy as np
import torch

SKETCH_SEED = 20261028
SKETCH_K_MAX = 256
SKETCH_K_DEFAULT = 64
ROW_BLOCK = 256
ROW_START_CAP = 1920
OPERATOR_CACHE_BYTES = 4 * 1024**3

_OPERATOR_CACHE = OrderedDict()
_CACHE_BYTES = 0
_CUDA_OPERATOR_CACHE = OrderedDict()
_CUDA_CACHE_BYTES = 0
CUDA_OPERATOR_CACHE_BYTES = 2 * 1024**3


def sketch_operator(dim: int, side: int, k: int = SKETCH_K_DEFAULT) -> np.ndarray:
    """Return a prefix of the fixed orthonormal operator for this axis."""
    global _CACHE_BYTES
    if dim <= 0 or side not in (0, 1) or not 1 <= k <= SKETCH_K_MAX:
        raise ValueError('expected dim > 0, side in {0, 1}, and 1 <= k <= 256')
    key = (dim, side)
    operator = _OPERATOR_CACHE.pop(key, None)
    if operator is None:
        g = np.random.default_rng([SKETCH_SEED, dim, side]).standard_normal(
            (SKETCH_K_MAX, dim)
        )
        q, r = np.linalg.qr(g.T)
        q *= np.sign(np.diag(r))
        operator = np.ascontiguousarray(q.T, dtype=np.float32)
        operator.flags.writeable = False
        if operator.nbytes <= OPERATOR_CACHE_BYTES:
            while _OPERATOR_CACHE and _CACHE_BYTES + operator.nbytes > OPERATOR_CACHE_BYTES:
                _, oldest = _OPERATOR_CACHE.popitem(last=False)
                _CACHE_BYTES -= oldest.nbytes
            _OPERATOR_CACHE[key] = operator
            _CACHE_BYTES += operator.nbytes
    else:
        _OPERATOR_CACHE[key] = operator
    return operator[:min(k, dim)]


def _matrix(w: torch.Tensor) -> np.ndarray:
    if w.ndim != 2 or min(w.shape) <= 0:
        raise ValueError('expected a nonempty matrix')
    return w.detach().to(device='cpu', dtype=torch.float64).numpy()


def two_sided_sketch(w: torch.Tensor, k: int = SKETCH_K_DEFAULT) -> np.ndarray:
    if w.is_cuda:
        return _cuda_two_sided_sketch(w, k)
    matrix = _matrix(w)
    left = sketch_operator(matrix.shape[0], 0, k).astype(np.float64)
    right = sketch_operator(matrix.shape[1], 1, k).astype(np.float64)
    return (left @ matrix @ right.T).astype(np.float32)


def _cuda_operator(dim, side, k, device):
    """Cache the original NumPy operator on the compute device, bounded in bytes."""
    global _CUDA_CACHE_BYTES
    key = (dim, side, k, str(device))
    operator = _CUDA_OPERATOR_CACHE.pop(key, None)
    if operator is None:
        fixed = sketch_operator(dim, side, k).astype(np.float64)
        operator = torch.from_numpy(fixed).to(device)
        size = operator.numel() * operator.element_size()
        if size > CUDA_OPERATOR_CACHE_BYTES:
            return operator
        while _CUDA_OPERATOR_CACHE and _CUDA_CACHE_BYTES + size > CUDA_OPERATOR_CACHE_BYTES:
            _, oldest = _CUDA_OPERATOR_CACHE.popitem(last=False)
            _CUDA_CACHE_BYTES -= oldest.numel() * oldest.element_size()
        _CUDA_CACHE_BYTES += size
    _CUDA_OPERATOR_CACHE[key] = operator
    return operator


def _cuda_two_sided_sketch(w, k):
    if w.ndim != 2 or min(w.shape) <= 0:
        raise ValueError('expected a nonempty matrix')
    matrix = w.detach().double()
    left = _cuda_operator(matrix.shape[0], 0, k, matrix.device)
    right = _cuda_operator(matrix.shape[1], 1, k, matrix.device)
    return (left @ matrix @ right.T).float().cpu().numpy()


def factored_sketch(b: torch.Tensor, a: torch.Tensor,
                    k: int = SKETCH_K_DEFAULT) -> np.ndarray:
    if b.is_cuda or a.is_cuda:
        device = b.device if b.is_cuda else a.device
        left_factor = b.detach().to(device=device, dtype=torch.float64)
        right_factor = a.detach().to(device=device, dtype=torch.float64)
        if (left_factor.ndim != 2 or right_factor.ndim != 2
                or min(left_factor.shape) <= 0 or min(right_factor.shape) <= 0
                or left_factor.shape[1] != right_factor.shape[0]):
            raise ValueError('expected compatible nonempty matrix factors')
        left = _cuda_operator(left_factor.shape[0], 0, k, device)
        right = _cuda_operator(right_factor.shape[1], 1, k, device)
        return ((left @ left_factor) @ (right_factor @ right.T)).float().cpu().numpy()
    left_factor, right_factor = _matrix(b), _matrix(a)
    if left_factor.shape[1] != right_factor.shape[0]:
        raise ValueError('incompatible factor shapes')
    left = sketch_operator(left_factor.shape[0], 0, k).astype(np.float64)
    right = sketch_operator(right_factor.shape[1], 1, k).astype(np.float64)
    return ((left @ left_factor) @ (right_factor @ right.T)).astype(np.float32)


def sketch_cosine(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError('expected two matrix sketches')
    rows, cols = min(x.shape[0], y.shape[0]), min(x.shape[1], y.shape[1])
    if rows == 0 or cols == 0:
        return 0.0
    a = np.asarray(x[:rows, :cols], dtype=np.float64).ravel()
    b = np.asarray(y[:rows, :cols], dtype=np.float64).ravel()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / norm, -1.0, 1.0)) if norm else 0.0


def row_block_bounds(rows: int, b: int = ROW_BLOCK) -> tuple[int, int]:
    if rows < 0 or b <= 0:
        raise ValueError('expected rows >= 0 and b > 0')
    start = min((rows - b) // 2, ROW_START_CAP) if rows > b else 0
    return start, min(rows, start + b)
