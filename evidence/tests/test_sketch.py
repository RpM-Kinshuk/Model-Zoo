import numpy as np
import pytest
import torch

from evidence.sketch import (sketch_operator, two_sided_sketch, factored_sketch,
                             sketch_cosine, row_block_bounds)


def test_operator_orthonormal_nested_and_side_specific():
    a64, a256 = sketch_operator(4096, 0, 64), sketch_operator(4096, 0, 256)
    assert np.allclose(a256 @ a256.T, np.eye(256), atol=1e-5)
    assert np.array_equal(a64, a256[:64])
    assert not np.allclose(sketch_operator(4096, 0, 64), sketch_operator(4096, 1, 64))

def test_sketch_is_linear():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(300, 128, generator=g), torch.randn(300, 128, generator=g)
    lhs = two_sided_sketch(0.3 * x + 0.7 * y)
    rhs = 0.3 * two_sided_sketch(x) + 0.7 * two_sided_sketch(y)
    assert np.allclose(lhs, rhs, rtol=0, atol=1e-5 * np.abs(lhs).max())

def test_sizes_compare_on_common_block():
    w = torch.randn(512, 384)
    s64, s128 = two_sided_sketch(w, 64), two_sided_sketch(w, 128)
    assert np.allclose(s128[:64, :64], s64, atol=1e-5 * np.abs(s64).max())
    assert sketch_cosine(s64, s128) > 0.9999

def test_factored_sketch_matches_dense():
    b, a = torch.randn(300, 8), torch.randn(8, 200)
    d = two_sided_sketch(b @ a)
    assert np.allclose(factored_sketch(b, a), d, atol=1e-4 * np.abs(d).max())

def test_unrelated_sketches_near_orthogonal():
    g = torch.Generator().manual_seed(1)
    a, b = (two_sided_sketch(torch.randn(512, 512, generator=g)) for _ in range(2))
    assert abs(sketch_cosine(a, b)) < 4 / 64

@pytest.mark.parametrize("rows,expected", [
    (4096, (1920, 2176)), (151936, (1920, 2176)), (1024, (384, 640)), (256, (0, 256)), (128, (0, 128))])
def test_row_block_bounds(rows, expected):
    assert row_block_bounds(rows) == expected


def test_head_sketch_cosine_uses_every_shared_column():
    x = np.array([[1.0, 1.0]])
    y = np.array([[1.0, -1.0, 4.0], [2.0, 3.0, 4.0]])
    assert sketch_cosine(x, y) == pytest.approx(0.0)


def test_cuda_sketch_dispatch_keeps_device_tensor(monkeypatch):
    import evidence.sketch as engine
    class DeviceMatrix:
        is_cuda = True
        ndim = 2
        shape = (8, 12)
    matrix = DeviceMatrix()
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    calls = []
    def cuda_sketch(w, k):
        calls.append((w, k))
        return expected
    monkeypatch.setattr(engine, '_cuda_two_sided_sketch', cuda_sketch)
    assert engine.two_sided_sketch(matrix, 4) is expected
    assert calls == [(matrix, 4)]
