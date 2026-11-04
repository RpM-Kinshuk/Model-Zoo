import numpy as np
import pytest
import torch

from evidence.delta import block_delta, gavish_donoho_rank


def lora_case(r, rho, rows=256, cols=4096, hetero=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = 0.02 * torch.randn(rows, cols, generator=g)
    if hetero:  # lognormal row/col scales plus 4 outlier columns
        w = w * torch.exp(0.5 * torch.randn(rows, 1, generator=g)) * torch.exp(0.5 * torch.randn(1, cols, generator=g))
        w[:, :4] *= 20
    upd = torch.randn(rows, r, generator=g) @ torch.randn(r, cols, generator=g)
    upd = upd * (rho * w.norm() / upd.norm())
    parent = w.bfloat16()
    return (parent.float() + upd).bfloat16(), parent

@pytest.mark.parametrize("r", [8, 64])
def test_rank_recovered_in_working_regime(r):
    out = block_delta(*lora_case(r, 1e-2), torch.bfloat16)
    assert out["status"] == "ok" and out["rank"] == r

def test_heteroscedastic_parent():
    out = block_delta(*lora_case(8, 1e-2, hetero=True), torch.bfloat16)
    assert out["status"] == "ok" and out["rank"] == 8

def test_below_floor_abstains():
    out = block_delta(*lora_case(8, 1e-3), torch.bfloat16)
    assert out["status"] == "below_floor" and out["rank"] == -1 and out["frac_identical"] > 0.5

def test_rank_one_never_confident_unless_exact():
    out = block_delta(*lora_case(1, 1e-2), torch.bfloat16)
    assert out["rank"] == 1 or out["rank_confidence"] == "low"

def test_identical_blocks():
    p = torch.randn(64, 64).bfloat16()
    out = block_delta(p.clone(), p, torch.bfloat16)
    assert out["status"] == "identical" and out["frac_identical"] == 1.0 and out["rel_update_norm"] == 0.0

def test_gd_rule_on_float32_noise():
    g = torch.Generator().manual_seed(0)
    low = torch.randn(256, 3, generator=g) @ torch.randn(3, 2048, generator=g)
    sv = torch.linalg.svdvals(low + 0.01 * torch.randn(256, 2048, generator=g)).numpy()
    assert gavish_donoho_rank(sv, 256, 2048) == 3


@pytest.mark.parametrize("rho", [0, 1e-3, 1e-2])
def test_explicit_cpu_delta_matches_default_exactly(rho):
    child, parent = lora_case(8, rho, rows=32, cols=96)
    expected = block_delta(child, parent, torch.bfloat16)
    actual = block_delta(child, parent, torch.bfloat16, compute_device="cpu")
    for name in expected:
        if name == "top_singular_values":
            np.testing.assert_array_equal(actual[name], expected[name])
        else:
            assert actual[name] == expected[name]


def test_cuda_routes_only_exact_float64_svd_inputs_and_preserves_results(monkeypatch):
    import evidence.delta as engine
    child, parent = lora_case(8, 1e-2, rows=32, cols=96, hetero=True)
    original_svd = torch.linalg.svdvals
    original_to = torch.Tensor.to
    cpu_inputs, cuda_inputs = [], []

    def cpu_svd(matrix, **kwargs):
        assert not kwargs
        cpu_inputs.append(matrix.clone())
        return original_svd(matrix)

    monkeypatch.setattr(torch.linalg, "svdvals", cpu_svd)
    expected = block_delta(child, parent, torch.bfloat16)
    assert len(cpu_inputs) == 2

    class SimulatedCuda:
        is_cuda = True
        def __init__(self, matrix):
            self.matrix = matrix

    def to(matrix, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "cuda":
            assert torch.device(device) == torch.device("cuda:1")
            assert matrix.dtype == torch.float64 and matrix.device.type == "cpu"
            return SimulatedCuda(matrix)
        return original_to(matrix, *args, **kwargs)

    def cuda_svd(matrix, **kwargs):
        assert isinstance(matrix, SimulatedCuda)
        assert kwargs == {"driver": "gesvd"}
        cuda_inputs.append(matrix.matrix.clone())
        return original_svd(matrix.matrix)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.linalg, "svdvals", cuda_svd)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    # Keep the upload boundary local in this offline test. Real CUDA parity is
    # checked separately; the SVD routing and exact whitening assertions remain.
    monkeypatch.setattr(engine, "_delta_stored", lambda tensor, dtype, device:
                        original_to(tensor.detach(), device='cpu', dtype=dtype).contiguous())
    actual = block_delta(child, parent, torch.bfloat16, compute_device="cuda:1")
    assert len(cuda_inputs) == 2
    for cpu, cuda in zip(cpu_inputs, cuda_inputs):
        assert torch.equal(cpu, cuda)
    for name in expected:
        if name == "top_singular_values":
            np.testing.assert_array_equal(actual[name], expected[name])
        else:
            assert actual[name] == expected[name]


@pytest.mark.parametrize("device", ["bogus", "mps", "cpu:1", "cuda:-1", "cuda:3"])
def test_invalid_delta_device_rejected_even_for_identical_inputs(device, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    parent = torch.ones(16, 16)
    with pytest.raises(ValueError, match="device|CUDA"):
        block_delta(parent, parent, torch.float32, compute_device=device)


def test_unavailable_delta_cuda_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parent = torch.ones(16, 16)
    with pytest.raises(ValueError, match="CUDA"):
        block_delta(parent, parent, torch.float32, compute_device="cuda:0")


@pytest.mark.parametrize('rho', [0, 1e-3, 1e-2])
def test_cuda_prepares_stored_inputs_on_requested_device(monkeypatch, rho):
    import evidence.delta as engine
    child, parent = lora_case(8, rho, rows=32, cols=96, hetero=True)
    expected = block_delta(child, parent, torch.bfloat16)
    staged = []
    def staged_on_cpu(tensor, dtype, device):
        staged.append((tensor, dtype, str(device)))
        return tensor.detach().to(device='cpu', dtype=dtype).contiguous()
    monkeypatch.setattr(engine, '_delta_stored', staged_on_cpu)
    monkeypatch.setattr(engine, '_compute_device', lambda value: torch.device('cuda:1'))
    # Simulate only transfer/SVD boundaries; arithmetic and storage rounding are real.
    monkeypatch.setattr(engine, '_delta_svdvals', lambda matrix, device: torch.linalg.svdvals(matrix).numpy())
    actual = block_delta(child, parent, torch.bfloat16, compute_device='cuda:1')
    assert len(staged) == 2
    assert [x[0] is y for x, y in zip(staged, [child, parent])] == [True, True]
    assert all(dtype == torch.bfloat16 and device == 'cuda:1' for _, dtype, device in staged)
    for key in expected:
        if key == 'top_singular_values':
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]
