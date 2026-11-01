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
