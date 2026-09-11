"""Offline tests for the independent, bounded public-LoRA pilot reference."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/adapter_checkpoint_pilot.py"
spec = importlib.util.spec_from_file_location("adapter_checkpoint_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def ordinary_case(transposed=False):
    base = {"projection.weight": torch.arange(12).reshape(4, 3).float(), "other.weight": torch.ones(2, 2)}
    if transposed:
        base["projection.weight"] = base["projection.weight"].T
    adapter = {"base_model.model.projection.lora_A.weight": torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
               "base_model.model.projection.lora_B.weight": torch.arange(8).reshape(4, 2).float()}
    config = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "fan_in_fan_out": transposed}
    return base, adapter, config


def test_downloads_are_pinned_and_small():
    assert len(pilot.SPECS) == 2
    total = 0
    for case in pilot.SPECS.values():
        for role in ("adapter", "base"):
            entry = case[role]
            assert len(entry["revision"]) == 40
            assert int(entry["revision"], 16)
            total += sum(entry["files"].values())
    assert total == 5794073


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("rslora", [False, True])
def test_reference_uses_the_correct_scaled_matrix_product_without_mutating_inputs(transposed, rslora):
    base, adapter, config = ordinary_case(transposed)
    config["use_rslora"] = rslora
    original = {name: tensor.clone() for name, tensor in base.items()}
    expected, updates = pilot.manual_lora_state(base, adapter, config)
    scale = 4 / (2**.5 if rslora else 2)
    product = torch.tensor([[4., 5., 6.], [14., 19., 24.], [24., 33., 42.], [34., 47., 60.]], dtype=torch.float64)
    if transposed:
        product = product.T
    torch.testing.assert_close(expected["projection.weight"], base["projection.weight"].double() + scale * product)
    torch.testing.assert_close(expected["other.weight"], base["other.weight"].double())
    for name in base:
        torch.testing.assert_close(base[name], original[name], rtol=0, atol=0)
    assert updates[0]["scaling"] == scale


@pytest.mark.parametrize("defect", ["missing_b", "surplus", "base_overwrite", "nonfinite", "wrong_rank"])
def test_reference_rejects_malformed_adapter_payloads(defect):
    base, adapter, config = ordinary_case()
    key = "base_model.model.projection.lora_B.weight"
    if defect == "missing_b":
        del adapter[key]
    elif defect == "surplus":
        adapter["base_model.model.unused.lora_A.weight"] = torch.ones(2, 3)
    elif defect == "base_overwrite":
        adapter["base_model.model.projection.base_layer.weight"] = torch.ones(4, 3)
    elif defect == "nonfinite":
        adapter[key][0, 0] = torch.nan
    else:
        config["r"] = 3
    with pytest.raises(ValueError):
        pilot.manual_lora_state(base, adapter, config)


@pytest.mark.parametrize("field,value", [("use_dora", True), ("bias", "all"),
    ("modules_to_save", ["head"]), ("rank_pattern", {"projection": 1})])
def test_reference_does_not_guess_untested_adapter_formulas(field, value):
    base, adapter, config = ordinary_case()
    config[field] = value
    with pytest.raises(ValueError, match="outside"):
        pilot.manual_lora_state(base, adapter, config)


def test_changed_public_manifest_is_rejected_before_weight_download(monkeypatch, tmp_path):
    entry = pilot.SPECS["opt_lora"]["adapter"]
    info = SimpleNamespace(private=False, gated=False, sha=entry["revision"], siblings=[])
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda **kwargs: SimpleNamespace(model_info=lambda *args, **kwargs: info),
        snapshot_download=lambda *args, **kwargs: pytest.fail("Must validate size before downloading")))
    with pytest.raises(ValueError, match="manifest"):
        pilot.download_pinned(entry, tmp_path, offline=False)


def test_existing_output_directory_is_preserved(tmp_path):
    with pytest.raises(SystemExit):
        pilot.main(["--output-dir", str(tmp_path)])


def test_no_op_adapter_cannot_pass_the_nonzero_merge_control():
    updates = [{"module_name": "projection", "max_abs_delta": 0.}]
    assert pilot.validate_control(updates, "zero_delta_preservation") == 0
    with pytest.raises(AssertionError, match="control role"):
        pilot.validate_control(updates, "nonzero_merge")


@pytest.mark.parametrize("record", [{}, {"status": "skipped", "measurement_names": []},
    {"status": "analyzed", "measurement_names": []}])
def test_pilot_cannot_pass_if_the_adapted_module_is_omitted(record):
    updates = [{"module_name": "projection", "max_abs_delta": .1}]
    with pytest.raises(AssertionError, match="coverage"):
        pilot.validate_control(updates, "nonzero_merge", [{"module_name": "projection", **record}])


def test_analyzed_qkv_slices_count_as_coverage_of_the_adapted_module():
    updates = [{"module_name": "projection", "max_abs_delta": .1}]
    records = [{"module_name": "projection", "status": "analyzed",
                "measurement_names": ["projection_q", "projection_k", "projection_v"]}]
    assert pilot.validate_control(updates, "nonzero_merge", records) == 1
