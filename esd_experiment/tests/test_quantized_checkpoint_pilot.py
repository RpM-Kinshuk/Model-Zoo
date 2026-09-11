"""Offline safety controls for the public pre-quantized checkpoint pilot."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/quantized_checkpoint_pilot.py"
spec = importlib.util.spec_from_file_location("quantized_checkpoint_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def metadata(case):
    return SimpleNamespace(private=False, gated=False, sha=case["revision"], siblings=[
        SimpleNamespace(rfilename=name, size=size) for name, size in case["files"].items()
    ])


def config(case):
    return {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "quantization_config": {"quant_method": case["quant_method"],
                                    "load_in_4bit": True, "bnb_4bit_quant_type": case.get("quant_type", "nf4")}}


def test_public_pilot_has_pinned_safe_file_manifests_with_negative_controls():
    assert set(pilot.CASES) == {"gptq", "nf4", "bnb_fp4"}
    assert sum(sum(case["files"].values()) for case in pilot.CASES.values()) == 4554349
    assert all(set(case["files"]) == set(pilot.FILES) for case in pilot.CASES.values())
    assert all(len(case["revision"]) == 40 and int(case["revision"], 16) for case in pilot.CASES.values())
    assert all(set(case["file_sha256"]) == set(pilot.FILES) for case in pilot.CASES.values())
    assert all(len(checksum) == 64 and int(checksum, 16)
               for case in pilot.CASES.values() for checksum in case["file_sha256"].values())


def test_source_provenance_includes_executed_pilot_helpers():
    assert "esd_experiment/scripts/backend_pilot.py" in pilot.source_hashes()


@pytest.mark.parametrize("case", pilot.CASES.values())
def test_verified_metadata_and_declared_quantization_are_accepted(case):
    pilot.validate_metadata(metadata(case), case)
    assert pilot.validate_config(config(case), case)["quant_method"] == case["quant_method"]


@pytest.mark.parametrize("field,value", [("private", True), ("gated", "auto"), ("sha", "b" * 40)])
def test_public_metadata_gate_rejects_private_gated_or_changed_revisions(field, value):
    case = pilot.CASES["nf4"]
    info = metadata(case)
    setattr(info, field, value)
    with pytest.raises(ValueError, match="public"):
        pilot.validate_metadata(info, case)


def test_size_manifest_is_checked_before_weight_download():
    case = pilot.CASES["nf4"]
    info = metadata(case)
    info.siblings[-1].size = 10**9
    with pytest.raises(ValueError, match="manifest"):
        pilot.validate_metadata(info, case)


@pytest.mark.parametrize("mutation", ["remote", "architecture", "unquantized", "fp4"])
def test_remote_code_or_different_representation_is_not_silently_loaded(mutation):
    case = pilot.CASES["nf4"]
    value = config(case)
    if mutation == "remote":
        value["auto_map"] = {"AutoModelForCausalLM": "custom.Model"}
    elif mutation == "architecture":
        value["architectures"] = ["OtherModel"]
    elif mutation == "unquantized":
        value.pop("quantization_config")
    else:
        value["quantization_config"]["bnb_4bit_quant_type"] = "fp4"
    with pytest.raises(ValueError):
        pilot.validate_config(value, case)


@pytest.mark.parametrize("method,quant_type,suffix", [
    ("gptq", "", ".qweight"),
    ("bitsandbytes", "nf4", ".weight.quant_state.bitsandbytes__nf4"),
    ("bitsandbytes", "fp4", ".weight.quant_state.bitsandbytes__fp4"),
])
def test_packed_safe_tensor_keys_preserve_full_quantized_module_names(method, quant_type, suffix):
    names = ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"]
    assert pilot.quantized_module_names([name + suffix for name in names], method, quant_type) == names
    with pytest.raises(ValueError, match="no declared packed"):
        pilot.quantized_module_names(["model.layers.0.self_attn.q_proj.weight"], method, quant_type)


def test_existing_output_directory_is_never_reused(tmp_path):
    with pytest.raises(SystemExit):
        pilot.main(["--output-dir", str(tmp_path)])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("timeout", ["0", "-1", "301", "nan", "inf"])
def test_subprocess_budget_cannot_be_unbounded(tmp_path, timeout):
    destination = tmp_path / "new"
    with pytest.raises(SystemExit):
        pilot.main(["--output-dir", str(destination), "--case-timeout", timeout])
    assert not destination.exists()
