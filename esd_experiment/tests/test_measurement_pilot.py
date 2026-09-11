"""Offline safety and numerical-reference checks for the bounded HF pilot."""

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/measurement_pilot.py"
spec = importlib.util.spec_from_file_location("measurement_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_manifest_is_pinned_and_below_the_download_budget():
    assert len(pilot.SPECS) == 4
    assert len({entry[0] for entry in pilot.SPECS}) == 4
    assert all(len(entry[2]) == 40 and int(entry[2], 16) for entry in pilot.SPECS)
    assert sum(entry[-1] for entry in pilot.SPECS) == 7495424
    assert pilot.FILES == ["config.json", "pytorch_model.bin"]


def test_json_missing_values_are_standard_null():
    result = pilot.json_safe({"nan": math.nan, "inf": np.float64(math.inf), "keys": {"b", "a"}})
    assert json.dumps(result, allow_nan=False) == '{"nan": null, "inf": null, "keys": ["a", "b"]}'


@pytest.mark.parametrize("remote_config", [False, True])
def test_offline_download_is_local_and_rejects_remote_code_configs(tmp_path, monkeypatch, remote_config):
    config = {"auto_map": {"AutoModel": "custom.Model"}} if remote_config else {"model_type": "bert"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "pytorch_model.bin").write_bytes(b"test")
    calls = []

    def snapshot_download(repo, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    def forbidden_api(**kwargs):
        raise AssertionError("Offline pilot must not fetch metadata")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=forbidden_api, snapshot_download=snapshot_download,
    ))
    entry = ("encoder", "test/public", "a" * 40, "BertModel",
             sum((tmp_path / name).stat().st_size for name in pilot.FILES))
    if remote_config:
        with pytest.raises(ValueError, match="Remote-code"):
            pilot.download_snapshot(entry, tmp_path, offline=True)
    else:
        assert pilot.download_snapshot(entry, tmp_path, offline=True) == tmp_path
    assert calls[0]["local_files_only"] is True
    assert calls[0]["token"] is False
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["allow_patterns"] == pilot.FILES


def test_online_metadata_rejects_a_larger_manifest_before_downloading(monkeypatch, tmp_path):
    info = SimpleNamespace(
        private=False, gated=False, sha="a" * 40,
        siblings=[SimpleNamespace(rfilename=name, size=10**9) for name in pilot.FILES],
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda **kwargs: SimpleNamespace(model_info=lambda *args, **kwargs: info),
        snapshot_download=lambda *args, **kwargs: pytest.fail("Unexpected download"),
    ))
    with pytest.raises(ValueError, match="manifest"):
        pilot.download_snapshot(("encoder", "test/public", "a" * 40, "BertModel", 100), tmp_path)


@pytest.mark.parametrize("shape", [(4, 3), (4, 3, 2), (4, 3, 2, 2), (4, 3, 2, 2, 2)])
def test_weightwatcher_reference_uses_explicit_float64_slices(shape):
    weight = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    original = weight.clone()

    def svd_vals(array, method):
        assert array.dtype == np.float64
        assert array.ndim == 2
        assert method == "accurate"
        return np.linalg.svd(array, compute_uv=False)

    eigs = pilot.reference_spectrum(weight, svd_vals)
    assert eigs.sum() == pytest.approx(weight.double().square().sum().item() * (1 if len(shape) == 2 else .5))
    torch.testing.assert_close(weight, original, rtol=0, atol=0)


def test_pilot_refuses_to_overwrite_an_existing_report(tmp_path):
    existing = tmp_path / "pilot_report.json"
    existing.write_text("existing report")
    with pytest.raises(SystemExit):
        pilot.main(["--output-dir", str(tmp_path)])
    assert existing.read_text() == "existing report"
