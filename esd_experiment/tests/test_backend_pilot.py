"""Small offline controls for the matrix-scale pilot; never require CUDA."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/backend_pilot.py"
spec = importlib.util.spec_from_file_location("backend_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_default_cases_are_bounded_and_include_realistic_dimensions():
    shapes = {entry["shape"] for entry in pilot.CASES.values()}
    assert {(768, 768), (3072, 768), (4096, 256), (2048, 2048)} <= shapes
    assert max(rows * cols for rows, cols in shapes) == 2048**2
    assert pilot.SPECTRUM_TOLERANCE == 1e-4


@pytest.mark.parametrize("device", ["cuda:0", "cuda:2", "cuda", "gpu", "cuda:-1"])
def test_requested_cuda_never_falls_back_to_cpu(device):
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
    with pytest.raises(ValueError):
        pilot.validate_device(device, fake)
    assert pilot.validate_device("cpu", fake) is None


def test_cuda_device_index_is_checked():
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
    assert pilot.validate_device("cuda:0", fake) == 0
    with pytest.raises(ValueError, match="unavailable"):
        pilot.validate_device("cuda:1", fake)


@pytest.mark.parametrize("name", ["low_rank", "ill_conditioned", "near_constant", "filter_boundary"])
def test_source_weights_are_repeatable_float32(name):
    first, second = pilot.build_matrix(name), pilot.build_matrix(name)
    assert first.dtype == np.float32
    assert first.shape == pilot.CASES[name]["shape"]
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("cutoff", ["xmin_mid", None, "xmin_peak"])
def test_independent_reference_includes_all_cutoff_ties(cutoff):
    eigs = np.array([1., 2., 2., 2., 4., 8.])
    result = pilot.reference_fit(eigs, cutoff)
    assert result["n_tail"] == sum(eigs >= result["fit_xmin"])
    if cutoff == "xmin_mid":
        assert result["fit_xmin"] == 2
        assert result["n_tail"] == 5


def test_rank_reference_applies_the_requested_precision_convention():
    eigs = np.array([1e-20, 1., 1.])
    single = pilot.reference_metrics(eigs, (8, 3), "float32", False, "xmin_mid")
    double = pilot.reference_metrics(eigs, (8, 3), "float64", False, "xmin_mid")
    assert single["matrix_rank"] == 2
    assert double["matrix_rank"] == 3
    assert single["entropy"] == pytest.approx(1)
    assert double["entropy"] == pytest.approx(math.log(2) / math.log(3))


def test_small_cpu_case_uses_independent_reference_and_reports_sensitivity():
    result = pilot.run_case("filter_boundary", "cpu", torch, None)
    assert result["status"] == "sensitivity_control_completed"
    assert "scipy.linalg.svdvals" in result["reference"]["method"]
    variants = result["variants"]
    assert variants["float32_svd"]["num_evals"] == 1
    assert variants["unfiltered_svd"]["num_evals"] == 4
    assert "fit_availability_changed" in variants["unfiltered_svd"]["baseline_sensitivity_flags"]
    for row in variants.values():
        assert row["spectrum_check_passed"] is None
        assert row["peak_gpu_allocated_bytes"] is None
        assert row["spectrum_max_abs_error_over_reference_lambda_max"] < 1e-6
    json.dumps(pilot.json_safe(result), allow_nan=False)


def test_well_conditioned_small_case_has_predeclared_spectrum_gate(monkeypatch):
    monkeypatch.setitem(pilot.CASES, "tiny", {"shape": (8, 8), "kind": "shifted_gaussian", "well_conditioned": True})
    result = pilot.run_case("tiny", "cpu", torch, None)
    assert result["status"] == "passed_spectrum_checks"
    assert all(row["spectrum_check_passed"] for row in result["variants"].values())
    assert all(row["lambda_max_relative_error"] < pilot.SPECTRUM_TOLERANCE for row in result["variants"].values())


def test_pilot_rejects_existing_directory_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "configure_runtime", lambda *args: pytest.fail("Unexpected numerical work"))
    with pytest.raises(SystemExit):
        pilot.main(["--output-dir", str(tmp_path)])


def test_output_is_strict_json_and_existing_reports_are_preserved(tmp_path):
    output = tmp_path / "report.json"
    pilot.write_report(output, {"missing": math.nan, "infinite": np.float64(math.inf)})
    assert json.loads(output.read_text()) == {"missing": None, "infinite": None}
    with pytest.raises(FileExistsError):
        pilot.write_report(output, {"replacement": True})
    assert json.loads(output.read_text()) == {"missing": None, "infinite": None}


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeouts_are_rejected_without_importing_backend(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(pilot, "configure_runtime", lambda *args: pytest.fail("Unexpected numerical work"))
    with pytest.raises(SystemExit):
        pilot.main(["--output-dir", str(tmp_path / "fresh"), "--case-timeout", timeout])


def test_case_timeout_is_recorded_and_returns_failure(tmp_path, monkeypatch):
    output = tmp_path / "fresh"
    monkeypatch.setattr(pilot, "configure_runtime", lambda device: (torch, None))
    monkeypatch.setattr(pilot.platform, "platform", lambda: "test-platform")

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 2
        assert kwargs["check"] is True
        raise pilot.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(pilot.subprocess, "run", timeout)
    assert pilot.main(["--output-dir", str(output), "--cases", "near_constant", "--case-timeout", "2"]) == 1
    report = json.loads((output / "backend_report.json").read_text())
    assert report["passed"] is False
    assert report["cases"] == []
    assert report["errors"][0]["case"] == "near_constant"
    assert report["errors"][0]["type"] == "TimeoutExpired"
