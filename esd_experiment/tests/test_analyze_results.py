"""Small result pairs exercise indexing without loading models or spectra."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))
from measurement_config import FORMAT_VERSION, NUMERICS_VERSION, measurement_config

spec = importlib.util.spec_from_file_location("result_summary_test", EXPERIMENT / "utils/analyze_results.py")
analyze_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_results)


def write_pair(run_dir, stem="org--model", *, coverage=True, loading_info=None, **settings):
    """Match the worker's two-file schema; /alpha is deliberately unavailable."""
    csv_path = run_dir / "stats" / f"{stem}.csv"
    h5_path = run_dir / "metrics" / f"{stem}.h5"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    model_id = stem.replace("--", "/")
    config = measurement_config(SimpleNamespace(), model_id=model_id)
    config.update(settings)
    config["runtime"] = {
        "model_class": "example.EncoderDecoder",
        "model_config_commit_hash": "a" * 40,
        "loading_info": loading_info,
    }
    frame = pd.DataFrame({
        "model_id": [model_id] * 3,
        "longname": ["encoder.layers.0.q_proj", "decoder.layers.0.q_proj", "classifier"],
        "module_name": ["encoder.layers.0.q_proj", "decoder.layers.0.q_proj", "classifier"],
        "fit_status": ["fitted", "constant_spectrum", "fitted"],
        "alpha": [2., np.nan, 4.],
        "alpha_weighted": [1., np.nan, 3.],
        "entropy": [0.5, 1., 0.75],
        "params": [10, 20, 30],
    })
    frame.to_csv(csv_path, index=False)
    with h5py.File(h5_path, "w") as h5:
        h5.attrs.update(full_name=model_id, source_model=config.get("source_model", ""),
                        base_model_relation=config.get("base_model_relation") or "base",
                        format_version=FORMAT_VERSION, numerics_version=NUMERICS_VERSION,
                        measurement_config_json=json.dumps(config))
        h5.create_dataset("alpha", data=np.empty((0, 0))).attrs["view_status"] = "unavailable"
        for name in frame.columns.drop("model_id"):
            values = frame[name].to_numpy()
            if values.dtype.kind == "O":
                h5.create_dataset(f"layers/{name}", data=values, dtype=h5py.string_dtype())
            else:
                h5.create_dataset(f"layers/{name}", data=values)
        h5.create_dataset("eigs", (3,), dtype=h5py.vlen_dtype("float32"))
        if coverage:
            counts = dict(candidate_modules=4, eligible_modules=3, analyzed_modules=3,
                          partially_analyzed_modules=0, skipped_modules=1,
                          analyzed_measurements=3, fitted_measurements=2)
            h5.create_dataset("coverage", data=json.dumps({"counts": counts, "modules": []}))
            for name, count in counts.items():
                h5.attrs[f"coverage_{name}"] = count
    return csv_path, h5_path


def test_reader_uses_all_canonical_records_without_reading_spectra(tmp_path, monkeypatch):
    csv_path, h5_path = write_pair(tmp_path)
    original_getitem = h5py.Dataset.__getitem__

    def read_scalar_columns_only(dataset, key):
        assert dataset.name not in ("/alpha", "/eigs"), f"Read unnecessary payload: {dataset.name}"
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read_scalar_columns_only)
    summary = analyze_results.read_model_summary(csv_path, h5_path)
    assert summary["analyzed_measurements"] == 3
    assert summary["measured_modules"] == 3
    assert summary["fitted_measurements"] == 2
    assert summary["missing_fit_measurements"] == 1
    assert summary["alpha_mean"] == 3
    assert summary["coverage_status"] == "recorded"
    assert summary["candidate_modules"] == 4
    assert summary["skipped_modules"] == 1
    assert "total_params" not in summary
    assert "num_layers" not in summary
    assert Path(summary["csv_path"]) == csv_path
    assert Path(summary["h5_path"]) == h5_path
    for name in ("base_repo_loaded", "base_revision_loaded", "base_resolved_commit_hash"):
        assert summary[name] is None


def test_summary_distinguishes_qkv_measurements_and_valid_fits():
    frame = pd.DataFrame({
        "longname": ["attention_q", "attention_k", "attention_v", "head", "other", "last"],
        "module_name": ["attention"] * 3 + ["head", "other", "last"],
        "fit_status": ["fitted", "fitted", "constant_spectrum", "fitted", "fitted", "fitted"],
        "alpha": [2., 4., 100., np.inf, 1., np.nan],
        "alpha_weighted": [3., 5., 100., 100., 100., 100.],
        "entropy": [0.5] * 6,
        "params": [10] * 6,
    })
    summary = analyze_results.compute_model_summary(frame)
    assert summary["analyzed_measurements"] == 6
    assert summary["measured_modules"] == 4
    assert summary["fitted_measurements"] == 2
    assert summary["missing_fit_measurements"] == 4
    assert summary["alpha_mean"] == summary["alpha_median"] == 3
    assert summary["alpha_std"] == pytest.approx(np.sqrt(2))
    assert summary["alpha_min"] == 2 and summary["alpha_max"] == 4
    assert summary["alpha_q25"] == 2.5 and summary["alpha_q75"] == 3.5
    assert summary["alpha_weighted_mean"] == 4
    assert summary["entropy_mean"] == 0.5
    assert "total_params" not in summary and "num_layers" not in summary


def test_absent_coverage_is_unknown_not_complete(tmp_path):
    summary = analyze_results.read_model_summary(*write_pair(tmp_path, coverage=False))
    assert summary["coverage_status"] == "unknown"
    for name in ("candidate_modules", "analyzed_modules", "partially_analyzed_modules", "skipped_modules"):
        assert summary[name] is None


def test_no_fitted_measurements_remain_in_summary():
    summary = analyze_results.compute_model_summary(pd.DataFrame({
        "longname": ["classifier"], "module_name": ["classifier"],
        "fit_status": ["constant_spectrum"], "alpha": [np.nan],
    }))
    assert summary["analyzed_measurements"] == summary["missing_fit_measurements"] == 1
    assert summary["fitted_measurements"] == 0
    assert np.isnan(summary["alpha_mean"])


def test_reader_preserves_requested_identity_and_actual_config_provenance(tmp_path):
    loading_info = {"base_repo_loaded": "org/dense-base", "base_revision_loaded": "dense-v1",
                    "base_resolved_commit_hash": "a" * 40}
    summary = analyze_results.read_model_summary(*write_pair(
        tmp_path, requested_revision="b" * 40, source_model="org/base@" + "c" * 40,
        base_model_relation="adapter", compute_dtype="float64", evals_thresh=0.001,
        loading_info=loading_info))
    assert summary["model_id"] == summary["repo_id"] == "org/model"
    assert summary["requested_revision"] == "b" * 40
    assert summary["source_model"] == "org/base@" + "c" * 40
    assert summary["base_model_relation"] == "adapter"
    assert summary["model_config_commit_hash"] == "a" * 40
    assert summary["model_class"] == "example.EncoderDecoder"
    assert summary["compute_dtype"] == "float64"
    assert summary["evals_thresh"] == 0.001
    for name, value in loading_info.items():
        assert summary[name] == value


@pytest.mark.parametrize("mutation", ["missing_csv", "old_version", "wrong_alpha", "wrong_coverage"])
def test_invalid_pairs_are_not_silently_read_as_csv(tmp_path, mutation):
    csv_path, h5_path = write_pair(tmp_path)
    if mutation == "missing_csv":
        csv_path.unlink()
    elif mutation == "wrong_alpha":
        frame = pd.read_csv(csv_path)
        frame.loc[0, "alpha"] = 99.
        frame.to_csv(csv_path, index=False)
    else:
        with h5py.File(h5_path, "a") as h5:
            if mutation == "old_version":
                h5.attrs["numerics_version"] = "old"
            else:
                report = json.loads(h5["coverage"][()])
                report["counts"]["fitted_measurements"] = 3
                del h5["coverage"]
                h5.create_dataset("coverage", data=json.dumps(report))
                h5.attrs["coverage_fitted_measurements"] = 3
    with pytest.raises(ValueError):
        analyze_results.read_model_summary(csv_path, h5_path)


def test_main_indexes_valid_and_missing_pairs_and_can_be_rerun(tmp_path):
    write_pair(tmp_path)
    missing_csv, _ = write_pair(tmp_path, "org--missing")
    missing_csv.unlink()
    for _ in range(2):
        assert analyze_results.main(["--results_dir", str(tmp_path)]) == 1
        summary = pd.read_csv(tmp_path / "summary.csv")
        assert len(summary) == 2
        assert set(summary["artifact_status"]) == {"valid", "invalid"}
        invalid = summary.loc[summary["artifact_status"] == "invalid"].iloc[0]
        assert isinstance(invalid["artifact_error"], str) and invalid["artifact_error"]


@pytest.mark.parametrize("output", ["stats/org--model.csv", "metrics/index.csv", "summary.h5"])
def test_output_cannot_overwrite_input_artifacts(tmp_path, output):
    csv_path, h5_path = write_pair(tmp_path)
    original = csv_path.read_bytes(), h5_path.read_bytes()
    try:
        result = analyze_results.main(["--results_dir", str(tmp_path), "--output", str(tmp_path / output)])
    except SystemExit as exc:
        result = exc.code
    assert result != 0
    assert (csv_path.read_bytes(), h5_path.read_bytes()) == original


def test_failed_summary_write_preserves_previous_output(tmp_path, monkeypatch):
    output_path = tmp_path / "summary.csv"
    analyze_results.write_summary(pd.DataFrame({"model_id": ["previous"]}), output_path)
    original = output_path.read_bytes()

    def fail_after_partial_write(frame, handle, **kwargs):
        handle.write("incomplete CSV")
        raise OSError("simulated disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated disk full"):
        analyze_results.write_summary(pd.DataFrame({"model_id": ["new"]}), output_path)
    assert output_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output_path]


@pytest.mark.parametrize("mixed", [False, True])
def test_mixed_measurement_warning_ignores_identity_differences(tmp_path, capsys, recwarn, mixed):
    write_pair(tmp_path)
    write_pair(tmp_path, "different--model", requested_revision="other",
               fix_fingers="DKS" if mixed else "xmin_mid")
    assert analyze_results.main(["--results_dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    messages = [str(warning.message).lower() for warning in recwarn]
    assert any("mixed" in message and "settings" in message for message in messages) is mixed
    assert ("Mean of model fitted-alpha means:" in captured.out) is not mixed
