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


def write_pair(run_dir, stem="org--model", *, coverage=True, loading_info=None, weight_usage=None, **settings):
    """Match the worker's two-file schema; /alpha is deliberately unavailable."""
    csv_path = run_dir / "stats" / f"{stem}.csv"
    h5_path = run_dir / "metrics" / f"{stem}.h5"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    model_id = stem.replace("--", "/")
    config = measurement_config(SimpleNamespace(), model_id=model_id)
    config.update(settings)
    config["runtime"] = {
        "analysis_source": "model", "filter_type": config["filter_type"],
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
            report = {"counts": counts, "modules": []}
            if weight_usage is not None:
                report["weight_usage"] = weight_usage
            h5.create_dataset("coverage", data=json.dumps(report))
            for name, count in counts.items():
                h5.attrs[f"coverage_{name}"] = count
    return csv_path, h5_path


def test_auto_results_from_different_sources_are_not_pooled(tmp_path):
    write_pair(tmp_path, "org--loaded")
    _, h5_path = write_pair(tmp_path, "org--tensors")
    with h5py.File(h5_path, "a") as h5:
        config = json.loads(h5.attrs["measurement_config_json"])
        config["runtime"].update(analysis_source="checkpoint", filter_type=False, model_class=None,
                                 fallback={"stage": "load", "reason": "checkpoint_config_unsupported"})
        h5.attrs["measurement_config_json"] = json.dumps(config)
        h5["layers/module_name"][:] = ["", "", ""]
        h5["coverage"][()] = json.dumps({"scope": "checkpoint_tensors", "counts": {
            "stored_tensors": 3, "analyzed_tensors": 3, "skipped_tensors": 0,
            "analyzed_measurements": 3, "fitted_measurements": 2,
        }})
    with pytest.warns(UserWarning, match="Mixed measurement settings"):
        assert analyze_results.main(["--results_dir", str(tmp_path)]) == 0
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary["analysis_policy"].unique().tolist() == ["auto"]
    assert set(summary["analysis_source"]) == {"model", "checkpoint"}



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


def test_summary_distinguishes_measured_modules_and_valid_fits():
    frame = pd.DataFrame({
        "longname": ["attention.q_proj", "attention.k_proj", "attention.v_proj", "head", "other", "last"],
        "module_name": ["attention.q_proj", "attention.k_proj", "attention.v_proj", "head", "other", "last"],
        "fit_status": ["fitted", "fitted", "constant_spectrum", "fitted", "fitted", "fitted"],
        "alpha": [2., 4., 100., np.inf, 1., np.nan],
        "alpha_weighted": [3., 5., 100., 100., 100., 100.],
        "entropy": [0.5] * 6,
        "params": [10] * 6,
    })
    summary = analyze_results.compute_model_summary(frame)
    assert summary["analyzed_measurements"] == 6
    assert summary["measured_modules"] == 6
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


@pytest.mark.parametrize("coverage", [False, True])
def test_absent_weight_usage_is_unknown_even_when_module_coverage_exists(tmp_path, coverage):
    summary = analyze_results.read_model_summary(*write_pair(tmp_path, coverage=coverage))
    assert summary["weight_usage_status"] == "unknown"
    assert all(summary[name] is None for name in analyze_results.WEIGHT_COUNTS)


def test_summary_records_loaded_tensor_counts(tmp_path):
    counts = dict(registered_tensors=8, measured_tensors=3, skipped_tensors=1,
                  not_applicable_tensors=3, unresolved_tensors=1, shared_tensors=2,
                  unmapped_measurements=0)
    usage = {"scope": "loaded_registered_tensors", "counts": counts}

    summary = analyze_results.read_model_summary(*write_pair(tmp_path, weight_usage=usage))

    assert summary["weight_usage_status"] == "recorded"
    assert all(summary[name] == count for name, count in counts.items())


def test_weight_usage_survives_real_spectra_writer_and_summary_round_trip(tmp_path):
    import torch
    from net_esd import net_esd_estimator
    from net_esd.utils import weight_usage_report
    from test_worker import load_worker_module

    layer = torch.nn.Linear(4, 4, bias=False)
    layer.extra = torch.nn.Parameter(torch.eye(4))
    model = torch.nn.ModuleDict({"first": layer, "shared": layer})
    modules = []
    metrics = net_esd_estimator(model, parallel=False, coverage=modules)
    worker = load_worker_module()
    coverage = worker.coverage_report(modules, metrics)
    coverage["weight_usage"] = weight_usage_report(model, modules, metrics["longname"])
    csv_path, h5_path = tmp_path / "model.csv", tmp_path / "model.h5"
    config = measurement_config(SimpleNamespace(), model_id="org/model", revision="a" * 40)
    config["runtime"] = {"analysis_source": "model", "filter_type": config["filter_type"]}

    worker.save_results(metrics, csv_path, "org/model", False, h5_output_path=h5_path,
                        save_eigs=True, measurement_config=config, coverage=coverage)
    summary = analyze_results.read_model_summary(csv_path, h5_path)

    assert summary["weight_usage_status"] == "recorded"
    assert summary["registered_tensors"] == summary["shared_tensors"] == 2
    assert summary["measured_tensors"] == summary["unresolved_tensors"] == 1
    assert summary["unmapped_measurements"] == 0
    with h5py.File(h5_path) as h5:
        assert json.loads(h5["coverage"].asstr()[()]) == coverage
        np.testing.assert_array_equal(h5["eigs"][0], metrics["eigs"][0])


@pytest.mark.parametrize("defect", [None, "shape", "no_spectrum"])
def test_attention_records_and_partial_module_counts_survive_round_trip(tmp_path, monkeypatch, defect):
    import torch
    import net_esd
    from net_esd.utils import weight_usage_report
    from test_worker import load_worker_module

    attention = torch.nn.MultiheadAttention(8, 2, kdim=4, vdim=6)
    model = torch.nn.ModuleDict({"attention": attention, "shared_attention": attention})
    if defect == "shape":
        attention.k_proj_weight = torch.nn.Parameter(torch.ones(8, 5))
    elif defect == "no_spectrum":
        compute = net_esd.compute_esd_for_weight
        monkeypatch.setattr(net_esd, "compute_esd_for_weight", lambda name, *args:
                            None if name == "attention.k_proj_weight" else compute(name, *args))
    weights = []
    metrics = net_esd.net_esd_estimator(model, parallel=False, coverage=weights)
    worker = load_worker_module()
    coverage = worker.coverage_report(weights, metrics)
    usage = weight_usage_report(model, weights, metrics["longname"])
    coverage["weight_usage"] = usage
    paths = tmp_path / "model.csv", tmp_path / "model.h5"
    config = measurement_config(SimpleNamespace(), model_id="org/model", revision="a" * 40)
    config["runtime"] = {"analysis_source": "model", "filter_type": config["filter_type"]}

    worker.save_results(metrics, paths[0], "org/model", False, h5_output_path=paths[1],
                        save_eigs=True, measurement_config=config, coverage=coverage)
    summary = analyze_results.read_model_summary(*paths)

    assert summary["candidate_modules"] == summary["measured_modules"] == 2
    assert summary["analyzed_modules"] == (2 if defect is None else 1)
    assert summary["partially_analyzed_modules"] == (0 if defect is None else 1)
    assert summary["analyzed_measurements"] == (4 if defect is None else 3)
    assert summary["skipped_modules"] == 0
    linked = next(record for record in usage["tensors"] if record["name"] == "attention.q_proj_weight")
    assert linked["aliases"] == ["shared_attention.q_proj_weight"]
    assert linked["measurement_names"] == ["attention.q_proj_weight"]
    key = next(record for record in usage["tensors"] if record["name"] == "attention.k_proj_weight")
    assert key["measurement_names"] == ([] if defect else ["attention.k_proj_weight"])
    assert key["status"] == ("skipped" if defect else "measured")
    with h5py.File(paths[1]) as h5:
        names = h5["layers/longname"].asstr()[:].tolist()
        assert names == metrics["longname"]
        assert h5["layers/weight_attribute"].asstr()[:].tolist() == metrics["weight_attribute"]
        assert len(set(zip(metrics["module_name"], metrics["weight_attribute"], metrics["slice"]))) == len(names)
        for index, spectrum in enumerate(metrics["eigs"]):
            np.testing.assert_array_equal(h5["eigs"][index], spectrum)


@pytest.mark.parametrize("field,value", [("registered_tensors", 9), ("measured_tensors", -1),
                                        ("skipped_tensors", True), ("shared_tensors", 10),
                                        ("unmapped_measurements", 4)])
def test_invalid_weight_usage_counts_are_rejected(tmp_path, field, value):
    counts = dict(registered_tensors=8, measured_tensors=3, skipped_tensors=1,
                  not_applicable_tensors=3, unresolved_tensors=1, shared_tensors=2,
                  unmapped_measurements=0)
    counts[field] = value
    usage = {"scope": "loaded_registered_tensors", "counts": counts}

    with pytest.raises(ValueError, match="[Ww]eight-usage"):
        analyze_results.read_model_summary(*write_pair(tmp_path, weight_usage=usage))


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


@pytest.mark.parametrize("output", ["stats/org--model.csv", "metrics/index.csv", "summary.h5",
                                  "models.csv", "logs/summary.csv"])
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


def test_summary_joins_selected_models_with_artifacts_and_last_recorded_outcomes(tmp_path):
    write_pair(tmp_path, "org--success", requested_revision="a" * 40)
    write_pair(tmp_path, "org--extra", requested_revision="a" * 40)
    models = [dict(model_id=f"org/{name}", revision_norm="a" * 40, family="encoder")
              for name in ("success", "failed", "blocked", "unrecorded", "lost")]
    models.append(dict(model_id="org/unpinned", pin_status="error", pin_error="revision unavailable"))
    pd.DataFrame(models).fillna("").to_csv(tmp_path / "models.csv", index=False)
    terminal_dir = tmp_path / "logs/terminal_status"
    terminal_dir.mkdir(parents=True)
    for name, status in (("success", "failed"), ("failed", "failed"), ("blocked", "blocked"), ("lost", "success")):
        record = dict(model_id=f"org/{name}", status=status, reason="example", stage="load")
        if name == "blocked":
            record.update(revision="a" * 40, source_model="", stage="preflight")
        (terminal_dir / f"org--{name}.json").write_text(json.dumps(record))

    assert analyze_results.main(["--results_dir", str(tmp_path)]) == 1  # Lost output is incomplete.
    summary = pd.read_csv(tmp_path / "summary.csv", keep_default_na=False).set_index("model_id")
    assert summary["outcome"].to_dict() == {
        "org/success": "success", "org/extra": "success", "org/failed": "failed",
        "org/blocked": "blocked", "org/unrecorded": "unrecorded", "org/lost": "incomplete",
        "org/unpinned": "blocked",
    }
    assert summary["selected"].sum() == 6
    assert not summary.loc["org/extra", "selected"]
    assert summary.loc["org/failed", "input_family"] == "encoder"
    assert summary.loc["org/failed", "outcome_pin_status"] == "unknown"
    assert summary.loc["org/failed", "analyzed_measurements"] == ""
    assert summary.loc["org/blocked", "outcome_pin_status"] == "matched"
    assert summary.loc["org/unpinned", "outcome_reason"] == "pin_error"
    assert summary.loc["org/success", "outcome_pin_status"] == "matched"


@pytest.mark.parametrize("evidence", ["artifact", "terminal"])
@pytest.mark.parametrize("changed_pin", ["revision", "base"])
def test_other_checkpoint_pins_do_not_count_as_selected_outcomes(tmp_path, evidence, changed_pin):
    selected = dict(model_id="org/model", revision_norm="a" * 40, source_model="org/base@" + "b" * 40)
    revision = "c" * 40 if changed_pin == "revision" else selected["revision_norm"]
    source = "org/base@" + "d" * 40 if changed_pin == "base" else selected["source_model"]
    if evidence == "artifact":
        write_pair(tmp_path, requested_revision=revision, source_model=source)
        with pytest.warns(UserWarning, match="measurement setting differs"):
            row = analyze_results.summarize_outcome(tmp_path, "org--model", selected)
        assert row["outcome"] == "incomplete"
        assert row["artifact_status"] == "invalid"
    else:
        path = tmp_path / "logs/terminal_status/org--model.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(model_id="org/model", status="failed", revision=revision, source_model=source)))
        row = analyze_results.summarize_outcome(tmp_path, "org--model", selected)
        assert row["outcome"] == "unrecorded"
        assert row["outcome_pin_status"] == "mismatch"


def test_status_only_run_exposes_corrupt_record_without_needing_artifact_directories(tmp_path):
    path = tmp_path / "logs/terminal_status/org--model.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    assert analyze_results.main(["--results_dir", str(tmp_path)]) == 1
    row = pd.read_csv(tmp_path / "summary.csv").iloc[0]
    assert row["outcome"] == "unrecorded"
    assert row["terminal_error"]
    assert row["terminal_status_path"] == str(path)


def test_external_model_list_is_checked_before_replacing_summary(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    models = tmp_path / "selected.csv"
    models.write_text("model_id,revision_norm\norg/model," + "a" * 40 + "\n")
    args = ["--results_dir", str(run_dir), "--model_list", str(models)]
    assert analyze_results.main(args) == 0  # Preparation-only runs are useful too.
    output = run_dir / "summary.csv"
    original = output.read_bytes()
    models.write_text(models.read_text() + "org/model," + "a" * 40 + "\n")
    assert analyze_results.main(args) == 1
    assert output.read_bytes() == original
    original_models = models.read_bytes()
    assert analyze_results.main(args + ["--output", str(models)]) == 1
    assert models.read_bytes() == original_models
