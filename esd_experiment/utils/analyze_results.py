#!/usr/bin/env python3
"""Summarize selected models, recorded outcomes and canonical layer scalars.

summary.csv is a rebuildable index, not completion evidence or a live monitor.
Eigenvalues stay in their HDF5 files.
"""
import argparse
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
import warnings

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from measurement_config import artifact_compatibility, measurement_config, safe_filename, validate_model_pin

# Use the same convention fields as the worker and resume checks.
MEASUREMENT_FIELDS = tuple(measurement_config(SimpleNamespace()))
MODULE_COUNTS = ("candidate_modules", "analyzed_modules",
                 "partially_analyzed_modules", "skipped_modules")
WEIGHT_COUNTS = ("registered_tensors", "measured_tensors", "skipped_tensors",
                 "not_applicable_tensors", "unresolved_tensors", "shared_tensors",
                 "unmapped_measurements")
CHECKPOINT_COUNTS = ("stored_tensors", "analyzed_tensors", "skipped_tensors")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="Run root containing models.csv, logs/ and/or stats/ and metrics/")
    parser.add_argument("--output", type=Path,
                        help="Summary CSV path (default: results_dir/summary.csv)")
    parser.add_argument("--model_list", type=Path,
                        help="Selected pinned CSV (default: results_dir/models.csv, if present)")
    parser.add_argument("--verbose", action="store_true", help="Show the first 20 valid model summaries")
    return parser.parse_args(argv)


def compute_model_summary(layers):
    """Summarize measurements, not architectural depth or unique parameters."""
    alpha = pd.to_numeric(layers["alpha"], errors="raise")
    fitted = layers["fit_status"].eq("fitted") & np.isfinite(alpha) & (alpha > 1)
    summary = {
        "analyzed_measurements": len(layers),
        "measured_modules": layers["module_name"].nunique(),
        "fitted_measurements": int(fitted.sum()),
        "missing_fit_measurements": int((~fitted).sum()),
    }
    alpha_values = alpha[fitted]
    summary.update(
        alpha_mean=alpha_values.mean(), alpha_median=alpha_values.median(),
        alpha_std=alpha_values.std(), alpha_min=alpha_values.min(),
        alpha_max=alpha_values.max(), alpha_q25=alpha_values.quantile(.25),
        alpha_q75=alpha_values.quantile(.75),
    )
    for metric, statistics in {
        "alpha_weighted": ("mean", "median"),
        "log_alpha_norm": ("mean", "median"),
        "D": ("median", "max"),
        "n_tail": ("min", "median"),
        "spectral_norm": ("mean", "max"),
        "log_spectral_norm": ("mean",),
        "stable_rank": ("mean", "median"),
        "entropy": ("mean", "median"),
    }.items():
        if metric not in layers:
            continue
        values = pd.to_numeric(layers[metric], errors="raise")
        if metric in ("alpha_weighted", "log_alpha_norm", "D", "n_tail"):
            values = values[fitted]
        values = values[np.isfinite(values)]
        for statistic in statistics:
            summary[f"{metric}_{statistic}"] = getattr(values, statistic)()
    # Per-measurement params can omit biases or repeat tied weights. Summing
    # them would not give the model's unique parameter count.
    return summary


def read_model_summary(csv_path, h5_path, expected=None):
    """Validate one pair and read its canonical scalars; never read /eigs or /alpha."""
    compatible, reason = artifact_compatibility(csv_path, h5_path, expected)
    if not compatible:
        raise ValueError(reason)
    with h5py.File(h5_path, "r") as h5:
        config = json.loads(h5.attrs["measurement_config_json"])
        settings = {name: config[name] for name in MEASUREMENT_FIELDS}
        json.dumps(settings, allow_nan=False)
        model_id = h5.attrs["full_name"]
        if isinstance(model_id, bytes):
            model_id = model_id.decode("utf-8")
        if not model_id or model_id.partition("@")[0] != config["repo_id"]:
            raise ValueError("Model identity disagrees with measurement configuration")
        layer_records = h5["layers"]
        required = {"longname", "module_name", "fit_status", "alpha"}
        if not required.issubset(layer_records):
            raise ValueError("Missing canonical identity or fit-status fields")
        layers = pd.DataFrame({
            name: dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]
            for name, dataset in layer_records.items()
        })
        summary = compute_model_summary(layers)
        runtime = config.get("runtime") or {}
        loading_info = runtime.get("loading_info") or {}
        summary.update(
            model_id=model_id, repo_id=config["repo_id"],
            requested_revision=config["requested_revision"],
            source_model=config.get("source_model", ""),
            base_model_relation=config.get("base_model_relation", ""),
            model_class=runtime.get("model_class"),
            model_config_commit_hash=runtime.get("model_config_commit_hash"),
            analysis_source=runtime["analysis_source"],
            effective_filter_type=runtime["filter_type"],
            fallback_reason=(runtime.get("fallback") or {}).get("reason"),
            base_repo_loaded=loading_info.get("base_repo_loaded"),
            base_revision_loaded=loading_info.get("base_revision_loaded"),
            base_resolved_commit_hash=loading_info.get("base_resolved_commit_hash"),
            csv_path=str(csv_path.resolve()), h5_path=str(h5_path.resolve()),
            coverage_status="unknown", weight_usage_status="unknown", **settings,
        )
        summary.update({name: None for name in (*MODULE_COUNTS, *WEIGHT_COUNTS, *CHECKPOINT_COUNTS)})
        if runtime["analysis_source"] == "checkpoint":
            summary["measured_modules"] = None
            if "coverage" not in h5:
                raise ValueError("Missing checkpoint-tensor coverage")
            coverage = json.loads(h5["coverage"].asstr()[()])
            if coverage.get("scope") != "checkpoint_tensors":
                raise ValueError("Checkpoint mode requires checkpoint-tensor coverage")
            counts = coverage["counts"]
            for name in (*CHECKPOINT_COUNTS, "analyzed_measurements", "fitted_measurements"):
                if type(counts.get(name)) is not int or counts[name] < 0:
                    raise ValueError(f"Invalid checkpoint coverage count: {name}")
            if counts["stored_tensors"] != counts["analyzed_tensors"] + counts["skipped_tensors"]:
                raise ValueError("Checkpoint tensor counts do not add up")
            if (counts["analyzed_tensors"] != summary["analyzed_measurements"]
                    or any(counts[name] != summary[name] for name in ("analyzed_measurements", "fitted_measurements"))):
                raise ValueError("Checkpoint coverage disagrees with canonical records")
            summary.update({name: counts[name] for name in CHECKPOINT_COUNTS})
            summary.update(coverage_status="checkpoint_tensors", weight_usage_status="not_applicable")
            return summary
        if "coverage" in h5:
            coverage = json.loads(h5["coverage"].asstr()[()])
            counts = coverage["counts"]
            for name in (*MODULE_COUNTS, "analyzed_measurements", "fitted_measurements"):
                if type(counts.get(name)) is not int or counts[name] < 0:
                    raise ValueError(f"Invalid coverage count: {name}")
            for name in ("analyzed_measurements", "fitted_measurements"):
                if counts[name] != summary[name]:
                    raise ValueError(f"Coverage disagrees with canonical records: {name}")
            if counts["candidate_modules"] != sum(counts[name] for name in MODULE_COUNTS[1:]):
                raise ValueError("Coverage module counts do not add up")
            if counts["analyzed_modules"] + counts["partially_analyzed_modules"] != summary["measured_modules"]:
                raise ValueError("Coverage disagrees with measured module identities")
            summary.update({name: counts[name] for name in MODULE_COUNTS})
            summary["coverage_status"] = "recorded"
            usage = coverage.get("weight_usage")
            if usage is not None:
                if usage.get("scope") != "loaded_registered_tensors":
                    raise ValueError("Unknown weight-usage scope")
                counts = usage["counts"]
                for name in WEIGHT_COUNTS:
                    if type(counts.get(name)) is not int or counts[name] < 0:
                        raise ValueError(f"Invalid weight-usage count: {name}")
                if counts["registered_tensors"] != sum(counts[name] for name in (
                        "measured_tensors", "skipped_tensors", "not_applicable_tensors", "unresolved_tensors")):
                    raise ValueError("Weight-usage tensor counts do not add up")
                if (counts["shared_tensors"] > counts["registered_tensors"]
                        or counts["unmapped_measurements"] > summary["analyzed_measurements"]):
                    raise ValueError("Weight-usage counts exceed available tensors or measurements")
                summary.update({name: counts[name] for name in WEIGHT_COUNTS})
                summary["weight_usage_status"] = "recorded"
    return summary


def read_selected_models(path):
    """Keep input metadata under input_* names, without importing model loaders."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "model_id" not in frame or frame.empty:
        raise ValueError("Model list must contain nonempty model_id rows")
    selected = {}
    for row in frame.to_dict("records"):
        model_id = row["model_id"] = row["model_id"].strip()
        if not model_id:
            raise ValueError("Model list contains an empty model_id")
        row["revision_norm"] = row.get("revision_norm", "").strip() or model_id.partition("@")[2]
        row["source_model"] = row.get("source_model", "").strip()
        if row.get("pin_status") != "error":
            validate_model_pin(model_id, row["revision_norm"], row["source_model"],
                               row.get("base_model_relation", ""), row.get("loader_scenario", ""))
        name = safe_filename(model_id)
        if name in selected:
            raise ValueError(f"Model rows share an output filename: {model_id}")
        selected[name] = row
    return selected


def summarize_outcome(results_dir, name, selected=None):
    """Prefer verified artifacts; keep last-recorded failures and unknowns explicit."""
    csv_path, h5_path = results_dir / "stats" / f"{name}.csv", results_dir / "metrics" / f"{name}.h5"
    status_path = results_dir / "logs" / "terminal_status" / f"{name}.json"
    row = {
        "model_id": selected["model_id"] if selected else "", "selected": selected is not None,
        "outcome": "unrecorded", "outcome_reason": "no_artifacts_or_terminal_record",
        "outcome_stage": "", "outcome_message": "", "outcome_pin_status": "unknown",
        "artifact_status": "missing", "artifact_error": "",
        "csv_path": str(csv_path), "h5_path": str(h5_path),
        "terminal_status_path": "", "terminal_error": "",
    }
    expected = None
    if selected:
        row.update({f"input_{key}": value for key, value in selected.items()})
        expected = dict(repo_id=selected["model_id"].partition("@")[0],
                        requested_revision=selected["revision_norm"], source_model=selected["source_model"])

    if csv_path.exists() or h5_path.exists():
        try:
            summary = read_model_summary(csv_path, h5_path, expected)
            if selected and summary["model_id"] != selected["model_id"]:
                raise ValueError("Artifact model_id differs from selected model_id")
            row.update(summary, artifact_status="valid")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            warnings.warn(f"{name}: {error}")
            row.update(artifact_status="invalid", artifact_error=str(error))

    terminal = None
    if status_path.exists():
        row["terminal_status_path"] = str(status_path)
        try:
            terminal = json.loads(status_path.read_text(encoding="utf-8"))
            if (not isinstance(terminal, dict) or not isinstance(terminal.get("model_id"), str)
                    or safe_filename(terminal["model_id"]) != name
                    or (row["model_id"] and terminal["model_id"] != row["model_id"])
                    or terminal.get("status") not in {"success", "failed", "blocked"}):
                raise ValueError("Invalid terminal identity or status")
            row["model_id"] = terminal["model_id"]
            row["terminal_timestamp"] = terminal.get("ts", "")
            row["terminal_revision"] = terminal.get("revision", "")
            row["terminal_source_model"] = terminal.get("source_model", "")
        except (OSError, ValueError, TypeError) as error:
            warnings.warn(f"{status_path}: {error}")
            row["terminal_error"] = str(error)
            terminal = None

    if row["artifact_status"] == "valid":
        row.update(outcome="success", outcome_reason="valid_artifact_pair",
                   outcome_pin_status="matched" if selected else "not_checked")
    elif row["artifact_status"] == "invalid":
        row.update(outcome="incomplete", outcome_reason="invalid_artifact_pair",
                   outcome_message=row["artifact_error"])
    elif selected and selected.get("pin_status") == "error":
        row.update(outcome="blocked", outcome_stage="prepare", outcome_reason="pin_error",
                   outcome_message=selected.get("pin_error", ""))
    elif terminal is not None:
        pin_status = "unknown"
        if expected and terminal.get("revision") and "source_model" in terminal:
            pin_status = "matched" if (terminal["revision"] == expected["requested_revision"]
                                       and terminal["source_model"] == expected["source_model"]) else "mismatch"
        row["outcome_pin_status"] = pin_status
        if pin_status == "mismatch":
            row["outcome_reason"] = "terminal_pin_mismatch"
        elif terminal["status"] == "success":
            row.update(outcome="incomplete", outcome_reason="success_without_artifacts")
        else:
            row.update(outcome=terminal["status"], outcome_reason=terminal.get("reason", ""),
                       outcome_stage=terminal.get("stage", ""), outcome_message=terminal.get("message", ""))
    elif row["terminal_error"]:
        row["outcome_reason"] = "unreadable_terminal_record"
    return row


def print_model_stats(summary, verbose=False, mixed_settings=False):
    print("Recorded outcomes: " + "; ".join(f"{status}: {count}" for status, count in
                                          summary["outcome"].value_counts().items()))
    valid = summary.loc[summary["artifact_status"] == "valid"]
    invalid = summary["artifact_status"].eq("invalid").sum()
    print(f"Valid artifact pairs: {len(valid)}; invalid/incomplete pairs: {invalid}; "
          f"without artifacts: {summary['artifact_status'].eq('missing').sum()}")
    if valid.empty:
        return
    print(f"Measurements: {valid['analyzed_measurements'].sum():.0f}; "
          f"fitted: {valid['fitted_measurements'].sum():.0f}; "
          f"missing fits: {valid['missing_fit_measurements'].sum():.0f}")
    unknown = (valid["coverage_status"] == "unknown").sum()
    if unknown:
        print(f"Module coverage is unknown for {unknown} models.")
    unknown_usage = (valid["weight_usage_status"] == "unknown").sum()
    if unknown_usage:
        print(f"Loaded weight usage is unknown for {unknown_usage} models.")
    print(f"Unresolved loaded tensors: {valid['unresolved_tensors'].sum():.0f}; "
          f"measurements without a registered-tensor link: {valid['unmapped_measurements'].sum():.0f} "
          "(recorded reports only)")
    if mixed_settings:
        warnings.warn("Mixed measurement settings: filter summary.csv by settings before comparing models. "
                      "Pooled metric statistics are not shown.")
        return
    alpha = valid["alpha_mean"].dropna()
    if not alpha.empty:
        print(f"Mean of model fitted-alpha means: {alpha.mean():.4f}")
    if verbose:
        print("First 20 valid models (not a model-quality ranking):")
        print(valid[["model_id", "analyzed_measurements", "fitted_measurements",
                     "alpha_mean", "coverage_status", "skipped_modules"]].head(20).to_string(index=False))


def write_summary(summary, output_path):
    """Replace a previous summary only after the new CSV is fully written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_path.parent,
                                prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            summary.to_csv(handle, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    stats_dir, metrics_dir = results_dir / "stats", results_dir / "metrics"
    terminal_dir = results_dir / "logs" / "terminal_status"
    model_list = args.model_list or results_dir / "models.csv"
    output_path = (args.output or results_dir / "summary.csv").resolve()
    if output_path.suffix.lower() != ".csv" or any(
        output_path.is_relative_to(directory.resolve()) for directory in (stats_dir, metrics_dir, results_dir / "logs")
    ) or output_path in (model_list.resolve(), (results_dir / "models.csv").resolve()):
        print("Summary output must be a .csv outside stats/, metrics/ and logs/, and cannot replace the model list.")
        return 1
    try:
        selected = read_selected_models(model_list) if args.model_list or model_list.exists() else {}
    except (OSError, ValueError) as error:
        print(f"Cannot read selected models: {error}; existing summary left unchanged.")
        return 1
    model_names = sorted(set(selected) | {
        path.stem for directory, suffix in ((stats_dir, "*.csv"), (metrics_dir, "*.h5"), (terminal_dir, "*.json"))
        for path in directory.glob(suffix) if path.is_file() and not path.name.startswith(".")
    })
    if not model_names:
        print("No selected models, artifacts or terminal records found; existing summary left unchanged.")
        return 1

    summaries, measurement_settings = [], set()
    for name in model_names:
        summary = summarize_outcome(results_dir, name, selected.get(name))
        if summary["artifact_status"] == "valid":
            comparison_fields = (*MEASUREMENT_FIELDS, "analysis_source", "effective_filter_type")
            measurement_settings.add(json.dumps({key: summary[key] for key in comparison_fields}, sort_keys=True))
        summaries.append(summary)

    # Only small per-model rows accumulate; layer tables are released after
    # each model, and eigenvalues remain in their original HDF5 files.
    summary = pd.DataFrame(summaries)
    first_columns = ["model_id", "selected", "outcome", "outcome_stage", "outcome_reason", "outcome_pin_status",
                     "artifact_status", "artifact_error", "csv_path", "h5_path"]
    summary = summary[first_columns + [name for name in summary if name not in first_columns]]
    write_summary(summary, output_path)
    print(f"Saved summary to: {output_path}")
    if selected:
        print(f"Selected models: {len(selected)}; additional stored models: {len(summary) - len(selected)}")
    print_model_stats(summary, args.verbose, mixed_settings=len(measurement_settings) > 1)
    return int(summary["outcome"].eq("incomplete").any() or summary["terminal_error"].ne("").any())


if __name__ == "__main__":
    raise SystemExit(main())
