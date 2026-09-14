#!/usr/bin/env python3
"""Summarize a run's canonical layer records without loading its eigenvalues.

summary.csv doubles as a small results index: one row per artifact pair, with
measurement settings, coverage, scalar summaries and paths back to the data.
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
from measurement_config import artifact_compatibility, measurement_config

# Use the same convention fields as the worker and resume checks.
MEASUREMENT_FIELDS = tuple(measurement_config(SimpleNamespace()))
MODULE_COUNTS = ("candidate_modules", "analyzed_modules",
                 "partially_analyzed_modules", "skipped_modules")
WEIGHT_COUNTS = ("registered_tensors", "measured_tensors", "skipped_tensors",
                 "not_applicable_tensors", "unresolved_tensors", "shared_tensors",
                 "unmapped_measurements")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="Run directory containing stats/ and metrics/")
    parser.add_argument("--output", type=Path,
                        help="Summary CSV path (default: results_dir/summary.csv)")
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


def read_model_summary(csv_path, h5_path):
    """Validate one pair and read its canonical scalars; never read /eigs or /alpha."""
    compatible, reason = artifact_compatibility(csv_path, h5_path)
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
            base_repo_loaded=loading_info.get("base_repo_loaded"),
            base_revision_loaded=loading_info.get("base_revision_loaded"),
            base_resolved_commit_hash=loading_info.get("base_resolved_commit_hash"),
            csv_path=str(csv_path.resolve()), h5_path=str(h5_path.resolve()),
            coverage_status="unknown", weight_usage_status="unknown", **settings,
        )
        summary.update({name: None for name in (*MODULE_COUNTS, *WEIGHT_COUNTS)})
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


def print_model_stats(summary, verbose=False, mixed_settings=False):
    valid = summary.loc[summary["artifact_status"] == "valid"]
    print(f"Valid artifact pairs: {len(valid)}; invalid/incomplete pairs: {len(summary) - len(valid)}")
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
    output_path = (args.output or results_dir / "summary.csv").resolve()
    if not stats_dir.is_dir() and not metrics_dir.is_dir():
        print("No stats/ or metrics/ directory found. Pass the run root as --results_dir.")
        return 1
    if output_path.suffix.lower() != ".csv" or any(
        output_path.is_relative_to(directory.resolve()) for directory in (stats_dir, metrics_dir)
    ):
        print("Summary output must be a .csv outside the input stats/ and metrics/ directories.")
        return 1
    model_names = sorted({
        path.stem for directory, suffix in ((stats_dir, "*.csv"), (metrics_dir, "*.h5"))
        for path in directory.glob(suffix) if path.is_file() and not path.name.startswith(".")
    })
    if not model_names:
        print("No model artifacts found; existing summary left unchanged.")
        return 1

    summaries, measurement_settings = [], set()
    for name in model_names:
        csv_path, h5_path = stats_dir / f"{name}.csv", metrics_dir / f"{name}.h5"
        try:
            summary = read_model_summary(csv_path, h5_path)
            measurement_settings.add(json.dumps({key: summary[key] for key in MEASUREMENT_FIELDS}, sort_keys=True))
            summary.update(artifact_status="valid", artifact_error="")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            warnings.warn(f"{name}: {error}")
            summary = {"model_id": "", "csv_path": str(csv_path), "h5_path": str(h5_path),
                       "artifact_status": "invalid", "artifact_error": str(error)}
        summaries.append(summary)

    # Only small per-model rows accumulate; layer tables are released after
    # each model, and eigenvalues remain in their original HDF5 files.
    summary = pd.DataFrame(summaries)
    first_columns = ["model_id", "artifact_status", "artifact_error", "csv_path", "h5_path"]
    summary = summary[first_columns + [name for name in summary if name not in first_columns]]
    write_summary(summary, output_path)
    print(f"Saved summary to: {output_path}")
    print_model_stats(summary, args.verbose, mixed_settings=len(measurement_settings) > 1)
    return int((summary["artifact_status"] != "valid").any())


if __name__ == "__main__":
    raise SystemExit(main())
