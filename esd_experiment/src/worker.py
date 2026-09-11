#!/usr/bin/env python3
"""
Worker script for analyzing a single model.
This is called by the main experiment runner for each model.
"""
import sys
import os
import argparse
import math
import warnings
import traceback
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import torch
import pandas as pd
import numpy as np
import h5py
import json
import re

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up to ESD root
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from model_loader import LoaderFailure, load_model, parse_model_string, safe_filename
from measurement_config import (
    FORMAT_VERSION, NUMERICS_VERSION, artifact_compatibility,
    measurement_config as build_measurement_config,
)
from net_esd import net_esd_estimator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze single model ESD metrics")
    
    # Model specification
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--revision", type=str, default="", help="Optional model revision")
    parser.add_argument("--base_model_relation", type=str, default="", help="Adapter relation type")
    parser.add_argument("--source_model", type=str, default="", help="Base model for adapters")
    parser.add_argument("--loader_scenario", type=str, default="", help="Curated loader scenario hint")
    parser.add_argument("--primary_type_bucket", type=str, default="", help="Curated type bucket")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    
    # ESD parameters
    parser.add_argument("--fix_fingers", choices=["xmin_mid", "xmin_peak", "DKS"], default="xmin_mid", help="xmin selection method")
    parser.add_argument("--evals_thresh", type=float, default=1e-5, help="Eigenvalue threshold")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins")
    parser.add_argument("--filter_zeros", action=argparse.BooleanOptionalAction, default=True, help="Filter the measurement/fit spectrum, not saved eigenvalues")
    parser.add_argument("--parallel_esd", action=argparse.BooleanOptionalAction, default=True, help="Use parallel ESD")
    parser.add_argument("--use_svd", action=argparse.BooleanOptionalAction, default=True, help="Use SVD (default); --no-use_svd selects Gram eigenvalues")
    parser.add_argument("--save_eigs", action="store_true", default=False, help="Save full computed spectra in HDF5")
    parser.add_argument("--load_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto", help="Checkpoint/framework-selected loading precision by default")
    parser.add_argument("--compute_dtype", choices=["float32", "float64"], default="float32", help="SVD/Gram precision; float64 for reference checks")
    
    # Model loading
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for loading (auto uses GPU when CUDA_VISIBLE_DEVICES is set)")
    parser.add_argument("--max_retries", type=int, default=0, help="Max retry attempts")
    
    args = parser.parse_args()
    try:
        build_measurement_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def coverage_report(records, metrics):
    """Keep module coverage separate from per-measurement fit availability."""
    counts = {
        "candidate_modules": len(records),
        "eligible_modules": sum(bool(record["measurement_names"]) for record in records),
        "analyzed_modules": sum(record["status"] == "analyzed" for record in records),
        "partially_analyzed_modules": sum(record["status"] == "partially_analyzed" for record in records),
        "skipped_modules": sum(record["status"] == "skipped" for record in records),
        "analyzed_measurements": len(metrics.get("longname", [])),
        "fitted_measurements": sum(status == "fitted" for status in metrics.get("fit_status", [])),
    }
    return {"counts": counts, "modules": records}


def runtime_provenance(model, args):
    """Record observed loading/runtime details without claiming checkpoint-native identity."""
    config = getattr(model, "config", None)
    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    return {
        "model_config_commit_hash": getattr(config, "_commit_hash", None),
        "model_config_name_or_path": getattr(config, "_name_or_path", None),
        "torch_version": getattr(torch, "__version__", None),
        "numpy_version": getattr(np, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "device_map_requested": args.device_map,
        "parallel_esd": args.parallel_esd,
        "loaded_parameter_dtypes": sorted({str(getattr(p, "dtype", "unknown")) for p in model.parameters()}),
        "cuda_matmul_allow_tf32": getattr(matmul, "allow_tf32", None),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
    }


def resolve_model_revision(model_id: str, revision_override: str = ""):
    """Resolve repo ID and effective revision, honoring curated overrides."""
    repo_id, revision = parse_model_string(model_id)
    if revision_override and revision_override.strip():
        revision = revision_override.strip()
    return repo_id, revision


# ------------------------------------------------------------
# Canonical layer records and a derived, compatibility-only alpha matrix.
# ------------------------------------------------------------

PREFIX_CANDIDATES = {"layers", "layer", "h", "block", "blocks"}
MAX_ALPHA_VIEW_CELLS = 1_000_000  # Optional dense view must not dominate storage/memory.

def parse_longname(longname: str):
    """
    Parse a module longname into (layer:int, module:str).
    Examples it can handle (tokens before the layer index include one of PREFIX_CANDIDATES):
        model.layers.5.mlp.up_proj
        transformer.h.10.attn.q_proj
        model.decoder.layers.3.self_attn.q_proj
        blocks.7.mlp.fc_in
    Returns (None, None) if it cannot parse.
    """
    if not isinstance(longname, str):
        return (None, None)
    tokens = longname.strip().split(".")
    for i in range(len(tokens) - 2):
        prefix, maybe_idx = tokens[i], tokens[i + 1]
        if prefix in PREFIX_CANDIDATES and re.fullmatch(r"\d+", maybe_idx):
            layer = int(maybe_idx)
            module = ".".join(tokens[i + 2:])
            return (layer, module) if module else (None, None)
    # fallback
    for i, tk in enumerate(tokens):
        if re.fullmatch(r"\d+", tk) and i > 0 and tokens[i - 1] in PREFIX_CANDIDATES:
            layer = int(tk)
            module = ".".join(tokens[i + 1:])
            return (layer, module) if module else (None, None)
    return (None, None)


def _validate_layer_names(longnames):
    """Full module paths, including the empty root path, are canonical identities."""
    if not all(isinstance(name, str) for name in longnames):
        raise ValueError("Layer longnames must be strings")
    if len(set(longnames)) != len(longnames):
        raise ValueError("Duplicate canonical layer longnames")


def build_tensor_from_pairs(longnames, alphas):
    """
    Build a derived depth-by-module matrix without merging distinct identities:
      - Rows: 0..max_layer; Columns: sorted unique module names.
      - Prefix-qualify modules shared by multiple stacks (e.g. encoder/decoder).
      - Missing fits do not remove parsed layers or modules.
      - Unparseable names stay in canonical records, not this optional view.
    Returns (mat [L,M], module_names [list[str]], num_layers [int]).
    """
    if len(longnames) != len(alphas):
        raise ValueError("Layer longnames and alphas must have equal lengths")
    _validate_layer_names(longnames)
    parsed = []
    prefixes_by_module = {}
    for name, alpha in zip(longnames, alphas):
        layer, module = parse_longname(name)
        if layer is None:
            continue
        tokens = name.strip().split(".")
        # parse_longname selects the first recognized depth index.
        index = next(i for i in range(1, len(tokens))
                     if tokens[i - 1] in PREFIX_CANDIDATES and tokens[i].isdigit())
        prefix = ".".join(tokens[:index])
        prefixes_by_module.setdefault(module, set()).add(prefix)
        parsed.append((name, layer, module, prefix, alpha))

    if not parsed:
        return np.empty((0, 0), dtype=float), [], 0

    namespaces = {(prefix, module) for _, _, module, prefix, _ in parsed}
    labels = {(prefix, module): f"{prefix}.{module}" if len(prefixes_by_module[module]) > 1 else module
              for prefix, module in namespaces}
    if len(set(labels.values())) != len(labels):
        # Rare nested namespaces can resemble another namespace's qualified
        # label. JSON pairs are unambiguous even for names containing punctuation.
        labels = {namespace: json.dumps(namespace, ensure_ascii=False) for namespace in namespaces}
    entries = [(name, layer, labels[prefix, module], alpha)
               for name, layer, module, prefix, alpha in parsed]
    # Very unusual nested names can collide even after prefix qualification.
    # Use their full identities as columns rather than ever averaging them.
    cells = {}
    for name, layer, column, alpha in entries:
        cells.setdefault((layer, column), []).append(name)
    if any(len(names) > 1 for names in cells.values()):
        entries = [(name, layer, name, alpha) for name, layer, _, alpha in entries]

    module_names = sorted({column for _, _, column, _ in entries})
    num_layers = max(layer for _, layer, _, _ in entries) + 1
    if num_layers * len(module_names) > MAX_ALPHA_VIEW_CELLS:
        return np.empty((0, 0), dtype=float), [], 0
    mat = np.full((num_layers, len(module_names)), np.nan, dtype=float)
    module_index = {module: index for index, module in enumerate(module_names)}
    occupied = set()
    for name, layer, column, alpha in entries:
        cell = (layer, module_index[column])
        if cell in occupied:
            raise ValueError(f"Ambiguous derived alpha cell for {name!r}")
        occupied.add(cell)
        mat[cell] = float(alpha) if alpha is not None else np.nan
    return mat, module_names, num_layers


def _prepare_layer_records(metrics):
    """Validate row alignment before any output is written; never truncate data."""
    if "longname" not in metrics or "alpha" not in metrics:
        raise ValueError("Layer records require longname and alpha columns")
    records = {}
    num_rows = len(metrics["longname"])
    for key, values in metrics.items():
        if not isinstance(key, str) or "/" in key:
            raise ValueError(f"Invalid layer metric name: {key!r}")
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            raise ValueError(f"Layer metric {key!r} must be a sequence")
        if len(values) != num_rows:
            raise ValueError(f"Layer metric {key!r} has {len(values)} rows; expected {num_rows}")
        records[key] = list(values)
    # The only supported legacy aggregate is an explicitly marked, aligned row.
    if num_rows and records["longname"][-1] is None:
        records = {key: values[:-1] for key, values in records.items()}
    _validate_layer_names(records["longname"])
    if not records["longname"]:
        raise ValueError("No canonical layer records to save")
    return records


def _write_layer_dataset(group, key, values):
    """Store scalar columns natively; retain structured per-layer values as JSON."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    if any(isinstance(value, str) for value in values) and all(
        value is None or isinstance(value, str) for value in values
    ):
        dataset = group.create_dataset(key, data=[value if value is not None else "" for value in values],
                                       dtype=string_dtype)
        if any(value is None for value in values):
            dataset.attrs["missing_value"] = ""
    elif all(value is None or (np.isscalar(value) and not isinstance(value, (str, bytes)))
             for value in values):
        group.create_dataset(key, data=np.asarray([np.nan if value is None else value for value in values]))
    else:
        dataset = group.create_dataset(
            key, data=[json.dumps(value, ensure_ascii=False) for value in values], dtype=string_dtype,
        )
        dataset.attrs["encoding"] = "json"


def save_h5(
    h5_path: Path,
    mat: np.ndarray,
    module_names,
    num_layers: int,
    file_attrs: dict,
    eigs=None,
    layer_records=None,
    measurement_config=None,
    coverage=None,
):
    if layer_records is not None:
        layer_records = _prepare_layer_records(layer_records)
        if eigs is not None and len(eigs) != len(layer_records["longname"]):
            raise ValueError("Eigenvalues must be aligned with canonical layer records")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        dset = h5.create_dataset("alpha", data=mat)
        dset.attrs["num_layers"] = int(num_layers)
        dset.attrs["num_modules"] = int(len(module_names))
        dset.attrs["missing_value"] = "NaN"
        dset.attrs["module_names_json"] = json.dumps(module_names, ensure_ascii=False)
        dset.attrs["canonical_records"] = "/layers"
        if layer_records is not None:
            layers = h5.create_group("layers")
            layers.attrs["canonical_identity"] = "longname"
            layers.attrs["num_records"] = len(layer_records["longname"])
            for key, values in layer_records.items():
                if key != "eigs":
                    _write_layer_dataset(layers, key, values)
            unmapped = [name for name in layer_records["longname"]
                        if mat.size == 0 or parse_longname(name)[0] is None]
            dset.attrs["view_status"] = (
                "unavailable" if len(unmapped) == len(layer_records["longname"])
                else "partial" if unmapped else "complete"
            )
            dset.attrs["unmapped_layer_count"] = len(unmapped)
            h5.create_dataset("alpha_unmapped_longname", data=unmapped, dtype=h5py.string_dtype("utf-8"))
        if eigs is not None:
            # Do not double storage for float32-computed spectra. Preserve
            # float64 reference spectra (and ordinary Python float inputs).
            spectrum_dtype = np.dtype("float32") if all(
                values is None or np.asarray(values).dtype == np.dtype("float32") for values in eigs
            ) else np.dtype("float64")
            vlen_float = h5py.vlen_dtype(spectrum_dtype)
            eigs_dset = h5.create_dataset("eigs", (len(eigs),), dtype=vlen_float)
            eigs_dset.attrs["aligned_with"] = "/layers/longname"
            eigs_dset.attrs["spectrum"] = "full_computed_spectrum"
            eigs_dset.attrs["storage_dtype"] = str(spectrum_dtype)
            for i, values in enumerate(eigs):
                eigs_dset[i] = np.asarray(values if values is not None else [], dtype=spectrum_dtype)
        if measurement_config is not None:
            h5.attrs["measurement_config_json"] = json.dumps(measurement_config, ensure_ascii=False, sort_keys=True)
        if coverage is not None:
            h5.create_dataset("coverage", data=json.dumps(coverage, ensure_ascii=False),
                              dtype=h5py.string_dtype("utf-8"))
            for key, value in coverage.get("counts", {}).items():
                h5.attrs[f"coverage_{key}"] = int(value)
        h5.attrs["format_version"] = FORMAT_VERSION
        for k, v in (file_attrs or {}).items():
            try:
                h5.attrs[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
            except Exception:
                pass


def save_results(
    metrics: dict,
    output_path: Path,
    model_id: str,
    is_adapter: bool,
    source_model: Optional[str] = None,
    base_model_relation: str = "",
    fix_fingers: str = "",
    h5_output_path: Optional[Path] = None,
    save_eigs: bool = False,
    measurement_config: Optional[dict] = None,
    coverage: Optional[dict] = None,
):
    """
    Save scalar metrics to CSV and canonical aligned layer records to HDF5.
    Eigenvalues, when requested, are stored only in HDF5. The root alpha matrix
    is a derived compatibility view; /layers/longname is the identity authority.
    
    Args:
        metrics: Dictionary of metrics from net_esd_estimator
        output_path: Path to save CSV
        model_id: Model identifier
        is_adapter: Whether this is an adapter model
        source_model: Base model for adapters
        base_model_relation: Relation tag (e.g., adapter/base/finetune)
        fix_fingers: xmin strategy used (DKS/xmin_mid/xmin_peak)
    """
    records = _prepare_layer_records(metrics)
    longnames, alphas = records["longname"], records["alpha"]
    eigs = records.get("eigs") if save_eigs else None
    if save_eigs and eigs is None:
        raise ValueError("save_eigs requires eigenvalues aligned with layer records")
    mat, module_names, num_layers = build_tensor_from_pairs(longnames, alphas)
    df = pd.DataFrame({key: values for key, values in records.items() if key != "eigs"})
    df["alpha"] = pd.to_numeric(df["alpha"], errors="raise")
    
    # Add metadata columns
    df.insert(0, "model_id", model_id)
    df.insert(1, "is_adapter", is_adapter)
    if source_model:
        df.insert(2, "source_model", source_model)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")
    
    # Print summary statistics
    if "alpha" in df.columns:
        alpha_values = df.loc[np.isfinite(df["alpha"]) & (df["alpha"] > 1), "alpha"]
        print(f"Finite power-law fits: {len(alpha_values)}/{len(df)} layers")
        if len(alpha_values) > 0:
            print(f"Alpha statistics:")
            print(f"  Mean: {alpha_values.mean():.4f}")
            print(f"  Median: {alpha_values.median():.4f}")
            print(f"  Std: {alpha_values.std():.4f}")
            print(f"  Range: [{alpha_values.min():.4f}, {alpha_values.max():.4f}]")
            print(f"  Layers: {len(alpha_values)}")

    if h5_output_path is None:
        h5_path = output_path.parent.parent / "metrics" / f"{safe_filename(model_id)}.h5"
    else:
        h5_path = h5_output_path
    relation_attr = base_model_relation.strip() or ("adapter" if is_adapter else "base")
    file_attrs = {
        "full_name": model_id,
        "source_model": source_model or "",
        "base_model_relation": relation_attr,
        "fix_fingers": fix_fingers,
        "alpha_only": "false",
        "save_eigs": str(save_eigs).lower(),
        "numerics_version": NUMERICS_VERSION,
    }
    save_h5(h5_path, mat, module_names, num_layers, file_attrs, eigs=eigs,
            layer_records=records, measurement_config=measurement_config, coverage=coverage)
    print(f"Saved H5 layer records to: {h5_path}")


def temp_output_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.tmp")


def finalize_output_path(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.replace(final_path)


def cleanup_temp_path(path: Path) -> None:
    if path.exists():
        path.unlink()


def cleanup_output_artifacts(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _terminal_status_path(output_dir: Path, model_id: str) -> Path:
    return output_dir / "logs" / "terminal_status" / f"{safe_filename(model_id)}.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    temp_path.replace(path)



def write_worker_heartbeat(
    heartbeat_path: Path,
    model_id: str,
    state: str,
    stage: str,
    pid: Optional[int] = None,
    stage_entered_at: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "updated_at": now,
        "stage_entered_at": stage_entered_at or now,
        "model_id": model_id,
        "state": state,
        "stage": stage,
        "pid": os.getpid() if pid is None else pid,
        "origin": "worker",
    }
    _write_json_atomic(Path(heartbeat_path), payload)


class HeartbeatReporter:
    def __init__(self, heartbeat_path: Optional[str], model_id: str, interval_seconds: int = 30):
        self.path = Path(heartbeat_path) if heartbeat_path else None
        self.model_id = model_id
        self.interval_seconds = interval_seconds
        self.state = "starting"
        self.stage = "start"
        self.stage_entered_at = datetime.now(timezone.utc).isoformat()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    def start(self, stage: str = "start", state: str = "running") -> None:
        if self.path is None:
            return
        self.update(stage=stage, state=state)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, stage: str, state: Optional[str] = None) -> None:
        if self.path is None or self._stopped:
            return
        with self._lock:
            if stage != self.stage:
                self.stage_entered_at = datetime.now(timezone.utc).isoformat()
            self.stage = stage
            if state is not None:
                self.state = state
            current_state = self.state
            current_stage = self.stage
            current_stage_entered_at = self.stage_entered_at
        write_worker_heartbeat(
            self.path,
            self.model_id,
            current_state,
            current_stage,
            stage_entered_at=current_stage_entered_at,
        )

    def stop(self, state: str = "stopped", stage: Optional[str] = None) -> None:
        if self.path is None or self._stopped:
            return
        with self._lock:
            self._stopped = True
            self.state = state
            if stage is not None and stage != self.stage:
                self.stage_entered_at = datetime.now(timezone.utc).isoformat()
                self.stage = stage
            elif stage is not None:
                self.stage = stage
            current_state = self.state
            current_stage = self.stage
            current_stage_entered_at = self.stage_entered_at
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        write_worker_heartbeat(
            self.path,
            self.model_id,
            current_state,
            current_stage,
            stage_entered_at=current_stage_entered_at,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                current_state = self.state
                current_stage = self.stage
                current_stage_entered_at = self.stage_entered_at
            try:
                write_worker_heartbeat(
                    self.path,
                    self.model_id,
                    current_state,
                    current_stage,
                    stage_entered_at=current_stage_entered_at,
                )
            except Exception:
                pass


def record_terminal_status(
    output_dir: Path,
    model_id: str,
    status: str,
    stage: str,
    reason: str,
    message: str,
    attempt: int = 0,
):
    """Record the final terminal outcome for a model."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "status": status,
        "stage": stage,
        "reason": reason,
        "message": message,
        "attempt": attempt,
    }
    _write_json_atomic(_terminal_status_path(output_dir, model_id), record)


def record_failure(
    output_dir: Path,
    model_id: str,
    stage: str,
    reason: str,
    message: str,
    attempt: int,
):
    """Record machine-readable failure details and keep a text summary."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = logs_dir / "failure_records.jsonl"
    text_path = logs_dir / "failed_models.txt"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "stage": stage,
        "reason": reason,
        "message": message,
        "attempt": attempt,
    }

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"{model_id}\t{stage}\t{reason}\t{message}\n")

    print(f"Recorded failure for {model_id}: {stage}/{reason}")


def validate_metrics_output(metrics: dict):
    """Accept useful spectral measurements even when every tail fit is missing."""
    longnames = metrics.get("longname", []) or []
    alphas = metrics.get("alpha", []) or []
    if longnames and longnames[-1] is None:
        longnames = longnames[:-1]
    if alphas and alphas[-1] is None:
        alphas = alphas[:-1]
    if not longnames or len(longnames) != len(alphas):
        return ("analyze", "analysis_empty")
    usable_pairs = zip(longnames, alphas)
    usable_alpha_count = sum(
        1
        for longname, alpha in usable_pairs
        if longname is not None and alpha is not None and math.isfinite(alpha) and alpha > 1
    )
    norms = metrics.get("norm", []) or []
    counts = metrics.get("num_evals", []) or []
    usable_spectrum_count = sum(
        1
        for longname, norm, count in zip(longnames, norms, counts)
        if longname is not None
        and norm is not None and math.isfinite(norm) and norm >= 0
        and count is not None and math.isfinite(count) and count >= 0
    )
    if usable_alpha_count == 0 and usable_spectrum_count == 0:
        return ("analyze", "analysis_empty")
    return None


def classify_retryable_failure(stage: str, reason: str) -> bool:
    non_retryable_by_stage = {
        "load": {
            "unsupported_loader_scenario",
            "adapter_base_unresolved",
            "repo_missing_or_private",
            "repo_gated",
        },
        "analyze": {"analysis_empty"},
    }
    retryable_by_stage = {
        "load": {"model_load_error", "cuda_oom"},
        "analyze": {"analysis_exception", "cuda_oom"},
        "save": {"save_error", "cuda_oom"},
    }

    if reason in non_retryable_by_stage.get(stage, set()):
        return False
    if reason in retryable_by_stage.get(stage, set()):
        return True
    return False


def classify_runtime_error(stage: str, error: Exception):
    message = str(error)
    lowered = message.lower()
    if "out of memory" in lowered and "cuda" in lowered:
        return stage, "cuda_oom", message
    if stage == "load":
        return stage, "model_load_error", message
    if stage == "save":
        return stage, "save_error", message
    return stage, "analysis_exception", message


def cleanup_model(model):
    """Cleanup model and free memory."""
    try:
        del model
    except:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Main worker function."""
    args = parse_args()
    measurement = build_measurement_config(
        args, model_id=args.model_id, revision=args.revision,
        source_model=args.source_model, base_model_relation=args.base_model_relation,
        loader_scenario=args.loader_scenario,
    )
    
    # Parse model ID (may include revision) and allow curated revision override
    repo_id, revision = resolve_model_revision(args.model_id, args.revision)
    display_name = args.model_id
    
    # Setup output path
    output_dir = Path(args.output_dir)
    output_file = output_dir / "stats" / f"{safe_filename(args.model_id)}.csv"
    temp_output_file = temp_output_path(output_file)
    metrics_file = output_dir / "metrics" / f"{safe_filename(args.model_id)}.h5"
    temp_metrics_file = temp_output_path(metrics_file)
    heartbeat = HeartbeatReporter(os.environ.get("WORKER_HEARTBEAT_FILE"), display_name)
    heartbeat.start(stage="prepare")
    
    # Check if already done
    try:
        if args.overwrite:
            if output_file.exists() or metrics_file.exists() or temp_output_file.exists() or temp_metrics_file.exists():
                print("Overwrite requested; clearing existing artifacts before regeneration")
                cleanup_output_artifacts(
                    temp_output_file,
                    temp_metrics_file,
                    output_file,
                    metrics_file,
                )
        else:
            if output_file.exists() or metrics_file.exists():
                compatible, reason = artifact_compatibility(output_file, metrics_file, measurement)
                if not compatible:
                    message = (
                        f"Existing results are incompatible ({reason}). Use a fresh output directory "
                        "or explicitly pass --overwrite; existing artifacts were not changed."
                    )
                    print(message)
                    record_terminal_status(output_dir, display_name, "failed", "save",
                                           "incompatible_results", message, attempt=0)
                    heartbeat.stop(state="failed", stage="save")
                    return 1
                print(f"Results already exist: {output_file}")
                record_terminal_status(
                    output_dir,
                    display_name,
                    status="success",
                    stage="skip",
                    reason="already_complete",
                    message="Results already exist",
                    attempt=0,
                )
                heartbeat.stop(state="success", stage="skip")
                return 0
    except Exception as exc:
        stage, reason, message = classify_runtime_error("save", exc)
        record_failure(output_dir, display_name, stage, reason, message, attempt=0)
        record_terminal_status(output_dir, display_name, "failed", stage, reason, message, attempt=0)
        heartbeat.stop(state="failed", stage=stage)
        return 1
    
    print("=" * 80)
    print(f"Analyzing model: {display_name}")
    print("=" * 80)
    
    # Print GPU assignment info
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"Device map: {args.device_map}")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes ({torch.cuda.device_count()} devices)")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"CUDA available: No (will use CPU)")
    print("=" * 80)
    
    model = None
    success = False
    
    for attempt in range(1, args.max_retries + 2):
        current_stage = "load"
        heartbeat.update(stage=current_stage)
        try:
            print(f"\nAttempt {attempt}/{args.max_retries + 1}")
            
            # Load model
            print(f"Loading model: {repo_id}")
            base_relation = args.base_model_relation if args.base_model_relation else None
            source_model = args.source_model if args.source_model else None
            
            try:
                model, is_adapter = load_model(
                    repo_id=repo_id,
                    base_model_relation=base_relation,
                    source_model=source_model,
                    device_map=args.device_map,
                    torch_dtype="auto" if measurement["load_dtype"] == "auto" else getattr(torch, measurement["load_dtype"]),
                    revision=revision,
                    loader_scenario=args.loader_scenario if args.loader_scenario else None,
                )
            except LoaderFailure as exc:
                raise exc
            except Exception as exc:
                stage, reason, message = classify_runtime_error("load", exc)
                raise LoaderFailure(stage, reason, message) from exc
            
            current_stage = "analyze"
            heartbeat.update(stage=current_stage)
            print(f"Model loaded successfully (adapter: {is_adapter})")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            measurement["runtime"] = runtime_provenance(model, args)
            
            # Report which device the model is on
            model_devices = set()
            for param in model.parameters():
                model_devices.add(str(param.device))
            print(f"Model devices: {', '.join(sorted(model_devices))}")
            
            # Run ESD analysis
            print("\nRunning ESD analysis...")
            fix_fingers_value = None if args.fix_fingers == "DKS" else args.fix_fingers
            layer_coverage = []
            
            try:
                metrics = net_esd_estimator(
                    model,
                    EVALS_THRESH=args.evals_thresh,
                    bins=args.bins,
                    fix_fingers=fix_fingers_value,
                    filter_zeros=args.filter_zeros,
                    use_svd=args.use_svd,
                    save_eigs=getattr(args, "save_eigs", False),
                    parallel=args.parallel_esd,
                    compute_dtype=measurement["compute_dtype"],
                    coverage=layer_coverage,
                )
            except Exception as exc:
                stage, reason, message = classify_runtime_error("analyze", exc)
                raise LoaderFailure(stage, reason, message) from exc

            coverage = coverage_report(layer_coverage, metrics)
            coverage.update(model_id=display_name, measurement_config=measurement)
            # Preserve missingness information even if every candidate is skipped.
            _write_json_atomic(output_dir / "logs" / "coverage" / f"{safe_filename(display_name)}.json", coverage)
            print(f"Coverage: {json.dumps(coverage['counts'], sort_keys=True)}")
            validation_failure = validate_metrics_output(metrics)
            if validation_failure is not None:
                stage, reason = validation_failure
                raise LoaderFailure(stage, reason, "ESD analysis returned no usable layer measurements")
            
            print(f"ESD analysis completed successfully")
            print(f"Analyzed {len(metrics.get('longname', []))} layers")
            
            # Save results
            current_stage = "save"
            heartbeat.update(stage=current_stage)
            cleanup_temp_path(temp_output_file)
            cleanup_temp_path(temp_metrics_file)
            try:
                save_results(
                    metrics,
                    temp_output_file,
                    display_name,
                    is_adapter,
                    source_model if is_adapter else None,
                    base_model_relation=args.base_model_relation or "",
                    fix_fingers=args.fix_fingers or "",
                    h5_output_path=temp_metrics_file,
                    save_eigs=getattr(args, "save_eigs", False),
                    measurement_config=measurement,
                    coverage=coverage,
                )
                finalize_output_path(temp_output_file, output_file)
                if temp_metrics_file.exists():
                    finalize_output_path(temp_metrics_file, metrics_file)
            except Exception as exc:
                cleanup_output_artifacts(
                    temp_output_file,
                    temp_metrics_file,
                    output_file,
                    metrics_file,
                )
                stage, reason, message = classify_runtime_error("save", exc)
                raise LoaderFailure(stage, reason, message) from exc
            
            success = True
            break
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            cleanup_model(model)
            cleanup_temp_path(temp_output_file)
            cleanup_temp_path(temp_metrics_file)
            record_terminal_status(
                output_dir,
                display_name,
                status="failed",
                stage="save",
                reason="interrupted",
                message="Interrupted by user",
                attempt=attempt,
            )
            heartbeat.stop(state="failed", stage="interrupted")
            return 1
            
        except LoaderFailure as e:
            error_msg = str(e)
            print(f"\nAttempt {attempt} failed: {error_msg}")

            retryable = classify_retryable_failure(e.stage, e.reason)
            if retryable and attempt <= args.max_retries:
                print("Retrying...")
                warnings.warn(f"Attempt {attempt} failed for {display_name}: {error_msg}")
            else:
                print("\nAll attempts failed!")
                print("Full traceback:")
                traceback.print_exc()
                record_failure(output_dir, display_name, e.stage, e.reason, error_msg, attempt)
                record_terminal_status(
                    output_dir,
                    display_name,
                    status="failed",
                    stage=e.stage,
                    reason=e.reason,
                    message=error_msg,
                    attempt=attempt,
                )
                heartbeat.stop(state="failed", stage=e.stage)
                break
        except Exception as e:
            error_msg = str(e)
            print(f"\nAttempt {attempt} failed: {error_msg}")

            stage, reason, message = classify_runtime_error(current_stage, e)
            retryable = classify_retryable_failure(stage, reason)
            if retryable and attempt <= args.max_retries:
                print("Retrying...")
                warnings.warn(f"Attempt {attempt} failed for {display_name}: {error_msg}")
            else:
                print("\nAll attempts failed!")
                print("Full traceback:")
                traceback.print_exc()
                record_failure(output_dir, display_name, stage, reason, message, attempt)
                record_terminal_status(
                    output_dir,
                    display_name,
                    status="failed",
                    stage=stage,
                    reason=reason,
                    message=message,
                    attempt=attempt,
                )
                heartbeat.stop(state="failed", stage=stage)
                break
        
        finally:
            # Cleanup
            cleanup_model(model)
    
    cleanup_temp_path(temp_output_file)
    cleanup_temp_path(temp_metrics_file)

    if not success:
        print(f"\nFailed to analyze {display_name}")
        if not _terminal_status_path(output_dir, display_name).exists():
            record_terminal_status(
                output_dir,
                display_name,
                status="failed",
                stage="save",
                reason="analysis_failed",
                message="Failed to analyze model",
                attempt=args.max_retries + 1,
            )
        heartbeat.stop(state="failed")
        return 1
    
    print(f"\n{'=' * 80}")
    print(f"Successfully completed: {display_name}")
    print(f"{'=' * 80}")
    record_terminal_status(
        output_dir,
        display_name,
        status="success",
        stage="save",
        reason="completed",
        message="Successfully completed",
        attempt=attempt,
    )
    heartbeat.stop(state="success", stage="save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
