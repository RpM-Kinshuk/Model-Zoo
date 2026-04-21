#!/usr/bin/env python3
"""
Worker script for analyzing a single model.
This is called by the main experiment runner for each model.
"""
import sys
import os
import argparse
import warnings
import traceback
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
    parser.add_argument("--fix_fingers", type=str, default="xmin_mid", help="xmin selection method")
    parser.add_argument("--evals_thresh", type=float, default=1e-5, help="Eigenvalue threshold")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins")
    parser.add_argument("--filter_zeros", action="store_true", default=True, help="Filter zeros")
    parser.add_argument("--parallel_esd", action="store_true", default=True, help="Use parallel ESD")
    parser.add_argument("--use_svd", action="store_true", default=True, help="Use SVD for ESD")
    
    # Model loading
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for loading (auto uses GPU when CUDA_VISIBLE_DEVICES is set)")
    parser.add_argument("--max_retries", type=int, default=0, help="Max retry attempts")
    
    return parser.parse_args()


def resolve_model_revision(model_id: str, revision_override: str = ""):
    """Resolve repo ID and effective revision, honoring curated overrides."""
    repo_id, revision = parse_model_string(model_id)
    if revision_override and revision_override.strip():
        revision = revision_override.strip()
    return repo_id, revision


# ------------------------------------------------------------
# Minimal helpers to build and save alpha matrices as .h5 files
# (mirrors ESD-Independence/Classification/run_metric.py format)
# ------------------------------------------------------------

PREFIX_CANDIDATES = {"layers", "layer", "h", "block", "blocks"}

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

def build_tensor_from_pairs(longnames, alphas):
    """
    Convert (longname, alpha) lists into a dense matrix:
      - Deduplicate (layer, module) by averaging alpha.
      - Rows: 0..max_layer; Columns: sorted unique module names.
    Returns (mat [L,M], module_names [list[str]], num_layers [int]).
    """
    if (not longnames) or (not alphas) or (len(longnames) != len(alphas)):
        raise RuntimeError("after deduplication and averaging")

    df = pd.DataFrame({"longname": longnames, "alpha": alphas})
    df = df.dropna(subset=["longname", "alpha"])  # type: ignore[arg-type]

    parsed = df["longname"].apply(parse_longname)
    df["layer"] = [p[0] for p in parsed]
    df["module"] = [p[1] for p in parsed]
    df = df.dropna(subset=["layer", "module"])  # type: ignore[arg-type]
    df["layer"] = df["layer"].astype(int)

    if df.empty:
        raise RuntimeError("No valid (layer, module) rows after parsing longname")

    df_pairs = (
        df.groupby(["layer", "module"], as_index=False)
          .agg(alpha=("alpha", "mean"))
          .sort_values(by=["layer", "module"]).reset_index(drop=True)
    )

    module_names = sorted(df_pairs["module"].unique().tolist())
    num_modules = len(module_names)
    num_layers = int(df_pairs["layer"].max()) + 1

    mat = np.full((num_layers, num_modules), np.nan, dtype=float)
    module_index = {m: j for j, m in enumerate(module_names)}
    for _, row in df_pairs.iterrows():
        i = int(row["layer"]); j = module_index[row["module"]]
        mat[i, j] = float(row["alpha"])

    return mat, module_names, num_layers

def save_h5(h5_path: Path, mat: np.ndarray, module_names, num_layers: int, file_attrs: dict):
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        dset = h5.create_dataset("alpha", data=mat)
        dset.attrs["num_layers"] = int(num_layers)
        dset.attrs["num_modules"] = int(len(module_names))
        dset.attrs["missing_value"] = "NaN"
        dset.attrs["module_names_json"] = json.dumps(module_names, ensure_ascii=False)
        h5.attrs["format_version"] = "1.0"
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
):
    """
    Save ESD metrics to CSV file and write alpha matrix to HDF5.
    
    Args:
        metrics: Dictionary of metrics from net_esd_estimator
        output_path: Path to save CSV
        model_id: Model identifier
        is_adapter: Whether this is an adapter model
        source_model: Base model for adapters
        base_model_relation: Relation tag (e.g., adapter/base/finetune)
        fix_fingers: xmin strategy used (DKS/xmin_mid/xmin_peak)
    """
    # Prepare data for DataFrame
    data = {}
    
    # Get layer names
    longnames = metrics.get("longname", [])
    
    # Add all metrics
    for key in metrics.keys():
        if key == "eigs":
            # Skip raw eigenvalues (too large)
            continue
        values = metrics[key]
        
        # Handle summary row (last element is often aggregate)
        if key == "longname" and values and values[-1] is None:
            values = values[:-1]
        elif len(values) > len(longnames):
            values = values[:len(longnames)]
        
        data[key] = values
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
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
        alpha_values = df["alpha"].dropna()
        if len(alpha_values) > 0:
            print(f"Alpha statistics:")
            print(f"  Mean: {alpha_values.mean():.4f}")
            print(f"  Median: {alpha_values.median():.4f}")
            print(f"  Std: {alpha_values.std():.4f}")
            print(f"  Range: [{alpha_values.min():.4f}, {alpha_values.max():.4f}]")
            print(f"  Layers: {len(alpha_values)}")

    # ---- Also write per-model H5 (alpha matrix) in output_dir/metrics ----
    longnames = metrics.get("longname", [])
    alphas = metrics.get("alpha", [])
    # strip trailing None if present
    if longnames and longnames[-1] is None:
        longnames = longnames[:-1]
    if alphas and alphas[-1] is None:
        alphas = alphas[:-1]
    if len(longnames) != len(alphas):
        n = min(len(longnames), len(alphas))
        longnames, alphas = longnames[:n], alphas[:n]

    if longnames and alphas:
        mat, module_names, num_layers = build_tensor_from_pairs(longnames, alphas)
        if h5_output_path is None:
            h5_dir = output_path.parent.parent / "metrics"
            h5_path = h5_dir / f"{safe_filename(model_id)}.h5"
        else:
            h5_path = h5_output_path
        relation_attr = base_model_relation.strip() or ("adapter" if is_adapter else "base")
        file_attrs = {
            "full_name": model_id,
            "source_model": source_model or "",
            "base_model_relation": relation_attr,
            "fix_fingers": fix_fingers,
            "alpha_only": "true",
        }
        save_h5(h5_path, mat, module_names, num_layers, file_attrs)
        print(f"Saved H5 alpha matrix to: {h5_path}")
    else:
        print("Skipping H5 save (no longname/alpha)")


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
    longnames = metrics.get("longname", []) or []
    alphas = metrics.get("alpha", []) or []
    if longnames and longnames[-1] is None:
        longnames = longnames[:-1]
    if alphas and alphas[-1] is None:
        alphas = alphas[:-1]
    if not longnames:
        return ("analyze", "analysis_empty")
    usable_pairs = zip(longnames, alphas)
    usable_alpha_count = sum(
        1
        for longname, alpha in usable_pairs
        if longname is not None and not pd.isna(alpha)
    )
    if usable_alpha_count == 0:
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
    
    # Parse model ID (may include revision) and allow curated revision override
    repo_id, revision = resolve_model_revision(args.model_id, args.revision)
    display_name = args.model_id
    
    # Setup output path
    output_dir = Path(args.output_dir)
    output_file = output_dir / "stats" / f"{safe_filename(args.model_id)}.csv"
    temp_output_file = temp_output_path(output_file)
    metrics_file = output_dir / "metrics" / f"{safe_filename(args.model_id)}.h5"
    temp_metrics_file = temp_output_path(metrics_file)
    
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
            if output_file.exists() and metrics_file.exists():
                print(f"Results already exist: {output_file}")
                return 0
            if output_file.exists() or metrics_file.exists():
                print("Incomplete existing outputs detected; clearing stale artifacts before regeneration")
                cleanup_output_artifacts(
                    temp_output_file,
                    temp_metrics_file,
                    output_file,
                    metrics_file,
                )
    except Exception as exc:
        stage, reason, message = classify_runtime_error("save", exc)
        record_failure(output_dir, display_name, stage, reason, message, attempt=0)
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
                    torch_dtype=torch.float16,
                    revision=revision,
                    loader_scenario=args.loader_scenario if args.loader_scenario else None,
                )
            except LoaderFailure as exc:
                raise exc
            except Exception as exc:
                stage, reason, message = classify_runtime_error("load", exc)
                raise LoaderFailure(stage, reason, message) from exc
            
            current_stage = "analyze"
            print(f"Model loaded successfully (adapter: {is_adapter})")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Report which device the model is on
            model_devices = set()
            for param in model.parameters():
                model_devices.add(str(param.device))
            print(f"Model devices: {', '.join(sorted(model_devices))}")
            
            # Run ESD analysis
            print("\nRunning ESD analysis...")
            fix_fingers_value = None if args.fix_fingers == "DKS" else args.fix_fingers
            
            try:
                metrics = net_esd_estimator(
                    model,
                    EVALS_THRESH=args.evals_thresh,
                    bins=args.bins,
                    fix_fingers=fix_fingers_value,
                    filter_zeros=args.filter_zeros,
                    use_svd=args.use_svd,
                    parallel=args.parallel_esd,
                )
            except Exception as exc:
                stage, reason, message = classify_runtime_error("analyze", exc)
                raise LoaderFailure(stage, reason, message) from exc

            validation_failure = validate_metrics_output(metrics)
            if validation_failure is not None:
                stage, reason = validation_failure
                raise LoaderFailure(stage, reason, "ESD analysis returned no layer metrics")
            
            print(f"ESD analysis completed successfully")
            print(f"Analyzed {len(metrics.get('longname', []))} layers")
            
            # Save results
            current_stage = "save"
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
                break
        
        finally:
            # Cleanup
            cleanup_model(model)
    
    cleanup_temp_path(temp_output_file)
    cleanup_temp_path(temp_metrics_file)

    if not success:
        print(f"\nFailed to analyze {display_name}")
        return 1
    
    print(f"\n{'=' * 80}")
    print(f"Successfully completed: {display_name}")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
