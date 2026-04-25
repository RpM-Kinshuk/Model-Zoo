#!/usr/bin/env python3
"""
Large-scale ESD analysis experiment with GPU resource management.

This script:
1. Reads a list of models from CSV
2. Dispatches analysis jobs to available GPUs using gputracker
3. Handles both standard models and PEFT adapters robustly
4. Supports resume by skipping already-analyzed models
5. Saves per-model ESD metrics as CSV files

Usage:
    python run_experiment.py --model_list models.csv --output_dir results/ --gpus 0 1 2 3
"""
import argparse
import importlib
import importlib.util
import itertools
import json
import os
import pandas as pd
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

# Add shells directory to path for gputracker
SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent  # Go up to ESD root
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "shells"))

from gputracker.gputracker import get_logger, DispatchThread, GPUDispatcher
from model_preflight import classify_row_preflight


BACKEND_PROBE_TIMEOUT_SECONDS = 30


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _backend_import_probe(module_names: tuple[str, ...]) -> bool:
    """Check backend imports in a GPU-hidden child process."""
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({module_name!r})" for module_name in module_names
    )
    env = os.environ.copy()
    # Optional quantization packages can initialize CUDA on import. Keep the
    # parent runner GPU-clean so the dispatcher can see truly free devices.
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=BACKEND_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _available_backends() -> set[str]:
    backends = set()
    if importlib.util.find_spec("gptqmodel") is not None:
        if _backend_import_probe(("gptqmodel",)):
            backends.add("gptq")
    if importlib.util.find_spec("autoawq") is not None:
        if _backend_import_probe(("autoawq",)):
            backends.add("awq")
    if importlib.util.find_spec("gguf") is not None:
        if _backend_import_probe(("gguf",)):
            backends.add("gguf")
    if importlib.util.find_spec("compressed_tensors") is not None:
        if _backend_import_probe(("compressed_tensors",)):
            backends.add("compressed_tensors")
        elif (
            importlib.util.find_spec("gptqmodel") is not None
            and _backend_import_probe(("gptqmodel", "compressed_tensors"))
        ):
            backends.add("gptq")
            backends.add("compressed_tensors")
    return backends


@dataclass(frozen=True)
class RunOutcomes:
    success_count: int
    failure_count: int
    completed_models: FrozenSet[str]
    failed_models: FrozenSet[str]


def _completed_models_from_artifacts(output_dir: Path) -> set[str]:
    completed = set()

    stats_dir = output_dir / "stats"
    metrics_dir = output_dir / "metrics"

    if stats_dir.exists():
        for csv_file in stats_dir.glob("*.csv"):
            metrics_file = metrics_dir / f"{csv_file.stem}.h5"
            if not metrics_file.exists():
                continue
            model_id = csv_file.stem.replace("--", "/").replace("__", "@")
            completed.add(model_id)

    return completed


def _decode_model_id_from_terminal_file(path: Path) -> str:
    return path.stem.replace("--", "/").replace("__", "@")


def _terminal_failed_models(output_dir: Path) -> set[str]:
    failed_models = set()
    terminal_dir = output_dir / "logs" / "terminal_status"
    if not terminal_dir.exists():
        return failed_models

    success_statuses = {"success", "succeeded", "completed", "done"}

    for status_file in terminal_dir.glob("*.json"):
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            status = _normalize_text(record.get("status") or record.get("state") or record.get("outcome"))
            if status in success_statuses or not status:
                continue
            model_id = _normalize_text(record.get("model_id")) or _decode_model_id_from_terminal_file(status_file)
            failed_models.add(model_id)

    return failed_models


def _legacy_failed_models(output_dir: Path) -> set[str]:
    failed_models = set()
    failed_file = output_dir / "logs" / "failed_models.txt"
    if not failed_file.exists():
        return failed_models

    for line in failed_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        model_id = line.strip().split("\t", 1)[0].strip()
        if model_id:
            failed_models.add(model_id)

    return failed_models


def collect_run_outcomes(output_dir: Path) -> RunOutcomes:
    completed_models = frozenset(_completed_models_from_artifacts(output_dir))
    failed_models = set(_terminal_failed_models(output_dir) | _legacy_failed_models(output_dir))
    failed_models -= set(completed_models)
    return RunOutcomes(
        success_count=len(completed_models),
        failure_count=len(failed_models),
        completed_models=completed_models,
        failed_models=frozenset(failed_models),
    )


def _row_backend_status(
    row: pd.Series,
    available_backends: set[str],
    effective_loader: str = "",
) -> str:
    backend_status = _normalize_text(row.get("backend_status"))

    loader_scenario = _normalize_text(row.get("loader_scenario"))
    resolved_loader = _normalize_text(effective_loader)
    tags_blob = " ".join(
        _normalize_text(value)
        for value in (
            row.get("tags"),
            row.get("tags_lb"),
            row.get("Type"),
            row.get("Type_lb"),
            row.get("model_id"),
            row.get("source_model"),
            row.get("base_model"),
            row.get("base_model_name_or_path"),
            row.get("parent_model"),
        )
        if _normalize_text(value)
    ).lower()
    file_blob = " ".join(
        _normalize_text(value)
        for value in (
            row.get("files"),
            row.get("file_names"),
            row.get("repo_file_names"),
            row.get("repo_files"),
        )
        if _normalize_text(value)
    ).lower()
    required_backend = ""
    if resolved_loader == "gguf" or loader_scenario == "gguf" or ".gguf" in file_blob:
        required_backend = "gguf"
    elif resolved_loader == "compressed_tensors" or "compressed-tensors" in tags_blob or "compressed_tensors" in tags_blob:
        required_backend = "compressed_tensors"
    elif resolved_loader in {"awq", "gptq"}:
        required_backend = resolved_loader
    elif "awq" in tags_blob:
        required_backend = "awq"
    elif "gptq" in tags_blob:
        required_backend = "gptq"
    if required_backend:
        return "available" if required_backend in available_backends else "missing"
    if backend_status:
        return backend_status
    if loader_scenario == "quantized_transformers_native":
        return ""
    return ""


def apply_preflight(model_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if model_df.empty:
        empty = model_df.iloc[0:0].copy()
        for column in ("preflight_eligible", "preflight_reason", "preflight_effective_loader"):
            empty[column] = []
        return empty, empty.copy()

    available_backends = _available_backends()
    runnable_rows = []
    blocked_rows = []

    for _, row in model_df.iterrows():
        row_dict = row.to_dict()
        initial_decision = classify_row_preflight(row_dict)
        row_dict["backend_status"] = _row_backend_status(
            row,
            available_backends,
            effective_loader=initial_decision.effective_loader,
        )
        decision = classify_row_preflight(row_dict)
        annotated_row = dict(row_dict)
        annotated_row["preflight_eligible"] = decision.eligible
        annotated_row["preflight_reason"] = decision.reason
        annotated_row["preflight_effective_loader"] = decision.effective_loader
        if decision.eligible:
            runnable_rows.append(annotated_row)
        else:
            blocked_rows.append(annotated_row)

    return pd.DataFrame(runnable_rows), pd.DataFrame(blocked_rows)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run large-scale ESD analysis with GPU resource management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Supported model list CSV formats:

        Minimal / legacy:
            model_id,base_model_relation,source_model
            meta-llama/Llama-2-7b-hf,,
            some/adapter-model,adapter,meta-llama/Llama-2-7b-hf
            org/model@revision,,

        Curated table (preferred):
            model_id,revision_norm,base_model_relation,source_model,loader_scenario,primary_type_bucket,files,pipeline_tag,Architecture,Available on the hub
            meta-llama/Llama-2-7b-hf,main,source,,,base_source
            some/adapter-model,main,adapter,meta-llama/Llama-2-7b-hf,adapter_requires_base,adapter
            org/model,commit-sha,,,,quantized

        Columns:
            - model_id: HuggingFace repo ID (required)
            - revision_norm: explicit revision override (optional)
            - base_model_relation: lineage / adapter relation (optional)
            - source_model: base model for adapters (optional, inferred when possible)
            - loader_scenario: curated loader hint such as standard_transformers or adapter_requires_base (optional)
            - primary_type_bucket: curated type bucket for logging / analysis (optional)
            - files / repo_files / file_names: optional artifact hints used by preflight for adapters and gguf
            - pipeline_tag / Architecture / model_type fields: optional routing hints for seq2seq, classification, or multimodal models
            - Available on the hub: optional curated availability gate

        Notes:
            - Preflight may replace loader_scenario with a more specific effective loader before dispatch.
            - Quantized-native rows are only backend-gated when they resolve to an explicit gptq or awq path.
        """
    )
    
    # Required arguments
    parser.add_argument("--model_list", type=str, required=True, help="Path to CSV file with model list (must have 'model_id' column)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    
    # GPU configuration
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="List of GPU indices to use (default: [0])")
    parser.add_argument("--num_gpus_per_job", type=int, default=1, help="Number of GPUs needed per model analysis job (default: 1)")
    parser.add_argument("--max_concurrent_jobs", type=int, default=None, help="Maximum number of model analysis jobs to run at once (default: GPU-limited)")
    parser.add_argument("--gpu_memory_threshold", type=int, default=500, help="GPU memory threshold in MB for considering GPU as free (default: 500)")
    parser.add_argument("--max_check", type=int, default=5, help="Number of checks to confirm GPU is free (default: 5)")
    
    # ESD configuration
    parser.add_argument("--fix_fingers", type=str, default="xmin_mid", choices=["xmin_mid", "xmin_peak", "DKS"], help="Method to select xmin for power law fitting (default: xmin_mid)")
    parser.add_argument("--evals_thresh", type=float, default=1e-5, help="Threshold for filtering eigenvalues (default: 1e-5)")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for histogram (default: 100)")
    parser.add_argument("--filter_zeros", action="store_true", default=True, help="Filter near-zero eigenvalues (default: True)")
    parser.add_argument("--use_svd", action="store_true", default=False, help="Use SVD for ESD (default: True)")
    parser.add_argument("--parallel_esd", action="store_true", default=True, help="Use parallel ESD computation across multiple GPUs (experimental)")
    
    # Experiment control
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N models (for testing)")
    parser.add_argument("--skip_failed", action="store_true", default=True, help="Skip models that previously failed (default: True)")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for logs (default: output_dir/logs)")
    
    return parser.parse_args()


def load_model_list(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load and validate model list from CSV.
    
    Expected columns:
        - model_id (required): HuggingFace repository ID
        - base_model_relation (optional): "adapter", "lora", "peft" for adapters
        - source_model (optional): Base model for adapters
        - revision_norm (optional): Curated revision override
        - loader_scenario (optional): Curated loader dispatch hint
        - primary_type_bucket (optional): Curated type bucket
        - optional routing/probe fields such as files, repo_files, pipeline_tag, Architecture,
          model_type/config_model_type, and Available on the hub
    
    Returns:
        DataFrame with model information
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if "model_id" not in df.columns:
        raise ValueError(
            f"CSV must have 'model_id' column. Found columns: {list(df.columns)}"
        )

    optional_columns = [
        "base_model_relation",
        "source_model",
        "revision_norm",
        "loader_scenario",
        "primary_type_bucket",
        "lineage_status",
        "candidate_source",
    ]
    for column in optional_columns:
        if column not in df.columns:
            df[column] = ""

    # Clean up data
    df["model_id"] = df["model_id"].map(_normalize_text)
    for column in optional_columns:
        df[column] = df[column].map(_normalize_text)
    
    # Remove empty rows
    df = df[df["model_id"] != ""]
    df = df[df["model_id"] != "nan"]
    
    # Apply limit
    if limit is not None and limit > 0:
        df = df.head(limit)
    
    return df


def generate_commands(model_df: pd.DataFrame, output_dir: Path, args) -> list:
    """
    Generate bash commands for each model analysis.
    
    Args:
        model_df: DataFrame with model information
        output_dir: Output directory path
        args: Command line arguments
    
    Returns:
        List of bash command strings
    """
    commands = []
    worker_script = SCRIPT_DIR / "worker.py"
    
    for idx, row in model_df.iterrows():
        model_id = _normalize_text(row["model_id"])
        base_relation = _normalize_text(row["base_model_relation"])
        source_model = _normalize_text(row["source_model"])
        revision_norm = _normalize_text(row.get("revision_norm", ""))
        loader_scenario = _normalize_text(
            row.get("preflight_effective_loader", "") or row.get("loader_scenario", "")
        )
        primary_type_bucket = _normalize_text(row.get("primary_type_bucket", ""))
        
        # Build command
        cmd_parts = [
            "python",
            str(worker_script),
            f"--model_id '{model_id}'",
            f"--output_dir '{output_dir}'",
            f"--fix_fingers {args.fix_fingers}",
            f"--evals_thresh {args.evals_thresh}",
            f"--bins {args.bins}",
        ]

        if args.filter_zeros: cmd_parts.append("--filter_zeros")
        if args.use_svd: cmd_parts.append("--use_svd")
        if args.parallel_esd: cmd_parts.append("--parallel_esd")
        if args.overwrite: cmd_parts.append("--overwrite")
        if revision_norm: cmd_parts.append(f"--revision '{revision_norm}'")
        if loader_scenario: cmd_parts.append(f"--loader_scenario '{loader_scenario}'")
        if primary_type_bucket: cmd_parts.append(f"--primary_type_bucket '{primary_type_bucket}'")
        if base_relation: cmd_parts.append(f"--base_model_relation '{base_relation}'")
        if source_model: cmd_parts.append(f"--source_model '{source_model}'")
        
        # Join command parts
        cmd = " ".join(cmd_parts)
        commands.append(cmd)
    
    return commands


def get_completed_models(output_dir: Path, skip_failed: bool = True) -> set:
    """
    Get set of already-completed model IDs.
    
    Args:
        output_dir: Output directory to check
        skip_failed: Whether to skip models in failed list
    
    Returns:
        Set of completed model IDs
    """
    completed = set()
    
    stats_dir = output_dir / "stats"
    metrics_dir = output_dir / "metrics"

    if stats_dir.exists():
        for csv_file in stats_dir.glob("*.csv"):
            metrics_file = metrics_dir / f"{csv_file.stem}.h5"
            if not metrics_file.exists():
                continue
            model_id = csv_file.stem.replace("--", "/").replace("__", "@")
            completed.add(model_id)
    
    # Remove failed models if requested
    if skip_failed:
        failed_file = output_dir / "logs" / "failed_models.txt"
        if failed_file.exists():
            with open(failed_file, "r") as f:
                failed = {
                    line.strip().split("\t", 1)[0]
                    for line in f
                    if line.strip()
                }
            completed -= failed
    
    return completed


def filter_models_to_run(model_df: pd.DataFrame, output_dir: Path, overwrite: bool = False, skip_failed: bool = True) -> pd.DataFrame:
    """
    Filter model list to only include models that need to be run.
    
    Args:
        model_df: Full model DataFrame
        output_dir: Output directory
        overwrite: Whether to overwrite existing results
        skip_failed: Whether to skip previously failed models
    
    Returns:
        Filtered DataFrame
    """
    if overwrite: return model_df
    
    completed = get_completed_models(output_dir, skip_failed)
    if not completed: return model_df
    
    # Filter out completed models
    mask = ~model_df["model_id"].isin(completed)
    filtered_df = model_df[mask].reset_index(drop=True)
    
    skipped = len(model_df) - len(filtered_df)
    if skipped > 0:
        print(f"Skipping {skipped} already-completed models")
    
    return filtered_df


# NEW: Helper to bootstrap the JSON config
def create_runtime_config(args, config_path):
    """
    Creates the dynamic config file based on CLI arguments.
    This allows the script to start with CLI args, but allows the user 
    to modify this file during runtime to change behavior.
    """
    config = {
        "available_gpus": args.gpus,
        "max_checks": args.max_check,
        "memory_threshold_mb": args.gpu_memory_threshold,
        "max_concurrent_jobs": args.max_concurrent_jobs,
    }
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Initialized runtime config at: {config_path}")
    except Exception as e:
        print(f"Warning: Could not create config file: {e}")


def main():
    """Main experiment runner."""
    args = parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = str(output_dir / "gpu_config.json")
    create_runtime_config(args, config_path)
    
    # Setup logger
    logger = get_logger(str(log_dir), "esd_experiment.log")

    dispatcher = GPUDispatcher(config_path=config_path)
    dispatcher.setup_signals()

    logger.info("=" * 80)
    logger.info("Starting ESD Experiment")
    logger.info(f"Controls: Edit {config_path} and send SIGHUP to reload GPUs")
    logger.info("=" * 80)
    logger.info(f"Model list: {args.model_list}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"GPUs per job: {args.num_gpus_per_job}")
    logger.info(f"Max concurrent jobs: {args.max_concurrent_jobs if args.max_concurrent_jobs is not None else 'GPU-limited'}")
    logger.info(f"Fix fingers: {args.fix_fingers}")
    logger.info("=" * 80)
    
    # Load model list
    logger.info("Loading model list...")
    model_df = load_model_list(args.model_list, limit=args.limit)
    logger.info(f"Loaded {len(model_df)} models from CSV")
    
    # Filter models to run
    model_df = filter_models_to_run(
        model_df,
        output_dir,
        overwrite=args.overwrite,
        skip_failed=args.skip_failed
    )

    model_df, blocked_df = apply_preflight(model_df)
    if len(blocked_df) > 0:
        logger.info(f"Blocked by preflight: {len(blocked_df)} models")
        logger.info(
            "Preflight reasons: "
            + ", ".join(sorted(set(blocked_df["preflight_reason"].astype(str).tolist())))
        )
    
    if len(model_df) == 0:
        logger.info("No models to process (all completed, skipped, or blocked by preflight)")
        outcomes = collect_run_outcomes(output_dir)
        logger.info(f"Successfully analyzed: {outcomes.success_count} models")
        logger.info(f"Failed: {outcomes.failure_count} models")
        logger.info(f"Results saved to: {output_dir}")
        return
    
    logger.info(f"Will process {len(model_df)} models")
    
    # Generate commands
    logger.info("Generating commands...")
    commands = generate_commands(model_df, output_dir, args)
    logger.info(f"Generated {len(commands)} commands")
    
    # Create and start dispatch thread
    logger.info("Starting GPU dispatch thread...")
    dispatch_thread = DispatchThread(
        name="ESD Analysis",
        bash_command_list=commands,
        logger=logger,
        dispatcher=dispatcher,
        config_path=config_path,
        num_gpus_needed=args.num_gpus_per_job,
        max_concurrent_jobs=args.max_concurrent_jobs,
    )
    
    # Start and wait for completion
    dispatch_thread.start()
    while dispatch_thread.is_alive():
        dispatch_thread.join(timeout=0.5)
    
    logger.info("=" * 80)
    logger.info("Experiment completed!")
    logger.info("=" * 80)
    
    # Print summary
    outcomes = collect_run_outcomes(output_dir)
    logger.info(f"Successfully analyzed: {outcomes.success_count} models")
    logger.info(f"Failed: {outcomes.failure_count} models")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
