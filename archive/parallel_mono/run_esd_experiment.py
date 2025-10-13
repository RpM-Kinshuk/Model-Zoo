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
    python run_esd_experiment.py --model_list models.csv --output_dir results/ --gpus 0 1 2 3
"""
import sys
import os
import argparse
import pandas as pd
import itertools
from pathlib import Path
from typing import Optional

# Add shells directory to path for gputracker
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "shells"))

from gputracker.gputracker import get_logger, DispatchThread


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run large-scale ESD analysis with GPU resource management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example model list CSV format:
    model_id,base_model_relation,source_model
    meta-llama/Llama-2-7b-hf,,
    some/adapter-model,adapter,meta-llama/Llama-2-7b-hf
    org/model@revision,,

Columns:
    - model_id: HuggingFace repo ID (required)
    - base_model_relation: "adapter", "lora", "peft" for adapters (optional)
    - source_model: Base model for adapters (optional, will be inferred if missing)
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_list",
        type=str,
        required=True,
        help="Path to CSV file with model list (must have 'model_id' column)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="List of GPU indices to use (default: [0])"
    )
    parser.add_argument(
        "--num_gpus_per_job",
        type=int,
        default=1,
        help="Number of GPUs needed per model analysis job (default: 1)"
    )
    parser.add_argument(
        "--gpu_memory_threshold",
        type=int,
        default=500,
        help="GPU memory threshold in MB for considering GPU as free (default: 500)"
    )
    parser.add_argument(
        "--max_check",
        type=int,
        default=10,
        help="Number of checks to confirm GPU is free (default: 10)"
    )
    
    # ESD configuration
    parser.add_argument(
        "--fix_fingers",
        type=str,
        default="xmin_mid",
        choices=["xmin_mid", "xmin_peak", "DKS"],
        help="Method to select xmin for power law fitting (default: xmin_mid)"
    )
    parser.add_argument(
        "--evals_thresh",
        type=float,
        default=1e-5,
        help="Threshold for filtering eigenvalues (default: 1e-5)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins for histogram (default: 100)"
    )
    parser.add_argument(
        "--filter_zeros",
        action="store_true",
        default=True,
        help="Filter near-zero eigenvalues (default: True)"
    )
    parser.add_argument(
        "--parallel_esd",
        action="store_true",
        help="Use parallel ESD computation across multiple GPUs (experimental)"
    )
    
    # Experiment control
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N models (for testing)"
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        default=True,
        help="Skip models that previously failed (default: True)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for logs (default: output_dir/logs)"
    )
    
    return parser.parse_args()


def load_model_list(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load and validate model list from CSV.
    
    Expected columns:
        - model_id (required): HuggingFace repository ID
        - base_model_relation (optional): "adapter", "lora", "peft" for adapters
        - source_model (optional): Base model for adapters
    
    Returns:
        DataFrame with model information
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if "model_id" not in df.columns:
        raise ValueError(
            f"CSV must have 'model_id' column. Found columns: {list(df.columns)}"
        )
    
    # Add optional columns if missing
    if "base_model_relation" not in df.columns:
        df["base_model_relation"] = ""
    if "source_model" not in df.columns:
        df["source_model"] = ""
    
    # Clean up data
    df["model_id"] = df["model_id"].astype(str).str.strip()
    df["base_model_relation"] = df["base_model_relation"].fillna("").astype(str).str.strip()
    df["source_model"] = df["source_model"].fillna("").astype(str).str.strip()
    
    # Remove empty rows
    df = df[df["model_id"] != ""]
    df = df[df["model_id"] != "nan"]
    
    # Apply limit
    if limit is not None and limit > 0:
        df = df.head(limit)
    
    return df


def generate_commands(
    model_df: pd.DataFrame,
    output_dir: Path,
    args
) -> list:
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
    worker_script = SCRIPT_DIR / "esd_worker.py"
    
    for idx, row in model_df.iterrows():
        model_id = row["model_id"]
        base_relation = row["base_model_relation"]
        source_model = row["source_model"]
        
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
        
        if args.filter_zeros:
            cmd_parts.append("--filter_zeros")
        
        if args.parallel_esd:
            cmd_parts.append("--parallel_esd")
        
        if args.overwrite:
            cmd_parts.append("--overwrite")
        
        if base_relation:
            cmd_parts.append(f"--base_model_relation '{base_relation}'")
        
        if source_model:
            cmd_parts.append(f"--source_model '{source_model}'")
        
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
    
    # Check for existing result CSVs
    if output_dir.exists():
        for csv_file in output_dir.glob("*.csv"):
            # Skip special files
            if csv_file.name in ["failed_models.txt", "summary.csv"]:
                continue
            # Extract model ID from filename (reverse safe_filename transformation)
            model_id = csv_file.stem.replace("--", "/").replace("__", "@")
            completed.add(model_id)
    
    # Remove failed models if requested
    if skip_failed:
        failed_file = output_dir / "failed_models.txt"
        if failed_file.exists():
            with open(failed_file, "r") as f:
                failed = {line.strip() for line in f if line.strip()}
            completed -= failed
    
    return completed


def filter_models_to_run(
    model_df: pd.DataFrame,
    output_dir: Path,
    overwrite: bool = False,
    skip_failed: bool = True
) -> pd.DataFrame:
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
    if overwrite:
        return model_df
    
    completed = get_completed_models(output_dir, skip_failed)
    
    if not completed:
        return model_df
    
    # Filter out completed models
    mask = ~model_df["model_id"].isin(completed)
    filtered_df = model_df[mask].reset_index(drop=True)
    
    skipped = len(model_df) - len(filtered_df)
    if skipped > 0:
        print(f"Skipping {skipped} already-completed models")
    
    return filtered_df


def main():
    """Main experiment runner."""
    args = parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_logger(str(log_dir), "esd_experiment.log")
    logger.info("=" * 80)
    logger.info("Starting ESD Experiment")
    logger.info("=" * 80)
    logger.info(f"Model list: {args.model_list}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"GPUs per job: {args.num_gpus_per_job}")
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
    
    if len(model_df) == 0:
        logger.info("No models to process (all completed or skipped)")
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
        gpu_m_th=args.gpu_memory_threshold,
        gpu_list=args.gpus,
        maxcheck=args.max_check,
        num_gpus_needed=args.num_gpus_per_job,
    )
    
    # Start and wait for completion
    dispatch_thread.start()
    dispatch_thread.join()
    
    logger.info("=" * 80)
    logger.info("Experiment completed!")
    logger.info("=" * 80)
    
    # Print summary
    completed = get_completed_models(output_dir, skip_failed=False)
    failed_file = output_dir / "failed_models.txt"
    num_failed = 0
    if failed_file.exists():
        with open(failed_file, "r") as f:
            num_failed = sum(1 for line in f if line.strip())
    
    logger.info(f"Successfully analyzed: {len(completed)} models")
    logger.info(f"Failed: {num_failed} models")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
