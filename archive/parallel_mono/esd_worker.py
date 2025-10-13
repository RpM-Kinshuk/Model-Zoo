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

import torch
import pandas as pd
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from model_loader import load_model, parse_model_string, safe_filename
from net_esd import net_esd_estimator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze single model ESD metrics")
    
    # Model specification
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--base_model_relation", type=str, default="", help="Adapter relation type")
    parser.add_argument("--source_model", type=str, default="", help="Base model for adapters")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    
    # ESD parameters
    parser.add_argument("--fix_fingers", type=str, default="xmin_mid", help="xmin selection method")
    parser.add_argument("--evals_thresh", type=float, default=1e-5, help="Eigenvalue threshold")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins")
    parser.add_argument("--filter_zeros", action="store_true", default=True, help="Filter zeros")
    parser.add_argument("--parallel_esd", action="store_true", default=True, help="Use parallel ESD")
    
    # Model loading
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for loading (auto uses GPU when CUDA_VISIBLE_DEVICES is set)")
    parser.add_argument("--max_retries", type=int, default=2, help="Max retry attempts")
    
    return parser.parse_args()


def save_results(
    metrics: dict,
    output_path: Path,
    model_id: str,
    is_adapter: bool,
    source_model: Optional[str] = None
):
    """
    Save ESD metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics from net_esd_estimator
        output_path: Path to save CSV
        model_id: Model identifier
        is_adapter: Whether this is an adapter model
        source_model: Base model for adapters
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


def record_failure(output_dir: Path, model_id: str, error: str):
    """Record failed model in failed_models.txt."""
    failed_file = output_dir / "failed_models.txt"
    failed_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(failed_file, "a") as f:
        f.write(f"{model_id}\t{error}\n")
    
    print(f"Recorded failure for {model_id}")


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
    
    # Parse model ID (may include revision)
    repo_id, revision = parse_model_string(args.model_id)
    display_name = args.model_id
    
    # Setup output path
    output_dir = Path(args.output_dir)
    output_file = output_dir / f"{safe_filename(args.model_id)}.csv"
    
    # Check if already done
    if output_file.exists() and not args.overwrite:
        print(f"Results already exist: {output_file}")
        return 0
    
    # Create placeholder to indicate work in progress
    if not args.overwrite:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.touch()
    
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
        try:
            print(f"\nAttempt {attempt}/{args.max_retries + 1}")
            
            # Load model
            print(f"Loading model: {repo_id}")
            base_relation = args.base_model_relation if args.base_model_relation else None
            source_model = args.source_model if args.source_model else None
            
            model, is_adapter = load_model(
                repo_id=repo_id,
                base_model_relation=base_relation,
                source_model=source_model,
                device_map=args.device_map,
                torch_dtype=torch.float16,
                revision=revision,
            )
            
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
            
            metrics = net_esd_estimator(
                model,
                EVALS_THRESH=args.evals_thresh,
                bins=args.bins,
                fix_fingers=fix_fingers_value,
                filter_zeros=args.filter_zeros,
                parallel=args.parallel_esd,
            )
            
            print(f"ESD analysis completed successfully")
            print(f"Analyzed {len(metrics.get('longname', []))} layers")
            
            # Save results
            save_results(
                metrics,
                output_file,
                display_name,
                is_adapter,
                source_model if is_adapter else None
            )
            
            success = True
            break
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            cleanup_model(model)
            if output_file.exists() and output_file.stat().st_size == 0:
                output_file.unlink()
            return 1
            
        except Exception as e:
            error_msg = str(e)
            print(f"\nAttempt {attempt} failed: {error_msg}")
            
            if attempt <= args.max_retries:
                print("Retrying...")
                warnings.warn(f"Attempt {attempt} failed for {display_name}: {error_msg}")
            else:
                print("\nAll attempts failed!")
                print("Full traceback:")
                traceback.print_exc()
                
                # Record failure
                record_failure(output_dir, display_name, error_msg)
        
        finally:
            # Cleanup
            cleanup_model(model)
    
    # Final cleanup of empty file if failed
    if not success:
        if output_file.exists() and output_file.stat().st_size == 0:
            output_file.unlink()
        print(f"\nFailed to analyze {display_name}")
        return 1
    
    print(f"\n{'=' * 80}")
    print(f"Successfully completed: {display_name}")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
