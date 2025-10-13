#!/usr/bin/env python3
"""
Analyze and summarize results from ESD experiment.

This script:
1. Reads all per-model CSV files from results directory
2. Computes model-level summary statistics
3. Generates visualizations and reports
"""
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze ESD experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for summary (default: results_dir/summary.csv)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics"
    )
    return parser.parse_args()


def load_all_results(results_dir: Path) -> List[pd.DataFrame]:
    """Load all result CSV files."""
    csv_files = list(results_dir.glob("*.csv"))
    
    # Filter out special files
    csv_files = [
        f for f in csv_files
        if f.name not in ["summary.csv", "batch_meta.csv", "failed_models.txt"]
    ]
    
    if not csv_files:
        print(f"No result files found in {results_dir}")
        return []
    
    print(f"Found {len(csv_files)} result files")
    
    results = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                results.append(df)
        except Exception as e:
            warnings.warn(f"Could not load {csv_file.name}: {e}")
    
    print(f"Successfully loaded {len(results)} result files")
    return results


def compute_model_summary(df: pd.DataFrame) -> Dict:
    """Compute summary statistics for a single model."""
    summary = {}
    
    # Basic info
    if "model_id" in df.columns:
        summary["model_id"] = df["model_id"].iloc[0] if len(df) > 0 else "unknown"
    
    if "is_adapter" in df.columns:
        summary["is_adapter"] = df["is_adapter"].iloc[0] if len(df) > 0 else False
    
    if "source_model" in df.columns and "is_adapter" in df.columns:
        if df["is_adapter"].iloc[0]:
            summary["source_model"] = df["source_model"].iloc[0] if len(df) > 0 else ""
    
    # Count layers
    summary["num_layers"] = len(df)
    
    # Alpha statistics
    if "alpha" in df.columns:
        alpha_values = df["alpha"].dropna()
        if len(alpha_values) > 0:
            summary["alpha_mean"] = alpha_values.mean()
            summary["alpha_median"] = alpha_values.median()
            summary["alpha_std"] = alpha_values.std()
            summary["alpha_min"] = alpha_values.min()
            summary["alpha_max"] = alpha_values.max()
            summary["alpha_q25"] = alpha_values.quantile(0.25)
            summary["alpha_q75"] = alpha_values.quantile(0.75)
        else:
            summary["alpha_mean"] = np.nan
            summary["alpha_median"] = np.nan
            summary["alpha_std"] = np.nan
            summary["alpha_min"] = np.nan
            summary["alpha_max"] = np.nan
            summary["alpha_q25"] = np.nan
            summary["alpha_q75"] = np.nan
    
    # Alpha weighted statistics
    if "alpha_weighted" in df.columns:
        aw_values = df["alpha_weighted"].dropna()
        if len(aw_values) > 0:
            summary["alpha_weighted_mean"] = aw_values.mean()
            summary["alpha_weighted_median"] = aw_values.median()
    
    # Spectral norm statistics
    if "spectral_norm" in df.columns:
        sn_values = df["spectral_norm"].dropna()
        if len(sn_values) > 0:
            summary["spectral_norm_mean"] = sn_values.mean()
            summary["spectral_norm_max"] = sn_values.max()
    
    # Log spectral norm
    if "log_spectral_norm" in df.columns:
        lsn_values = df["log_spectral_norm"].dropna()
        if len(lsn_values) > 0:
            summary["log_spectral_norm_mean"] = lsn_values.mean()
    
    # Stable rank statistics
    if "stable_rank" in df.columns:
        sr_values = df["stable_rank"].dropna()
        if len(sr_values) > 0:
            summary["stable_rank_mean"] = sr_values.mean()
            summary["stable_rank_median"] = sr_values.median()
    
    # Entropy statistics
    if "entropy" in df.columns:
        ent_values = df["entropy"].dropna()
        if len(ent_values) > 0:
            summary["entropy_mean"] = ent_values.mean()
            summary["entropy_median"] = ent_values.median()
    
    # Parameter count
    if "params" in df.columns:
        params = df["params"].dropna()
        if len(params) > 0:
            summary["total_params"] = params.sum()
    
    return summary


def print_model_stats(summary_df: pd.DataFrame, verbose: bool = False):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal models analyzed: {len(summary_df)}")
    
    if "is_adapter" in summary_df.columns:
        num_adapters = summary_df["is_adapter"].sum()
        num_base = len(summary_df) - num_adapters
        print(f"  - Base models: {num_base}")
        print(f"  - Adapter models: {num_adapters}")
    
    if "num_layers" in summary_df.columns:
        print(f"\nLayers per model:")
        print(f"  Mean: {summary_df['num_layers'].mean():.1f}")
        print(f"  Median: {summary_df['num_layers'].median():.0f}")
        print(f"  Range: [{summary_df['num_layers'].min():.0f}, {summary_df['num_layers'].max():.0f}]")
    
    if "alpha_mean" in summary_df.columns:
        print(f"\nAlpha (model averages):")
        alpha_means = summary_df["alpha_mean"].dropna()
        if len(alpha_means) > 0:
            print(f"  Mean: {alpha_means.mean():.4f}")
            print(f"  Median: {alpha_means.median():.4f}")
            print(f"  Std: {alpha_means.std():.4f}")
            print(f"  Range: [{alpha_means.min():.4f}, {alpha_means.max():.4f}]")
    
    if "alpha_weighted_mean" in summary_df.columns:
        print(f"\nAlpha Weighted (model averages):")
        aw_means = summary_df["alpha_weighted_mean"].dropna()
        if len(aw_means) > 0:
            print(f"  Mean: {aw_means.mean():.4f}")
            print(f"  Median: {aw_means.median():.4f}")
    
    if "stable_rank_mean" in summary_df.columns:
        print(f"\nStable Rank (model averages):")
        sr_means = summary_df["stable_rank_mean"].dropna()
        if len(sr_means) > 0:
            print(f"  Mean: {sr_means.mean():.2f}")
            print(f"  Median: {sr_means.median():.2f}")
    
    if verbose and "model_id" in summary_df.columns:
        print("\n" + "=" * 80)
        print("TOP 10 MODELS BY ALPHA (LOWEST)")
        print("=" * 80)
        top_low = summary_df.nsmallest(10, "alpha_mean")[["model_id", "alpha_mean", "num_layers"]]
        print(top_low.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("TOP 10 MODELS BY ALPHA (HIGHEST)")
        print("=" * 80)
        top_high = summary_df.nlargest(10, "alpha_mean")[["model_id", "alpha_mean", "num_layers"]]
        print(top_high.to_string(index=False))
    
    print("\n" + "=" * 80)


def main():
    """Main analysis function."""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Load all results
    print(f"Loading results from: {results_dir}")
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("No results to analyze")
        return 1
    
    # Compute summaries
    print("\nComputing summary statistics...")
    summaries = []
    for df in all_results:
        try:
            summary = compute_model_summary(df)
            summaries.append(summary)
        except Exception as e:
            warnings.warn(f"Could not compute summary: {e}")
    
    if not summaries:
        print("Could not compute any summaries")
        return 1
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summaries)
    
    # Sort by model_id
    if "model_id" in summary_df.columns:
        summary_df = summary_df.sort_values("model_id").reset_index(drop=True)
    
    # Save summary
    output_path = Path(args.output) if args.output else results_dir / "summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")
    
    # Print statistics
    print_model_stats(summary_df, verbose=args.verbose)
    
    return 0


if __name__ == "__main__":
    exit(main())
