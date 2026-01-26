#!/usr/bin/env python3
"""
Interactive scatter plot to compare a metric across two ESD result folders.

Creates a scatter plot of metric values from folder A (x-axis) vs folder B (y-axis),
with identity line, correlation stats, and optional density coloring.

Usage:
    # Save to file:
    python plot_metric_comparison.py \
        --dir_a path/to/svd_results \
        --dir_b path/to/gram_results \
        --metric alpha \
        --output plot.png

    # Interactive mode (use in Jupyter or with working display):
    python plot_metric_comparison.py \
        --dir_a path/to/svd_results \
        --dir_b path/to/gram_results \
        --metric alpha \
        --interactive

    # Or import and use in Jupyter:
    from plot_metric_comparison import plot_interactive
    plot_interactive('path/to/svd', 'path/to/gram', 'alpha')
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LogNorm
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    exit(1)

EPS = 1e-12


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ("longname", "model_id", "source_model"):
            continue
        if df[c].dtype.kind not in "biufc":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_and_align(a_csv: Path, b_csv: Path, metric: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load two CSVs, align by longname, and extract metric arrays.
    Returns (x_values, y_values, model_stem).
    """
    a = pd.read_csv(a_csv)
    b = pd.read_csv(b_csv)

    if "longname" not in a.columns or "longname" not in b.columns:
        raise ValueError(f"Missing 'longname' in {a_csv} or {b_csv}")

    a = _coerce_numeric_columns(a)
    b = _coerce_numeric_columns(b)

    # Deduplicate by averaging
    def dedup(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["longname"])
        num_cols = [c for c in df.columns if c != "longname" and df[c].dtype.kind in "biufc"]
        agg = df.groupby("longname", as_index=False)[num_cols].mean()
        return agg

    a_d = dedup(a)
    b_d = dedup(b)

    merged = pd.merge(a_d, b_d, on="longname", how="inner", suffixes=("_a", "_b"))

    if f"{metric}_a" not in merged.columns or f"{metric}_b" not in merged.columns:
        raise ValueError(f"Metric '{metric}' not found in both CSVs: {a_csv.stem}")

    x = merged[f"{metric}_a"].to_numpy(dtype=float)
    y = merged[f"{metric}_b"].to_numpy(dtype=float)

    return x, y, a_csv.stem


def find_matching_csvs(dir_a: Path, dir_b: Path) -> List[Tuple[Path, Path]]:
    a_map = {p.stem: p for p in dir_a.glob("*.csv")}
    b_map = {p.stem: p for p in dir_b.glob("*.csv")}
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    return [(a_map[k], b_map[k]) for k in common]


def compute_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    cx = x - x.mean()
    cy = y - y.mean()
    vx = np.dot(cx, cx)
    vy = np.dot(cy, cy)
    if vx <= EPS or vy <= EPS:
        return np.nan
    return float(np.dot(cx, cy) / np.sqrt(vx * vy))


def compute_rmse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y - x) ** 2)))


def compute_smape(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    abs_d = np.abs(y - x)
    return float(np.mean(2.0 * abs_d / (np.abs(x) + np.abs(y) + EPS)))


def plot_scatter(
    x_all: np.ndarray,
    y_all: np.ndarray,
    metric: str,
    ref: str,
    style: str,
    alpha_points: float,
    show_models: bool,
    model_labels: List[str],
    output_path: Path = None,
    interactive: bool = False,
):
    """Create scatter plot comparing ref vs test metric values.
    
    Args:
        output_path: If provided, saves to file. If None and interactive=True, shows plot.
        interactive: If True, displays interactive plot window (use plt.show())
    """
    # Filter finite values
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    x = x_all[mask]
    y = y_all[mask]
    if show_models:
        labels_filtered = [model_labels[i] for i in range(len(mask)) if mask[i]]
    else:
        labels_filtered = []

    if x.size == 0:
        print(f"No finite values to plot for metric '{metric}'")
        return None

    # Compute stats
    pearson = compute_pearson(x, y)
    rmse = compute_rmse(x, y)
    smape = compute_smape(x, y)

    # Determine axis limits with padding
    all_vals = np.concatenate([x, y])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    margin = (vmax - vmin) * 0.05
    lim_min = vmin - margin
    lim_max = vmax + margin

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot based on style
    if style == "scatter":
        scatter = ax.scatter(x, y, alpha=alpha_points, s=15, edgecolors="none", c="steelblue", picker=True)
    elif style == "hexbin":
        # Hexbin with log color scale for density
        hb = ax.hexbin(x, y, gridsize=50, cmap="Blues", mincnt=1, norm=LogNorm())
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Count (log scale)")
    elif style == "density":
        # 2D histogram with color intensity
        h, xedges, yedges = np.histogram2d(x, y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(
            h.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="Blues",
            interpolation="nearest",
        )
    else:
        raise ValueError(f"Unknown style: {style}")

    # Identity line
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2, alpha=0.7, label="y = x (identity)")

    # Labels and title
    label_a = "Reference (A)" if ref == "a" else "Test (A)"
    label_b = "Test (B)" if ref == "a" else "Reference (B)"
    ax.set_xlabel(f"{metric} — {label_a}", fontsize=13)
    ax.set_ylabel(f"{metric} — {label_b}", fontsize=13)
    ax.set_title(f"Comparison: {metric}\n(n={x.size} layers)", fontsize=15, fontweight="bold")

    # Stats annotation (make it moveable by clicking)
    stats_text = f"Pearson r = {pearson:.4f}\nRMSE = {rmse:.4g}\nsMAPE = {smape:.4f}"
    stats_box = ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Grid and limits
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
        if not interactive:
            plt.close(fig)
    
    if interactive:
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Use the toolbar to:")
        print("  - Pan/Zoom: Click the pan icon, then drag to move or zoom")
        print("  - Zoom box: Click zoom icon, drag rectangle to zoom in")
        print("  - Home: Reset view to original")
        print("  - Back/Forward: Navigate zoom history")
        print("  - Save: Save current view to file")
        print("="*60)
        print(f"Viewing {x.size} data points")
        print(f"Stats: Pearson r={pearson:.4f}, RMSE={rmse:.4g}, sMAPE={smape:.4f}")
        print("="*60 + "\n")
        plt.show()
        return fig
    else:
        plt.close(fig)
        return None


def plot_interactive(dir_a: str, dir_b: str, metric: str, ref: str = "a", style: str = "scatter", alpha_points: float = 0.3):
    """
    Load data and create interactive plot. Use this function in Jupyter notebooks.
    
    Args:
        dir_a: Path to folder A (e.g., SVD results)
        dir_b: Path to folder B (e.g., Gram results) 
        metric: Metric name to compare (e.g., 'alpha', 'spectral_norm')
        ref: Which folder is reference, 'a' or 'b' (default: 'a')
        style: Plot style - 'scatter', 'hexbin', or 'density' (default: 'scatter')
        alpha_points: Transparency for scatter points, 0-1 (default: 0.3)
    
    Returns:
        matplotlib Figure object (for Jupyter display)
    
    Example in Jupyter:
        >>> from plot_metric_comparison import plot_interactive
        >>> %matplotlib widget  # or %matplotlib notebook
        >>> fig = plot_interactive('svd_results/', 'gram_results/', 'alpha')
    """
    dir_a = Path(dir_a).resolve()
    dir_b = Path(dir_b).resolve()
    
    pairs = find_matching_csvs(dir_a, dir_b)
    if not pairs:
        print(f"No matching CSVs found between {dir_a} and {dir_b}")
        return None
    
    print(f"Found {len(pairs)} matching models")
    print(f"Comparing metric: {metric}")
    
    # Collect all values across models
    x_all = []
    y_all = []
    model_labels = []
    
    for a_csv, b_csv in pairs:
        try:
            x, y, model_stem = load_and_align(a_csv, b_csv, metric)
            # Swap if ref is b
            if ref == "b":
                x, y = y, x
            x_all.append(x)
            y_all.append(y)
            model_labels.extend([model_stem] * len(x))
        except Exception as e:
            warnings.warn(f"Skipping {a_csv.stem}: {e}")
    
    if not x_all:
        print(f"No data loaded for metric '{metric}'")
        return None
    
    x_concat = np.concatenate(x_all, axis=0)
    y_concat = np.concatenate(y_all, axis=0)
    
    return plot_scatter(
        x_concat,
        y_concat,
        metric,
        ref,
        style,
        alpha_points,
        False,
        model_labels,
        output_path=None,
        interactive=True,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Plot scatter comparison of a metric across two ESD result folders",
        epilog="""
Examples:
  # Save to file:
  python plot_metric_comparison.py --dir_a svd/ --dir_b gram/ --metric alpha --output alpha.png
  
  # Interactive mode (if display available):
  python plot_metric_comparison.py --dir_a svd/ --dir_b gram/ --metric alpha --interactive
  
  # In Jupyter notebook:
  from plot_metric_comparison import plot_interactive
  plot_interactive('svd/', 'gram/', 'alpha')
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir_a", required=True, type=str, help="Folder A (e.g., SVD results)")
    ap.add_argument("--dir_b", required=True, type=str, help="Folder B (e.g., Gram results)")
    ap.add_argument("--metric", required=True, type=str, help="Metric to compare (e.g., alpha, spectral_norm)")
    ap.add_argument("--output", type=str, help="Output plot file path (e.g., alpha_comparison.png)")
    ap.add_argument("--interactive", action="store_true", help="Show interactive plot window (requires working display)")
    ap.add_argument(
        "--ref", choices=["a", "b"], default="a", help="Which folder is reference (default: a)"
    )
    ap.add_argument(
        "--style",
        choices=["scatter", "hexbin", "density"],
        default="scatter",
        help="Plot style: scatter (points), hexbin (hex density), density (2D histogram)",
    )
    ap.add_argument(
        "--alpha_points",
        type=float,
        default=0.3,
        help="Transparency for scatter points (0-1, default: 0.3)",
    )
    ap.add_argument(
        "--show_models",
        action="store_true",
        help="Show model labels on hover (not implemented; reserved for interactive mode)",
    )
    args = ap.parse_args()
    
    if not args.output and not args.interactive:
        ap.error("Either --output or --interactive must be specified")

    dir_a = Path(args.dir_a).resolve()
    dir_b = Path(args.dir_b).resolve()

    pairs = find_matching_csvs(dir_a, dir_b)
    if not pairs:
        print(f"No matching CSVs found between {dir_a} and {dir_b}")
        return

    print(f"Found {len(pairs)} matching models")
    print(f"Comparing metric: {args.metric}")

    # Collect all values across models
    x_all = []
    y_all = []
    model_labels = []

    suffix_ref = "_a" if args.ref == "a" else "_b"
    suffix_tst = "_b" if args.ref == "a" else "_a"

    for a_csv, b_csv in pairs:
        try:
            x, y, model_stem = load_and_align(a_csv, b_csv, args.metric)
            # Swap if ref is b
            if args.ref == "b":
                x, y = y, x
            x_all.append(x)
            y_all.append(y)
            model_labels.extend([model_stem] * len(x))
        except Exception as e:
            warnings.warn(f"Skipping {a_csv.stem}: {e}")

    if not x_all:
        print(f"No data loaded for metric '{args.metric}'")
        return

    x_concat = np.concatenate(x_all, axis=0)
    y_concat = np.concatenate(y_all, axis=0)

    output_path = Path(args.output).resolve() if args.output else None
    if output_path and output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_scatter(
        x_concat,
        y_concat,
        args.metric,
        args.ref,
        args.style,
        args.alpha_points,
        args.show_models,
        model_labels,
        output_path=output_path,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
