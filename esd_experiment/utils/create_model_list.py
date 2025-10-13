#!/usr/bin/env python3
"""
Utility to create model lists from HuggingFace Hub or existing results.

This script helps you:
1. Search HuggingFace Hub for models
2. Import from existing CSV/list formats
3. Filter and organize model lists
"""
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create model list for ESD experiment")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # From text file
    text_parser = subparsers.add_parser("from-text", help="Create from text file (one model per line)")
    text_parser.add_argument("input", type=str, help="Input text file")
    text_parser.add_argument("--output", type=str, default="models.csv", help="Output CSV file")
    
    # From existing CSV
    csv_parser = subparsers.add_parser("from-csv", help="Convert existing CSV format")
    csv_parser.add_argument("input", type=str, help="Input CSV file")
    csv_parser.add_argument("--output", type=str, default="models.csv", help="Output CSV file")
    csv_parser.add_argument("--model_col", type=str, help="Column name for model IDs")
    csv_parser.add_argument("--relation_col", type=str, help="Column name for adapter relation")
    csv_parser.add_argument("--source_col", type=str, help="Column name for source model")
    
    # Manual creation
    manual_parser = subparsers.add_parser("create", help="Create empty template")
    manual_parser.add_argument("--output", type=str, default="models.csv", help="Output CSV file")
    manual_parser.add_argument("--models", nargs="+", help="Model IDs to add")
    
    # Merge multiple lists
    merge_parser = subparsers.add_parser("merge", help="Merge multiple model lists")
    merge_parser.add_argument("inputs", nargs="+", type=str, help="Input CSV files to merge")
    merge_parser.add_argument("--output", type=str, default="merged_models.csv", help="Output CSV file")
    merge_parser.add_argument("--deduplicate", action="store_true", help="Remove duplicates")
    
    return parser.parse_args()


def from_text_file(input_path: str, output_path: str):
    """Create model list from text file (one model per line)."""
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Reading models from: {input_path}")
    
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    # Parse lines
    models = []
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        
        models.append(line)
    
    print(f"Found {len(models)} models")
    
    # Create DataFrame
    df = pd.DataFrame({
        "model_id": models,
        "base_model_relation": "",
        "source_model": "",
    })
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved to: {output_path}")
    print(f"\nPreview:")
    print(df.head(10).to_string(index=False))


def from_csv_file(
    input_path: str,
    output_path: str,
    model_col: Optional[str] = None,
    relation_col: Optional[str] = None,
    source_col: Optional[str] = None
):
    """Convert existing CSV to standard format."""
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Reading CSV from: {input_path}")
    df = pd.read_csv(input_file)
    
    print(f"Available columns: {list(df.columns)}")
    
    # Auto-detect model column if not specified
    if model_col is None:
        for col in ["model_id", "full_name", "model_name", "model", "name"]:
            if col in df.columns:
                model_col = col
                print(f"Auto-detected model column: {model_col}")
                break
    
    if model_col is None or model_col not in df.columns:
        print(f"Error: Could not find model column. Please specify with --model_col")
        return
    
    # Create output DataFrame
    output_df = pd.DataFrame()
    output_df["model_id"] = df[model_col]
    
    # Add relation column
    if relation_col and relation_col in df.columns:
        output_df["base_model_relation"] = df[relation_col]
    else:
        output_df["base_model_relation"] = ""
    
    # Add source column
    if source_col and source_col in df.columns:
        output_df["source_model"] = df[source_col]
    else:
        output_df["source_model"] = ""
    
    # Clean up
    output_df["model_id"] = output_df["model_id"].astype(str).str.strip()
    output_df["base_model_relation"] = output_df["base_model_relation"].fillna("").astype(str).str.strip()
    output_df["source_model"] = output_df["source_model"].fillna("").astype(str).str.strip()
    
    # Remove invalid rows
    output_df = output_df[output_df["model_id"] != ""]
    output_df = output_df[output_df["model_id"] != "nan"]
    
    print(f"Converted {len(output_df)} models")
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_path}")
    print(f"\nPreview:")
    print(output_df.head(10).to_string(index=False))


def create_template(output_path: str, models: Optional[List[str]] = None):
    """Create empty template or with specified models."""
    if models:
        df = pd.DataFrame({
            "model_id": models,
            "base_model_relation": "",
            "source_model": "",
        })
        print(f"Created list with {len(models)} models")
    else:
        df = pd.DataFrame({
            "model_id": ["meta-llama/Llama-2-7b-hf", "microsoft/phi-2"],
            "base_model_relation": ["", ""],
            "source_model": ["", ""],
        })
        print("Created template with example models")
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved to: {output_path}")
    print(f"\nContents:")
    print(df.to_string(index=False))


def merge_lists(input_paths: List[str], output_path: str, deduplicate: bool = False):
    """Merge multiple model lists."""
    dfs = []
    
    for input_path in input_paths:
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"Warning: Skipping non-existent file: {input_path}")
            continue
        
        try:
            df = pd.read_csv(input_file)
            
            # Ensure required columns
            if "model_id" not in df.columns:
                print(f"Warning: Skipping {input_path} (no model_id column)")
                continue
            
            if "base_model_relation" not in df.columns:
                df["base_model_relation"] = ""
            if "source_model" not in df.columns:
                df["source_model"] = ""
            
            dfs.append(df[["model_id", "base_model_relation", "source_model"]])
            print(f"Loaded {len(df)} models from: {input_path}")
        
        except Exception as e:
            print(f"Warning: Could not load {input_path}: {e}")
    
    if not dfs:
        print("Error: No valid input files found")
        return
    
    # Merge
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal models before processing: {len(merged)}")
    
    # Deduplicate if requested
    if deduplicate:
        original_len = len(merged)
        merged = merged.drop_duplicates(subset=["model_id"], keep="first")
        print(f"Removed {original_len - len(merged)} duplicates")
    
    print(f"Final model count: {len(merged)}")
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"Saved to: {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == "from-text":
        from_text_file(args.input, args.output)
    
    elif args.command == "from-csv":
        from_csv_file(
            args.input,
            args.output,
            model_col=args.model_col,
            relation_col=args.relation_col,
            source_col=args.source_col
        )
    
    elif args.command == "create":
        create_template(args.output, models=args.models)
    
    elif args.command == "merge":
        merge_lists(args.inputs, args.output, deduplicate=args.deduplicate)
    
    else:
        print("Error: Please specify a command (from-text, from-csv, create, or merge)")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()
