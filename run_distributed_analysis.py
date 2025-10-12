#!/usr/bin/env python3
"""
Example script showing how to run ESD analysis with GPU management.
This handles shared GPU environments intelligently.
"""

import os
import sys
from pathlib import Path

# Example 1: Run with atlas_metadata.csv
def run_batch_analysis():
    """Run batch ESD analysis with GPU management."""
    cmd = [
        "python", "shells/kinexp.py",
        "--mode", "esd",
        "--model-list", "atlas_metadata.csv",
        "--max-gpus", "4",  # Use up to 4 GPUs
        "--gpus-per-job", "4",  # 4 GPU per model
        "--gpu-mem-threshold", "500",  # Consider GPU free if <500MB used
        "--output-dir", "./esd_results_managed",
        "--fix-fingers", "xmin_mid",
        "--overwrite"  # Overwrite existing results
    ]
    
    print("Running batch ESD analysis with GPU management...")
    print(" ".join(cmd))
    os.system(" ".join(cmd))


# Example 2: Run with specific GPU list
def run_with_specific_gpus():
    """Run using specific GPUs only."""
    cmd = [
        "python", "shells/kinexp.py",
        "--mode", "esd",
        "--model-list", "atlas_metadata.csv",
        "--gpu-list", "4", "5", "6", "7",  # Only use GPUs 4-7
        "--gpus-per-job", "4",  # 4 GPU per model
        "--gpu-mem-threshold", "1000",  # Higher threshold for busy environments
        "--max-checks", "15",  # More checks before considering GPU free
        "--output-dir", "./esd_results_specific",
    ]
    
    print("Running with specific GPU list...")
    print(" ".join(cmd))
    os.system(" ".join(cmd))


# Example 3: Single model analysis
def run_single_model():
    """Run analysis on a single model."""
    cmd = [
        "python", "shells/kinexp.py",
        "--mode", "esd",
        "--single-model", "openai-community/gpt2",
        "--max-gpus", "1",
        "--gpus-per-job", "1",
        "--output-dir", "./esd_single_test",
    ]
    
    print("Running single model analysis...")
    print(" ".join(cmd))
    os.system(" ".join(cmd))


# Example 4: Custom commands from file
def run_custom_commands():
    """Run custom commands with GPU management."""
    
    # Create a commands file
    commands_file = Path("custom_commands.txt")
    with open(commands_file, 'w') as f:
        # Write your custom commands
        f.write("python esd_analysis/run_analysis.py --single-model openai-community/gpt2 --output-dir ./test1\n")
        f.write("python esd_analysis/run_analysis.py --single-model EleutherAI/gpt-j-6B --output-dir ./test2\n")
        f.write("python esd_analysis/run_analysis.py --single-model facebook/opt-125m --output-dir ./test3\n")
    
    cmd = [
        "python", "shells/kinexp.py",
        "--mode", "custom",
        "--commands", str(commands_file),
        "--max-gpus", "4",
        "--gpus-per-job", "4",
        "--gpu-mem-threshold", "500",
    ]
    
    print("Running custom commands with GPU management...")
    print(" ".join(cmd))
    os.system(" ".join(cmd))


# Example 5: Production run with all features
def production_run():
    """Production run with comprehensive settings."""
    cmd = [
        "python", "shells/kinexp.py",
        "--mode", "esd",
        "--model-list", "atlas_metadata.csv",
        "--max-gpus", "8",  # Use all available GPUs
        "--gpu-mem-threshold", "500",  # 500MB threshold
        "--max-checks", "10",  # 10 checks before allocation
        "--gpus-per-job", "4",  # 4 GPU per model
        "--output-dir", "./esd_production_results",
        "--log-dir", "./logs/production",
        "--fix-fingers", "xmin_mid",
        "--evals-thresh", "1e-5",
        "--bins", "100",
        # "--no-parallel-gpu",  # Uncomment to disable internal GPU parallelization
    ]
    
    print("Running production ESD analysis...")
    print(" ".join(cmd))
    os.system(" ".join(cmd))


def main():
    """Main entry point with menu."""
    print("\n" + "="*60)
    print("ESD ANALYSIS WITH GPU MANAGEMENT")
    print("="*60)
    print("\nSelect an example to run:")
    print("1. Batch analysis with auto GPU detection")
    print("2. Analysis with specific GPU list")
    print("3. Single model test")
    print("4. Custom commands from file")
    print("5. Production run with all features")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    examples = {
        "1": run_batch_analysis,
        "2": run_with_specific_gpus,
        "3": run_single_model,
        "4": run_custom_commands,
        "5": production_run,
    }
    
    if choice in examples:
        print("\n" + "="*60)
        examples[choice]()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    # You can either run the menu
    main()
    
    # Or directly run a specific example:
    # run_batch_analysis()
    # production_run()
