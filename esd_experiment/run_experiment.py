#!/usr/bin/env python3
"""
Convenience wrapper for running experiments.
Calls src/run_experiment.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run
from run_experiment import main

if __name__ == "__main__":
    main()
