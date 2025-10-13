#!/usr/bin/env python3
"""
Convenience wrapper for analyzing results.
Calls utils/analyze_results.py
"""
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# Import and run
from analyze_results import main

if __name__ == "__main__":
    exit(main())
