#!/usr/bin/env python3
"""
Convenience wrapper for creating model lists.
Calls utils/create_model_list.py
"""
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# Import and run
from create_model_list import main

if __name__ == "__main__":
    main()
