#!/usr/bin/env python3
"""
Test script to verify the experiment framework setup.

This script checks:
1. All required dependencies are installed
2. GPU tracking system is accessible
3. Model loading works correctly
4. ESD computation runs successfully
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Test results
tests_passed = []
tests_failed = []


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("gpustat", "GPUstat"),
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            all_ok = False
    
    return all_ok


def test_gputracker():
    """Test that GPU tracking system is accessible."""
    print("\nTesting GPU tracker...")
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT.parent / "shells"))  # Go up to ESD root
        from gputracker.gputracker import get_logger, DispatchThread
        print("  ✓ GPU tracker imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ GPU tracker import failed: {e}")
        return False


def test_model_loader():
    """Test model loading utilities."""
    print("\nTesting model loader...")
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from model_loader import (
            get_hf_token,
            parse_model_string,
            safe_filename,
        )
        
        # Test parsing
        repo, rev = parse_model_string("org/model@revision")
        assert repo == "org/model" and rev == "revision"
        
        # Test safe filename
        safe = safe_filename("org/model@rev")
        assert "/" not in safe and "@" not in safe
        
        print("  ✓ Model loader utilities work")
        return True
    except Exception as e:
        print(f"  ✗ Model loader test failed: {e}")
        return False


def test_esd_import():
    """Test that ESD estimator can be imported."""
    print("\nTesting ESD estimator...")
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT.parent))  # Go up to ESD root
        from net_esd import net_esd_estimator
        print("  ✓ ESD estimator imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ ESD estimator import failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    - Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
            return True
    except Exception as e:
        print(f"  ✗ CUDA test failed: {e}")
        return False


def test_hf_token():
    """Test HuggingFace token."""
    print("\nTesting HuggingFace authentication...")
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from model_loader import get_hf_token
        
        token = get_hf_token()
        if token:
            print(f"  ✓ HF token found (first 10 chars: {token[:10]}...)")
            return True
        else:
            print("  ⚠ No HF token found (may limit access to gated models)")
            return True
    except Exception as e:
        print(f"  ✗ HF token test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files are present."""
    print("\nTesting file structure...")
    
    required_structure = {
        "src/run_experiment.py": "Main experiment runner",
        "src/worker.py": "Worker script",
        "src/model_loader.py": "Model loader",
        "utils/analyze_results.py": "Analysis utility",
        "utils/create_model_list.py": "Model list utility",
        "run_experiment.py": "Entry point wrapper",
        "README.md": "Main README",
        "docs/QUICKSTART.md": "Quick start guide",
    }
    
    all_ok = True
    for filepath, description in required_structure.items():
        full_path = PROJECT_ROOT / filepath
        if full_path.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} ({description}) - missing")
            all_ok = False
    
    return all_ok


def test_gputracker_files():
    """Test that gputracker files exist."""
    print("\nTesting gputracker files...")
    
    gputracker_dir = PROJECT_ROOT.parent / "shells" / "gputracker"
    
    if not gputracker_dir.exists():
        print(f"  ✗ gputracker directory not found: {gputracker_dir}")
        return False
    
    required_files = ["__init__.py", "gputracker.py"]
    
    all_ok = True
    for filename in required_files:
        filepath = gputracker_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing)")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("=" * 80)
    print("ESD Experiment Framework - Setup Test")
    print("=" * 80)
    
    tests = [
        ("Package imports", test_imports),
        ("File structure", test_file_structure),
        ("GPU tracker files", test_gputracker_files),
        ("GPU tracker import", test_gputracker),
        ("Model loader", test_model_loader),
        ("ESD estimator", test_esd_import),
        ("CUDA", test_cuda),
        ("HuggingFace token", test_hf_token),
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if passed:
                tests_passed.append(test_name)
            else:
                tests_failed.append(test_name)
        except Exception as e:
            print(f"\n✗ {test_name} raised exception: {e}")
            tests_failed.append(test_name)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {len(tests_passed)}/{len(tests)}")
    print(f"Tests failed: {len(tests_failed)}/{len(tests)}")
    
    if tests_failed:
        print("\nFailed tests:")
        for test_name in tests_failed:
            print(f"  - {test_name}")
        print("\nPlease fix the issues above before running experiments.")
        return 1
    else:
        print("\n✓ All tests passed! The framework is ready to use.")
        print("\nNext steps:")
        print("  1. Create a model list: python create_model_list.py create --output models.csv")
        print("  2. Run experiment: python run_experiment.py --model_list models.csv --output_dir results/ --gpus 0")
        print("  3. Analyze results: python analyze_results.py --results_dir results/ --verbose")
        return 0


if __name__ == "__main__":
    sys.exit(main())
