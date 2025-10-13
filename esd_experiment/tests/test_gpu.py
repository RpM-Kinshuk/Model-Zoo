#!/usr/bin/env python3
"""
Quick GPU test to verify CUDA and model loading work correctly.
Run this before starting your experiment to ensure GPU usage will work.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))  # ESD root for net_esd
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # For imports

def test_cuda():
    """Test CUDA availability."""
    print("=" * 60)
    print("Testing CUDA Setup")
    print("=" * 60)
    
    try:
        import torch
        print("\n✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"\n✓ CUDA is available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}:")
                print(f"    Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # Test allocation
            print("\n✓ Testing GPU allocation...")
            test_tensor = torch.randn(100, 100, device='cuda:0')
            print(f"  Successfully allocated tensor on GPU")
            print(f"  Tensor device: {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
            
            return True
        else:
            print("\n✗ CUDA is NOT available")
            print("  Models will run on CPU")
            print("\n  Possible reasons:")
            print("  - No GPU available")
            print("  - CUDA not installed")
            print("  - PyTorch CPU-only version installed")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_model_loading():
    """Test that models can load with device_map='auto'."""
    print("\n" + "=" * 60)
    print("Testing Model Loading with device_map='auto'")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        print("\n✓ Transformers imported successfully")
        
        # Use a very small model for testing
        model_name = "sshleifer/tiny-gpt2"
        print(f"\nLoading test model: {model_name}")
        print("(This is a tiny 1MB model just for testing)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        print("✓ Model loaded successfully")
        
        # Check device
        model_devices = set()
        for param in model.parameters():
            model_devices.add(str(param.device))
        
        print(f"  Model devices: {', '.join(sorted(model_devices))}")
        
        if any('cuda' in d for d in model_devices):
            print("\n✓ SUCCESS: Model loaded on GPU!")
            print("  Your experiments will use GPU")
        else:
            print("\n⚠ WARNING: Model loaded on CPU")
            print("  Your experiments will run slower on CPU")
            print("\n  This might be expected if:")
            print("  - No GPU is available")
            print("  - CUDA_VISIBLE_DEVICES limits GPU access")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return False


def test_gputracker_env():
    """Test gputracker environment simulation."""
    print("\n" + "=" * 60)
    print("Testing GPU Assignment (Simulating gputracker)")
    print("=" * 60)
    
    try:
        import os
        import torch
        from transformers import AutoModelForCausalLM
        
        # Simulate gputracker setting CUDA_VISIBLE_DEVICES
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("\nSimulating: export CUDA_VISIBLE_DEVICES=0")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"Visible GPUs after setting: {torch.cuda.device_count()}")
            
            # Load tiny model
            model_name = "sshleifer/tiny-gpt2"
            print(f"\nLoading {model_name} with device_map='auto'...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            model_devices = set()
            for param in model.parameters():
                model_devices.add(str(param.device))
            
            print(f"Model devices: {', '.join(sorted(model_devices))}")
            
            if any('cuda' in d for d in model_devices):
                print("\n✓ SUCCESS: GPU assignment works correctly!")
                print("  gputracker will successfully assign models to GPUs")
            else:
                print("\n⚠ Model still on CPU despite CUDA_VISIBLE_DEVICES")
            
            del model
            torch.cuda.empty_cache()
            
            # Clean up
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
            
            return True
        else:
            print("\nSkipping (no GPU available)")
            return True
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all GPU tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "GPU SETUP VERIFICATION" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    
    results = []
    
    # Test 1: CUDA
    results.append(("CUDA availability", test_cuda()))
    
    # Test 2: Model loading
    results.append(("Model loading", test_model_loading()))
    
    # Test 3: GPU assignment
    results.append(("GPU assignment", test_gputracker_env()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        print("\nYour setup is ready for GPU-accelerated experiments.")
        print("\nNext steps:")
        print("  1. Run: bash example_workflow.sh")
        print("  2. Check worker output shows 'Model devices: cuda:0'")
    else:
        print("\n⚠ Some tests failed")
        print("\nYour experiments may run on CPU instead of GPU.")
        print("\nTo troubleshoot:")
        print("  1. Check GPU: nvidia-smi")
        print("  2. Verify CUDA installation")
        print("  3. Reinstall PyTorch with CUDA support")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
