#!/usr/bin/env python3
"""Small, offline CPU installation check. No token access, downloads or GPU use."""
import importlib
import os
from pathlib import Path
import sys


def main():
    # Set before importing torch or any optional library that may initialize CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    project = Path(__file__).resolve().parents[2]
    for path in (project, project / "esd_experiment", project / "esd_experiment/src"):
        sys.path.insert(0, str(path))

    try:
        for name in ("torch", "torchvision", "transformers", "accelerate", "peft",
                     "huggingface_hub", "safetensors", "numpy", "scipy", "pandas",
                     "h5py", "gpustat", "psutil", "tqdm",
                     "esd_experiment.src.run_experiment", "esd_experiment.src.worker",
                     "esd_experiment.utils.analyze_results"):
            module = importlib.import_module(name)
            print(f"OK {name} {getattr(module, '__version__', '')}".rstrip())

        import numpy as np
        import torch
        from net_esd import net_esd_estimator

        torch.set_num_threads(1)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.diag(torch.arange(1., 5.)))
        result = net_esd_estimator(model, parallel=False, fix_fingers="xmin_mid")
        np.testing.assert_allclose(result["eigs"][0], [1., 4., 9., 16.])
        assert result["fit_status"] == ["fitted"]
        print("OK CPU spectrum and fit")
    except Exception as error:
        print(f"Setup check failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print("CPU setup passed. This does not verify CUDA or optional quantization backends.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
