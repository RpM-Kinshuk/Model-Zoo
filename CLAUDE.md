# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model-Zoo is a high-performance framework for analyzing Empirical Spectral Density (ESD) metrics across large-scale neural network models from HuggingFace. It computes power-law exponents (α), spectral norms, stable ranks, and other spectral properties to understand model behavior and training dynamics.

### Architecture

The repository has a two-tier structure:

1. **`net_esd/`** - Core ESD computation library
   - `core.py`: Vectorized ESD algorithms with multi-GPU support (Gram matrix or SVD methods)
   - `utils.py`: Helper functions for rank, entropy, layer filtering
   - `constants.py`: Configuration constants and result keys

2. **`esd_experiment/`** - Large-scale experiment framework
   - `src/run_experiment.py`: Main orchestrator that dispatches jobs to GPUs
   - `src/worker.py`: Per-model analysis worker (loads model, runs ESD, saves results)
   - `src/model_loader.py`: Robust HuggingFace model loading with PEFT/LoRA adapter support
   - `gputracker/gputracker.py`: Dynamic GPU allocation with signal-based runtime control
   - `utils/analyze_results.py`: Post-processing for aggregating results

### Execution Flow

1. `run_experiment.py` reads model list CSV and creates job queue
2. `GPUDispatcher` monitors GPU memory and assigns free GPUs to jobs
3. For each model, spawns `worker.py` as subprocess with `CUDA_VISIBLE_DEVICES` set
4. Worker loads model → extracts Conv/Linear layers → calls `net_esd.core.compute_esd_for_weight()`
5. Results saved as CSV (per-layer metrics) and HDF5 (alpha matrices)

## Environment Setup

### Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate esd_ind
```

### Pip Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.10+ with CUDA
- Transformers 5.0+
- PEFT 0.18+
- gpustat, pandas, numpy

## Common Commands

### Running Experiments

**Always run from the project root** (not from subdirectories):

```bash
# Basic usage
python esd_experiment/run_experiment.py \
    --model_list esd_experiment/examples/atlas_models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3

# With custom ESD parameters
python esd_experiment/run_experiment.py \
    --model_list models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3 \
    --fix_fingers xmin_peak \
    --evals_thresh 1e-6 \
    --bins 100 \
    --use_svd \
    --parallel_esd

# Test with limited models
python esd_experiment/run_experiment.py \
    --model_list models.csv \
    --output_dir test_results/ \
    --limit 5 \
    --gpus 0
```

### Testing

```bash
# Verify framework setup
python esd_experiment/tests/test_setup.py

# Test GPU configuration
python esd_experiment/tests/test_gpu.py
```

### Analyzing Results

```bash
# Generate summary statistics
python esd_experiment/analyze_results.py \
    --results_dir results/ \
    --verbose

# Compare two experiment runs
python scatter.py \
    --dir_a results_a/stats/ \
    --dir_b results_b/stats/ \
    --metric alpha \
    --output comparison.png
```

### Runtime GPU Control

The framework supports signal-based runtime control:

```bash
# Start experiment
python esd_experiment/run_experiment.py \
    --model_list models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3 &

PID=$!

# Modify GPU pool at runtime
echo '{"available_gpus": [2, 3], "max_checks": 5, "memory_threshold_mb": 500}' \
    > results/gpu_config.json
kill -HUP $PID  # Reload config

# Graceful shutdown (finish current jobs)
kill -USR1 $PID

# Force stop
kill -TERM $PID
```

## Code Architecture Details

### GPU Resource Management

The `gputracker` module provides:
- **Singleton pattern**: `GPUDispatcher` ensures only one instance manages GPU pool
- **Memory monitoring**: Uses `gpustat` to check GPU memory usage
- **Dynamic allocation**: Waits for `num_gpus_per_job` free GPUs before dispatching
- **Signal handling**: SIGHUP (reload config), SIGUSR1 (drain mode), SIGTERM/SIGINT (hard stop)

The dispatcher tracks occupied GPUs to prevent double-allocation and maintains a list of active worker PIDs for cleanup.

### ESD Computation Methods

Three methods for selecting `xmin` (power-law tail cutoff):

1. **`xmin_mid`** (default): Divides spectrum at midpoint - fast, good for clean spectra
2. **`xmin_peak`**: Peak histogram method - more robust to noisy spectra
3. **`DKS`**: Full Kolmogorov-Smirnov scan - most accurate, slowest

The core computation (`net_esd/core.py`) uses:
- **Vectorized power-law fitting**: Computes alpha across all possible xmin values in parallel
- **Pinned memory**: Accelerates CPU→GPU transfers
- **Batched convolution**: Reshapes Conv2d/Conv1d kernels into stacks of 2D matrices

### Model Loading

`model_loader.py` handles:
- **PEFT adapters**: Detects via `base_model_relation` column, loads base model first, then merges adapter
- **Revisions**: Supports `model_id@revision` syntax
- **Retry logic**: Up to 3 attempts with exponential backoff for transient HuggingFace Hub errors
- **Memory management**: Explicit cleanup and cache clearing between models

### Output Format

Each model produces:
- **CSV** (`results/stats/model--name.csv`): Per-layer metrics (alpha, spectral_norm, stable_rank, entropy, etc.)
- **HDF5** (`results/metrics/model--name.h5`): Alpha matrices with metadata for ML pipelines
- **Summary** (`results/summary.csv`): Model-level aggregated statistics

Failed models are logged to `results/logs/failed_models.txt` with error messages.

## Important Patterns

### Import Structure

Always import from project root context:

```python
# Correct (run from Model-Zoo root)
from net_esd.core import compute_esd_for_weight
from esd_experiment.gputracker.gputracker import GPUDispatcher

# Incorrect - do not use relative imports from subdirectories
```

### Model List CSV Format

```csv
model_id,base_model_relation,source_model
meta-llama/Llama-2-7b-hf,,
microsoft/phi-2,,
some/lora-adapter,adapter,meta-llama/Llama-2-7b-hf
another/model@main,,
```

- `model_id` (required): HuggingFace repo ID, optionally with `@revision`
- `base_model_relation` (optional): "adapter", "lora", or "peft" for adapter models
- `source_model` (optional): Base model for adapters (inferred from adapter config if missing)

### Adding Custom Metrics

To add new metrics, modify:
1. `net_esd/core.py`: Compute metric in `compute_esd_for_weight()` function
2. `net_esd/constants.py`: Add new column name to `RESULT_KEYS`
3. Worker will automatically include new metrics in CSV output

### Resume Capability

The framework automatically skips models with existing output files. To force reanalysis:

```bash
python esd_experiment/run_experiment.py \
    --model_list models.csv \
    --output_dir results/ \
    --overwrite
```

Or manually delete specific model CSV files from `results/stats/`.

## Troubleshooting

### Models running on CPU

Check CUDA availability and ensure `device_map="auto"` is set in `model_loader.py`:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python esd_experiment/tests/test_gpu.py
```

Verify gputracker is setting `CUDA_VISIBLE_DEVICES` in logs.

### Import errors

Always run commands from the repository root (`Model-Zoo/`), not from subdirectories. The code uses absolute imports that assume project root is in `sys.path`.

### Out of memory

Options:
1. Use fewer GPUs per job: `--num_gpus_per_job 1`
2. Increase memory threshold: `--gpu_memory_threshold 1000`
3. Use SVD sparingly: `--use_svd` is more accurate but memory-intensive
4. For Conv layers, the code processes all kernels in batch - modify `net_esd/core.py` if needed

### Slow analysis

- Thread backend (default) is faster than process backend for most cases
- `--use_svd` is slower but more numerically stable (only use when precision is critical)
- Enable `--parallel_esd` to distribute layers across multiple GPUs (experimental)
- Check if models are sharded (`.safetensors.index.json`) - these load slower

### Signal handling issues

On some systems, signal handling may not work as expected. If runtime GPU control fails:
- Check that the process is running (not a detached subprocess)
- Use `kill -l` to verify signal numbers on your system
- For hard stop, use `kill -9 $PID` as fallback

## Testing Before Large Runs

Always validate setup before running on hundreds of models:

```bash
# 1. Verify dependencies and structure
python esd_experiment/tests/test_setup.py

# 2. Test GPU allocation
python esd_experiment/tests/test_gpu.py

# 3. Run on 3-5 small models
python esd_experiment/run_experiment.py \
    --model_list test_models.csv \
    --output_dir test_results/ \
    --limit 3 \
    --gpus 0

# 4. Check output format
ls test_results/stats/
python esd_experiment/analyze_results.py --results_dir test_results/ --verbose
```

## Performance Considerations

- **Largest models first**: The framework sorts models by layer count (descending) for better load balancing
- **Pinned memory**: Used automatically for CPU→GPU transfers when available
- **Numerical stability**: Gram matrices are symmetrized; small jitter added for singular cases
- **Conv layer batching**: All conv kernels processed together - reduces kernel launch overhead but increases memory

For production runs on hundreds of models, monitor:
1. GPU utilization: `watch -n 1 nvidia-smi`
2. Progress: `tail -f results/logs/esd_experiment.log`
3. Failures: `wc -l results/logs/failed_models.txt`
