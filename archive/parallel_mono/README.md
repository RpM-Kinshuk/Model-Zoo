# Large-Scale ESD Model Analysis Framework

A robust, GPU-aware framework for analyzing Empirical Spectral Density (ESD) metrics across hundreds of neural network models from HuggingFace.

## Features

- ✅ **GPU Resource Management**: Automatic GPU scheduling using the proven gputracker system
- ✅ **PEFT Adapter Support**: Robust loading and merging of LoRA/PEFT adapters
- ✅ **Resume Capability**: Skip already-analyzed models automatically
- ✅ **Parallel Processing**: Dispatch multiple models across available GPUs
- ✅ **Error Handling**: Retry logic and failure tracking
- ✅ **Clean & Simple**: Easy to understand and modify code structure

## Quick Start

### 1. Prepare Your Model List

Create a CSV file with your models:

```csv
model_id,base_model_relation,source_model
meta-llama/Llama-2-7b-hf,,
microsoft/phi-2,,
some/lora-adapter,adapter,meta-llama/Llama-2-7b-hf
```

**Columns:**
- `model_id` (required): HuggingFace repository ID, optionally with revision (`org/model@commit`)
- `base_model_relation` (optional): Set to "adapter", "lora", or "peft" for adapter models
- `source_model` (optional): Base model for adapters (auto-detected if not provided)

### 2. Run the Experiment

```bash
python run_esd_experiment.py \
    --model_list sample_models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3 \
    --num_gpus_per_job 1 \
    --fix_fingers xmin_mid
```

### 3. View Results

Each model gets its own CSV file in the output directory with per-layer metrics:
- `alpha`: Power law exponent
- `spectral_norm`: Largest singular value
- `stable_rank`: Effective rank
- `entropy`: Spectral entropy
- And more...

## Configuration Options

### GPU Configuration

```bash
--gpus 0 1 2 3              # GPU indices to use
--num_gpus_per_job 1        # GPUs per model (1 recommended)
--gpu_memory_threshold 500  # MB threshold for "free" GPU
--max_check 10              # Checks before considering GPU free
```

### ESD Parameters

```bash
--fix_fingers xmin_mid      # Method: xmin_mid, xmin_peak, or DKS
--evals_thresh 1e-5         # Eigenvalue filtering threshold
--bins 100                  # Histogram bins
--filter_zeros              # Filter near-zero eigenvalues
--parallel_esd              # Multi-GPU ESD computation (experimental)
```

### Experiment Control

```bash
--overwrite                 # Recompute existing results
--limit 10                  # Process only first N models
--skip_failed               # Skip previously failed models
--log_dir logs/             # Custom log directory
```

## Architecture

### Component Overview

```
esd_experiment/
├── run_esd_experiment.py   # Main experiment runner
│   └── Uses gputracker to dispatch jobs
│
├── esd_worker.py           # Per-model analysis worker
│   └── Loads model and runs ESD analysis
│
├── model_loader.py         # Robust model loading
│   ├── Handles standard models
│   ├── Handles PEFT adapters
│   └── Auto-detects adapter base models
│
└── sample_models.csv       # Example model list
```

### How It Works

1. **Main Runner** (`run_esd_experiment.py`):
   - Reads model list CSV
   - Filters out already-completed models (resume)
   - Generates worker commands for each model
   - Uses `DispatchThread` to schedule jobs across GPUs

2. **GPU Tracker** (from `shells/gputracker`):
   - Monitors GPU memory usage
   - Waits for free GPUs
   - Dispatches jobs with `CUDA_VISIBLE_DEVICES` set
   - Tracks occupied GPUs to avoid conflicts

3. **Worker** (`esd_worker.py`):
   - Loads single model (adapter or standard)
   - Runs `net_esd_estimator` from `net_esd.py`
   - Saves per-layer metrics to CSV
   - Records failures for retry/skip

4. **Model Loader** (`model_loader.py`):
   - Detects if model is PEFT adapter
   - Loads and merges adapter weights robustly
   - Handles HuggingFace authentication
   - Parses model IDs with revisions

## Advanced Usage

### Analyzing Adapter Models

The framework automatically detects and handles PEFT adapters:

```csv
model_id,base_model_relation,source_model
some/lora-adapter,adapter,base-model/name
another/adapter,lora,
```

If `source_model` is empty, the framework will:
1. Check the adapter's `PeftConfig`
2. Check the adapter's `AutoConfig`
3. Raise an error if base model cannot be determined

### Resume Interrupted Experiments

Simply rerun the same command - already-completed models are automatically skipped:

```bash
# Run 1: Analyzes models 1-50, crashes
python run_esd_experiment.py --model_list models.csv --output_dir results/

# Run 2: Automatically skips models 1-50, starts from 51
python run_esd_experiment.py --model_list models.csv --output_dir results/
```

To force recomputation:
```bash
python run_esd_experiment.py --model_list models.csv --output_dir results/ --overwrite
```

### Custom Analysis Pipeline

You can easily modify `esd_worker.py` to add custom analysis:

```python
# In esd_worker.py, after net_esd_estimator:

# Add custom metric computation
custom_metrics = compute_my_metrics(model)

# Add to results
df["my_metric"] = custom_metrics
```

### Batch Analysis Script

Create a simple analysis script:

```python
import pandas as pd
from pathlib import Path

results_dir = Path("results/")
all_results = []

for csv_file in results_dir.glob("*.csv"):
    df = pd.read_csv(csv_file)
    # Compute model-level statistics
    model_stats = {
        "model_id": df["model_id"].iloc[0],
        "mean_alpha": df["alpha"].mean(),
        "median_alpha": df["alpha"].median(),
        "num_layers": len(df),
    }
    all_results.append(model_stats)

summary = pd.DataFrame(all_results)
summary.to_csv(results_dir / "summary.csv", index=False)
print(summary)
```

## Integration with Existing Code

This framework builds on your existing codebase:

- **GPU Scheduling**: Exact implementation from `shells/kinexp.py` and `shells/gputracker/gputracker.py`
- **Adapter Loading**: Heavy inspiration from `ESD-Independence/calculate_adapters.py` and `Classification/run_metric.py`
- **ESD Analysis**: Uses your improved `net_esd.py` estimator
- **Large-Scale Patterns**: Follows `WW_LLMs-main/run_hf_model_alphas.py` structure

## Troubleshooting

### GPU Not Releasing

If GPUs aren't being released properly:
- Check `logs/esd_experiment.log` for errors
- Ensure worker processes are completing
- Increase `--max_check` for more conservative GPU allocation

### Adapter Loading Failures

If adapters fail to load:
1. Check if `source_model` is correctly specified
2. Verify adapter has proper `adapter_config.json`
3. Check HuggingFace token is set: `export HF_TOKEN=your_token`

### Out of Memory

If you run out of GPU memory:
- Use `--num_gpus_per_job 2` for larger models
- Set `device_map="cpu"` in `esd_worker.py` for CPU analysis
- Reduce number of parallel jobs with fewer `--gpus`

### Failed Models

Check `results/failed_models.txt` for errors:
```bash
cat results/failed_models.txt
```

To retry failed models:
```bash
# Remove from failed list
rm results/failed_models.txt

# Rerun (will skip successful, retry failed)
python run_esd_experiment.py --model_list models.csv --output_dir results/
```

## Environment Setup

```bash
# Required packages
pip install torch transformers peft accelerate
pip install pandas numpy gpustat
pip install huggingface_hub

# Set HuggingFace token (for private/gated models)
export HF_TOKEN=your_token_here

# Optional: Set cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

## Design Philosophy

1. **Simple & Modular**: Each component has a single responsibility
2. **Robust by Default**: Extensive error handling and retry logic
3. **Research-Friendly**: Easy to modify and extend
4. **Production-Ready**: Handles hundreds of models reliably

## Citation

If you use this framework in your research, please cite the underlying ESD methodology and tools.

## License

Follows the license of the parent project.
