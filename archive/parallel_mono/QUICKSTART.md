# Quick Start Guide

Get started with large-scale ESD analysis in 5 minutes.

## Installation

```bash
# Navigate to project directory
cd /Users/kigoel/Projects/mlc/ESD

# Install dependencies (if not already installed)
pip install torch transformers peft accelerate huggingface_hub
pip install pandas numpy gpustat

# Set your HuggingFace token (for gated models)
export HF_TOKEN=your_token_here
```

## Basic Usage

### 1. Create Your Model List

**Option A: Start with the sample**
```bash
cd esd_experiment
cat sample_models.csv
```

**Option B: Create your own**
```bash
python create_model_list.py create --output my_models.csv --models \
    "meta-llama/Llama-2-7b-hf" \
    "microsoft/phi-2" \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Option C: Convert from existing file**
```bash
# From text file (one model per line)
python create_model_list.py from-text models.txt --output models.csv

# From existing CSV
python create_model_list.py from-csv old_format.csv \
    --output models.csv \
    --model_col "full_name"
```

### 2. Run the Experiment

**Minimal command:**
```bash
python run_esd_experiment.py \
    --model_list sample_models.csv \
    --output_dir results/ \
    --gpus 0
```

**Full control:**
```bash
python run_esd_experiment.py \
    --model_list my_models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3 \
    --num_gpus_per_job 1 \
    --fix_fingers xmin_mid \
    --filter_zeros
```

### 3. Analyze Results

```bash
python analyze_results.py \
    --results_dir results/ \
    --verbose
```

This creates `results/summary.csv` with model-level statistics.

## Understanding the Output

### Per-Model Results

Each model gets a CSV file: `results/model--name.csv`

Key columns:
- `alpha`: Power law exponent per layer (lower = heavier tails)
- `spectral_norm`: Largest singular value
- `stable_rank`: Effective rank of weight matrices
- `entropy`: Spectral entropy
- `log_spectral_norm`: Log of spectral norm
- `alpha_weighted`: Alpha weighted by spectral norm

### Summary Statistics

`results/summary.csv` contains model-level aggregates:
- `alpha_mean`: Average alpha across layers
- `alpha_median`: Median alpha
- `alpha_std`: Standard deviation of alpha
- `num_layers`: Number of analyzed layers
- `total_params`: Total parameters

## Common Workflows

### Analyzing Adapters

```csv
model_id,base_model_relation,source_model
meta-llama/Llama-2-7b-hf,,
some/lora-adapter,adapter,meta-llama/Llama-2-7b-hf
```

The framework automatically:
1. Detects adapter models
2. Loads the base model
3. Merges adapter weights
4. Analyzes the merged model

### Resuming Interrupted Runs

Just rerun the same command - completed models are automatically skipped:

```bash
# First run: analyzes 50/100 models, then crashes
python run_esd_experiment.py --model_list models.csv --output_dir results/

# Second run: automatically skips first 50, analyzes remaining 50
python run_esd_experiment.py --model_list models.csv --output_dir results/
```

### Testing with a Small Subset

```bash
python run_esd_experiment.py \
    --model_list models.csv \
    --output_dir test_results/ \
    --limit 5 \
    --gpus 0
```

### Different ESD Methods

```bash
# Use xmin_peak method
python run_esd_experiment.py \
    --model_list models.csv \
    --output_dir results_peak/ \
    --fix_fingers xmin_peak

# Use DKS method
python run_esd_experiment.py \
    --model_list models.csv \
    --output_dir results_dks/ \
    --fix_fingers DKS
```

## Monitoring Progress

### Watch the logs
```bash
tail -f results/logs/esd_experiment.log
```

### Check GPU usage
```bash
watch -n 1 nvidia-smi
```

### Count completed models
```bash
ls results/*.csv | wc -l
```

### View failed models
```bash
cat results/failed_models.txt
```

## Tips & Tricks

### 1. Start Small
Test with 3-5 models first to ensure everything works:
```bash
python run_esd_experiment.py \
    --model_list sample_models.csv \
    --output_dir test/ \
    --gpus 0 \
    --limit 3
```

### 2. Optimize GPU Usage
- Use `--num_gpus_per_job 1` for most models
- Use `--num_gpus_per_job 2` for very large models (>30B params)
- Adjust `--gpu_memory_threshold` based on your GPU memory

### 3. Handle Failures
If many models fail:
- Check `results/failed_models.txt` for error patterns
- Verify HuggingFace token is set correctly
- Try with `--device_map cpu` for memory issues

### 4. Batch Analysis

Analyze results in Python:
```python
import pandas as pd
from pathlib import Path

# Load all results
results_dir = Path("results/")
all_data = []

for csv_file in results_dir.glob("*.csv"):
    if csv_file.name not in ["summary.csv", "failed_models.txt"]:
        df = pd.read_csv(csv_file)
        all_data.append(df)

# Combine
combined = pd.concat(all_data, ignore_index=True)

# Analyze
print("Overall alpha distribution:")
print(combined["alpha"].describe())

# Compare models
summary = pd.read_csv("results/summary.csv")
print("\nTop 5 models by alpha:")
print(summary.nsmallest(5, "alpha_mean")[["model_id", "alpha_mean"]])
```

## Troubleshooting

### "No module named gputracker"
Make sure you're running from the correct directory:
```bash
cd /Users/kigoel/Projects/mlc/ESD/esd_experiment
python run_esd_experiment.py ...
```

### "Could not resolve base model for adapter"
Add the base model explicitly in your CSV:
```csv
model_id,base_model_relation,source_model
adapter-repo,adapter,base-model-repo
```

### Out of Memory
- Use CPU mode: edit `esd_worker.py` and set `device_map="cpu"`
- Reduce parallel jobs by using fewer GPUs
- Use smaller models first

### GPU Not Releasing
- Check if processes are stuck: `nvidia-smi`
- Kill stuck processes: `pkill -f esd_worker.py`
- Increase `--max_check` for more conservative allocation

## Next Steps

- Read the full [README.md](README.md) for advanced usage
- Explore `example_workflow.sh` for a complete pipeline
- Modify `esd_worker.py` to add custom metrics
- Check out the analysis utilities in `analyze_results.py`

## Getting Help

Common issues and solutions:
1. **Authentication errors**: Set `HF_TOKEN` environment variable
2. **GPU selection**: Verify GPU IDs with `nvidia-smi`
3. **Import errors**: Ensure you're in the `esd_experiment/` directory
4. **Failed models**: Check error messages in `results/failed_models.txt`
