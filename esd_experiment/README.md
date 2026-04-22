# Large-Scale ESD Analysis Framework

A clean, organized framework for analyzing Empirical Spectral Density (ESD) metrics across hundreds of neural network models from HuggingFace.

> **📌 GPU Usage**: The framework uses `device_map="auto"` to automatically utilize GPUs when available. See [docs/GPU_FIX.md](docs/GPU_FIX.md) for details.

## Quick Start

```bash
# 1. Test your setup
python tests/test_setup.py

# 2. Run the experiment with the canonical curated list
python run_experiment.py \
    --model_list ../data/curated/model_zoo_phase2.csv \
    --output_dir ../analysis_runs/phase2/example_run \
    --gpus 0 1 2 3

# 3. Analyze results
python analyze_results.py --results_dir ../analysis_runs/phase2/example_run --verbose
```

Canonical phase-2 outputs belong under `../analysis_runs/phase2/`.

Phase-2 runs now use a preflight eligibility step before GPU dispatch. The explicit loader paths are `standard_causal`, `seq2seq`, `multimodal`, `adapter_requires_base`, `gptq`, `awq`, and `gguf`. Loader hints remain primary, but preflight and loading also use optional curated fields such as `files`, `pipeline_tag`, `Architecture`, `model_type`, and `Available on the hub` when present. Run summaries are derived from per-model success artifacts plus terminal/failure records, not dispatcher logs alone.

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Full Documentation](docs/README.md)** - Comprehensive user guide
- **[Architecture Overview](docs/OVERVIEW.md)** - Technical architecture details
- **[GPU Setup Guide](docs/GPU_FIX.md)** - GPU configuration and troubleshooting

## Repository Structure

```
esd_experiment/
├── README.md
├── STRUCTURE.md
├── analyze_results.py
├── docs/
├── examples/
├── gputracker/
├── requirements.txt
├── run_experiment.py
├── src/
├── tests/
└── utils/
```

## Features

✅ **GPU Resource Management** - Automatic scheduling using gputracker  
✅ **PEFT Adapter Support** - Robust loading and merging of LoRA/PEFT adapters  
✅ **Multimodal Support** - Llava-style image-text-to-text models load through the multimodal auto class  
✅ **Quantized-Native Support** - Common HF-native 4-bit repos work when the required quant backend is available  
✅ **Resume Capability** - Skip already-analyzed models automatically  
✅ **Parallel Processing** - Dispatch multiple models across GPUs  
✅ **Error Handling** - Retry logic and failure tracking  
✅ **Clean Code** - Well-organized, easy to understand and modify  

## Example Workflow

```bash
# Run the complete example workflow
cd examples
bash workflow.sh
```

This will:
1. Create a sample model list
2. Run ESD analysis on example models
3. Generate summary statistics
4. Display results

## Model List Format

Create a CSV file with your models:

```csv
model_id,revision_norm,base_model_relation,source_model,loader_scenario,primary_type_bucket
meta-llama/Llama-2-7b-hf,main,source,,,base_source
some/lora-adapter,main,adapter,meta-llama/Llama-2-7b-hf,adapter_requires_base,adapter
```

**Columns:**
- `model_id` (required): HuggingFace repository ID
- `revision_norm` (optional): Curated revision override
- `base_model_relation` (optional): adapter or lineage relation
- `source_model` (optional): Base model for adapters
- `loader_scenario` (optional): curated loader hint such as `standard_transformers` or `adapter_requires_base`
- `primary_type_bucket` (optional): curated type bucket for logging / downstream analysis
- optional routing/probe fields such as `files`, `repo_files`, `pipeline_tag`, `Architecture`, `model_type`, and `Available on the hub`

Legacy three-column CSVs (`model_id,base_model_relation,source_model`) remain supported.

### Loader Notes

- `multimodal_transformers` routes through `AutoModelForImageTextToText`.
- `quantized_transformers_native` only gets backend-gated once the row resolves to an explicit `gptq` or `awq` path; otherwise it stays on the standard causal path until the loader has stronger evidence.
- When a quantized backend is missing or incompatible, the worker records a structured load failure instead of leaving partial outputs behind.

## Output

Each successful model generates:
- `../analysis_runs/phase2/example_run/stats/*.csv` for per-layer metrics
- `../analysis_runs/phase2/example_run/metrics/*.h5` for alpha matrices

Failures are recorded in:
- `../analysis_runs/phase2/example_run/logs/failed_models.txt`
- `../analysis_runs/phase2/example_run/logs/failure_records.jsonl`

The per-layer CSV contains metrics such as:
- `alpha` - Power law exponent
- `spectral_norm` - Largest singular value
- `stable_rank` - Effective rank
- `entropy` - Spectral entropy
- And more...

Summary statistics are saved to `../analysis_runs/phase2/example_run/summary.csv`.

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch (with CUDA for GPU support)
- Transformers
- PEFT
- Pandas, NumPy
- GPUstat

## Troubleshooting

### Models running on CPU?
```bash
# Verify GPU setup
python tests/test_gpu.py

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Import errors?
Make sure you're running from the repository root:
```bash
cd /path/to/esd_experiment
python run_experiment.py ...
```

### Need help?
See [docs/QUICKSTART.md](docs/QUICKSTART.md) for common issues and solutions.

## Citation

If you use this framework in your research, please cite the underlying ESD methodology and tools.

## License

Follows the license of the parent ESD project.
