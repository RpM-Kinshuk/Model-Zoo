# Large-Scale ESD Analysis Framework

A clean, organized framework for analyzing Empirical Spectral Density (ESD) metrics across hundreds of neural network models from HuggingFace.

> **📌 GPU Usage**: The framework uses `device_map="auto"` to automatically utilize GPUs when available. See [docs/GPU_FIX.md](docs/GPU_FIX.md) for details.

## Quick Start

```bash
# 1. Test your setup
python tests/test_setup.py

# 2. Create a model list
python create_model_list.py create --output my_models.csv \
    --models "meta-llama/Llama-2-7b-hf" "microsoft/phi-2"

# 3. Run the experiment
python run_experiment.py \
    --model_list my_models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3

# 4. Analyze results
python analyze_results.py --results_dir results/ --verbose
```

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Full Documentation](docs/README.md)** - Comprehensive user guide
- **[Architecture Overview](docs/OVERVIEW.md)** - Technical architecture details
- **[GPU Setup Guide](docs/GPU_FIX.md)** - GPU configuration and troubleshooting

## Repository Structure

```
esd_experiment/
│
├── src/                      # Core framework code
│   ├── run_experiment.py     # Main experiment orchestrator
│   ├── worker.py             # Per-model analysis worker
│   └── model_loader.py       # Robust model loading with adapter support
│
├── utils/                    # Utility scripts
│   ├── analyze_results.py    # Results aggregation and analysis
│   └── create_model_list.py  # Model list management
│
├── docs/                     # Documentation
│   ├── README.md             # Full user guide
│   ├── QUICKSTART.md         # Quick start guide
│   ├── OVERVIEW.md           # Architecture overview
│   └── GPU_FIX.md            # GPU troubleshooting
│
├── examples/                 # Example files
│   ├── workflow.sh           # Complete workflow example
│   ├── config.py             # Configuration patterns
│   └── example_models.csv    # Sample model list
│
├── tests/                    # Testing scripts
│   ├── test_setup.py         # Framework setup verification
│   └── test_gpu.py           # GPU setup testing
│
├── run_experiment.py         # Main entry point (wrapper)
├── analyze_results.py        # Analysis entry point (wrapper)
├── create_model_list.py      # Model list entry point (wrapper)
├── requirements.txt          # Python dependencies
└── CHANGELOG.md              # Version history
```

## Features

✅ **GPU Resource Management** - Automatic scheduling using gputracker  
✅ **PEFT Adapter Support** - Robust loading and merging of LoRA/PEFT adapters  
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
model_id,base_model_relation,source_model
meta-llama/Llama-2-7b-hf,,
microsoft/phi-2,,
some/lora-adapter,adapter,meta-llama/Llama-2-7b-hf
```

**Columns:**
- `model_id` (required): HuggingFace repository ID
- `base_model_relation` (optional): "adapter", "lora", or "peft" for adapters
- `source_model` (optional): Base model for adapters

## Output

Each model generates a CSV file with per-layer metrics:
- `alpha` - Power law exponent
- `spectral_norm` - Largest singular value
- `stable_rank` - Effective rank
- `entropy` - Spectral entropy
- And more...

Summary statistics are saved to `results/summary.csv`.

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
