# ESD Analysis Pipeline for Large Language Models

A high-performance batch processing pipeline for computing Empirical Spectral Density (ESD) metrics on HuggingFace models, with full support for LoRA/PEFT adapters and multi-GPU acceleration.

## Features

- **Fast ESD Computation**: Uses optimized multi-GPU parallel processing for eigenvalue decomposition
- **Adapter Support**: Automatically detects and merges LoRA/PEFT adapters before analysis
- **Batch Processing**: Process hundreds of models efficiently with checkpointing and resume capabilities
- **Smart Model Loading**: Auto-detects model types and handles various HuggingFace model formats
- **Memory Management**: Intelligent memory management with automatic cleanup and RAM monitoring
- **Comprehensive Logging**: Detailed logs with progress tracking and error reporting

## Installation

```bash
# Clone the repository
cd /Users/kigoel/Projects/mlc/ESD

# Install dependencies
pip install -r esd_analysis/requirements.txt

# Set HuggingFace token (optional, for private models)
export HF_TOKEN=your_token_here
```

## Quick Start

### Single Model Analysis

```bash
# Analyze a base model
python esd_analysis/run_analysis.py \
    --single-model openai-community/gpt2 \
    --output-dir ./results

# Analyze an adapter model
python esd_analysis/run_analysis.py \
    --single-model peft-internal-testing/gpt2-lora-random \
    --base-model openai-community/gpt2 \
    --output-dir ./results
```

### Batch Processing

```bash
# Process models from CSV file
python esd_analysis/run_analysis.py \
    --model-list atlas_metadata.csv \
    --output-dir ./results \
    --num-workers 4

# Process with GPU parallelization
python esd_analysis/run_analysis.py \
    --model-list models.csv \
    --output-dir ./results \
    --device-ids 0 1 2 3 \
    --max-gpu-workers 4
```

## CSV Format

The input CSV file should have the following columns:

- `full_name` (required): HuggingFace model ID or path
- `source_model` (optional): Base model for adapters
- `base_model_relation` (optional): Type of model ('base', 'adapter', 'lora', 'peft')

Example:
```csv
full_name,source_model,base_model_relation
openai-community/gpt2,,base
peft-internal-testing/gpt2-lora-random,openai-community/gpt2,adapter
meta-llama/Llama-2-7b-hf,,base
```

## Configuration

### ESD Parameters

- `--fix-fingers`: Method for xmin selection (`xmin_mid`, `xmin_peak`, `DKS`)
- `--evals-thresh`: Threshold for filtering eigenvalues (default: 1e-5)
- `--bins`: Number of bins for histogram (default: 100)
- `--no-filter-zeros`: Disable filtering of near-zero eigenvalues

### Performance Options

- `--num-workers`: Number of parallel model processing workers
- `--device-ids`: GPU IDs to use for computation
- `--max-gpu-workers`: Maximum workers for parallel GPU computation
- `--no-parallel-gpu`: Disable parallel GPU computation

### Processing Options

- `--overwrite`: Overwrite existing results
- `--limit`: Limit number of models to process (for testing)
- `--checkpoint-freq`: Save checkpoint every N models (default: 10)

## Output Structure

```
output_dir/
├── esd_results_summary.csv       # Summary metrics for all models
├── per_model_csv/                 # Individual model results
│   ├── model1.csv
│   ├── model2.csv
│   └── ...
├── checkpoint.csv                 # Processing checkpoint
├── failed_models.csv             # List of failed models (if any)
└── logs/                         # Detailed logs
    └── batch_run_YYYYMMDD_HHMMSS.log
```

## Output Metrics

The analysis produces the following metrics for each layer:

- **Alpha**: Power law exponent of the ESD
- **D**: Kolmogorov-Smirnov distance
- **Spectral Norm**: Largest eigenvalue
- **Stable Rank**: Ratio of Frobenius norm to spectral norm
- **Entropy**: Matrix entropy
- **Log Norms**: Various logarithmic norm metrics

Summary statistics (mean, std, min, max) are computed across all layers.

## Python API

```python
from esd_analysis.batch_analyzer import BatchESDAnalyzer, ModelInfo

# Create analyzer
analyzer = BatchESDAnalyzer(
    output_dir="./results",
    fix_fingers="xmin_mid",
    parallel=True,
    device_ids=[0, 1, 2, 3]
)

# Define models
models = [
    ModelInfo(full_name="openai-community/gpt2"),
    ModelInfo(
        full_name="peft-internal-testing/gpt2-lora-random",
        source_model="openai-community/gpt2",
        base_model_relation="adapter"
    )
]

# Process batch
results_df = analyzer.process_batch(models, num_workers=2)
```

## Advanced Usage

### Custom Model Loading

```python
from esd_analysis.model_utils import load_model_smart

# Load with custom settings
model = load_model_smart(
    model_name="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    device_map="auto"
)
```

### Using the Fast ESD Implementation Directly

```python
from net_esd import net_esd_estimator

# Run ESD analysis
metrics = net_esd_estimator(
    net=model,
    EVALS_THRESH=1e-5,
    bins=100,
    fix_fingers="xmin_mid",
    filter_zeros=True,
    parallel=True,
    device_ids=[0, 1, 2, 3]
)
```

## Troubleshooting

### Out of Memory Errors

- Reduce `--num-workers` for batch processing
- Use `--device-map cpu` to load models on CPU
- Increase checkpoint frequency to save progress

### Slow Processing

- Enable GPU parallelization with `--device-ids`
- Increase `--num-workers` for batch processing
- Use `--no-filter-zeros` if appropriate

### Authentication Issues

- Set HuggingFace token: `export HF_TOKEN=your_token`
- Or use: `huggingface-cli login`

## Performance Tips

1. **Multi-GPU Setup**: Use all available GPUs for maximum speed
   ```bash
   python run_analysis.py --model-list models.csv --device-ids 0 1 2 3
   ```

2. **Memory Management**: Monitor RAM usage and adjust workers
   ```bash
   python run_analysis.py --model-list models.csv --num-workers 2
   ```

3. **Large Models**: Process large models sequentially
   ```bash
   python run_analysis.py --model-list large_models.csv --num-workers 1
   ```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{Model-Zoo,
  title = {ESD Analysis Pipeline for Large Language Models},
  author = {Kinshuk Goel},
  year = {2025},
  url = {https://github.com/RpM-Kinshuk/Model-Zoo}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

This implementation builds upon the WeightWatcher framework and incorporates optimizations for large-scale model analysis.
