# Model-Zoo: Large-Scale Neural Network Spectral Analysis

A high-performance framework for analyzing Empirical Spectral Density (ESD) metrics across hundreds of neural network models from HuggingFace. Compute power-law exponents (α), spectral norms, stable ranks, and other spectral properties to understand model behavior and training dynamics.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 What This Does

This framework analyzes the weight matrices of neural networks using spectral methods (eigenvalue/singular value analysis) to compute:

- **Power-law exponent (α)**: Measures heavy-tailed behavior in weight spectra
- **Spectral norm**: Largest singular value (affects model stability)
- **Stable rank**: Effective dimensionality of weight matrices
- **Matrix entropy**: Diversity of singular values
- **Alpha-weighted metrics**: Combined spectral properties

These metrics provide insights into:
- Model capacity and expressiveness
- Training stability and convergence
- Fine-tuning vs. base model differences
- Adapter (LoRA/PEFT) impact on model structure

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone <your-model-zoo-remote> Model-Zoo
cd Model-Zoo

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate esd_ind

# Or install with pip
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Test your setup
python esd_experiment/tests/test_setup.py

# 2. Run analysis with the canonical curated list
python esd_experiment/run_experiment.py \
    --model_list data/curated/model_zoo_phase2.csv \
    --output_dir analysis_runs/phase2/example_run \
    --gpus 0 1 2 3

# 3. Analyze results
python esd_experiment/analyze_results.py --results_dir analysis_runs/phase2/example_run --verbose
```

Legacy three-column CSVs (`model_id,base_model_relation,source_model`) are still accepted, but curated tables are now the preferred input.

Canonical phase-2 outputs belong under `analysis_runs/phase2/`.
Phase-2 ESD runs use `data/curated/model_zoo_phase2.csv`, run a preflight eligibility step before dispatch, and keep output-root accounting under `analysis_runs/phase2/<run_name>/`.
Preflight also consumes optional curated routing/probe fields such as `files`, `repo_files`, `pipeline_tag`, `Architecture`, `model_type`, and `Available on the hub` when they are present.

## 📁 Repository Structure

```
Model-Zoo/
├── data/curated/               # Canonical phase-1 artifacts and synced views
├── analysis_runs/phase2/       # Canonical phase-2 run outputs
├── docs/operations/            # Human operational docs for phases 1 and 2
├── net_esd/                      # Core ESD computation library
│   ├── core.py                   # Main ESD algorithms (vectorized, multi-GPU)
│   ├── utils.py                  # Helper functions (rank, entropy, layer filtering)
│   ├── constants.py              # Configuration and result keys
│   └── archive/                  # Legacy implementations
│
├── esd_experiment/               # Large-scale experiment framework
│   ├── src/
│   │   ├── run_experiment.py    # Main orchestrator (GPU dispatch, job queuing)
│   │   ├── worker.py             # Per-model analysis worker
│   │   └── model_loader.py       # Robust HF model loading (handles adapters)
│   │
│   ├── gputracker/              # GPU resource management
│   │   └── gputracker.py        # Dynamic GPU allocation with signal handling
│   │
│   ├── utils/
│   │   └── analyze_results.py   # Results aggregation and statistics
│   │
│   ├── tests/
│   │   ├── test_setup.py        # Framework verification
│   │   └── test_gpu.py          # GPU setup testing
│   │
│   ├── examples/
│   │   ├── workflow.sh          # Complete workflow example
│   │   ├── sample.sh            # GPU scheduling examples
│   │   └── atlas_models.csv     # Sample model list
│   │
│   └── docs/                    # Detailed documentation
│       ├── README.md            # Full user guide
│       ├── QUICKSTART.md        # 5-minute tutorial
│       ├── OVERVIEW.md          # Architecture details
│       └── GPU_FIX.md           # GPU troubleshooting
│
├── scatter.py                   # Interactive metric comparison tool
├── atlas_metadata.csv           # Large-scale model metadata
├── environment.yml              # Conda environment specification
└── requirements.txt             # Python dependencies
```

## 🚀 Key Features

### 1. High-Performance Computation
- **Vectorized operations**: Batch computation of power-law fits across eigenvalue spectrum
- **Multi-GPU parallel processing**: Distributes layers across available GPUs
- **Two backends**: Thread-based (shared memory) or process-based (isolated contexts)
- **SVD or Gram matrix methods**: Choose speed vs. numerical precision

### 2. Intelligent GPU Management
- **Dynamic GPU allocation**: Monitors GPU memory and assigns jobs automatically
- **Runtime reconfiguration**: Modify GPU pool without restarting (via SIGHUP signal)
- **Graceful shutdown**: SIGUSR1 for drain mode, SIGTERM/SIGINT for hard stop
- **Per-job GPU assignment**: Control how many GPUs each model analysis uses

### 3. Robust Model Loading
- **PEFT/LoRA adapter support**: Automatically detects and merges adapters with base models
- **Multimodal support**: Routes Llava-style image-text-to-text repos through the appropriate auto model class
- **Quantized-native support**: Supports common HF-native quantized repos when the required backend is available, and records structured incompatibility failures otherwise
- **Config-aware routing**: Uses loader hints first, then config/task metadata such as `quantization_config`, `pipeline_tag`, and architectures to choose the most appropriate loader path
- **Revision support**: Analyze specific model versions (e.g., `model@revision`)
- **Retry logic**: Handles transient HuggingFace Hub errors
- **Memory management**: Automatic cleanup and cache clearing

### 4. Production-Ready Workflow
- **Resume capability**: Automatically skips already-analyzed models
- **Failure tracking**: Records failed models in both `logs/failed_models.txt` and machine-readable `logs/failure_records.jsonl`
- **Progress logging**: Detailed logs for debugging and monitoring
- **Output formats**: CSV (per-layer metrics) + HDF5 (alpha matrices for ML)

## 📊 Understanding the Output

### Per-Model CSV Files

Each model produces a CSV with one row per layer under the chosen run directory, for example `analysis_runs/phase2/example_run/stats/*.csv`:

| Metric | Description | Use Case |
|--------|-------------|----------|
| `alpha` | Power-law exponent (α > 1) | Model capacity indicator; higher α → more regularized |
| `spectral_norm` | Largest singular value | Training stability (lower is more stable) |
| `stable_rank` | Frobenius norm / spectral norm | Effective matrix rank (higher → more expressive) |
| `entropy` | Spectral entropy | Weight distribution diversity |
| `log_alpha_norm` | Log of α-weighted norm | Combined metric for model quality |
| `D` | Kolmogorov-Smirnov statistic | Quality of power-law fit |
| `num_evals` | Number of eigenvalues | Matrix size indicator |

### Alpha Matrix HDF5 Files

Structured format for machine learning pipelines:
```python
import json
import h5py
with h5py.File('analysis_runs/phase2/example_run/metrics/model.h5', 'r') as f:
    alpha_matrix = f['alpha'][:]  # Shape: (num_layers, num_modules)
    module_names = json.loads(f['alpha'].attrs['module_names_json'])
    print(f"Model: {f.attrs['full_name']}")
```

### Summary Statistics

Aggregated metrics across all analyzed models for easy comparison. Canonical phase-2 summaries live under `analysis_runs/phase2/<run_name>/summary.csv`.

## 🔧 Advanced Usage

### Custom ESD Parameters

```bash
python esd_experiment/run_experiment.py \
    --model_list data/curated/model_zoo_phase2.csv \
    --output_dir analysis_runs/phase2/example_run \
    --gpus 0 1 2 3 \
    --fix_fingers xmin_peak \
    --evals_thresh 1e-6 \
    --bins 100 \
    --use_svd \
    --parallel_esd
```

Optional tuning:
- `--fix_fingers xmin_mid` or `DKS`
- `--evals_thresh 1e-6`
- `--bins 100`
- `--use_svd`

**ESD Methods:**
- `xmin_mid`: Divide spectrum at midpoint (fast, default)
- `xmin_peak`: Peak histogram method (more accurate for noisy spectra)
- `DKS`: Full Kolmogorov-Smirnov scan (most accurate, slowest)

### Runtime GPU Control

```bash
# Start experiment
python esd_experiment/run_experiment.py \
    --model_list data/curated/model_zoo_phase2.csv \
    --output_dir analysis_runs/phase2/example_run \
    --gpus 0 1 2 3 4 5 6 7 &

PID=$!

# Edit GPU pool during runtime
echo '{"available_gpus": [4, 5, 6, 7], "max_checks": 5, "memory_threshold_mb": 500}' \
    > analysis_runs/phase2/example_run/gpu_config.json

# Reload configuration
kill -HUP $PID

# Graceful shutdown (finish current jobs)
kill -USR1 $PID

# Force stop
kill -TERM $PID
```

### Working with Adapters

```bash
# Automatically detects adapters
cat > adapters.csv << EOF
model_id,base_model_relation,source_model
some-user/llama-lora,adapter,meta-llama/Llama-2-7b-hf
another/phi-peft,adapter,microsoft/phi-2
EOF

python esd_experiment/run_experiment.py \
    --model_list adapters.csv \
    --output_dir analysis_runs/phase2/example_run \
    --gpus 0 1
```

The framework automatically:
1. Loads the base model
2. Loads and merges the adapter
3. Analyzes the merged model weights
4. Records metadata (base model, adapter type)

### Analyzing Results

```bash
# Generate summary statistics
python esd_experiment/analyze_results.py \
    --results_dir analysis_runs/phase2/example_run \
    --verbose

# Compare two experiment runs (e.g., SVD vs Gram method)
python scatter.py \
    --dir_a analysis_runs/phase2/example_run/stats/ \
    --dir_b analysis_runs/phase2/example_run_alt/stats/ \
    --metric alpha \
    --output alpha_comparison.png

# Interactive plot (for Jupyter or local)
python scatter.py \
    --dir_a analysis_runs/phase2/example_run/stats/ \
    --dir_b analysis_runs/phase2/example_run_alt/stats/ \
    --metric alpha \
    --interactive
```

## 🧪 Testing

```bash
# Test basic setup
python esd_experiment/tests/test_setup.py

# Test GPU configuration
python esd_experiment/tests/test_gpu.py

# Run on small model list
python esd_experiment/run_experiment.py \
    --model_list data/curated/model_zoo_phase2.csv \
    --output_dir analysis_runs/phase2/example_run \
    --limit 5 \
    --gpus 0
```

## 📚 Documentation

- **[Quick Start Guide](esd_experiment/docs/QUICKSTART.md)**: Get running in 5 minutes
- **[Full User Guide](esd_experiment/docs/README.md)**: Comprehensive documentation
- **[Architecture Overview](esd_experiment/docs/OVERVIEW.md)**: Technical implementation details
- **[GPU Troubleshooting](esd_experiment/docs/GPU_FIX.md)**: Common GPU issues and solutions

## 🛠️ Technical Details

### Core Algorithms

The framework implements efficient spectral analysis:

1. **Weight extraction**: Filters Conv1d/Conv2d/Linear layers from model
2. **Eigenvalue computation**:
   - Gram matrix method (fast): Computes eigenvalues of AA^T or A^T A
   - SVD method (accurate): Direct singular value decomposition
3. **Power-law fitting**: Vectorized Clauset-Shalizi-Newman estimator
4. **Metric computation**: Parallel computation across GPU pool

### Performance Optimizations

- **Task ordering**: Largest layers scheduled first (better load balancing)
- **Pinned memory**: Accelerates CPU→GPU transfers
- **Batched conv layers**: Processes conv kernels as batched 2D matrices
- **Numerical stability**: Symmetric Gram matrices, jitter for singular cases

### Model Compatibility

Tested architectures:
- **Transformers**: GPT-2, Llama, Mistral, Phi, Gemma, Qwen
- **Vision**: ResNet, ViT (any model with Linear/Conv layers)
- **Adapters**: LoRA, PEFT, any adapter supported by HuggingFace

## 🐛 Troubleshooting

### Models running on CPU?

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Test GPU setup
python esd_experiment/tests/test_gpu.py

# Verify CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

### Out of memory errors?

```python
# Use gradient checkpointing (in model_loader.py)
model.gradient_checkpointing_enable()

# Or reduce batch size for conv layers in net_esd/core.py
# (currently processes all conv kernels at once)
```

### Import errors?

```bash
# Always run from project root
cd /path/to/Model-Zoo
python esd_experiment/run_experiment.py ...

# Not from subdirectories
```

### Slow analysis?

- Use thread backend (default) instead of process backend
- Enable `--use_svd` only if numerical precision is critical
- Increase GPU pool size
- Check if models are sharded (slower to load)

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{model_zoo_esd,
  title = {Model-Zoo: Large-Scale Neural Network Spectral Analysis},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/RpM-Kinshuk/Model-Zoo.git}
}
```

## 📄 License

This project follows the license of the parent ESD research project. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional spectral metrics (e.g., matrix condition number, nuclear norm)
- Support for quantized models
- Distributed analysis across multiple nodes
- Web dashboard for results visualization

## 🙏 Acknowledgments

Built on:
- **WeightWatcher**: Original ESD methodology for neural networks
- **HuggingFace**: Model hub and transformers library
- **PyTorch**: Deep learning framework
- **PEFT**: Parameter-efficient fine-tuning library

---

**Questions?** Check the [documentation](esd_experiment/docs/) or open an issue.
