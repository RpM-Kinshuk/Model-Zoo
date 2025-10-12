# GPU-Managed ESD Analysis Pipeline

## Overview

This integrated system combines the fast ESD (Empirical Spectral Density) analysis pipeline with intelligent GPU resource management for shared computing environments. It automatically handles:

- **Non-exclusive GPU usage** - Respects other users' jobs
- **Automatic GPU allocation** - Finds and allocates free GPUs based on memory thresholds
- **LoRA/PEFT adapter merging** - Automatically detects and merges adapters before analysis
- **Batch processing** - Efficiently processes hundreds of models
- **Fault tolerance** - Includes checkpointing and automatic retries

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   kinexp.py                          │
│  (Experiment Orchestrator & Command Generator)       │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│              gputracker/gputracker.py                │
│         (GPU Resource Manager & Scheduler)           │
│                                                      │
│  • Monitors GPU memory usage                        │
│  • Allocates GPUs when available                    │
│  • Manages job queue and dispatch                   │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│           esd_analysis/run_analysis.py              │
│              (ESD Analysis Runner)                   │
│                                                      │
│  • Loads models (with adapter merging)              │
│  • Runs fast net_esd.py implementation              │
│  • Saves results                                    │
└──────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Usage - Process Model List

```bash
python shells/kinexp.py \
    --mode esd \
    --model-list atlas_metadata.csv \
    --max-gpus 4 \
    --output-dir ./esd_results
```

### 2. Shared GPU Environment

```bash
# Use specific GPUs with higher memory threshold
python shells/kinexp.py \
    --mode esd \
    --model-list atlas_metadata.csv \
    --gpu-list 4 5 6 7 \
    --gpu-mem-threshold 1000 \
    --max-checks 15
```

### 3. Single Model Test

```bash
python shells/kinexp.py \
    --mode esd \
    --single-model openai-community/gpt2 \
    --max-gpus 1
```

## Configuration Options

### GPU Management

| Option | Default | Description |
|--------|---------|-------------|
| `--max-gpus` | 4 | Maximum number of GPUs to use |
| `--gpu-list` | Auto-detect | Specific GPU IDs to use (e.g., `0 1 2 3`) |
| `--gpu-mem-threshold` | 500 | Memory threshold in MB to consider GPU free |
| `--max-checks` | 10 | Number of checks before considering GPU free |
| `--gpus-per-job` | 1 | Number of GPUs needed per job |

### ESD Analysis

| Option | Default | Description |
|--------|---------|-------------|
| `--fix-fingers` | xmin_mid | Method for xmin selection (xmin_mid, xmin_peak, DKS) |
| `--evals-thresh` | 1e-5 | Threshold for filtering eigenvalues |
| `--bins` | 100 | Number of bins for histogram |
| `--no-parallel-gpu` | False | Disable parallel GPU computation within ESD |

### Input/Output

| Option | Description |
|--------|-------------|
| `--model-list` | CSV file with models to process |
| `--single-model` | Process a single model |
| `--output-dir` | Directory for results |
| `--log-dir` | Directory for log files |
| `--overwrite` | Overwrite existing results |

## CSV Format

The model list CSV should have these columns:

```csv
full_name,source_model,base_model_relation
openai-community/gpt2,,base
peft-internal-testing/gpt2-lora-random,openai-community/gpt2,adapter
meta-llama/Llama-2-7b-hf,,base
```

## GPU Allocation Strategy

The system uses a smart allocation strategy:

1. **Monitor Phase**: Continuously checks GPU memory usage
2. **Validation Phase**: Ensures GPU stays free for `max_checks` consecutive checks
3. **Allocation Phase**: Marks GPU as occupied and assigns to job
4. **Release Phase**: Automatically releases GPU after job completion

### Memory Threshold Recommendations

- **Exclusive cluster**: 100-200 MB (very low threshold)
- **Shared with light usage**: 500-1000 MB (default)
- **Heavily shared environment**: 1000-2000 MB (conservative)
- **Very busy shared cluster**: 2000-4000 MB (very conservative)

## Advanced Usage

### Custom Commands Mode

Create a file with custom commands:

```bash
# commands.txt
python esd_analysis/run_analysis.py --single-model model1 --output-dir ./out1
python esd_analysis/run_analysis.py --single-model model2 --output-dir ./out2
python custom_script.py --arg1 value1
```

Run with GPU management:

```bash
python shells/kinexp.py \
    --mode custom \
    --commands commands.txt \
    --max-gpus 4
```

### Production Configuration

For production runs with many models:

```bash
python shells/kinexp.py \
    --mode esd \
    --model-list production_models.csv \
    --max-gpus 8 \
    --gpu-mem-threshold 500 \
    --max-checks 10 \
    --output-dir ./production_results \
    --log-dir ./logs/production \
    --fix-fingers xmin_mid \
    --evals-thresh 1e-5 \
    --bins 100
```

## Monitoring & Logs

### Log Files

The system creates detailed logs in the specified log directory:

- `esd_experiment_YYYYMMDD_HHMMSS.log` - Main experiment log
- `gpu_metrics.json` - GPU usage statistics

### Real-time Monitoring

The console shows:
- GPU allocation status
- Job progress
- Wait time estimates
- Success/failure counts

### Metrics

After completion, `gpu_metrics.json` contains:
- Total jobs processed
- Success/failure rates
- Average GPU wait time
- Total runtime

## Troubleshooting

### Common Issues

1. **GPUs never become available**
   - Increase `--gpu-mem-threshold`
   - Reduce `--max-checks` for faster allocation
   - Check if other processes are using GPUs

2. **Jobs failing due to OOM**
   - Reduce model batch size
   - Use `--no-parallel-gpu` to disable internal parallelization
   - Process large models with `--gpus-per-job 2`

3. **Slow processing**
   - Increase `--max-gpus`
   - Lower `--gpu-mem-threshold` (if safe)
   - Enable internal GPU parallelization (remove `--no-parallel-gpu`)

### Debug Mode

For detailed debugging:

```bash
# Set logging to DEBUG level
export LOG_LEVEL=DEBUG

# Run with verbose output
python shells/kinexp.py \
    --mode esd \
    --model-list test.csv \
    --max-gpus 1 \
    --output-dir ./debug_test
```

## Performance Tips

1. **Optimal GPU allocation**: Set `--gpu-mem-threshold` based on your typical model size
2. **Batch size**: Process models of similar size together
3. **Checkpointing**: Results are saved incrementally, so you can resume if interrupted
4. **Memory management**: The system automatically cleans up after each model

## Integration with Existing Code

### Using as a Library

```python
from shells.kinexp import ESDExperiment, ExperimentConfig

# Create configuration
config = ExperimentConfig(
    mode='esd',
    model_list='models.csv',
    output_dir='./results',
    gpu_list=[0, 1, 2, 3],
    gpu_memory_threshold=500,
    max_checks=10,
)

# Run experiment
experiment = ESDExperiment(config)
experiment.run()
```

### Custom Processing Pipeline

You can extend the system for custom processing:

```python
class CustomExperiment(ESDExperiment):
    def _build_esd_command(self, row):
        # Customize command generation
        cmd = super()._build_esd_command(row)
        cmd += " --custom-arg value"
        return cmd
    
    def _print_summary(self):
        # Add custom summary statistics
        super()._print_summary()
        print("Custom metrics...")
```

## Best Practices

1. **Start conservative**: Begin with higher memory thresholds and adjust down
2. **Test first**: Run a small batch to verify settings
3. **Monitor resources**: Keep an eye on GPU memory and system RAM
4. **Use checkpoints**: Enable checkpointing for long runs
5. **Clean up**: The system auto-cleans, but verify after large runs

## FAQ

**Q: How does it handle GPU failures?**
A: The system will retry allocation and skip models that repeatedly fail.

**Q: Can I use this on a single GPU?**
A: Yes, set `--max-gpus 1` and it will process models sequentially.

**Q: What if I need multiple GPUs per model?**
A: Use `--gpus-per-job N` where N is the number of GPUs needed.

**Q: How do I prioritize certain GPUs?**
A: Use `--gpu-list` to specify preferred GPUs in order.

**Q: Can I pause and resume?**
A: The system checkpoints progress. Stop with Ctrl+C and restart with the same command.

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Verify GPU availability with `nvidia-smi`
3. Test with a single small model first
4. Review the metrics in `gpu_metrics.json`

## License

This integrated system is provided as-is for research purposes.
