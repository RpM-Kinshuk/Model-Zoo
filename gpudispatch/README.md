# gpudispatch

**Universal GPU orchestration - from laptop to supercomputer.**

[![Tests](https://img.shields.io/badge/tests-491%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

gpudispatch is a production-ready GPU orchestration library that provides:

- **Job Orchestration**: Queue, schedule, and execute GPU workloads with dependency management
- **Hyperparameter Experiments**: Grid search, random search with extensible strategies
- **Multi-Backend Support**: Local machines, SLURM clusters (stub), Kubernetes (planned)
- **Zero Config**: Auto-detects GPUs and environment, works out of the box

## Installation

```bash
pip install gpudispatch
```

Or install from source:

```bash
git clone https://github.com/yourusername/gpudispatch.git
cd gpudispatch
pip install -e ".[dev]"
```

## Quick Start

### Simple GPU Jobs

```python
from gpudispatch import Dispatcher

# Create dispatcher with specific GPUs
with Dispatcher(gpus=[0, 1, 2, 3]) as d:
    # Submit a job - runs on available GPU
    job = d.submit(train_model, args=(config,), gpu=1)

    # Submit dependent job
    eval_job = d.submit(evaluate, args=(job,), gpu=1, after=[job])
```

### Hyperparameter Experiments (Beginner)

```python
from gpudispatch import experiment

@experiment(lr=[1e-4, 1e-3, 1e-2], batch_size=[16, 32, 64])
def train(lr, batch_size):
    """Your training function - receives params as kwargs."""
    model = train_model(lr=lr, batch_size=batch_size)
    return {"loss": model.loss, "accuracy": model.accuracy}

# Run all 9 combinations automatically
results = train()

# Analyze results
print(results.best_params)   # {"lr": 0.001, "batch_size": 32}
print(results.best_metrics)  # {"loss": 0.05, "accuracy": 0.98}
print(results.df)            # pandas DataFrame for analysis
```

### Hyperparameter Experiments (Advanced)

```python
from gpudispatch import Experiment, Sweep, Log, Choice, Uniform
from gpudispatch.experiments.storage import SQLiteStorage
from gpudispatch.experiments.strategies import RandomStrategy

# Define sophisticated search space
exp = Experiment(
    fn=train_model,
    search_space=Sweep(
        lr=Log(1e-5, 1e-1),           # Log-uniform for learning rate
        dropout=Uniform(0.0, 0.5),     # Uniform for dropout
        optimizer=Choice(["adam", "sgd", "rmsprop"]),
    ),
    strategy=RandomStrategy(n_trials=100),
    storage=SQLiteStorage("experiments.db"),
    metric="loss",
    maximize=False,
)

results = exp.run()
print(results.summary())
```

### Auto-Dispatch (Environment Detection)

```python
from gpudispatch import auto_dispatcher

# Automatically detects: Local, SLURM, K8s, Cloud
dispatcher = auto_dispatcher()

with dispatcher:
    gpus = dispatcher.allocate_gpus(2)
    # Use GPUs...
    dispatcher.release_gpus(gpus)
```

## Core Concepts

### Search Spaces

Define hyperparameter spaces with Grid (exhaustive) or Sweep (sampled):

```python
from gpudispatch.experiments import Grid, Sweep, Log, Uniform, Int, Choice, Range

# Grid: Exhaustive combinations
grid = Grid(lr=[1e-4, 1e-3], batch_size=[16, 32])  # 4 combinations

# Sweep: Sampled distributions
sweep = Sweep(
    lr=Log(1e-5, 1e-1),        # Log-uniform (ideal for learning rates)
    dropout=Uniform(0.0, 0.5), # Uniform continuous
    layers=Int(4, 16),         # Integer range
    optimizer=Choice(["adam", "sgd"]),  # Categorical
    warmup=Range(0.1, 1.0, 0.1),        # Stepped range
)
```

### Strategies

Control how the search space is explored:

```python
from gpudispatch.experiments.strategies import GridStrategy, RandomStrategy

# Exhaustive grid search
GridStrategy()

# Random sampling with n trials
RandomStrategy(n_trials=50)

# Custom strategy (implement Strategy ABC)
class BayesianStrategy(Strategy):
    def suggest(self, search_space, completed_trials):
        # Use Optuna, GPyOpt, etc.
        return next_params_or_none
```

### Storage Backends

Persist experiments across sessions:

```python
from gpudispatch.experiments.storage import MemoryStorage, FileStorage, SQLiteStorage

# In-memory (testing)
MemoryStorage()

# File-based (human-readable CSV + JSON)
FileStorage("./experiments")

# SQLite (queryable database)
SQLiteStorage("experiments.db")
```

### Observability Hooks

Monitor events across the system:

```python
from gpudispatch.observability.hooks import hooks, EventHook, LoggingHook

# Built-in logging
hooks.register(LoggingHook())

# Custom hook
my_hook = EventHook(
    on_job_start=lambda job_id, job_name, **kw: print(f"Started: {job_name}"),
    on_experiment_complete=lambda experiment_id, total_jobs, **kw:
        notify_slack(f"Done: {experiment_id}"),
)
hooks.register(my_hook)
```

## CLI

```bash
# Show GPU status
gpudispatch status

# List all experiments
gpudispatch list

# Show experiment details
gpudispatch show train_20240101_120000
```

## Extension Points

gpudispatch is designed to be extended:

| Component | Interface | Built-in | Extend for |
|-----------|-----------|----------|------------|
| Backends | `Backend` ABC | Local, SLURM (stub) | K8s, AWS, GCP, Custom clusters |
| Strategies | `Strategy` ABC | Grid, Random | Bayesian, Evolutionary, NSGA-II |
| Storage | `Storage` ABC | Memory, File, SQLite | S3, MLflow, Weights & Biases |
| Hooks | `EventHook` | Logging | Prometheus, OpenTelemetry, Slack |

### Example: Custom Backend

```python
from gpudispatch.backends import Backend

class KubernetesBackend(Backend):
    @property
    def name(self):
        return "kubernetes"

    def allocate_gpus(self, count, memory=None):
        # Create K8s pod with GPU resources
        ...

    def release_gpus(self, gpus):
        # Delete K8s pod
        ...
```

### Example: Custom Strategy

```python
from gpudispatch.experiments.strategies import Strategy

class BayesianStrategy(Strategy):
    def suggest(self, search_space, completed_trials):
        # Build surrogate model from completed_trials
        # Return next promising configuration
        return {"lr": 0.001, "batch_size": 32}

    @property
    def name(self):
        return "bayesian"
```

## Runtime Control (Unix Signals)

Control running experiments without restart:

```bash
# Start experiment
python run_experiment.py &
PID=$!

# Reload GPU configuration
echo '{"available_gpus": [2, 3]}' > gpu_config.json
kill -HUP $PID

# Enter drain mode (finish current, reject new)
kill -USR1 $PID

# Graceful shutdown
kill -TERM $PID
```

## Reproducibility

```python
from gpudispatch.experiments import set_seeds, capture_context

# Set all random seeds
set_seeds(42)

# Capture execution context
context = capture_context()
# Returns: {
#   "git_commit": "abc123",
#   "python_version": "3.11.0",
#   "torch_version": "2.0.0",
#   "timestamp": "2024-01-01T12:00:00",
#   ...
# }
```

## Project Structure

```
gpudispatch/
├── src/gpudispatch/
│   ├── core/               # Job, Queue, Dispatcher, Signals
│   ├── experiments/        # Search spaces, Trials, Results, Strategies, Storage
│   ├── backends/           # Local, SLURM (stub)
│   ├── observability/      # Event hooks
│   ├── cli/                # Command-line interface
│   ├── utils/              # GPU detection utilities
│   ├── auto.py             # Environment auto-detection
│   └── decorators.py       # @gpu decorator
├── tests/                  # 491 tests
└── docs/
    └── ARCHITECTURE.md     # Comprehensive technical guide
```

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Comprehensive technical documentation
- [API Reference](docs/API.md) - (Coming soon)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/unit/experiments/ -v

# Run with coverage
pytest tests/ --cov=gpudispatch
```

## Requirements

- Python 3.9+
- PyTorch (optional, for GPU detection)
- gpustat (optional, for memory monitoring)
- pandas (optional, for Results.df)
- click (for CLI)

## License

Apache 2.0

## Contributing

Contributions welcome! Please read the [Architecture Guide](docs/ARCHITECTURE.md) to understand the codebase before contributing.

Key areas for contribution:
- [ ] SLURM backend implementation
- [ ] Kubernetes backend
- [ ] Bayesian optimization strategy
- [ ] MLflow/W&B storage integrations
- [ ] Prometheus/OpenTelemetry hooks
