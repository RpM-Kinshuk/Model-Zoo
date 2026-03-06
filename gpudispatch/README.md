# gpudispatch

Universal GPU orchestration - from laptop to supercomputer.

## Installation

```bash
pip install gpudispatch
```

## Quick Start

### Simple GPU Jobs

```python
from gpudispatch import Dispatcher

with Dispatcher(gpus=[0, 1]) as d:
    job = d.submit(train_model, args=(config,), gpu=1)
    # Job runs on available GPU
```

### Hyperparameter Experiments

```python
from gpudispatch import experiment

# Beginner: Grid search with one decorator
@experiment(lr=[1e-4, 1e-3, 1e-2], batch_size=[16, 32])
def train(lr, batch_size):
    model = train_model(lr=lr, batch_size=batch_size)
    return {"accuracy": model.accuracy}

results = train()  # Runs all 6 combinations
print(results.best)  # Best params + metrics
print(results.df)    # pandas DataFrame for analysis
```

### Advanced Experiments

```python
from gpudispatch import Experiment, Sweep, Log, Choice
from gpudispatch.experiments.storage import SQLiteStorage

exp = Experiment(
    fn=train_model,
    search_space=Sweep(
        lr=Log(1e-5, 1e-1),          # Log-uniform sampling
        dropout=Choice([0.1, 0.3]),   # Categorical
    ),
    strategy="random",
    storage=SQLiteStorage("experiments.db"),
    trials=50,
)

results = exp.run()
```

## Features

- **Zero-config start** - Auto-detects GPUs, sensible defaults
- **Beginner-friendly** - `@experiment` decorator for quick HPO
- **Expert control** - Full access to strategies, storage, backends
- **Extensible** - Plugin interfaces for strategies, storage, backends

## Core Components

### Dispatcher
Manages GPU pool and job queue with thread-safe allocation.

```python
from gpudispatch import Dispatcher, auto_dispatcher

# Explicit GPUs
d = Dispatcher(gpus=[0, 1, 2, 3])

# Auto-detect environment
d = auto_dispatcher()  # Local, SLURM, K8s, Cloud
```

### Search Spaces

```python
from gpudispatch.experiments import Grid, Sweep, Log, Uniform, Int, Choice, Range

# Grid search (exhaustive)
Grid(lr=[1e-4, 1e-3], layers=[4, 6, 8])

# Random/Bayesian search (sampled)
Sweep(
    lr=Log(1e-5, 1e-1),      # Log-uniform
    dropout=Uniform(0, 0.5),  # Uniform
    layers=Int(4, 12),        # Integer range
    optim=Choice(["adam", "sgd"]),
)
```

### Strategies

```python
from gpudispatch.experiments.strategies import GridStrategy, RandomStrategy

# Built-in
GridStrategy()      # Exhaustive grid search
RandomStrategy(50)  # 50 random samples

# Custom (implement Strategy ABC)
class MyStrategy(Strategy):
    def suggest(self, search_space, completed_trials):
        # Return next params or None when done
        ...
```

### Storage

```python
from gpudispatch.experiments.storage import MemoryStorage, FileStorage, SQLiteStorage

MemoryStorage()                    # Quick tests
FileStorage("./experiments")       # Human-readable CSV/JSON
SQLiteStorage("experiments.db")    # Queryable database
```

## Extension Points

gpudispatch is designed to be extended:

| Component | Interface | Example |
|-----------|-----------|---------|
| Backends | `Backend` ABC | SLURM, K8s, Cloud |
| Strategies | `Strategy` ABC | Bayesian, NSGA-II |
| Storage | `Storage` ABC | S3, MLflow, W&B |
| Observability | `EventHook` | Prometheus, OpenTelemetry |

### Adding a Custom Backend

```python
from gpudispatch.backends import Backend

class MyClusterBackend(Backend):
    @property
    def name(self) -> str:
        return "my-cluster"

    def allocate_gpus(self, count, memory=None):
        # Your allocation logic
        ...
```

### Adding a Custom Strategy

```python
from gpudispatch.experiments.strategies import Strategy

class BayesianStrategy(Strategy):
    def suggest(self, search_space, completed_trials):
        # Use Optuna, GPyOpt, etc.
        ...
```

## CLI

```bash
gpudispatch status          # Show GPU status
gpudispatch list            # List experiments
gpudispatch show <name>     # Show experiment details
```

## Reproducibility

```python
from gpudispatch.experiments import set_seeds, capture_context

set_seeds(42)  # Sets random, numpy, torch seeds
context = capture_context()  # Captures git commit, python version, etc.
```

## License

Apache 2.0
