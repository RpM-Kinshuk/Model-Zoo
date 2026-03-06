# gpudispatch Experiments Design

> Phase 3: Experiment Primitives for GPU orchestration library

## Overview

A hybrid API for ML experimentation that serves both beginners and experts:
- **Beginners**: `@experiment` decorator with lists of values
- **Experts**: `Experiment` class with full control over strategies, storage, and optimization

**Core principles:**
1. Zero-config for simple cases
2. Progressive disclosure of complexity
3. DataFrame-first analysis
4. Auto-save everything (no lost experiments)
5. Pluggable for cutting-edge research

---

## 1. Core API

### 1.1 The `@experiment` Decorator

```python
from gpudispatch import experiment

# Simplest case - list values to try
@experiment(learning_rate=[1e-4, 1e-3, 1e-2])
def train(learning_rate):
    model = train_model(lr=learning_rate)
    return {"accuracy": model.accuracy}

results = train()
print(results.best)  # {"learning_rate": 1e-3, "accuracy": 0.94}
```

### 1.2 The `Experiment` Class

```python
from gpudispatch import Experiment, Sweep, Log, Choice

exp = Experiment(
    fn=train_model,
    name="transformer-ablation",
    search_space=Sweep(
        learning_rate=Log(1e-5, 1e-1),
        batch_size=Choice([16, 32, 64]),
    ),
    strategy="bayesian",
    trials=100,
    gpu=1,
)

results = exp.run()
```

### 1.3 Conversion Between Forms

```python
# Decorator creates Experiment internally
@experiment(lr=[1e-4, 1e-3])
def train(lr): ...

# Access underlying Experiment
train.experiment.strategy = "random"

# Build from function
exp = Experiment(train).with_sweep(lr=Log(1e-5, 1e-1))
```

---

## 2. Search Space Primitives

### 2.1 Basic Types

```python
from gpudispatch.experiments import Grid, Sweep, Choice, Range, Log, Uniform, Int

# Lists = grid search (exhaustive)
@experiment(lr=[1e-4, 1e-3], batch_size=[16, 32])
def train(lr, batch_size): ...  # 4 combinations

# Distributions = sweep (sampled)
Log(1e-5, 1e-1)        # Log-uniform
Uniform(0.0, 0.5)      # Uniform
Choice([16, 32, 64])   # Categorical
Int(4, 32)             # Integer range
Range(0.1, 1.0, 0.1)   # Stepped range
```

### 2.2 Conditional Parameters

```python
Conditional(
    optimizer=Choice(["adam", "sgd"]),
    momentum=If("optimizer", "sgd", Uniform(0.8, 0.99)),
)
```

### 2.3 Mixed Grid + Sweep

```python
@experiment(
    model_size=["small", "large"],  # Grid: try both
    lr=Log(1e-5, 1e-1),             # Sweep: sample
    trials=20,                       # 20 trials per model_size
)
def train(model_size, lr): ...
```

---

## 3. Search Strategies

### 3.1 Built-in (String Names)

```python
strategy="grid"      # Exhaustive (default for lists)
strategy="random"    # Random sampling (default for distributions)
strategy="bayesian"  # TPE-based Bayesian optimization
strategy="halving"   # Successive halving with early stopping
```

### 3.2 Strategy Objects

```python
from gpudispatch.experiments.strategies import (
    RandomStrategy, BayesianStrategy, HalvingStrategy, NSGAStrategy
)

BayesianStrategy(
    sampler="tpe",
    n_startup_trials=10,
    multivariate=True,
)

NSGAStrategy()  # Multi-objective Pareto optimization
```

### 3.3 Pluggable Interface

```python
from gpudispatch.experiments import Strategy

class MyStrategy(Strategy):
    def suggest(self, trial_id: int, search_space: SearchSpace) -> dict:
        ...

    def report(self, trial_id: int, metrics: dict) -> None:
        ...
```

### 3.4 Early Stopping

```python
@experiment(
    lr=Log(1e-5, 1e-1),
    pruner="median",  # Stop if below median
)
def train(lr):
    for epoch in range(100):
        loss = train_epoch()
        yield {"epoch": epoch, "loss": loss}  # Intermediate report
    return {"final_loss": loss}
```

---

## 4. Results & Storage

### 4.1 Smart Defaults

```python
from gpudispatch import set_experiment_dir

# Project-level config
set_experiment_dir("./experiments")

# Auto-saves to ./experiments/<name>/
#   ├── config.json      # Search space, strategy, git commit
#   ├── trials.csv       # All trials (human-readable)
#   ├── best.json        # Best params + metrics
#   └── history.sqlite   # Full history (large experiments)
```

### 4.2 Results Object

```python
results = exp.run()

results.best              # Best trial
results.best_params       # {"lr": 0.001}
results.best_metrics      # {"accuracy": 0.94}
results.df                # pandas DataFrame (primary analysis)

results.top(5)            # Top 5 trials
results.failed            # Failed trials
results.successful        # Successful trials
```

### 4.3 Storage Backends

```python
from gpudispatch.experiments.storage import (
    MemoryStorage,    # In-memory (quick tests)
    FileStorage,      # JSON/CSV files (default)
    SQLiteStorage,    # Database (large experiments)
)
```

### 4.4 Loading & Comparison

```python
from gpudispatch import experiments

experiments.list()                    # All experiments
exp = experiments.load("lr-search")   # Load by name
experiments.compare(["a", "b"], metric="accuracy")
experiments.diff("v1", "v2")          # Show changes
```

---

## 5. Reproducibility

### 5.1 Auto-Captured (No Config)

Every experiment automatically captures:
- Random seeds (numpy, torch, python)
- Git commit + dirty status
- Search space and config
- Python version, key package versions

### 5.2 Reproduce Trials

```python
results = experiments.load("lr-search")
results.trials[0].reproduce()  # Re-run with exact seeds

results.export_environment("requirements.txt")
```

---

## 6. Dispatcher Integration

```python
# Auto-uses dispatcher
@experiment(lr=[1e-4, 1e-3], gpu=1)
def train(lr): ...

# Shared dispatcher
with Dispatcher(gpus=[0,1,2,3]) as d:
    r1, r2 = d.run_experiments([exp1, exp2])
```

---

## 7. Error Handling

```python
results = exp.run()
results.failed           # Failed trials with errors
results.retry_failed()   # Retry

# Resume interrupted
exp.run(resume=True)
```

---

## 8. Module Structure

```
gpudispatch/
├── experiments/
│   ├── __init__.py          # Public API exports
│   ├── experiment.py        # Experiment class
│   ├── decorator.py         # @experiment decorator
│   ├── search_space.py      # Grid, Sweep, distributions
│   ├── trial.py             # Trial dataclass
│   ├── results.py           # Results container
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py          # Strategy ABC
│   │   ├── grid.py          # GridStrategy
│   │   ├── random.py        # RandomStrategy
│   │   ├── bayesian.py      # BayesianStrategy
│   │   └── halving.py       # HalvingStrategy
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py          # Storage ABC
│   │   ├── memory.py        # MemoryStorage
│   │   ├── file.py          # FileStorage
│   │   └── sqlite.py        # SQLiteStorage
│   └── reproducibility.py   # Seed capture, environment export
```

---

## 9. Phase 3 Scope

**In scope:**
- `@experiment` decorator
- `Experiment` class
- Search space primitives (Grid, Sweep, Log, Choice, Uniform, Int, Range)
- Strategies: grid, random, bayesian (via Optuna)
- Storage: Memory, File, SQLite
- Results with DataFrame analysis
- Basic reproducibility (seeds, git, config)
- Dispatcher integration

**Deferred to future phases:**
- Multi-objective optimization (NSGA-II)
- Pipelines (multi-stage experiments)
- Conditional parameters
- W&B/MLflow integrations
- Advanced pruners
- Distributed strategies
