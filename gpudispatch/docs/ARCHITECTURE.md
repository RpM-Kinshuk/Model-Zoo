# gpudispatch Architecture Guide

A comprehensive technical guide to understanding the gpudispatch library.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Layers](#architecture-layers)
3. [Core Module](#core-module)
4. [Experiments Module](#experiments-module)
5. [Backends Module](#backends-module)
6. [Observability Module](#observability-module)
7. [CLI Module](#cli-module)
8. [Data Flow](#data-flow)
9. [Extension Points](#extension-points)
10. [Design Patterns](#design-patterns)

---

## Overview

gpudispatch is a universal GPU orchestration library designed to work seamlessly from laptops to supercomputers. It provides:

- **Job orchestration**: Queue, schedule, and execute GPU workloads
- **Hyperparameter experiments**: Grid search, random search, with extensible strategies
- **Multi-backend support**: Local, SLURM (stub), Kubernetes (planned), Cloud (planned)
- **Observability**: Hook-based event system for monitoring

### Design Philosophy

1. **Zero-config start**: Works out of the box with sensible defaults
2. **Progressive complexity**: Simple API for beginners, full control for experts
3. **Extensible**: Plugin architecture for strategies, storage, backends, hooks
4. **Portable**: Same code works on laptop, cluster, or cloud

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Interface                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ @experiment │  │ @gpu        │  │ Experiment  │  │ CLI         │ │
│  │ decorator   │  │ decorator   │  │ class       │  │ commands    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        Orchestration Layer                           │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐   │
│  │     Dispatcher      │  │          Experiment Engine          │   │
│  │  - Job Queue        │  │  - Search Space                     │   │
│  │  - GPU Allocation   │  │  - Strategy Selection               │   │
│  │  - Signal Handling  │  │  - Trial Execution                  │   │
│  └─────────────────────┘  └─────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                          Backend Layer                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │   Local    │  │   SLURM    │  │ Kubernetes │  │   Cloud    │     │
│  │  Backend   │  │  (stub)    │  │ (planned)  │  │ (planned)  │     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│                        Storage Layer                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │   Memory   │  │    File    │  │   SQLite   │  │  Custom    │     │
│  │  Storage   │  │  Storage   │  │  Storage   │  │ (extend)   │     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│                      Observability Layer                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                        HookRegistry                            │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │ │
│  │  │ Logging  │  │Prometheus│  │  OTel    │  │  Custom  │       │ │
│  │  │  Hook    │  │ (extend) │  │ (extend) │  │  Hooks   │       │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Module

Located in `src/gpudispatch/core/`

### Job (`job.py`)

A **Job** represents a unit of work to be executed on GPU resources.

```python
@dataclass
class Job:
    fn: Callable[..., Any]      # Function to execute
    args: tuple                  # Positional arguments
    kwargs: dict                 # Keyword arguments
    gpu_count: int = 1           # Number of GPUs required
    memory: Optional[Memory]     # Memory requirement
    priority: int = 0            # Higher = more important
    name: Optional[str]          # Human-readable name
    dependencies: set[str]       # Job IDs that must complete first

    # Internal state
    id: str                      # Unique identifier (8-char UUID)
    status: JobStatus            # PENDING → QUEUED → RUNNING → COMPLETED/FAILED

    # Timing
    created_at: datetime
    queued_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Results
    result: Optional[Any]        # Return value on success
    error: Optional[str]         # Error message on failure
```

**JobStatus Lifecycle:**
```
PENDING → QUEUED → RUNNING → COMPLETED
                          ↘ FAILED
                          ↘ CANCELLED
```

### Queue (`queue.py`)

Thread-safe job queue implementations.

**FIFOQueue**: First-in-first-out ordering
```python
queue = FIFOQueue()
queue.put(job)
job = queue.get()  # Returns oldest job
```

**PriorityQueue**: Priority-based ordering with FIFO tie-breaking
```python
queue = PriorityQueue()
queue.put(high_priority_job)  # priority=10
queue.put(low_priority_job)   # priority=1
job = queue.get()  # Returns high_priority_job first
```

Implementation details:
- Uses `heapq` for O(log n) insertion and O(log n) extraction
- Negates priority for max-heap behavior (higher priority = extracted first)
- Maintains sequence numbers for FIFO tie-breaking
- Lazy deletion: removed jobs are filtered on extraction

### Dispatcher (`dispatcher.py`)

The **Dispatcher** is the central orchestrator that manages GPU allocation and job execution.

```python
class Dispatcher:
    def __init__(
        self,
        gpus: Optional[list[int]] = None,    # GPU indices (None = auto-detect)
        memory_threshold: str = "500MB",      # Free memory threshold
        queue: Optional[JobQueue] = None,     # Custom queue (default: PriorityQueue)
        polling_interval: float = 5.0,        # Seconds between queue checks
    ): ...

    # Job management
    def submit(fn, args, kwargs, gpu, memory, priority, name, after) -> Job
    def cancel(job_id) -> bool
    def stats() -> DispatcherStats

    # Lifecycle
    def start() -> None       # Start dispatch loop (spawns background thread)
    def shutdown(wait=True)   # Stop dispatcher
    def drain()               # Stop accepting jobs, finish current ones

    # Runtime control
    def reload_config(config) # Hot-reload GPU pool and settings
    def setup_signals(config_path) -> SignalHandler
```

**Dispatch Loop** (runs in background thread):
```
while not shutdown:
    job = queue.peek()
    if job and job.dependencies_satisfied and gpus_available:
        gpus = allocate_gpus(job.gpu_count)
        spawn_worker_thread(job, gpus)
    sleep(polling_interval)
```

### Resources (`resources.py`)

Resource primitives for GPU and memory management.

```python
@dataclass
class GPU:
    index: int                    # Device index (e.g., 0, 1, 2)
    memory: Optional[int] = None  # Total memory in MB

class Memory:
    def __init__(self, mb: int): ...

    @classmethod
    def from_string(cls, s: str) -> Memory:
        # Parses "500MB", "2GB", "1024" (assumed MB)
        ...

    @property
    def mb(self) -> int: ...
    @property
    def gb(self) -> float: ...
```

### Signals (`signals.py`)

Unix signal handling for runtime control.

| Signal | Action | Use Case |
|--------|--------|----------|
| `SIGHUP` | Reload config | Update GPU pool without restart |
| `SIGUSR1` | Drain mode | Gracefully wind down |
| `SIGTERM` | Shutdown | Stop immediately |
| `SIGINT` | Shutdown | Ctrl+C handling |

```python
# Setup signal handling
handler = dispatcher.setup_signals("gpu_config.json")

# Runtime control via signals:
# kill -HUP <pid>   → Reload gpu_config.json
# kill -USR1 <pid>  → Enter drain mode
# kill -TERM <pid>  → Shutdown
```

---

## Experiments Module

Located in `src/gpudispatch/experiments/`

### Search Space (`search_space.py`)

Defines the hyperparameter space for optimization.

**Distributions** (for random/Bayesian search):

| Class | Description | Example |
|-------|-------------|---------|
| `Choice` | Categorical selection | `Choice(["adam", "sgd", "rmsprop"])` |
| `Log` | Log-uniform (for learning rates) | `Log(1e-5, 1e-1)` |
| `Uniform` | Uniform continuous | `Uniform(0.0, 0.5)` |
| `Int` | Integer range [low, high] | `Int(4, 32)` |
| `Range` | Stepped range (like arange) | `Range(0.1, 1.0, 0.1)` |

**Grid** (exhaustive combinations):
```python
grid = Grid(lr=[1e-4, 1e-3], batch_size=[16, 32, 64])
# Produces 6 combinations: lr × batch_size
for params in grid:
    print(params)  # {"lr": 1e-4, "batch_size": 16}, ...
```

**Sweep** (sampled distributions):
```python
sweep = Sweep(
    lr=Log(1e-5, 1e-1),
    dropout=Uniform(0.0, 0.5),
    layers=Int(4, 12),
)
params = sweep.sample()  # {"lr": 0.00234, "dropout": 0.31, "layers": 7}
```

**SearchSpace** (combined grid + sweep):
```python
space = SearchSpace.from_dict({
    "model": ["small", "medium", "large"],  # Grid
    "lr": Log(1e-5, 1e-1),                   # Sweep
})
# Iterates grid points, samples sweep distributions for each
```

### Trial (`trial.py`)

Represents a single experiment execution.

```python
@dataclass
class Trial:
    id: int                              # Sequential trial ID
    params: Dict[str, Any]               # Hyperparameters used
    metrics: Dict[str, Any]              # Results (e.g., {"loss": 0.5, "accuracy": 0.92})
    status: TrialStatus                  # PENDING/RUNNING/COMPLETED/FAILED/PRUNED
    error: Optional[str]                 # Error message if failed
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    @property
    def duration_seconds(self) -> Optional[float]: ...

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict) -> Trial: ...
```

### Results (`results.py`)

Container for analyzing experiment outcomes.

```python
results = Results(trials, metric="loss", maximize=False)

# Access best trial
results.best          # Trial with lowest loss
results.best_params   # {"lr": 0.001, "batch_size": 32}
results.best_metrics  # {"loss": 0.05, "accuracy": 0.98}

# Filter trials
results.successful    # List of COMPLETED trials
results.failed        # List of FAILED trials
results.top(5)        # Top 5 trials by metric

# Analysis
results.df            # pandas DataFrame with all trials
results.summary()     # Human-readable summary string
```

### Strategies (`strategies/`)

Strategies suggest which parameter configurations to try next.

**Strategy ABC:**
```python
class Strategy(ABC):
    @abstractmethod
    def suggest(self, search_space, completed_trials) -> Optional[Dict]:
        """Return next params to try, or None if search complete."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'grid', 'random')."""
        ...
```

**Built-in Strategies:**

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `GridStrategy` | Exhaustive iteration | Small search spaces, reproducibility |
| `RandomStrategy(n)` | Random sampling | Large spaces, initial exploration |

**Extending with custom strategies:**
```python
class BayesianStrategy(Strategy):
    """Example: Bayesian optimization with Optuna."""

    def suggest(self, search_space, completed_trials):
        # Use completed_trials to build surrogate model
        # Return next promising configuration
        ...

    @property
    def name(self):
        return "bayesian"
```

### Storage (`storage/`)

Persistence layer for experiment data.

**Storage ABC:**
```python
class Storage(ABC):
    @abstractmethod
    def save_trial(self, experiment_name: str, trial: Trial) -> None: ...

    @abstractmethod
    def load_trial(self, experiment_name: str, trial_id: int) -> Optional[Trial]: ...

    @abstractmethod
    def load_trials(self, experiment_name: str) -> List[Trial]: ...

    @abstractmethod
    def save_config(self, experiment_name: str, config: Dict) -> None: ...

    @abstractmethod
    def load_config(self, experiment_name: str) -> Optional[Dict]: ...

    @abstractmethod
    def list_experiments(self) -> List[str]: ...
```

**Built-in Storage Backends:**

| Backend | Use Case | Format |
|---------|----------|--------|
| `MemoryStorage` | Testing, ephemeral | In-memory dict |
| `FileStorage(path)` | Human-readable, portable | CSV (trials) + JSON (config) |
| `SQLiteStorage(path)` | Queryable, single-file | SQLite database |

**FileStorage directory structure:**
```
experiments/
├── train_model_20240101_120000/
│   ├── config.json
│   └── trials.csv
└── another_experiment/
    ├── config.json
    └── trials.csv
```

### Experiment (`experiment.py`)

The main orchestration class.

```python
class Experiment:
    def __init__(
        self,
        fn: Callable,                    # Objective function
        name: Optional[str] = None,       # Auto-generated if None
        search_space: Optional[SearchSpace] = None,
        strategy: Optional[Strategy] = None,   # Auto-selected if None
        storage: Optional[Storage] = None,     # MemoryStorage if None
        metric: str = "loss",
        maximize: bool = False,
        gpu: int = 0,
    ): ...

    def run(self, trials: Optional[int] = None) -> Results:
        """Execute the experiment."""
        ...

    @classmethod
    def load(cls, name: str, storage: Storage) -> Optional[Experiment]:
        """Load existing experiment from storage."""
        ...
```

**Execution flow:**
```
1. Initialize experiment with fn, search_space, strategy, storage
2. Auto-select strategy if not provided:
   - Pure grid space → GridStrategy
   - Sweep or mixed → RandomStrategy(n=10)
3. While trials < max_trials:
   a. params = strategy.suggest(search_space, completed_trials)
   b. if params is None: break (strategy exhausted)
   c. trial = execute_trial(params)
   d. storage.save_trial(trial)
   e. completed_trials.append(trial)
4. Return Results(completed_trials)
```

### Decorator (`decorator.py`)

Syntactic sugar for function-based experiments.

```python
@experiment(lr=[1e-4, 1e-3], batch_size=[16, 32])
def train(lr, batch_size):
    """Objective function - receives params as kwargs."""
    model = train_model(lr=lr, batch_size=batch_size)
    return {"loss": model.loss, "accuracy": model.accuracy}

# Run experiment
results = train()           # Executes all 4 combinations
results = train(trials=2)   # Limit to 2 trials

# Access underlying Experiment object
train.experiment.name       # "train_20240101_120000"
train.experiment.strategy   # GridStrategy()
```

**How it works:**
1. `@experiment` creates a `SearchSpace.from_dict()` from kwargs
2. Wraps `fn` to accept `Dict[str, Any]` and call `fn(**params)`
3. Creates an `Experiment` instance with auto-generated name
4. Returns wrapper function that calls `experiment.run()`

---

## Backends Module

Located in `src/gpudispatch/backends/`

### Backend ABC (`base.py`)

All backends implement this interface.

```python
class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def allocate_gpus(self, count: int, memory: Optional[Memory]) -> List[GPU]: ...

    @abstractmethod
    def release_gpus(self, gpus: List[GPU]) -> None: ...

    @abstractmethod
    def list_available(self) -> List[GPU]: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    # Context manager support
    def __enter__(self) -> Backend: ...
    def __exit__(self, ...): ...
```

### LocalBackend (`local.py`)

Single-machine GPU management.

```python
backend = LocalBackend(
    gpus="auto",              # or [0, 1, 2, 3]
    memory_threshold="500MB", # GPU considered free if usage < 500MB
    polling_interval=5,       # Seconds between availability checks
    process_mode="subprocess" # or "thread"
)

with backend:
    gpus = backend.allocate_gpus(2)
    # Execute work on GPUs
    backend.release_gpus(gpus)
```

**GPU availability check:**
1. GPU must be in configured pool
2. GPU must not be currently allocated
3. GPU memory usage must be below threshold (checked via gpustat)

### SLURM Backend (`slurm.py`)

Extensible stub for HPC clusters.

```python
class SLURMBackend(Backend):
    def __init__(
        self,
        partition: str = "gpu",
        account: Optional[str] = None,
        time_limit: str = "1:00:00",
        nodes: int = 1,
        gpus_per_node: int = 1,
    ): ...
```

Currently raises `NotImplementedError` with helpful messages pointing to extension methods. Designed to be subclassed:

```python
class MySLURMBackend(SLURMBackend):
    def allocate_gpus(self, count, memory=None):
        # sbatch job submission
        # squeue monitoring
        # GPU node allocation
        ...
```

### Auto-Detection (`auto.py`)

Environment detection and backend selection.

```python
from gpudispatch import auto_dispatcher

# Auto-detect environment
dispatcher = auto_dispatcher()

# Force specific backend
dispatcher = auto_dispatcher(force_backend="local")
dispatcher = auto_dispatcher(force_backend="slurm")  # NotImplementedError
```

**Detection priority:**
1. SLURM: `SLURM_JOB_ID` or `SLURM_NODELIST` env vars
2. Kubernetes: `KUBERNETES_SERVICE_HOST` env var
3. AWS/GCP: Instance metadata (planned)
4. Default: LocalBackend

---

## Observability Module

Located in `src/gpudispatch/observability/`

### EventHook (`hooks.py`)

Callback-based event system.

```python
@dataclass
class EventHook:
    on_job_start: Optional[Callable] = None
    on_job_complete: Optional[Callable] = None
    on_job_failed: Optional[Callable] = None
    on_experiment_start: Optional[Callable] = None
    on_experiment_complete: Optional[Callable] = None
```

### HookRegistry

Global registry for event hooks.

```python
from gpudispatch.observability.hooks import hooks, EventHook

# Register a custom hook
my_hook = EventHook(
    on_job_start=lambda job_id, job_name, **kw: print(f"Started: {job_name}"),
    on_job_complete=lambda job_id, job_name, runtime_seconds, **kw:
        print(f"Done: {job_name} in {runtime_seconds}s"),
)
hooks.register(my_hook)

# Built-in logging hook
from gpudispatch.observability.hooks import LoggingHook
hooks.register(LoggingHook())
```

### Extending for Prometheus/OpenTelemetry

```python
class PrometheusHook(EventHook):
    def __init__(self, registry):
        self.job_duration = Histogram(
            'gpudispatch_job_duration_seconds',
            'Job duration',
            registry=registry
        )
        super().__init__(
            on_job_complete=self._on_complete,
        )

    def _on_complete(self, job_id, job_name, runtime_seconds, **kw):
        self.job_duration.observe(runtime_seconds)
```

---

## CLI Module

Located in `src/gpudispatch/cli/`

### Commands

```bash
# Show GPU status
gpudispatch status

# List all experiments
gpudispatch list

# Show experiment details
gpudispatch show <experiment_name>
```

Built with Click framework. Uses `FileStorage` with default path `./experiments`.

---

## Data Flow

### Job Execution Flow

```
User Code                  Dispatcher               Worker Thread
    │                          │                          │
    │  submit(fn, args)        │                          │
    ├─────────────────────────>│                          │
    │                          │  create Job              │
    │                          │  status = QUEUED         │
    │                          │  add to queue            │
    │                          │                          │
    │                          │  [dispatch loop]         │
    │                          │  check dependencies      │
    │                          │  find available GPUs     │
    │                          │  allocate GPUs           │
    │                          │                          │
    │                          │  spawn thread ──────────>│
    │                          │                          │ set CUDA_VISIBLE_DEVICES
    │                          │                          │ status = RUNNING
    │                          │                          │ execute fn(*args)
    │                          │                          │
    │                          │                          │ on success:
    │                          │                          │   result = return value
    │                          │                          │   status = COMPLETED
    │                          │                          │
    │                          │                          │ on error:
    │                          │                          │   error = str(exception)
    │                          │                          │   status = FAILED
    │                          │                          │
    │                          │<─────────────────────────│ release GPUs
    │                          │                          │
```

### Experiment Execution Flow

```
@experiment decorator          Experiment              Strategy
        │                          │                      │
        │  train()                 │                      │
        ├─────────────────────────>│                      │
        │                          │  suggest() ─────────>│
        │                          │<──────────────────── params
        │                          │                      │
        │                          │  execute_trial(params)
        │                          │    fn(params)
        │                          │    capture metrics
        │                          │    save to storage
        │                          │                      │
        │                          │  [repeat until max_trials or None]
        │                          │                      │
        │<─────────────────────────│  Results             │
        │                          │                      │
```

---

## Extension Points

### Custom Strategy

```python
from gpudispatch.experiments.strategies import Strategy

class MyStrategy(Strategy):
    def suggest(self, search_space, completed_trials):
        # Analyze completed_trials
        # Use surrogate model, genetic algorithm, etc.
        # Return next params or None
        return {"lr": 0.001, "batch_size": 32}

    @property
    def name(self):
        return "my_strategy"
```

### Custom Storage

```python
from gpudispatch.experiments.storage import Storage

class S3Storage(Storage):
    def __init__(self, bucket, prefix):
        self.bucket = bucket
        self.prefix = prefix

    def save_trial(self, experiment_name, trial):
        key = f"{self.prefix}/{experiment_name}/trials/{trial.id}.json"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=trial.to_json())

    # ... implement other methods
```

### Custom Backend

```python
from gpudispatch.backends import Backend

class KubernetesBackend(Backend):
    def allocate_gpus(self, count, memory=None):
        # Create K8s pod with nvidia.com/gpu resource request
        # Wait for pod to be scheduled
        # Return GPU objects
        ...

    def release_gpus(self, gpus):
        # Delete K8s pod
        ...
```

### Custom Hook

```python
from gpudispatch.observability.hooks import EventHook, hooks

class SlackNotificationHook(EventHook):
    def __init__(self, webhook_url):
        super().__init__(
            on_experiment_complete=self._notify,
        )
        self.webhook_url = webhook_url

    def _notify(self, experiment_id, total_jobs, **kw):
        requests.post(self.webhook_url, json={
            "text": f"Experiment {experiment_id} complete: {total_jobs} trials"
        })

hooks.register(SlackNotificationHook("https://..."))
```

---

## Design Patterns

### Abstract Factory (Backends)

`auto_dispatcher()` uses the Abstract Factory pattern to create the appropriate backend based on environment detection.

### Strategy Pattern (Search Strategies)

The `Strategy` ABC allows interchangeable algorithms for hyperparameter search without modifying the `Experiment` class.

### Observer Pattern (Hooks)

The `HookRegistry` implements the Observer pattern, allowing multiple observers (hooks) to react to events without tight coupling.

### Template Method (Storage)

The `Storage` ABC defines the skeleton of persistence operations, with concrete implementations providing the specific behavior.

### Decorator Pattern (@experiment, @gpu)

Python decorators wrap functions to add experiment or GPU execution capabilities without modifying the original function.

### Singleton Pattern (HookRegistry)

The global `hooks` registry is a module-level singleton, ensuring a single point of registration for all hooks.

---

## Thread Safety

All core classes use `threading.Lock` or `threading.RLock` for thread-safe operations:

- `Dispatcher`: Protects job queue, GPU allocation, and running jobs dict
- `JobQueue` implementations: Protects queue operations
- `LocalBackend`: Protects GPU pool and occupied set

This enables concurrent job submission and execution without race conditions.

---

## Performance Considerations

1. **PriorityQueue**: O(log n) insertion, O(log n) extraction with lazy deletion
2. **GPU detection**: Cached on backend start, not re-detected per allocation
3. **Memory threshold**: Checked per-allocation using gpustat (subprocess call)
4. **Storage writes**: Per-trial (not batched) for durability

For high-throughput scenarios:
- Use `MemoryStorage` and batch-save at end
- Increase `polling_interval` to reduce CPU usage
- Use `process_mode="thread"` for lower-latency job spawning
