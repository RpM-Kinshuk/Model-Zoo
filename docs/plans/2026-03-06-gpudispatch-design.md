# gpudispatch: Universal GPU Orchestration Library

**Design Document**
**Date:** 2026-03-06
**Status:** Approved
**Author:** Model-Zoo Team

---

## Executive Summary

`gpudispatch` is a production-grade, researcher-friendly GPU orchestration library designed to be the universal solution for ML workload management. It provides a unified interface that works seamlessly from a single laptop GPU to multi-node HPC clusters and cloud deployments.

**Vision:** GPU orchestration that just works - from laptop to supercomputer.

**Core Principles:**
1. Zero-config start - Works out of the box with sensible defaults
2. Progressive disclosure - Simple things simple, complex things possible
3. Research-first - Designed for experimentation workflows
4. Backend-agnostic - Write once, run anywhere
5. Observable by default - Know what's happening without extra setup

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Target Users](#2-target-users)
3. [Architecture](#3-architecture)
4. [API Design](#4-api-design)
5. [Experiment Primitives](#5-experiment-primitives)
6. [Backend Specifications](#6-backend-specifications)
7. [Innovative Features](#7-innovative-features)
8. [Observability](#8-observability)
9. [CLI Interface](#9-cli-interface)
10. [Configuration System](#10-configuration-system)
11. [Package Structure](#11-package-structure)
12. [Testing Strategy](#12-testing-strategy)
13. [Documentation Plan](#13-documentation-plan)
14. [Roadmap](#14-roadmap)
15. [Model-Zoo Integration](#15-model-zoo-integration)

---

## 1. Problem Statement

Researchers and ML engineers face fragmented tooling for GPU resource management:

| Tool | Limitation |
|------|------------|
| Ray | Too heavy for simple GPU queueing |
| SLURM | No local mode, complex setup |
| Dask | CPU-focused, GPU support bolted on |
| submitit | Facebook-internal patterns, limited backends |

**gpudispatch** fills this gap with a unified, lightweight, yet powerful solution.

---

## 2. Target Users

| User | Pain Point We Solve |
|------|---------------------|
| PhD student | "I just want to run on whatever GPUs are free" |
| ML engineer | "I need to queue 500 experiments overnight" |
| Research lab | "We share 8 GPUs and it's chaos" |
| MLOps team | "We need visibility into GPU utilization" |
| Principal scientist | "I need reproducible experiments at scale" |

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           USER LAYER                                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │ @gpu()      │ │ Experiment  │ │ Sweep       │ │ CLI                 │  │
│  │ decorator   │ │ Grid        │ │ Search      │ │ gpudispatch run ... │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│                         EXPERIMENT LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ ExperimentManager                                                    │  │
│  │ - Grid search, random search, Bayesian optimization                 │  │
│  │ - Hyperparameter sweeps with early stopping                         │  │
│  │ - Checkpointing, resumption, fault tolerance                        │  │
│  │ - Experiment versioning and reproducibility                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│                          CORE LAYER                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ Dispatcher   │ │ JobQueue     │ │ Resource     │ │ MemoryManager    │  │
│  │              │ │ - Priority   │ │ Manager      │ │ - Estimation     │  │
│  │ - Lifecycle  │ │ - FIFO/LIFO  │ │ - GPU/CPU    │ │ - OOM prevention │  │
│  │ - Signals    │ │ - Fair share │ │ - Memory     │ │ - Adaptive       │  │
│  │ - Events     │ │ - Deadlines  │ │ - Topology   │ │ - Profiling      │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│                         BACKEND LAYER                                       │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ Local  │ │ SLURM  │ │ PBS    │ │ K8s    │ │ AWS    │ │ GCP    │ ...   │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
├────────────────────────────────────────────────────────────────────────────┤
│                      OBSERVABILITY LAYER                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │ Prometheus  │ │ OpenTelem   │ │ Structured  │ │ Dashboards          │  │
│  │ Metrics     │ │ Traces      │ │ Logging     │ │ (Grafana/Web UI)    │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Design Principles

- **Layered Architecture**: Each layer has clear responsibilities
- **Plugin System**: Backends are plugins implementing a standard interface
- **Event-Driven**: Core uses events for loose coupling
- **Async-First**: Non-blocking operations where possible
- **Type-Safe**: Full type hints with mypy strict compliance

---

## 4. API Design

### 4.1 Level 1: Dead Simple (90% of use cases)

```python
from gpudispatch import gpu, run

# Decorator - just works
@gpu(count=1)
def train_model(config):
    model = load_model(config)
    model.fit()
    return model.metrics

# Run it - automatically finds free GPU
result = train_model({"lr": 0.001})

# Or batch multiple configs
configs = [{"lr": lr} for lr in [0.001, 0.01, 0.1]]
results = run(train_model, configs)  # Queues and dispatches automatically
```

### 4.2 Level 2: More Control

```python
from gpudispatch import Dispatcher, GPU, Job

# Explicit dispatcher with config
dispatcher = Dispatcher(
    resources=[GPU(0), GPU(1), GPU(2), GPU(3)],
    memory_threshold="500MB",
)

# Submit jobs with requirements
job1 = dispatcher.submit(train_large, gpu=2, memory="24GB", priority=10)
job2 = dispatcher.submit(train_small, gpu=1, memory="8GB", priority=5)

# Dependencies
job3 = dispatcher.submit(evaluate, gpu=1, after=[job1, job2])

# Wait for completion
results = dispatcher.gather([job1, job2, job3])
```

### 4.3 Level 3: Full Control (Production/HPC)

```python
from gpudispatch import Dispatcher
from gpudispatch.backends import SLURMBackend
from gpudispatch.observers import PrometheusExporter, SlackNotifier

# HPC cluster setup
dispatcher = Dispatcher(
    backend=SLURMBackend(
        partition="gpu",
        account="my-lab",
        qos="high",
        modules=["cuda/12.0", "python/3.10"],
    ),
    observers=[
        PrometheusExporter(port=9090),
        SlackNotifier(webhook=SLACK_URL, on=["failure", "completion"]),
    ],
    memory_strategy="adaptive",
)

# Runtime control
dispatcher.reload_config()      # SIGHUP equivalent
dispatcher.drain()              # Finish current, stop new
dispatcher.shutdown()           # Graceful termination
```

### 4.4 Auto-Detection

```python
from gpudispatch import auto_dispatcher

# Automatically detects:
# - Local machine → LocalBackend with available GPUs
# - SLURM cluster → SLURMBackend with sane defaults
# - Kubernetes → K8sBackend reading from pod spec
# - AWS/GCP → Cloud backend from instance metadata

dispatcher = auto_dispatcher()  # Just works everywhere
```

---

## 5. Experiment Primitives

### 5.1 Grid Search

```python
from gpudispatch import Experiment, Grid

experiment = Experiment(
    name="transformer-ablation",
    fn=train_model,
    grid=Grid(
        learning_rate=[1e-4, 1e-3, 1e-2],
        batch_size=[16, 32, 64],
        num_layers=[6, 12, 24],
    ),
    resources={"gpu": 1, "memory": "16GB"},
)

results = experiment.run()
```

### 5.2 Hyperparameter Sweeps

```python
from gpudispatch import Experiment, Sweep, Log, Choice, Uniform, Int

experiment = Experiment(
    name="efficient-search",
    fn=train_model,
    sweep=Sweep(
        learning_rate=Log(1e-5, 1e-1),
        batch_size=Choice([16, 32, 64]),
        dropout=Uniform(0.0, 0.5),
        num_layers=Int(4, 32),
    ),
    strategy="bayesian",
    num_trials=100,
    pruner="median",
)

results = experiment.run(max_parallel=8)
```

### 5.3 Multi-Objective Optimization

```python
experiment = Experiment(
    name="pareto-search",
    fn=train_model,
    sweep=Sweep(...),
    objectives=[
        Maximize("accuracy"),
        Minimize("latency_ms"),
        Minimize("memory_mb"),
    ],
    strategy="nsga2",
)

pareto_front = experiment.run()
```

### 5.4 Experiment Pipelines

```python
from gpudispatch import Pipeline, Experiment

pipeline = Pipeline(
    name="full-research-workflow",
    stages=[
        Experiment(name="pretraining", fn=pretrain, grid=Grid(...)),
        Experiment(
            name="finetuning",
            fn=finetune,
            depends_on="pretraining",
            inherit_best=True,
        ),
        Experiment(
            name="evaluation",
            fn=evaluate,
            depends_on="finetuning",
            inherit_all=True,
        ),
    ],
)

pipeline.run()
```

---

## 6. Backend Specifications

### 6.1 Local Backend (Default)

```python
from gpudispatch.backends import LocalBackend

backend = LocalBackend(
    gpus="auto",
    memory_threshold="500MB",
    polling_interval=5,
    process_isolation=True,
)
```

### 6.2 SLURM Backend

```python
from gpudispatch.backends import SLURMBackend

backend = SLURMBackend(
    partition="gpu",
    account="my-lab-account",
    qos="normal",
    time_limit="24:00:00",
    modules=["cuda/12.0", "anaconda/2024"],
    conda_env="ml-research",
    constraint="a100",
    gpus_per_node=8,
    cpus_per_gpu=8,
    memory_per_gpu="64GB",
)
```

### 6.3 Kubernetes Backend

```python
from gpudispatch.backends import KubernetesBackend

backend = KubernetesBackend(
    namespace="ml-experiments",
    image="my-registry/ml-image:latest",
    gpu_resource="nvidia.com/gpu",
    node_selector={"gpu-type": "a100"},
    volumes={
        "data": {"pvc": "shared-data-pvc", "mount": "/data"},
        "checkpoints": {"pvc": "checkpoints-pvc", "mount": "/ckpts"},
    },
    min_replicas=0,
    max_replicas=16,
)
```

### 6.4 Cloud Backends

```python
from gpudispatch.backends import AWSBackend, GCPBackend

aws = AWSBackend(
    region="us-west-2",
    instance_types=["p4d.24xlarge", "p3.16xlarge"],
    spot=True,
    max_cost_per_hour=50.0,
)

gcp = GCPBackend(
    project="my-project",
    zone="us-central1-a",
    accelerator_type="nvidia-tesla-a100",
    preemptible=True,
)
```

---

## 7. Innovative Features

### 7.1 NVIDIA MIG Support

Partition modern GPUs (A100/H100) into isolated instances:

```python
from gpudispatch.resources import MIGInstance

dispatcher = Dispatcher(
    resources=[
        MIGInstance(gpu=0, profile="3g.40gb"),
        MIGInstance(gpu=0, profile="2g.20gb"),
        MIGInstance(gpu=0, profile="2g.20gb"),
    ],
)

# Or automatic partitioning
dispatcher = Dispatcher(
    mig_strategy="auto",
    mig_profiles={
        "small": "1g.10gb",
        "medium": "3g.40gb",
        "large": "7g.80gb",
    },
)
```

### 7.2 Predictive Memory Management

ML-based memory estimation to prevent OOM:

```python
from gpudispatch.memory import PredictiveMemoryManager

dispatcher = Dispatcher(
    memory_manager=PredictiveMemoryManager(
        model="transformer",
        features=["model_params", "batch_size", "sequence_length"],
        auto_learn=True,
        conservative=True,
        margin_percent=15,
    ),
)

# Memory prediction API
prediction = dispatcher.predict_memory(
    model_params=7e9,
    batch_size=32,
    sequence_length=2048,
)
print(f"Predicted: {prediction.peak_mb}MB")
print(f"Recommended batch size: {prediction.recommended_batch_size}")
```

### 7.3 Elastic Training

Handle node failures and dynamic scaling:

```python
from gpudispatch.distributed import ElasticTraining

@gpu(
    min_gpus=4,
    max_gpus=32,
    elastic=ElasticTraining(
        scale_up_threshold="queue_wait > 5m",
        scale_down_threshold="gpu_util < 30% for 10m",
        fault_tolerance="restart",
        min_healthy_nodes=2,
        maintain_global_batch_size=True,
    ),
)
def elastic_training(config, world_size):
    ...
```

### 7.4 Intelligent Failure Analysis

Automatic root cause analysis and remediation:

```python
from gpudispatch.diagnostics import FailureAnalyzer

dispatcher = Dispatcher(
    failure_analyzer=FailureAnalyzer(
        analyze_oom=True,
        analyze_nccl=True,
        analyze_nan=True,
        suggest_fixes=True,
        auto_fix={
            "oom": "reduce_batch_size",
            "nccl_timeout": "increase_timeout",
            "nan_loss": "reduce_lr_and_retry",
        },
    ),
)
```

### 7.5 Experiment Time Travel

Version control for experiments:

```python
from gpudispatch import Experiment

experiment = Experiment.load("transformer-ablation")
experiment.history()

# Reproduce past run
past_run = experiment.checkout(run_id=2)
past_run.reproduce()

# Compare runs
experiment.compare([1, 2, 3], metrics=["accuracy", "loss"])
```

### 7.6 Reproducibility Vault

Cryptographic guarantees:

```python
from gpudispatch.reproducibility import ReproducibilityVault

experiment = Experiment(
    name="critical-research",
    fn=train_model,
    reproducibility=ReproducibilityVault(
        capture=["code_snapshot", "environment", "config", "hardware", "random_seeds"],
        sign=True,
        certificate=True,
    ),
)

# Generate reproducibility report
experiment.generate_report("reproducibility_report.pdf")
```

### 7.7 Carbon-Aware Scheduling

Environmentally conscious computing:

```python
from gpudispatch.scheduling import CarbonAwareScheduler

dispatcher = Dispatcher(
    scheduler=CarbonAwareScheduler(
        carbon_api_key=CARBON_API_KEY,
        max_carbon_intensity=200,
        prefer_low_carbon=True,
        regions=["us-west-2", "eu-west-1", "eu-north-1"],
    ),
)
```

### 7.8 Communication-Efficient Distributed Training

```python
from gpudispatch.distributed import GradientCompression, HierarchicalAllReduce

@gpu(
    8,
    distributed={
        "compression": GradientCompression(algorithm="powersgd", rank=4),
        "overlap": OverlapStrategy.FULL,
        "topology": HierarchicalAllReduce(intra_node="nvlink", inter_node="ring"),
    },
)
def communication_efficient_training(config):
    ...
```

### 7.9 Multi-Fidelity Optimization

Transfer knowledge from smaller runs:

```python
from gpudispatch.experiments import TransferLearningSearch

experiment = Experiment(
    search=TransferLearningSearch(
        fidelity_schedule=[
            {"epochs": 1, "data_fraction": 0.1, "trials": 100},
            {"epochs": 5, "data_fraction": 0.3, "trials": 30},
            {"epochs": 20, "data_fraction": 1.0, "trials": 10},
        ],
        transfer_method="warm_start",
        cost_aware=True,
    ),
)
```

### 7.10 GPU Time-Sharing (MPS)

Share GPU for inference workloads:

```python
from gpudispatch.backends import MPSBackend

dispatcher = Dispatcher(
    backend=MPSBackend(
        gpu=0,
        max_clients=8,
        memory_limit_per_client="10GB",
    ),
)
```

---

## 8. Observability

```python
from gpudispatch.observers import (
    PrometheusExporter,
    OpenTelemetryTracer,
    StructuredLogger,
    SlackNotifier,
)

dispatcher = Dispatcher(
    observers=[
        PrometheusExporter(
            port=9090,
            metrics=["gpu_utilization", "jobs_queued", "job_runtime"],
        ),
        OpenTelemetryTracer(endpoint="http://jaeger:14268"),
        StructuredLogger(output="logs/gpudispatch.jsonl", level="INFO"),
        SlackNotifier(webhook=SLACK_URL, on=["job_failed", "experiment_completed"]),
    ],
)

# Built-in metrics query
stats = dispatcher.stats()
print(f"GPU utilization: {stats.gpu_utilization_percent}%")
print(f"Queue depth: {stats.jobs_queued}")
```

---

## 9. CLI Interface

```bash
# Quick start
$ gpudispatch run train.py --gpu 1

# Experiment management
$ gpudispatch experiment launch --name "lr-search" --script train.py \
    --grid "learning_rate=[1e-4,1e-3,1e-2]" --gpu 1

$ gpudispatch experiment status lr-search
$ gpudispatch experiment results lr-search --sort accuracy --top 5

# Queue management
$ gpudispatch queue
$ gpudispatch cancel <job_id>
$ gpudispatch priority <job_id> --high

# Resource monitoring
$ gpudispatch gpus
$ gpudispatch monitor  # Live TUI dashboard

# Configuration
$ gpudispatch init
$ gpudispatch config backend slurm --partition gpu
$ gpudispatch daemon start --port 8080
```

---

## 10. Configuration System

```yaml
# .gpudispatch.yaml
version: "1.0"

defaults:
  gpu: 1
  memory: "16GB"
  timeout: "24h"

backend:
  type: "auto"
  local:
    gpus: [0, 1, 2, 3]
    memory_threshold: "500MB"
  slurm:
    partition: "gpu"
    account: "my-lab"

experiments:
  checkpoint_dir: "./checkpoints"
  auto_checkpoint: true

observability:
  prometheus:
    enabled: true
    port: 9090
  logging:
    level: "INFO"

memory:
  strategy: "adaptive"
  oom_retry: true

scheduling:
  strategy: "smart"
```

---

## 11. Package Structure

```
gpudispatch/
├── pyproject.toml
├── README.md
├── LICENSE (Apache 2.0)
├── CHANGELOG.md
├── CONTRIBUTING.md
│
├── src/gpudispatch/
│   ├── __init__.py
│   ├── py.typed
│   │
│   ├── core/
│   │   ├── dispatcher.py
│   │   ├── job.py
│   │   ├── queue.py
│   │   ├── resources.py
│   │   ├── scheduler.py
│   │   └── signals.py
│   │
│   ├── backends/
│   │   ├── base.py
│   │   ├── local.py
│   │   ├── slurm.py
│   │   ├── pbs.py
│   │   ├── kubernetes.py
│   │   ├── aws.py
│   │   └── gcp.py
│   │
│   ├── experiments/
│   │   ├── experiment.py
│   │   ├── grid.py
│   │   ├── sweep.py
│   │   ├── search.py
│   │   └── pipeline.py
│   │
│   ├── memory/
│   │   ├── estimator.py
│   │   ├── profiler.py
│   │   └── adaptive.py
│   │
│   ├── observers/
│   │   ├── prometheus.py
│   │   ├── opentelemetry.py
│   │   └── notifiers.py
│   │
│   ├── distributed/
│   │   ├── torch.py
│   │   ├── elastic.py
│   │   └── compression.py
│   │
│   ├── ui/
│   │   ├── dashboard.py
│   │   ├── tui.py
│   │   └── notebook.py
│   │
│   └── cli/
│       ├── main.py
│       └── commands/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/
│   ├── quickstart.md
│   ├── user-guide/
│   └── api/
│
└── examples/
```

---

## 12. Testing Strategy

- **Unit tests**: Fast, isolated, mock external dependencies
- **Integration tests**: Backend mocking, subprocess testing
- **E2E tests**: Real GPU tests on CI with GPU runners
- **Property-based tests**: Hypothesis for scheduler fairness
- **Benchmarks**: Performance regression tests

Target: >90% code coverage, mypy strict compliance.

---

## 13. Documentation Plan

```
docs/
├── Getting Started
│   ├── Installation
│   ├── 5-Minute Quickstart
│   └── Choosing a Backend
├── User Guide
│   ├── Core Concepts
│   ├── Backends (Local, SLURM, K8s, Cloud)
│   ├── Experiments
│   └── Advanced Topics
├── API Reference (auto-generated)
├── Examples
└── Contributing
```

---

## 14. Roadmap

### Phase 1: Core Foundation (v0.1.0) - 2 weeks
- Core dispatcher, job queue, resource management
- Local backend with GPU detection
- MPS (GPU sharing) support
- Basic CLI and configuration

### Phase 2: HPC Integration (v0.2.0) - 2 weeks
- SLURM, PBS backends
- MIG (Multi-Instance GPU) support
- Signal handling and runtime control

### Phase 3: Experiments (v0.3.0) - 2 weeks
- Grid, Sweep, Bayesian optimization
- Multi-fidelity search (ASHA/Hyperband)
- Checkpointing and resumption

### Phase 4: Cloud & Scale (v0.4.0) - 2 weeks
- Kubernetes, AWS, GCP backends
- Elastic training support
- Spot instance handling

### Phase 5: Intelligence (v0.5.0) - 2 weeks
- Predictive memory management
- Intelligent failure analysis
- Communication-efficient distributed training

### Phase 6: Observability (v0.6.0) - 1 week
- Prometheus, OpenTelemetry
- Web dashboard and TUI
- Notebook integration

### Phase 7: Advanced Features (v0.7.0) - 2 weeks
- Reproducibility vault
- Carbon-aware scheduling
- Federated dispatch

### Phase 8: Polish (v1.0.0) - 2 weeks
- Comprehensive documentation
- Performance benchmarks
- Natural language interface (experimental)

---

## 15. Model-Zoo Integration

After extracting gpudispatch, Model-Zoo becomes a clean consumer:

```python
# esd_experiment/src/run_experiment.py (refactored)

from gpudispatch import Experiment
from net_esd import net_esd_estimator
from .worker import analyze_model

def run_esd_experiment(model_list: str, output_dir: str, **esd_params):
    models = load_model_list(model_list)

    experiment = Experiment(
        name="esd-analysis",
        fn=analyze_model,
        configs=[
            {"model_id": m.model_id, "output_dir": output_dir, **esd_params}
            for m in models
        ],
        resources={"gpu": 1, "memory": "auto"},
    )

    return experiment.run()
```

---

## Appendix A: Comparison with Existing Tools

| Feature | gpudispatch | Ray | SLURM | Dask | submitit |
|---------|-------------|-----|-------|------|----------|
| Zero-config start | Yes | No | No | Partial | No |
| Local GPU mode | Yes | Yes | No | Partial | No |
| HPC backends | Yes | Partial | Native | No | Yes |
| Cloud backends | Yes | Yes | No | Yes | No |
| Experiment primitives | Yes | Tune | No | No | No |
| Memory estimation | Yes | No | No | No | No |
| MIG support | Yes | No | Partial | No | No |
| Observability | Built-in | External | External | External | No |

---

## Appendix B: Performance Targets

- Job submission latency: <10ms
- GPU detection: <100ms
- Memory estimation accuracy: >90%
- Queue throughput: >1000 jobs/sec
- Dashboard refresh: <1s

---

## Appendix C: Security Considerations

- No credential storage in plaintext
- Support for secrets managers (Vault, AWS Secrets)
- Process isolation between jobs
- Network policies for K8s backend
- Audit logging for compliance

---

*Document approved for implementation: 2026-03-06*
