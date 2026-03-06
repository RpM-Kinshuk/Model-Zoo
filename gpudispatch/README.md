# gpudispatch

Universal GPU orchestration - from laptop to supercomputer.

## Installation

```bash
pip install gpudispatch
```

## Quick Start

```python
from gpudispatch import gpu, Dispatcher

@gpu(count=1)
def train_model(config):
    # Your training code here
    return metrics

# Run on available GPU
result = train_model({"lr": 0.001})
```

## Features

- Zero-config start - works out of the box
- Backend-agnostic - Local, SLURM, Kubernetes, AWS, GCP
- Researcher-friendly - experiments, sweeps, checkpointing
- Production-ready - observability, fault tolerance

## License

Apache 2.0
