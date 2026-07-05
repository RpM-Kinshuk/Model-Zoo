# Infra Overview

The experiment infra is intentionally small and reusable:

```
run_experiment.py -> gputracker -> src/worker.py -> src/model_loader.py -> net_esd
```

- `run_experiment.py` reads the model table, runs preflight, writes `gpu_config.json`, and creates worker jobs.
- `gputracker` schedules jobs onto GPUs, tracks active workers, reloads runtime config on `SIGHUP`, and cleans up process groups, active logs, and per-worker caches.
- `src/worker.py` loads one model, writes heartbeat stage updates, runs ESD, and records terminal status.
- `src/model_loader.py` isolates HuggingFace/model-format behavior from scheduling.
- `net_esd` computes spectral metrics.

Runtime state lives under `<output_dir>/logs/`. Results live under `<output_dir>/stats/` and `<output_dir>/metrics/`.

The infra is generic enough for future worker types: the scheduler only needs commands, GPU requirements, heartbeat/stage files, and terminal status records.
