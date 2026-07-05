# ESD Experiment Structure

Current layout:

```
esd_experiment/
├── run_experiment.py        # wrapper for src/run_experiment.py
├── analyze_results.py       # wrapper for utils/analyze_results.py
├── create_model_list.py     # helper for simple model-list creation
├── gputracker/              # scheduling, runtime config, worker supervision
├── src/
│   ├── run_experiment.py    # model-table normalization and orchestration
│   ├── worker.py            # one-model load/analyze/save worker
│   └── model_loader.py      # HuggingFace/model-format loading
├── utils/
│   └── analyze_results.py   # summary generation
├── tests/                   # scheduler, loader, worker, and setup tests
├── examples/                # lightweight examples
└── docs/                    # concise current references
```

Reusable infra boundary:

```
run_experiment.py -> gputracker -> worker.py -> model_loader.py -> net_esd
```

Responsibilities:

- `run_experiment.py`: read curated or legacy model CSVs, preflight rows, write `gpu_config.json`, create worker jobs.
- `gputracker`: assign GPUs, enforce concurrency, reload runtime config on `SIGHUP`, track active workers, detect stale workers, clean process groups and per-worker caches.
- `worker.py`: load one model, update heartbeat stage, run ESD, write `stats/*.csv` and `metrics/*.h5`, record terminal status.
- `model_loader.py`: isolate model-format decisions from scheduler logic.
- `net_esd`: spectral computation core.

Canonical phase-2 inputs and outputs are documented in `../docs/operations/analysis.md` from the repository root.
