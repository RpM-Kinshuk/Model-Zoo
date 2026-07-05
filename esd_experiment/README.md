# ESD Experiment Infra

This directory is the plug-and-play experiment runner for large-scale ESD analysis.

## Current Path

```
run_experiment.py -> gputracker -> src/worker.py -> src/model_loader.py -> net_esd
```

- `run_experiment.py`: entrypoint wrapper for `src/run_experiment.py`
- `gputracker/`: GPU scheduling, runtime config reload, worker process groups, active state, stale-worker handling, and cache cleanup
- `src/worker.py`: one-model worker for loading, ESD analysis, output writes, heartbeat stages, and terminal status
- `src/model_loader.py`: HuggingFace/model-format handling
- `../net_esd/`: reusable spectral-analysis core

## Quick Run

From the repository root:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/example_run \
  --gpus 0 1 2 3 \
  --num_gpus_per_job 1
```

From this directory:

```bash
python run_experiment.py \
  --model_list ../data/curated/model_zoo_phase2.csv \
  --output_dir ../analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```

`run_script.sh` at the repository root is the HPC-style reference wrapper.

## Runtime Config

The runner writes `<output_dir>/gpu_config.json`. Edit it and send `SIGHUP` to the main runner PID to reload:

```json
{
  "available_gpus": [0, 1, 2, 3],
  "max_checks": 1,
  "memory_threshold_mb": 500,
  "max_concurrent_jobs": 2,
  "stale_process_action": "log",
  "heartbeat_timeout_seconds": 7200,
  "stage_timeout_seconds": {
    "load": 7200,
    "analyze": 28800,
    "save": 1800,
    "default": 14400
  },
  "termination_grace_seconds": 30
}
```

- `heartbeat_timeout_seconds`: catches workers whose heartbeat stops.
- `stage_timeout_seconds`: catches workers whose heartbeat is alive but whose foreground stage is stuck.
- `stale_process_action`: use `log` while tuning, then `terminate` when the timeouts are trusted.

## Outputs

Successful models write:

- `stats/*.csv`: per-layer ESD metrics
- `metrics/*.h5`: alpha matrices

Failures and terminal states write:

- `logs/failed_models.txt`
- `logs/failure_records.jsonl`
- `logs/terminal_status/*.json`

Live monitoring files:

- `logs/current_state.json`
- `logs/active_workers/<run_id>/*`

Active worker logs, heartbeat files, and per-worker HF caches are removed when a worker finishes, fails, or is killed.

## Tests

```bash
python -m pytest esd_experiment/tests -q
```
