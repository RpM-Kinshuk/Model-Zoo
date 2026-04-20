# Analysis

Run phase 2 analysis from `data/curated/model_zoo_phase2.csv` and write outputs under `analysis_runs/phase2/<run_name>/`.

## Quick Spin-Up

If you just want to run phase 2, this is the default pattern:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```

Use a short run name in place of `example_run`.

## What To Check After The Run

- successful models should have both:
  - `stats/*.csv`
  - `metrics/*.h5`
- failures should appear in:
  - `logs/failed_models.txt`
  - `logs/failure_records.jsonl`
- `summary.csv` is useful for quick inspection, but it is not the success rule

## Completion Rule

A phase 2 run is complete only when both the per-model `stats/*.csv` and the matching `metrics/*.h5` exist for the run's successful models.

## From `esd_experiment/`

From `esd_experiment/`, use:

```bash
python run_experiment.py \
  --model_list ../data/curated/model_zoo_phase2.csv \
  --output_dir ../analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```
