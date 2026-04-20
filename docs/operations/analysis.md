# Analysis

Run phase 2 analysis from `data/curated/model_zoo_phase2.csv` and write outputs under `analysis_runs/phase2/<run_name>/`.

## Preferred command

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```

## Output layout

- Expected outputs:
  - `stats/*.csv`
  - `metrics/*.h5`
  - `logs/failed_models.txt`
  - `logs/failure_records.jsonl`
- Summary files are written below the chosen `--output_dir`.
- Use one subdirectory per run so results stay easy to compare and delete.

## Completion Rule

A phase 2 run is complete only when both the per-model `stats/*.csv` and the matching `metrics/*.h5` exist for the run's successful models.

## Relative-path variant

From `esd_experiment/`, use:

```bash
python run_experiment.py \
  --model_list ../data/curated/model_zoo_phase2.csv \
  --output_dir ../analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```
