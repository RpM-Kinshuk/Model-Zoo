# Analysis

Run phase-2 analysis from the curated table in `data/curated/` and write outputs under `analysis_runs/phase2/`.

## Preferred command

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```

## Output layout

- Stats, metrics, logs, and summary files are written below the chosen `--output_dir`.
- Use one subdirectory per run so results stay easy to compare and delete.

## Relative-path variant

From `esd_experiment/`, use:

```bash
python run_experiment.py \
  --model_list ../data/curated/model_zoo_phase2.csv \
  --output_dir ../analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```
