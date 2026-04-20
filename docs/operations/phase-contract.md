# Phase Contract

Model-Zoo uses the curated phase-2 tables in `data/curated/` as the source of truth for operational runs.

## Canonical inputs

- `data/curated/model_zoo_phase2.csv`
- `data/curated/model_zoo_phase2_analysis_ready.csv`
- `data/curated/model_zoo_phase2_prediction_ready.csv`
- `data/curated/model_zoo_phase2_probe_failures.csv`

## Canonical output root

- `analysis_runs/phase2/`

Use a run-specific subdirectory under that root, for example `analysis_runs/phase2/example_run/`.

## Scope

- Keep phase-2 run inputs in `data/curated/`.
- Keep phase-2 analysis artifacts under `analysis_runs/phase2/`.
- Do not write canonical outputs back into `data/curated/`.
