# Phase Contract

Model-Zoo owns phases 1 and 2. Model-Research owns phase 3.

## Phases

- Phase 1: data gathering, curation, enrichment, and probe status tracking in Model-Zoo.
- Phase 2: ESD analysis runs in Model-Zoo.
- Phase 3: downstream research and follow-on work in Model-Research.

## Ownership Split

- Model-Zoo is the source of truth for phase 1 inputs and phase 2 analysis runs.
- Model-Research consumes the phase 2 outputs for phase 3 work.

## Canonical Phase 1 Outputs

- `data/curated/model_zoo_phase2.csv`
- derived views in `data/curated/`, including:
  - `model_zoo_phase2_analysis_ready.csv`
  - `model_zoo_phase2_prediction_ready.csv`
  - `model_zoo_phase2_probe_failures.csv`
- `data/manual_audits/` if present, for manual audit artifacts that support phase 1

Phase 1 stops at curation, enrichment, and probe status tracking. It does not produce ESD run artifacts.

## Canonical Phase 2 Inputs

- `data/curated/model_zoo_phase2.csv`

## Canonical Phase 2 Outputs

- `analysis_runs/phase2/<run_name>/stats/*.csv`
- `analysis_runs/phase2/<run_name>/metrics/*.h5`
- `analysis_runs/phase2/<run_name>/logs/failed_models.txt`
- `analysis_runs/phase2/<run_name>/logs/failure_records.jsonl`

Use a run-specific subdirectory under `analysis_runs/phase2/`, for example `analysis_runs/phase2/example_run/`.

## Scope

- Keep phase 1 outputs in `data/curated/` and `data/manual_audits/` when present.
- Keep phase 2 analysis artifacts under `analysis_runs/phase2/`.
- Do not write canonical outputs back into `data/curated/`.
