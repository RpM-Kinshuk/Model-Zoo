# Curation

The curated phase-2 tables are the operational inputs for Model-Zoo.

## Files

- `model_zoo_phase2.csv` is the primary curated model list.
- `model_zoo_phase2_analysis_ready.csv` is the analysis-ready subset.
- `model_zoo_phase2_prediction_ready.csv` is the prediction-ready subset.
- `model_zoo_phase2_probe_failures.csv` records probe failures that should stay visible to operators.

## Rules

- Treat the curated CSVs as generated artifacts.
- Update them together when the phase-2 source table changes.
- Keep local experiment outputs out of `data/curated/`.

## Usage

- Use `data/curated/model_zoo_phase2.csv` for normal phase-2 runs.
- Use the ready subsets when a workflow needs a narrower input slice.
