# Curation

The curated tables in this folder are the canonical phase 1 outputs for Model-Zoo.

Phase 1 stops at curation, enrichment, and probe status tracking. It does not include ESD analysis outputs.

## Files

- `model_zoo_phase2.csv` is the primary curated model list and the main phase 1 handoff.
- `model_zoo_phase2_analysis_ready.csv` is a derived phase 1 view for analysis.
- `model_zoo_phase2_prediction_ready.csv` is a derived phase 1 view for prediction.
- `model_zoo_phase2_probe_failures.csv` records probe failures for phase 1 tracking.

## Rules

- Treat the curated CSVs as generated artifacts.
- Update them together when the phase-2 source table changes.
- Keep local experiment outputs out of `data/curated/`.

## Usage

- Use `data/curated/model_zoo_phase2.csv` for normal phase-2 runs.
- Use the ready subsets when a workflow needs a narrower input slice.
