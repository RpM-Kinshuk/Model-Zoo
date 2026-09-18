# Curation

The curated tables in this folder are the canonical phase 1 artifacts for Model-Zoo and the phase 2 operational inputs.

These CSVs are generated/synced outputs, not hand-edited files.

Phase 1 stops at curation, enrichment, and probe status tracking. It does not include ESD analysis outputs.

## Files

- `model_zoo_phase2.csv` is the primary curated model list and the main phase 1 handoff.
- `model_zoo_phase2_analysis_ready.csv` is a derived phase 1 view for analysis.
- `model_zoo_phase2_prediction_ready.csv` is a derived phase 1 view for prediction.
- `model_zoo_phase2_probe_failures.csv` records probe failures for phase 1 tracking.

## Rules

- Treat the curated CSVs as generated/synced artifacts.
- Update them together when the phase-2 source table changes.
- Keep local experiment outputs out of `data/curated/`.

## Usage

- Use `data/curated/model_zoo_phase2.csv` for normal phase-2 runs.
- Use the ready subsets when a workflow needs a narrower input slice.

## Architecture-label caveat

The current handoff's 8,467 Atlas-derived `Architecture` cells are truncated to
the first character of their source strings. The analysis-ready subset has
3,164 rows with blank primary architecture labels; all come from `seed_metadata`.
Neither field/subset is suitable for cross-family stratification as-is.

Keep these generated tables as source evidence. The existing `--prepare_only`
step now records revision-specific Hub config labels separately in `models.csv`.
Use those fields for new selections, with missing labels kept explicit; do not
guess a family from the initial or repository name. See the
[analysis guide](analysis.md#pinned-inputs-and-remote-code) for fields and provenance.
The curation exporter is absent from this checkout and still needs correction at
its source before these tables are regenerated.
