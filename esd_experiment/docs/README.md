# ESD Experiment Docs

The current user-facing guide is `../README.md`.

Use these docs as short references:

- `QUICKSTART.md`: minimal run command and output checks
- `OVERVIEW.md`: infra boundaries and data flow
- `GPU_FIX.md`: current GPU scheduling and stuck-worker checks
- `MIGRATION.md`: legacy filename mapping

Current entrypoints:

- repository root: `python esd_experiment/run_experiment.py ...`
- from `esd_experiment/`: `python run_experiment.py ...`
- worker implementation: `src/worker.py`
