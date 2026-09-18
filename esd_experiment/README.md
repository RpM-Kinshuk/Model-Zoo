# ESD experiment runner

From the repository root, prepare a small pinned list first (metadata/config
only; no weights or GPU jobs):

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/my_run --limit 20 --prepare_only
```

Review `models.csv`, fix or remove any `pin_status=error` rows, then run it:

```bash
python esd_experiment/run_experiment.py \
  --model_list analysis_runs/phase2/my_run/models.csv \
  --output_dir analysis_runs/phase2/my_run \
  --gpus 5 6 7 --num_gpus_per_job 1
```

Eigenvalues are saved by default; use `--no-save_eigs` for scalar-only output.
Results are per-layer CSV and canonical HDF5 records with spectra, missing-fit
status and coverage. Use a fresh output directory when settings change.
Model and adapter-base revisions must be full commit SHAs. Remote code is off;
use `--trust_remote_code` only for reviewed repositories.

`--analysis_source auto` is the default: strict model loading first, then stored
safetensors or memory-mapped PyTorch matrices for narrowly identified architecture-support failures.
Use `model` to disable fallback or `checkpoint` for direct matrix analysis.
Broken weights, OOM and network/backend errors are not fallback triggers.
The summary records the actual source and fallback reason; tensor-only results
are not verified model layers. See the [rules and limits](../docs/operations/analysis.md#architecture-independent-checkpoint-matrices).

Refresh `summary.csv`, the per-model index of metrics, settings, coverage and
artifact paths, without loading spectra:

```bash
python esd_experiment/analyze_results.py --results_dir analysis_runs/phase2/my_run
```

See the [analysis guide](../docs/operations/analysis.md) for measurement
definitions, supported loaders, output layout, resume rules and GPU supervision.
`run_script.sh` at the repository root is the local HPC wrapper.

## Code map

- `run_experiment.py`: entrypoint for `src/run_experiment.py` (model list,
  preflight and worker jobs).
- `gputracker/`: GPU scheduling, runtime config reload, worker supervision and cleanup.
- `src/worker.py`: one-model load/analyze/save lifecycle and terminal status.
- `src/model_loader.py`: checkpoint routing and integrity checks.
- `src/checkpoint_tensors.py`: explicit, file-by-file checkpoint matrix analysis.
- `utils/analyze_results.py`: canonical result reader and per-model summary.
- `../net_esd/`: reusable spectral-analysis core.
- `tests/`: offline regression tests and separate installation/GPU smoke checks.
- `scripts/backend_pilot.py`: repeatable, download-free CPU/GPU numerical check.
