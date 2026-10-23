# ESD experiment runner

Use the repository's [installation recipe](../README.md#1--create-the-environment)
and root `requirements.txt`; there is no separate experiment dependency list.

From the repository root, prepare a small pinned list first (metadata/config
only; no weights or GPU jobs):

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/my_run --limit 20 --prepare_only
```

Review `models.csv` (`preparation_status`, `preparation_reason`, and selected
weight sizes), then run it. Held rows remain in the list but are not dispatched:

```bash
python esd_experiment/run_experiment.py \
  --model_list analysis_runs/phase2/my_run/models.csv \
  --output_dir analysis_runs/phase2/my_run \
  --gpus 0 1 2 3 --num_gpus_per_job 1 \
  --worker_cache_root /scratch/$USER/model_zoo_worker_cache
```

Eigenvalues are saved by default; use `--no-save_eigs` for scalar-only output.
Supported matrices, including wide/tall Linear layers and ordinary dense 2D
registered parameters beyond `.weight`, are included by default. New parameter
matrices retain their full names and `weight_layout=parameter_matrix`: they
describe stored parameters, not inferred linear operators. Shared parameters
already accounted for are not measured again by this pass. `--filter_type` restores the older
restricted selection; it must be explicit when resuming runs made with that setting.
Built-in ViT token/position tables are also included with explicit `[0]` slices;
HF Swin's declared relative-position bias tables are measured as stored 2D
tables, not as attention operators. Selection 4 records generic parameter discovery; older
results remain readable but require a fresh directory under the new selection.
Unknown higher-dimensional parameters are not reshaped. Quantization/adapter
helpers and parametrization internals are excluded with recorded reasons.
Results are per-layer CSV and canonical HDF5 records with spectra, missing-fit
status and coverage. Use a fresh output directory when settings change.
Model and adapter-base revisions must be full commit SHAs. Remote code is off;
use `--trust_remote_code` only for reviewed repositories.

The normal workflow defaults to one worker, a 6 GiB preparation weight budget
(including adapter bases), and sampled per-worker limits of 16 GiB RAM and
12 GiB private cache. Configure `--max_weight_gib`, `--max_worker_ram_gib` and
`--max_worker_cache_gib`; zero disables the corresponding cap. RAM/cache checks
can overshoot between samples and are not OS quotas or GPU-memory predictions.
A breached cap stops that worker, records failure, cleans owned resources and
continues the queue when cleanup succeeds. Heartbeat/stage timeouts default to
`terminate`; `--stale_process_action log` deliberately keeps waiting instead.
Preparation checks files/access without downloading weights; its small JSON
caches are disposable. The ordinary runner does not need the pilot observer.

GPU dispatch requires two consecutive idle checks, five seconds apart, with no
GPU processes and memory below the configured threshold. Use `--max_check` to
require more checks on a busy shared host; occupancy is checked again for every job.

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

Build a collection-wide CSV index from multiple runs (spectra stay in HDF5):

```bash
python esd_experiment/analyze_results.py \
  --results_dir analysis_runs/phase2/*/ --output analysis_runs/phase2/catalog.csv
```

Each row keeps `run_dir`, pins, settings and artifact paths. Repeated model IDs
and failed attempts remain separate; mixed settings are not pooled. This reads
the original pairs and outcomes, not potentially stale per-run summaries.

Check `matrix_coverage` as well as `outcome`: a valid result can still be partial
(for example, embeddings measured but packed quantized projections skipped).
Partial/unknown coverage is excluded from printed pooled metrics, not deleted.
Use the [named-spectrum and cross-run query examples](../docs/operations/analysis.md#query-models-and-read-a-named-spectrum)
to select models from the index and read just the layer spectrum you need.

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
- `tests/`: offline regression tests and a small CPU installation check.
- `scripts/backend_pilot.py`: repeatable, download-free CPU/GPU numerical check.
