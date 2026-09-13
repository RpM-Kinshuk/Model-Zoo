# ESD analysis

Run from the repository root with a curated model list and a fresh output directory:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/my_run \
  --gpus 5 6 7 --num_gpus_per_job 1
```

`run_script.sh` is the local HPC wrapper; check its paths and flags before use.
Eigenvalues are saved by default. Use `--no-save_eigs` for scalar-only outputs.

## Measurement settings

| Setting | Default | Meaning |
|---|---|---|
| `--save_eigs` | on | Store full computed spectra in HDF5, including zeros and filtered values. |
| `--load_dtype` | `auto` | Checkpoint/framework-selected precision; not a guarantee of mixed-dtype preservation. |
| `--compute_dtype` | `float32` | SVD/Gram precision; use `float64` for reference checks. |
| `--use_svd` | on | CUDA uses precision-focused `gesvd`; `--no-use_svd` selects Gram eigenvalues. |
| `--filter_zeros` | on | Retain eigenvalues strictly above `--evals_thresh` for fitting and retained metrics. |
| `--evals_thresh` | `1e-5` | Absolute threshold; rescaling weights can change which eigenvalues survive. |
| `--fix_fingers` | `xmin_mid` | Cutoff selection: `xmin_mid`, `xmin_peak`, or full-KS minimization with `DKS`. |
| `--parallel_esd` | on | Parallel layer analysis; disable with `--no-parallel_esd`. |

Upcasting cannot restore precision lost while loading. Gram construction can
lose small-eigenvalue accuracy and inflate numerical rank; keep SVD as the
default. Gram jitter/fallback is not recorded per layer.

## Outputs and resume

Each successful model writes `stats/*.csv` and matching `metrics/*.h5`:

- `/layers/longname` is the canonical identity; every scalar metric is aligned
  with it. `/layers/module_name` and `/layers/slice` identify original modules
  and Q/K/V slices. Arbitrary names and missing fits remain present.
- `/eigs[i]` is the full spectrum for `/layers/longname[i]`, unless saving was
  disabled. Float32/float64 spectra retain their computed dtype. CSV contains
  scalars, not duplicated eigenvalue strings.
- `/coverage` records candidate modules, analyzed/skipped status and reasons.
  Candidate counts differ from measurement and finite-fit counts. A completed
  model can have partial coverage; inspect it before cross-model comparisons.
- Root `/alpha` is a derived depth-by-module view, **not** canonical storage.
  Its `view_status` is `complete`, `partial`, or `unavailable`; omitted names
  appear in `/alpha_unmapped_longname`. Large/unrecognized layouts do not block
  canonical records.
- `measurement_config_json` records requested settings and runtime provenance:
  actual model class, loading checks, installed backend versions and config
  commit. Per-layer fields record source/compute dtype, compute device, weight
  layout and fit status. Visible GPU names alone do not identify where a layer ran.

```python
import h5py

with h5py.File("metrics/org--model.h5", "r") as h5:
    names = h5["layers/longname"].asstr()[:]
    alphas = h5["layers/alpha"][:]
    first_spectrum = h5["eigs"][0]  # Unless --no-save_eigs was used.
```

Current output versions are numerics **5**, loader **2**, HDF5 format **2.0**.
Resume requires a compatible CSV/HDF5 pair: versions, canonical identities,
aligned alpha values and requested measurement settings must match. Changing
spectrum storage, precision, filtering or model revisions requires new outputs.
Runtime hardware differences are provenance, not a CPU/GPU equivalence claim.

Pin HF commit SHAs, including adapter bases. Offline resume does not re-resolve
moving `main` branches. Incompatible/incomplete artifacts stop the run without
deletion. Prefer a fresh directory; explicit `--overwrite` deletes the selected
models' previous outputs before loading. `summary.csv` alone is not completion.

## Reading a run

Build or refresh the existing per-model summary from the run root:

```bash
python esd_experiment/analyze_results.py \
  --results_dir analysis_runs/phase2/my_run
```

This writes `summary.csv`: one row per CSV/HDF5 pair, with artifact paths,
measurement settings, fitted/missing measurement counts, module coverage and
scalar summaries. It validates current artifacts and reads canonical `/layers`
one model at a time; it does not load eigenvalues or use the derived `/alpha`
view. Q/K/V slices count as separate measurements, not separate modules or depth.
Alpha and other fit-derived summaries include only finite fitted `alpha > 1`;
other metrics use their finite values across all measured rows.

Incomplete or incompatible pairs remain as `artifact_status=invalid` rows with
an `artifact_error`, and the command exits nonzero. Missing coverage is
`coverage_status=unknown`, never assumed complete. Mixed measurement settings
produce a warning and suppress pooled metric statistics; filter by settings
before comparing rows. Requested revisions and `model_config_commit_hash` are
separate fields; the latter is not proof that every checkpoint dependency was
pinned. Recorded adapter-base repository, revision and resolved commit are
included separately from the requested `source_model`. Full runtime provenance
remains in HDF5.

The summary is an artifact index, not a ledger of every attempted model: failures
that produced no pair remain in the runner's logs. The depth-only clustering
dashboard accepts complete `/alpha` views and warns when skipping partial or
unavailable views; arbitrary model structures remain accessible through `/layers`.

## Loading and coverage

Ordinary Transformers loads preserve the checkpoint's compatible built-in
architecture. Metadata loader scenarios are routing hints, not model identity.
Missing/mismatched weights, loading errors, unexplained extra keys and ambiguous
architecture declarations fail before analysis. Only the exact historical GPT2
`masked_bias` buffers are permitted as extra keys, and remain recorded.

Ordinary/RSLoRA matrix adapters are checked for exact configured keys, shapes,
finite values and correct loading after recorded dtype conversion; merging uses
`safe_merge=True`. This includes HF Conv1D, configured saved heads and
`bias="all"`. Adapter/base revisions remain separate, and provenance records
requested/loaded base references plus adapter configuration/tensor hashes.
An unspecified historical training-base revision cannot be reconstructed from
these checks.

DoRA, other PEFT types, `bias="lora_only"`, embedding/convolution LoRA,
unconfigured embedding dumps and unvalidated topology/initializer variants fail
explicitly. Adapter validation covers materialized CPU/GPU bases, not Accelerate
CPU/disk offloading or meta tensors. Existing quantized-to-dense upstream
substitution is recorded and must not be interpreted as quantized measurement
equivalence. Custom repositories may execute remote code; inspect them before
allowing execution.

Supported dense weights include Linear subclasses, embeddings, Conv1d/2d/3d
and HF Conv1D projections. Packed/custom representations are skipped with
reasons, including packed recurrent weights. There is no automatic quantized
reconstruction. The existing Linear aspect-ratio skip and name/shape-based QKV
split remain measurement conventions.

## Interpreting the metrics

- Unprefixed norm/rank/entropy fields describe the retained spectrum. `raw_*`
  fields describe the full computed spectrum. `norm` is the sum of eigenvalues
  (squared Frobenius norm for a 2D weight matrix).
- `fit_xmin` and `n_tail` describe the fitted positive tail, including all ties
  at the cutoff. `xmin`/`xmax` are retained-spectrum extrema, not fitted limits.
  `D` checks both sides of empirical-CDF jumps using stable float64 log-space
  calculations. Exhaustive cutoff search uses batches of 64: bounded
  `O(64 * spectrum_size)` workspace, but still quadratic total work.
- Numerical rank uses `max(rows, columns)` in the singular-value tolerance.
  Entropy normalizes the largest rank-many eigenvalues and divides Shannon
  entropy by `log(rank)`: evenness among numerically active directions. Rank-one
  entropy is zero; zero-rank entropy is undefined. Tiny changes across a rank
  threshold can produce large entropy jumps, including in `raw_entropy`.
- Undefined fits and logarithms use NaN, never numeric failure sentinels.
  `fit_status` distinguishes fitted, empty/insufficient, constant and invalid
  tails. Rank-one layers and layers with no retained eigenvalues remain stored.
  A model can complete with no finite fits if useful spectra were measured.
- Convolution spectra pool spatial kernel slices, each scaled by
  `sqrt(conv_norm)` (`conv_norm=0.5`). They are **not** the spectrum of the full
  convolution operator; this also applies to `raw_*` metrics. Match pooling,
  scaling, filtering and dtype before comparing with WeightWatcher.

A finite alpha or low fitted KS distance does not establish power-law behavior,
generalization, or an architecture-independent quality score. Aspect-ratio
correction and broad predictive validation remain outside this pipeline.

Definitions: [Clauset–Shalizi–Newman](https://arxiv.org/abs/0706.1062),
[SciPy KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html),
[LoRA](https://arxiv.org/abs/2106.09685),
[PEFT checkpoint format](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint).

## Scheduling and monitoring

`run_experiment.py → gputracker → worker.py → model_loader.py → net_esd`

The runner writes `<output_dir>/gpu_config.json`. Edit it and send
`kill -HUP <runner_pid>` to reload GPU/memory/concurrency and supervision limits.
Workers see only assigned GPUs: a single physical GPU becomes local `cuda:0`.
Each worker has its own process group for scoped termination.

- `heartbeat_timeout_seconds` detects stopped heartbeats.
- `stage_timeout_seconds` limits load/analyze/save stages even while heartbeats
  continue. `termination_grace_seconds` controls shutdown grace.
- Keep `stale_process_action="log"` while tuning; choose `terminate` only after
  timeout windows are trusted.
- `logs/current_state.json` records PID, PGID, assigned GPUs, stage and paths.
  `logs/failure_records.jsonl`, `logs/failed_models.txt` and
  `logs/terminal_status/*.json` retain terminal results. Empty-analysis coverage
  is preserved in `logs/coverage/`.

Active worker logs, heartbeats and per-worker HF caches are removed on completion,
failure or termination. Copy any needed diagnostic logs before cleanup.

## Validation

Offline regression tests cover numerical edge cases, layer identity, coverage,
checkpoint/adapter integrity and persistence:

```bash
python -m pytest esd_experiment/tests -q \
  --ignore=esd_experiment/tests/test_gpu.py \
  --ignore=esd_experiment/tests/test_setup.py
```

The reusable matrix check needs no downloads. It compares against SciPy float64
and records precision/filter/cutoff sensitivity, with a timeout per case:

```bash
python esd_experiment/scripts/backend_pilot.py \
  --device cpu --output-dir analysis_runs/validation/cpu_check
# For GPU: restrict CUDA_VISIBLE_DEVICES and select --device cuda:0.
```

Prior small public-checkpoint CPU/GPU checks exercised dense loading, LoRA
merging and FP4 partial coverage; they do not validate heterogeneous model
quality or full quantized support. GPTQ still needs a healthy checkpoint and a
consistent backend environment. One-off reports/checkpoint caches are disposable;
keep production data under `analysis_runs/phase2/`.
