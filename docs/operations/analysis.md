# Analysis

Run phase 2 analysis from `data/curated/model_zoo_phase2.csv` and write outputs under `analysis_runs/phase2/<run_name>/`.

## Quick Spin-Up

If you just want to run phase 2, this is the default pattern:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/numerics_v4 \
  --gpus 0 1 2 3 --save_eigs
```

Use a fresh run directory. `--save_eigs` is recommended for dataset collection;
without it, only scalar layer measurements are stored.

The runner reads `loader_scenario` first, but it also consumes optional curated fields such as `files`, `repo_files`, `pipeline_tag`, `Architecture`, `model_type`, and `Available on the hub` when they are present. Quantized-native rows are only blocked early when they resolve to an explicit `gptq` or `awq` backend requirement.

## Scheduling And Supervision

The phase-2 infra is meant to be reusable:

```
run_experiment.py -> gputracker -> worker.py -> model_loader.py -> net_esd
```

The runner writes `<output_dir>/gpu_config.json`. Edit it and send `SIGHUP` to the runner PID to reload scheduling policy:

```json
{
  "available_gpus": [0, 1, 2, 3],
  "max_checks": 1,
  "memory_threshold_mb": 500,
  "max_concurrent_jobs": 2,
  "stale_process_action": "log",
  "heartbeat_timeout_seconds": 7200,
  "stage_timeout_seconds": {
    "load": 7200,
    "analyze": 28800,
    "save": 1800,
    "default": 14400
  },
  "termination_grace_seconds": 30
}
```

- `heartbeat_timeout_seconds` catches workers that stop writing heartbeats.
- `stage_timeout_seconds` catches workers that are alive but stuck in one foreground stage.
- `stale_process_action` should be `log` while tuning and `terminate` once the windows are trusted.

Live state is in `logs/current_state.json`. Per-worker active logs, heartbeat files, and worker caches are removed when the worker finishes, fails, or is killed; terminal status and failure summaries remain.

## What To Check After The Run

- successful models should have both:
  - `stats/*.csv`
  - `metrics/*.h5`
- failures should appear in:
  - `logs/failed_models.txt`
  - `logs/failure_records.jsonl`
- `summary.csv` is useful for quick inspection, but it is not the success rule

## Completion Rule

A model is complete only when its CSV/HDF5 pair has the current schema and
numerics version, aligned canonical layer names and alpha values, and matching
requested measurement settings. Resume checks loading/computation precision,
filtering, fitting settings, spectrum storage, and requested model revision.
Recorded runtime details may differ across machines; they are provenance, not
a claim of CPU/GPU equivalence. Pin HF commit SHAs: a moving `main` branch is
not refreshed or re-resolved by an offline resume check.

Incompatible, unreadable, or incomplete existing artifacts stop the run with an
explanation. They are not deleted automatically. Use a fresh output directory,
or explicitly pass `--overwrite` to regenerate the selected models. Overwrite
removes those models' previous outputs before loading; it is not a migration.

## Numerical Results

New HDF5 outputs carry `numerics_version="4"` and `format_version="2.0"`.
Do not combine versions in a research run. Local `run_script.sh` changes are
left untouched; check its output directory and flags before using it.

### Layer identity and storage

- `/layers/longname` is the canonical measurement identity. All scalar metrics,
  including `/layers/alpha`, are aligned with it. `/layers/module_name` and
  `/layers/slice` distinguish original modules from emitted Q/K/V slices.
  Arbitrary names and missing fits are preserved; duplicate identities or
  unequal metric lengths are errors, not silently averaged/truncated data.
- Root `/eigs[i]`, when requested, is the **full computed spectrum** for
  `/layers/longname[i]`, including zeros and values below `evals_thresh`.
  Float32 spectra are stored as float32; float64 reference spectra remain
  float64. Eigenvalue arrays are not duplicated as strings in CSV.
- Root `/alpha` is only a derived depth-by-module compatibility view. Encoder
  and decoder namespaces are separated. Its `view_status` is `complete`,
  `partial`, or `unavailable`; `/alpha_unmapped_longname` lists omitted names.
  Unrecognized depth layouts and dense views exceeding one million cells do
  not prevent canonical records from being saved. New consumers should read
  `/layers`, not infer layer identities from this matrix.
- `/coverage` contains module names, layouts, analyzed/skipped status, and skip
  reasons. Counts distinguish candidate modules from emitted measurements and
  finite fits. `logs/coverage/<model>.json` also preserves this report and its
  measurement configuration when all candidates are skipped.

For example:

```python
import h5py

with h5py.File("metrics/org--model.h5", "r") as h5:
    names = h5["layers/longname"].asstr()[:]
    alphas = h5["layers/alpha"][:]
    first_spectrum = h5["eigs"][0]  # Only present with --save_eigs.
```

### Precision, filtering, and interpretation

- Loading defaults to `--load_dtype auto` (checkpoint/framework-selected),
  replacing forced float16. This does not guarantee preservation of mixed
  checkpoint dtypes. Other explicit choices are `float32`, `float16`, and
  `bfloat16`. `--compute_dtype float32` is the default SVD/Gram precision;
  `float64` is available for reference checks. Upcasting cannot restore
  information already lost during loading.
- `--use_svd` is now the consistent runner/worker default. `--no-use_svd`
  explicitly selects Gram eigenvalues; ill-conditioned spectra can lose
  precision through the Gram construction. `--no-parallel_esd` disables
  multi-device dispatch. The runner forwards both positive and negative flags.
- `--filter_zeros` retains values strictly above `evals_thresh` for the existing
  norm/rank/entropy and fitting metrics. `--no-filter_zeros` disables that
  absolute filter; fitting still requires positive values. Saved spectra are
  unfiltered in either case. `raw_norm`, `raw_spectral_norm`, `raw_matrix_rank`,
  `raw_entropy`, and `raw_num_evals` describe the full computed spectrum;
  existing unprefixed fields describe the retained spectrum. `norm` is
  `sum(eigenvalues)`, the squared Frobenius norm for a 2D weight matrix.
- HDF5 `measurement_config_json` records the requested conventions and runtime
  provenance. Per-layer fields include `source_dtype` (loaded tensor dtype),
  `compute_dtype`, `compute_device`, `weight_layout`, and `fit_status`.
  `runtime.model_config_commit_hash` identifies the loaded model config; for a
  merged adapter this may be the **base** config, not the adapter's commit.

- `D` is the full two-sided Kolmogorov–Smirnov distance, checking both sides of
  each empirical-CDF jump. MLE/CDF calculations use stable float64 log ratios
  and `expm1`. DKS minimizes this distance over candidate cutoffs;
  `xmin_mid` and `xmin_peak` retain their cutoff-selection rules. Exhaustive
  search batches at most 64 cutoffs at a time: workspace is bounded by
  `O(64 * spectrum_size)`, while exhaustive computation remains quadratic.
- Per-layer CSV fields `fit_xmin` and `n_tail` record the selected fitting cutoff
  and the number of positive retained eigenvalues at or above it. Equal values
  at the cutoff are all included. With no fit, `fit_xmin` is NaN and `n_tail` is
  zero. Existing `xmin`/`xmax` still describe the retained spectrum, not fit limits.
- Numerical rank uses `max(rows, columns)` in the singular-value tolerance.
  Entropy uses normalized probabilities over the largest numerically retained
  eigenvalues, divided by `log(matrix_rank)`. Rank-one entropy is zero;
  zero-rank entropy is undefined. This measures evenness among numerically
  active directions and can jump when numerical rank changes despite negligible
  energy change. `raw_entropy` uses the same rank-normalized definition without
  the absolute eigenvalue filter; it does not remove this discontinuity.
- Constant spectra and other undefined power-law fits have missing `alpha`, `D`,
  `alpha_weighted`, and `log_alpha_norm` values in CSV. The HDF5 alpha matrix uses
  NaN for missing fits. `fit_status` distinguishes `fitted`,
  `no_retained_eigenvalues`, `insufficient_positive_eigenvalues`,
  `constant_spectrum`, and `no_valid_cutoff`. Missing fits never remove
  canonical layer records.
- Rank-one layers remain in the output. With filtering enabled, values at or
  below `evals_thresh` stay excluded even when none survive. Such a layer has
  `num_evals=0`, zero retained norm and rank, and missing retained spectrum
  extrema. Its full spectrum and raw measurements remain available.
- Zero retained norms have NaN `log_norm`, `log_spectral_norm`, and `stable_rank`;
  these are undefined measurements, not ordinary numeric zeros.
- The worker reports the number of finite fits (`alpha > 1`). Models with useful
  spectral measurements can complete even when every fit is missing;
  `analysis_empty` means no usable layer measurements were returned. A finite
  exponent or a low `D` alone does not establish power-law behavior or model quality.
- `log_alpha_norm` is computed in log space to avoid overflow.
- Standard Conv1d/2d/3d, Linear subclasses, embeddings, and declared HF Conv1D
  projections are supported. Unknown/packed quantized representations are
  skipped with reasons, not treated as ordinary dense matrices. The existing
  Linear aspect-ratio skip and name/shape-based QKV split remain explicit
  conventions; arbitrary custom architectures may need declared adapters.
- Convolution normalization is unchanged: spatial-kernel slice spectra are
  pooled after multiplying each slice by `sqrt(conv_norm)` (default `0.5`).
  These are descriptors of normalized kernel slices, **not** the complete
  convolution operator; even `raw_*` refers to these slices. Match pooling,
  scaling, filtering, dtype, and fit conventions when comparing WeightWatcher.

### Small public-checkpoint pilot

The [v4 pilot summary](pilot_v4.md) records the initial results and limitations.

The reproducible CPU pilot downloads four pinned public checkpoints (~7.5 MB
total) covering encoder, decoder, encoder–decoder, and CNN measurements:

```bash
python esd_experiment/scripts/measurement_pilot.py \
  --output-dir analysis_runs/validation/my_pilot \
  --weightwatcher-path ../WeightWatcher
```

Use a fresh output directory. The script restricts downloads to config/weights
and disables remote code, uses explicit built-in model classes, verifies full
CSV/HDF5 alignment, and compares representative layers against local
WeightWatcher's accurate SVD on explicit float64 arrays. It also checks seeded
Pareto, lognormal, Gaussian, near-constant and low-rank controls. It does not
compare against WeightWatcher's legacy KS or entropy implementations.

This is a numerical/coverage/storage smoke test, not predictive validation,
automatic loader-routing validation, GPU equivalence, or a quantized-model
pilot. Most checkpoints are tiny random test models. Before scaling to 50k,
extend the pilot to representative real model sizes, actual GPU/backend
settings, quantized/custom implementations, threshold sensitivity, and related
checkpoint groups. No aspect-ratio correction or universal quality-score claim
is introduced here.

Definitions: [Clauset–Shalizi–Newman, §§3–4](https://arxiv.org/html/0706.1062v2),
[SciPy KS statistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html),
[numerical-rank tolerance](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html),
and [WeightWatcher normalization](https://github.com/CalculatedContent/WeightWatcher/blob/master/weightwatcher/weightwatcher.py).

## From `esd_experiment/`

From `esd_experiment/`, use:

```bash
python run_experiment.py \
  --model_list ../data/curated/model_zoo_phase2.csv \
  --output_dir ../analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```
