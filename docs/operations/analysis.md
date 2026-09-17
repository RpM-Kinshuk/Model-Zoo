# ESD analysis

From the repository root, first prepare a small, pinned model list in a fresh
output directory. This contacts HF for metadata and adapter JSON, not weights:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/my_run --limit 20 --prepare_only
```

Review `analysis_runs/phase2/my_run/models.csv`, then launch exactly that list:

```bash
python esd_experiment/run_experiment.py \
  --model_list analysis_runs/phase2/my_run/models.csv \
  --output_dir analysis_runs/phase2/my_run \
  --gpus 5 6 7 --num_gpus_per_job 1
```

`run_script.sh` is the local HPC wrapper; check its paths and flags before use.
Eigenvalues are saved by default. Use `--no-save_eigs` for scalar-only outputs.

## Pinned inputs and remote code

Preparation fills `revision_norm` with a full commit SHA and pins each
`source_model` as `repo@SHA`. It uses `revision_norm` first, then a revision in
`model_id`, then a full `Modelsha` from metadata, otherwise resolves `main`.
Adapter bases come from an explicit `source_model` or the pinned adapter config.
Repeated bases share one resolution within preparation. `revision_requested`
and `source_model_requested` preserve the references used to obtain the pins.
Resolving an unspecified base revision today does not recover its historical
training revision.

`pin_status=error` rows keep their `pin_error`; preparation exits nonzero if any
fail. Fix or explicitly remove those rows before launching. Duplicate output
identities are rejected. An existing `models.csv` is never replaced, even with
`--overwrite`: keep it for resume, or prepare another directory for new pins.
You can also supply a manually pinned CSV. Runner and standalone worker require
full model/base SHAs before starting; they do not resolve moving branches during
launch or resume.
Transformers' implicit adapter-config probes use the same pinned revision as
the requested model, including when loading an adapter's base.

Remote Python code is off by default, including metadata probes. Only add
`--trust_remote_code` when launching a list whose code you have reviewed; the
choice is recorded in the measurement configuration and checked by resume.
Same-repository custom code uses the model's pinned revision. Cross-repository
`auto_map` references are rejected because that SHA does not pin the other repo.
This is not a sandbox: explicitly trusted code can still execute arbitrary
Python and fetch its own dependencies. See HF's [custom-code guidance](https://huggingface.co/docs/transformers/models#custom-models)
and [revision metadata API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.model_info).

## Measurement settings

| Setting | Default | Meaning |
|---|---|---|
| `--analysis_source` | `auto` | Model loading first, with narrow tensor fallback. `model` disables fallback; `checkpoint` directly measures stored matrices. |
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
  with it. `/layers/module_name` identifies the original module and
  `/layers/weight_attribute` its stored weight. `/layers/slice` is empty for
  whole-weight measurements. Arbitrary names and missing fits remain present.
- `/eigs[i]` is the full spectrum for `/layers/longname[i]`, unless saving was
  disabled. Float32/float64 spectra retain their computed dtype. CSV contains
  scalars, not duplicated eigenvalue strings.
- `/coverage` records analyzed/skipped status and reasons. Its `modules` table
  has one entry per candidate weight (or an unsupported attribute group), so
  `module_name` can repeat. Counts group these entries by physical module; a
  module with both measured and skipped weights is partially analyzed.
  Module, measurement and finite-fit counts differ. A completed model can have
  partial coverage; inspect it before cross-model comparisons.
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

Current output versions are numerics **7**, loader **8**, HDF5 format **2.0**.
Resume requires a compatible CSV/HDF5 pair: versions, canonical identities,
aligned alpha values and requested measurement settings must match. Changing
spectrum storage, precision, filtering or model revisions requires new outputs.
Runtime hardware differences are provenance, not a CPU/GPU equivalence claim.

Numerics 7 adds declared MultiheadAttention projection matrices, measured whole,
and records their weight attributes. Loader 8 adds narrowly triggered automatic
fallback and separates requested policy from actual analysis source. Older outputs
need a fresh run. Incompatible/incomplete
artifacts stop the run without deletion. Prefer a fresh directory; explicit
`--overwrite` deletes the selected
models' previous outputs before loading. `summary.csv` alone is not completion.

## Reading a run

Build or refresh the existing per-model summary from the run root:

```bash
python esd_experiment/analyze_results.py \
  --results_dir analysis_runs/phase2/my_run
```

This rebuilds `summary.csv` from `models.csv`, artifact pairs and
`logs/terminal_status/*.json`. Use `--model_list path/to/pinned.csv` if your
selected list lives elsewhere. Every selected model gets a row, even without
outputs; extra stored models remain visible with `selected=False`. Original
metadata is preserved as `input_*` columns for family/coverage queries.

| `outcome` | Evidence |
|---|---|
| `success` | Valid CSV/HDF5 pair, matching the selected model/base pins when a list is supplied. Overrides stale failure records. |
| `failed` | Last terminal record reports a worker failure. |
| `blocked` | Recorded preflight block, or a preparation `pin_status=error` row. |
| `incomplete` | Invalid/incompatible pair, or a success record without artifacts. |
| `unrecorded` | No usable outcome for this selection; may be queued, running or never launched. |

`outcome_reason`, `outcome_stage`, `outcome_message` and artifact/terminal paths
explain the row. `outcome_pin_status=unknown` means the terminal record does not
establish the selected checkpoint/base revisions; existing worker failure records
have this limitation. A known terminal-pin mismatch stays `unrecorded`, not a
failure of the selected checkpoint. New preflight blocks record both pins in the
existing terminal directory; the reader never reruns environment-dependent checks.
This is a snapshot of persisted evidence, not a live monitor or attempt history.

Valid pairs retain measurement settings, fitted/missing measurement counts,
module coverage, loaded-tensor usage counts and scalar summaries. The reader
validates artifacts and reads canonical `/layers` one model at a time; it does
not load eigenvalues or use the derived `/alpha` view. Measurement counts are
not model depth.
Alpha and other fit-derived summaries include only finite fitted `alpha > 1`;
other metrics use their finite values across all measured rows.

Incomplete or incompatible pairs remain as `artifact_status=invalid` rows with
an `artifact_error`. Incomplete outputs or malformed terminal records make the
command exit nonzero; ordinary recorded failures/blocks do not mean indexing
failed. Missing coverage is
`coverage_status=unknown`, never assumed complete. Mixed measurement settings
produce a warning and suppress pooled metric statistics; filter by settings
before comparing rows. Requested revisions and `model_config_commit_hash` are
separate fields; the latter is not proof that every checkpoint dependency was
pinned. Recorded adapter-base repository, revision and resolved commit are
included separately from the requested `source_model`. Full runtime provenance
remains in HDF5.

`weight_usage_status=recorded` means a loaded-tensor inventory exists, not that
coverage is complete. Inspect `unresolved_tensors`, `skipped_tensors` and
`unmapped_measurements`; full names and reasons are in `/coverage` under
`weight_usage`. Older otherwise-compatible outputs can still be read/resumed,
but their absent inventory is `unknown`, not zero or complete. This additive
report does not change loading, tensor selection or numerical conventions.

Filter `selected` when computing coverage of your input list, and
`outcome == "success"` before comparing measurements. The summary is replaceable;
it does not affect resume or alter measurements/terminal records. Historical
text-only failure logs are not reconstructed into checkpoint-specific outcomes.
The depth-only clustering dashboard accepts complete `/alpha` views and warns
when skipping partial or unavailable views; arbitrary model structures remain
accessible through `/layers`.

## Loading and coverage

Ordinary Transformers loads preserve the checkpoint's compatible built-in
architecture. Metadata loader scenarios are routing hints, not model identity.
Missing/mismatched weights, loading errors, unexplained extra keys and ambiguous
architecture declarations fail before analysis. Only the exact historical GPT2
`masked_bias` buffers are permitted as extra keys, and remain recorded.

Ordinary BERT, RoBERTa, DistilBERT, ALBERT and DeBERTa (v1/v2) checkpoints without
`architectures` can be selected automatically from their complete safetensors
key/shape layout, including shards and omitted tied
aliases. Candidate built-in classes are constructed on the meta device, without
allocating their weights. The decoder config distinguishes masked from causal LM;
multiple matches (for example some one-output classification/multiple-choice
heads) fail explicitly. Declared architectures are never silently replaced.
The inspection uses the pinned HF download/cache path and loads only headers
into RAM; uncached weight files are still downloaded once in the load stage.
Selection evidence and actual input/output embedding tying are recorded in loading
provenance. Other families, legacy-only files, custom and quantized missing-metadata
cases are not covered by this new inference path. No architecture override exists yet.
Families without a supported causal-LM class are not inferred as decoders. Task
constraints are checked before trying a candidate structure, so a span-QA class
requiring two outputs cannot block a valid multiclass classifier. ALBERT's shared
modules are measured as stored, not repeated to manufacture logical depth.

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
CPU/disk offloading or meta tensors. Quantized adapter bases are never silently
replaced with a dense upstream model. If a dense base is scientifically intended,
select it explicitly in `source_model` before preparation; this is not quantized
measurement equivalence.

Supported dense weights include Linear subclasses, embeddings, Conv1d/2d/3d
and HF Conv1D projections. Packed/custom representations are skipped with
reasons, including packed recurrent weights. There is no automatic quantized
reconstruction. Each eligible matrix is measured whole, including fused QKV
projections. Names and a 3:1 shape do not establish the packing layout: separate
grouped-query key/value projections can have that shape too. The existing
Linear aspect-ratio skip remains a measurement convention.

The built-in `torch.nn.MultiheadAttention` is supported by default: its fused
`in_proj_weight` is one whole matrix, or its separate `q_proj_weight`,
`k_proj_weight` and `v_proj_weight` are measured individually when key/value
dimensions differ. `out_proj` remains its own Linear module. For example,
`attention.k_proj_weight` names a measurement whose module is `attention` and
weight attribute is `k_proj_weight`; no synthetic QKV split is made.
Shapes must agree with the declared dimensions, and an unsupported projection
does not hide valid siblings. Quantizable/custom MHA subclasses are not granted
this layout automatically: they may retain unused base-class parameters while
performing projections elsewhere. Optional `bias_k`/`bias_v` vectors are not
projection matrices; they remain visible as unresolved in loaded-tensor usage.
The layout follows [PyTorch's implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/activation.py).

For an explicit broader pass over a successfully loaded model, add
`--no-filter_type` to the runner or worker command in a fresh output directory.
This exposes the estimator's existing dense-matrix fallback: other ordinary
floating, dense 2D `.weight` tensors become eligible, and the legacy Linear
aspect-ratio skip is disabled. Standard convolution layouts remain supported;
unknown higher-dimensional layouts, packed/quantized representations and
undeclared nonstandard attributes are still skipped. This is not
architecture recovery or permission to use a partially loaded checkpoint.
Inspect coverage and weight-usage records for remaining gaps. `filter_type` is
stored in HDF5 and the summary, and must match on resume; the default remains
`True`, so existing compatible runs continue with their original selection.

Coverage also inventories loaded registered parameters and buffers, including
non-persistent buffers, without reading/copying their values. Each tensor has its
full name, shared aliases, shape/dtype and measurement links or a reason it was
not measured. Shared aliases mean the same Tensor object, not equal values or
overlapping storage; existing per-module measurements are not deduplicated.
Scalar/vector parameters and buffers not selected as weights are distinguished
from skipped weights and unresolved parameters. For example, an extra matrix
parameter beside a module's `.weight` is unresolved if the selector overlooked
it. Computed/unregistered weights can leave measurements explicitly unmapped.
Callable/packed non-tensor helpers remain in module coverage; this inventory
does not unpack them or prove that all checkpoint weights survived loading.
The loader's integrity gate remains separate and required. See
[PyTorch's named tensor traversal](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html#torch.nn.Module.named_parameters)
for alias enumeration with `remove_duplicate=False`.

### Architecture-independent checkpoint matrices

The default `--analysis_source auto` first attempts strict model loading. It can
fall back once to checkpoint matrices for three identified cases:

- A valid config declares a `model_type` that the installed library does not support.
- Resolving the config requires repository code that is disabled.
- Multiple supported encoder architectures fully match the checkpoint keys/shapes.

The last case differs from **no matching layout**, which may mean missing or
incorrect weights and remains a failure. Conflicting architecture declarations,
missing dependencies, integrity failures, quantization/adapter failures, OOM,
network errors, timeouts and numerical/save errors do not trigger fallback.
No skipped layers are filled in from raw tensors after a successful model load.
Generic exception messages are never enough to trigger this route.

Use `--analysis_source model` for strict model-only runs, or
`--analysis_source checkpoint` to measure stored matrices directly. Explicit
checkpoint analysis may describe tensors from a repository whose model cannot
be validated; it still does not establish model completeness. Neither route
uses a partially loaded model. Use a fresh output directory when policy changes.

`analysis_policy` records the request (`auto`, `model` or `checkpoint`). Runtime
provenance records actual `analysis_source`, effective `filter_type`, and the
original fallback stage/reason/message. The summary exposes `analysis_policy`,
`analysis_source`, `effective_filter_type` and `fallback_reason`; mixed actual
sources are not pooled even when both requests were `auto`. Resume validates this
record and does not try loading again to promote an existing tensor-only result.

Checkpoint mode reads pinned `model.safetensors` or `model.safetensors.index.json`
and validates all shard keys/shapes. It needs no recognized architecture, model
construction or repository Python. Missing/unknown architecture metadata is fine;
malformed files still fail. There is no pickle or arbitrary-file fallback.
Known adapter/quantization config and packed-weight markers are rejected; use
the supported model-loading path to merge or interpret those representations.

Every stored floating 2D tensor (float16, bfloat16, float32, float64) is eligible,
including direct parameters and possible buffers. Other ranks, empty tensors and
unsupported dtypes are recorded as skipped. Higher-dimensional tensors are not
reshaped or assumed to be convolution kernels. Unknown packing cannot be ruled
out from dtype alone: these are **stored-matrix descriptors**, not a claim about
effective model weights or universal model support.

The same numerical core, CSV/HDF5 writer and dispatcher are used. Input tensors
are read one at a time on one device per worker (`auto` selects the first assigned
GPU, otherwise CPU). Direct checkpoint mode requires one GPU per job. Automatic
fallback uses the first assigned device even if the model requested several;
it does not change reservations mid-job. Within-model parallelism is disabled;
worker concurrency, load/analyze timeouts, signals, resume and ephemeral cache
cleanup are unchanged. Automatic fallback stays within the same load-stage
timeout and cache; a failed fallback is not retried, even with `--max_retries`.
Auto/checkpoint workers accept `--device_map auto`, `cpu` or `cuda:<index>`; use
model-only policy for other model device maps.
Disk still holds the downloaded checkpoint until worker
cleanup; host memory retains accumulated spectra, not a constructed full model.
`load_dtype=auto` preserves each stored tensor's dtype. Actual `filter_type` is
false: no module-class or legacy Linear aspect-ratio filter applies here. An
automatic run retains its requested model filter separately in the config.

Exact checkpoint keys become `/layers/longname` and `weight_attribute`;
`module_name` is empty, and the derived depth view is unavailable. `/coverage`
has `scope=checkpoint_tensors` and one `tensors` record per stored key, with file,
shape/dtype, measurement link or skip reason. The summary exposes stored/analyzed/
skipped tensor counts; module and loaded-parameter coverage are not applicable.
Aliases omitted during saving cannot be reconstructed, and equal stored tensors
are not deduplicated. `params` is only the measured tensor's element count.
Source mode, file inventory, config hash and observed execution are recorded;
resume and summary comparison keep the two modes distinct.

Offline controls cover streaming lifetime, mixed precision, exact names, shards,
shared-key omissions, representation rejection and artifact/resume behavior.
Automatic-routing controls include unsupported/custom configs, ambiguous full
layouts, broken/missing weights, partial model coverage, failure/retry boundaries
and consistent source reporting. These tests use real tiny local checkpoints.
A sharded BERT control agrees exactly with model-mode spectra for the same
matrices. Earlier explicit-mode public CPU smoke checks used `hf-internal-testing/tiny-random-bert`
at `f171d7baecaf37b5da5a3616d8833b9969753535` (39 matrices, 100 stored tensors)
and `hf-internal-testing/tiny-random-gpt2` at
`71034c5d8bde858ff824298bdedc65515b97d2b9` (22 matrices, 64 stored tensors).
Every saved spectrum matched an independent CPU float64 SVD calculation;
summary reading and resume passed. Temporary outputs/caches were removed.
These random test checkpoints establish neither trained-model validity nor
GPU equivalence; those remain pilot questions.

This uses safetensors' [per-tensor reading API](https://huggingface.co/docs/safetensors/index);
see also its [shared-tensor limitations](https://huggingface.co/docs/safetensors/torch_shared_tensors).

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
Initial configuration write/load failures abort before dispatch. GPU selection
requires repeated low-memory readings **and no visible GPU processes**; a small
existing job is not an idle device.
`SIGUSR1` stops dispatching new models and lets current workers finish (drain).
`SIGINT`/`SIGTERM` stop the run and request termination of active worker groups.
The runner attempts supervisor cleanup before exiting nonzero. Workers get
`termination_grace_seconds` before remaining group members receive SIGKILL;
repeating SIGINT/SIGTERM does not bypass cleanup. Allow the current supervisor
poll (up to five seconds), grace period and cache deletion to finish. SIGKILL
of the runner itself cannot run cleanup and should be a last resort.
Resume with the same pinned input, output directory and measurement settings:
compatible completed models are skipped; interrupted models restart. There is
no within-model/layer checkpointing.
Workers see only assigned GPUs: a single physical GPU becomes local `cuda:0`.
Each worker has its own process group for scoped termination.

Status, supervision or cache-cleanup errors stop new launches, request shutdown
of the remaining workers and make the run exit nonzero. Finalization steps are
attempted independently: a status-write error must not skip cache cleanup, and
neither error strands the internal GPU reservation after confirmed termination.
If termination cannot be confirmed, caches and worker records are retained;
if status recording fails, diagnostic worker files are retained.

The main wait also checks for stuck supervisor I/O. An operation stalled longer
than `max(30, termination_grace_seconds)` requests shutdown. Shutdown has a shared
deadline of `grace + max(30, grace) + 10` seconds (130 seconds at the default).
If control threads remain blocked, the runner attempts SIGKILL on owned groups
and exits without waiting for file cleanup. This bounds the runtime supervisor
wait, not startup I/O or kernel-stuck processes. Inspect retained PIDs/logs/cache
before resuming after an infrastructure failure; unconfirmed cleanup is never
reported as successful.

- `heartbeat_timeout_seconds` detects stopped heartbeats.
- `stage_timeout_seconds` limits load/analyze/save stages even while heartbeats
  continue. `termination_grace_seconds` controls shutdown grace.
- Keep `stale_process_action="log"` while tuning; choose `terminate` only after
  timeout windows are trusted.
- `logs/current_state.json` records PID, PGID, assigned GPUs, stage and paths.
  `logs/failure_records.jsonl`, `logs/failed_models.txt` and
  `logs/terminal_status/*.json` retain terminal results, including preflight
  blocks. Empty-analysis coverage is preserved in `logs/coverage/`.

Runner-managed worker logs, heartbeats and per-worker HF caches are removed on
ordinary completion, failure or confirmed termination, with the error-path
exceptions above. Copy any needed diagnostic logs before cleanup.
Cache removal is deliberate: checkpoint storage previously grew without bound,
even over a few models. Re-downloading can be an acceptable tradeoff on this
shared HPC server. GPU availability checks, runtime signals, scoped shutdown
and resume from completed outputs are also intentional operating safeguards.

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

### Bounded pilot, 2026-09-14

`analysis_runs/validation/trained_pilot_20260914/` holds the pinned `models.csv`,
`pilot_report.json`, ordinary CSV/HDF5 outputs and reference checks. This was a
coverage/operability check: 18 trained checkpoints plus two adapter/quantization
controls, not a population sample. The current 15k curated input is heavily
LLM/adapter-oriented; this pilot deliberately also includes encoders and vision.

- **16 valid pairs:** 1,296 measurements, 757,362 full-spectrum eigenvalues,
  6.22 MiB of CSV/HDF5. One missing fit remains explicitly stored. All measured
  weights have registered-tensor links; deliberate skips remain (including the
  default Linear aspect-ratio filter and DeiT token/position parameters).
- **Four non-completions:** `prajjwal1/bert-tiny` lacks `model_type`;
  `albert/albert-base-v2` has pooler weights omitted by its declared loader;
  `microsoft/deberta-v3-xsmall` needs architecture inspection but has only `.bin`
  weights; `casperhansen/opt-125m-awq` is blocked by the backend preflight.
  All successful cases used model analysis; this pilot did not trigger automatic
  checkpoint fallback or establish quantized coverage.
- **Numerics:** four well-conditioned GPU controls passed the independent
  SciPy float64 reference; sensitivity controls retain rank/filter/cutoff flags.
  Six selected saved MiniLM/ResNet spectra matched pinned float32 weights with
  maximum error below 2e-7 times the reference largest eigenvalue. Convolution
  checks use the documented pooled kernel slices, not the complete operator.
- **Recovery:** drain preserved an active model's successful output. SIGTERM
  exposed a runner exit before supervisor cleanup; this was fixed and repeated
  successfully. The interrupted model then completed on resume, and prior
  CSV/HDF5 hashes stayed unchanged. Changed-precision compatibility checks
  rejected all 16 completed pairs without modifying them.
- **Cost/access:** one worker, `load_dtype=auto`, float32 SVD, CPU math threads=1,
  L40 GPUs 5 then 4 after other jobs occupied 5. The final resume took 765 seconds,
  with about 245 sampled worker seconds; the remainder includes startup,
  preflight, availability checks, dispatch and cleanup, not just avoidable idle time.
  Sampled peaks were 1.95 GiB process-tree RSS, 1.77 GiB GPU memory and 1.24 GiB
  cache. RSS can double-count shared mappings; cache bytes are not bytes transferred.
  Worker caches returned to zero after the fixed runs. The summary rebuilt in
  0.37 seconds; selected-spectrum reads had a 1.5 ms median. These small, warm
  filesystem reads and checkpoints do not establish 50k-model costs.

`auto` used float16 for Pythia/OPT and the merged adapter, bfloat16 for SmolLM2,
and float32 elsewhere. These are observed loaded dtypes, not a claim that every
stored checkpoint tensor retained its original precision. The independent
trained-weight comparisons above cover the six selected float32 weights only.

That pilot's shutdown fix passed 794 offline tests. The subsequent runtime audit
adds focused failure-path checks for status writes, cache cleanup, process
ownership and bounded shutdown. Numerical/loading policies and ephemeral caches
are unchanged. Predictive validity and broader coverage remain open.

## Roadmap: a heterogeneous model-spectra dataset

The original goal is an efficient, robust dataset/database of roughly 50,000
HF-metadata-sourced models, preserving ESD analyses and eigenvalues across many
model types. Trustworthy numbers must stay attached to the correct checkpoint
and weight under recorded conventions. Model loading serves that measurement
goal; providing task inference is not itself a requirement.

First principle: keep this human-readable, usable and maintainable from the
start, down to commands, variable names and errors. Prefer small helpers, plain
names and one clear way to do each job. The existing loader/worker, cache,
HDF5/CSV writer and summary reader are starting points, not permanent requirements.
For each change, consider keeping, simplifying, merging, replacing or removing
what exists before adding another mechanism. Redundant code, artifacts, docs and
obsolete workflows can go; branch history need not become a compatibility burden.
Check consumers and distinguish disposable outputs from source data and useful
research results before removal. Preserve the local run-script changes. No
50k-model run is authorized.

Shared-HPC constraints are part of correctness: retain bounded temporary disk
usage, respect other users' GPU jobs, and preserve runtime control and resumable
progress. Do not remove cache cleanup, availability checks, signal handling or
resume behavior merely to reduce overhead. A replacement must demonstrate the
same guarantees before the existing safeguard can be retired. Any cache reuse
proposal needs explicit disk limits, ownership and eviction/cleanup behavior;
an indefinitely growing shared cache is not an acceptable optimization.

### Already in place

Corrected numerics, full-spectrum storage, canonical identities, missing-fit and
module-coverage records, pinned model/base revisions and version-aware resume.
Checkpoint-shape selection covers the six encoder model types above. Offline
controls exercise standard heads, shards, shared weights, malformed checkpoints
and ambiguous layouts. Small public CPU checks verified preserved checkpoint
tensors and saved spectra; missing-metadata inference was tested separately with
local safetensors. These are implementation checks, not research validation.
Loaded-tensor accounting now exposes extra parameters missed by module coverage
and records shared aliases and measurement links. It reuses coverage JSON/HDF5
and the summary reader without a new storage layer.
The existing dense 2D `.weight` fallback is now accessible from the runner and
worker with `--no-filter_type`, under recorded settings and unchanged defaults.
Declared PyTorch MultiheadAttention projections now use the same estimator and
writer, with per-weight identities and module-grouped coverage counts.

### 1. Now: small correctness and operability steps

Continue architecture fallback and checkpoint-to-measurement weight accounting.
Alongside it, inspect the broader workflow below so the roadmap does not become
only a loader project. These are hypotheses to check, not diagnosed faults or
commitments to build every proposed feature. Prioritize demonstrated risks to
measurement identity, data durability and recovery, then measured cost and human
effort. Each slice should have a concrete check and a small, understandable diff.

#### Architecture fallback and weight usage

- [ ] **Map the remaining fallback cases.** Separate missing/contradictory
  architecture metadata, unsupported classes, backend failures and resource
  limits. Preserve declared classes when consistent. Use pinned config and
  checkpoint keys/shapes to narrow supported alternatives, without repeatedly
  loading full models. Add one recorded architecture override only where needed
  for ambiguity; an override must still pass weight-integrity checks. Backend
  recovery must not silently change architecture, precision, base or revision.
- [ ] **Account for checkpoint-to-measurement usage.** Compare checkpoint contents
  with loaded named modules, parameters, relevant buffers and shared aliases, then
  with the tensors actually passed to ESD and the saved records. Each in-scope
  weight must be accounted for as measured, shared with an identified measurement,
  deliberately skipped with a reason, or unresolved. Distinguish non-weight
  buffers and non-matrix parameters from missing analyzable weights. Preserve task
  heads and full names; do not confuse physical shared modules with execution
  depth. Inspecting only the loaded model cannot expose weights already discarded
  or newly initialized during loading.
  The loaded registered-tensor portion is implemented; broader checkpoint-name
  conversions, non-tensor packing and fallback eligibility remain to be checked.
  The explicit dense `.weight` fallback covers a limited subset of loaded-layer
  gaps. Declared MultiheadAttention projection attributes are now covered;
  arbitrary extra loaded parameters remain open. Failed loads follow the narrow
  automatic-fallback rules above; broader recovery needs pilot evidence.
- [x] **Direct matrices and narrow automatic fallback.** `--analysis_source auto`
  is the default; `model` and `checkpoint` remain explicit overrides. Tensor analysis streams
  ordinary floating safetensors matrices through the existing core/writer, with
  exact keys, complete stored-key accounting and separate summary/resume scope.
  Eligibility and limits are defined above. This does not salvage partial model
  loads, interpret unknown layouts or claim architectural recovery. Keep the
  strict gate for model mode, preserve failure reasons and do not promote tensor
  results on resume. Expand formats/layouts or triggers only for demonstrated needs.
- [ ] **Verify these paths with bounded controls.** Reuse current regression
  fixtures; add cases for each new fallback and for dropped/new weights, heads,
  shared aliases, unsupported layouts and partial coverage. Keep declared and
  inferred loading, and tensor-only measurements, distinguishable in
  summaries and resume checks. Only claimed-supported tensors should reach ESD.

Acceptance before the next pilot: supported cases produce correctly identified
measurements, uncertain cases remain explicit, and no fallback silently changes
the measured object. Record the selection reason, analysis source and coverage
in existing outputs. Verify time/memory bounds and failure/resume behavior; bump
the relevant policy version when behavior changes. Universal architecture support
is not required to proceed, but known gaps must be visible and scoped.

#### Workflow, storage and efficiency hypotheses

- [ ] **Human workflow and cleanup.** Walk through preparing a small input,
  launching it, checking progress, understanding a failure, resuming and reading
  one spectrum. Identify confusing defaults, repeated configuration, stale docs,
  redundant scripts and outputs. Consolidate or remove where that makes this
  path clearer; keep one primary operating guide. Check that errors explain what
  happened and what the operator can do next. New abstractions must earn their
  complexity, just as existing ones must earn their place.
- [ ] **Database and result access.** Test the existing summary index and HDF5
  reader against actual research questions: find models by metadata and coverage,
  compare compatible runs, join canonical layer records, and fetch selected
  spectra without reading every array. Check checkpoint/revision/measurement
  identity, duplicate handling, partial writes and rebuilding derived indexes.
  Separate authoritative measurements from replaceable summaries. Measure query
  time and storage at representative sizes before choosing whether to keep,
  simplify or replace the current layout; a new database service is not assumed.
- [ ] **Input preparation and loading.** Follow HF metadata through selection,
  pinned inputs, downloads, cache use and loading. Check malformed/duplicate rows,
  unavailable checkpoints, transient failures, shared adapter bases and cache
  reuse. Measure download/startup time, bytes transferred and peak host memory.
  Keep ephemeral cleanup as the baseline: it prevents the disk-growth problem
  already observed here. Measure repeated-download costs before considering any
  bounded reuse; preserve safe concurrency, disk limits and reproducibility.
  Do not add retries or prefetching without evidence and explicit bounds.
- [ ] **Workers, timeouts and recovery.** Exercise a failed launch, stalled
  download, slow-but-progressing analysis, worker exit/OOM and interrupted run.
  Check that stage and heartbeat limits distinguish these cases, release owned
  processes/resources, preserve completed outputs and leave a clear terminal
  record. Verify resume does not accept incomplete data or lose failures. Look
  for overlapping supervision/state mechanisms that can be merged or removed;
  avoid another watchdog merely to compensate for an unclear existing one.
- [ ] **Scheduling and idle time.** Separate lack of eligible GPU capacity from
  download, CPU/SVD, startup and dispatch delays. Measure queue/stage times,
  useful completed spectra per wall time, wasted work and resource peaks. Only
  then assess polling intervals, concurrency, placement or overlapping work.
  High GPU utilization alone is not success, and an idle wait imposed by resource
  ownership or safety is not automatically waste.

Start with a bounded workflow audit using existing commands, summaries, logs and
small failure controls. Add only measurements needed to answer an open question.
For each finding, record the evidence, smallest useful change (including deletion
or no change), and how to verify it. Fix critical demonstrated gaps before the
pilot; let the pilot resolve workload-dependent questions. This is not a mandate
for a scheduler rewrite, a new monitoring stack or a separate database layer.

Initial workflow inspection: per-worker caches are isolated and removed on exit,
so cross-worker reuse is not provided by that path, intentionally bounding cache
accumulation across completed jobs. Measure repeated downloads before proposing
an alternative with equally clear storage bounds. Heartbeats report liveness;
stage timeouts bound elapsed stage time, not per-layer progress. Calibrate them against slow valid
work before changing termination policy. Existing recovery/timeout controls are
covered by offline tests; workload costs still need pilot measurements.

### 2. Bounded trained-checkpoint pilots

Use roughly 20–30 pinned public checkpoints spanning encoders, decoders,
encoder-decoder models, CNNs and supported adapters. Select against the intended
HF metadata population, including different sizes and known loading challenges;
keep quantized/custom/composite coverage explicit. Finish corrected GPU checks
on an idle authorized device (3–7), with representative CPU/float64 comparisons.

Use the existing summaries and terminal records to measure completion, loaded
weight usage, skipped weights and missing fits by family, plus runtime, peak
memory and spectrum-storage size. Exercise interruption/resume and incompatible
outputs, cache reuse and selective result queries. Distinguish active work from
queue/download/startup delays, and record manual intervention needed to complete
the workflow. Use these results to accept or reject the efficiency and usability
hypotheses above. A finite alpha alone is not a successful scientific validation.

The first 20-entry coverage pass and follow-up runtime hardening are recorded above.
The existing summary now joins selected pins, artifacts and terminal outcomes.
Rebuilding the pilot index gives 16 verified successes, three recorded failures
with unknown terminal-pin provenance, and one unrecorded model: the AWQ preflight
block was not saved per model at the time. Future blocks are recorded; do not
rewrite historical evidence from today's backend availability.

Next, address the demonstrated encoder gaps:
safe legacy-checkpoint inspection and preserving ALBERT's extra stored weights,
without guessing architectures or weakening loading integrity. Reconsider automatic
fallback eligibility only with those concrete cases and distinct measurement scope.
Keep GPU checks and ephemeral caches; measure representative larger checkpoints
before optimizing the small-model startup/dispatch costs. Also audit the curated
input's single-character `Architecture` values before using them for stratification.

### 3. After that: cost and coverage gates for staged scale-up

Use pilot measurements and the target metadata distribution to estimate storage
and compute, with uncertainty for larger/untested models. Identify family-level
missingness and expensive failure patterns. Fix observed critical problems, then
check that the operating/query path remains simple and retire superseded paths
and disposable pilot artifacts when no longer useful. Seek approval for the next
bounded batch. Keep pinned inputs and consistent
measurement versions; do not grow concurrency or launch 50k models implicitly.

Predictive validity, alternative-distribution comparisons and aspect-ratio
correction remain separate research questions. None follows merely from a
working loader, matching formulas or a completed dataset run.

Inspection references: [safetensors metadata](https://huggingface.co/docs/safetensors/metadata_parsing)
supports reading tensor descriptions without full weight downloads;
[PyTorch serialization](https://docs.pytorch.org/docs/main/notes/serialization.html)
documents `FakeTensorMode` inspection and the restrictions of `weights_only`.
Verify these paths against the installed versions before choosing an implementation.
