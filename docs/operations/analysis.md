# ESD analysis

From the repository root, first prepare a small, pinned model list in a fresh
output directory. This contacts HF for metadata and adapter JSON, not weights:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/my_run --limit 20 --prepare_only
```

Review `analysis_runs/phase2/my_run/models.csv`, then launch exactly that list.
Held preparation rows stay visible but are not dispatched:

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
The curated literal `source_model=unknown` means unspecified, not a repository
to fetch. Its original hint stays in `source_model_requested`; an adapter still
needs a real base resolved from its config or supplied explicitly.
Repeated bases share one resolution within preparation. `revision_requested`
and `source_model_requested` preserve the references used to obtain the pins.
Resolving an unspecified base revision today does not recover its historical
training revision.

Preparation also records `config_model_type`, the complete JSON list
`config_architectures`, and `config_revision` from the same revision-specific
Hub metadata response. `config_metadata_status` is `recorded`, `missing`, or
`invalid`; recorded metadata can still omit either label. These are declarations,
not a verified weight layout. The original `Architecture` and leaderboard fields
remain unchanged. Routing prefers the prepared config labels only at their
recorded pin; missing, invalid or stale config evidence does not revive imported
architecture labels. Other routing hints and the worker's strict integrity checks
still apply. This adds no request per model beyond the existing metadata lookup
and downloads no weights. It does not infer an adapter's architecture from its base.
The existing summary carries these input fields as `input_config_*` for queries;
no separate metadata database or repair command is needed.

Preparation records `preparation_status=ready|held`, `preparation_reason`, the
selected `weight_format`, `weight_files` (JSON byte sizes), `weight_bytes` and
`download_bytes`. Adapter payloads and their pinned base files are counted
together; ordinary finetune lineage hints do not add a second model to the
budget. Standard safetensors files are preferred over PyTorch files. Sharded
checkpoints use their small index JSON, not all similarly named files in the
repository. The default `--max_weight_gib 6` holds oversized jobs; zero disables
the size cap, not missing-file/size/access checks. Gated/private repositories
need verified read access. These checks do not validate tensor layouts, predict
RAM/VRAM requirements, or establish quantization support. Runtime integrity and
coverage checks still apply. JSON downloads are deleted immediately after use;
shared base metadata and index reviews are reused within preparation.

`pin_status=error` rows keep their `pin_error`; preparation exits nonzero if any
fail, but the written manifest retains them as holds. Launch skips explicit
holds and records them as blocked; the summary can show them before launch.
Manually pinned older lists without preparation fields remain supported, but
unresolved rows in those lists must be fixed or removed. Duplicate output
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

### Query models and read a named spectrum

Refresh a run's index with `python esd_experiment/analyze_results.py --results_dir
<run_directory>` after collection or a retry. It checks the saved pairs without
reading spectra. Then use pandas for model selection and h5py for a chosen layer.
This example uses the collected encoder/vision slice:

```python
from pathlib import Path
import h5py
import pandas as pd

run = Path("analysis_runs/phase2/encoder_vision_batch_001")
index = pd.read_csv(run / "summary.csv", keep_default_na=False)
ready = index.query("artifact_status == 'valid' and matrix_coverage == 'complete' and analysis_source == 'model'")
row = ready.loc[ready.model_id == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"].iloc[0]

with h5py.File(row.h5_path, "r") as h5:
    names = h5["layers/longname"].asstr()[:].tolist()
    layer = "distilbert.transformer.layer.0.attention.q_lin"
    i = names.index(layer)  # Exact canonical name; print names to choose another.
    print(h5["layers/fit_status"].asstr()[i], h5["layers/n_tail"][i])
    spectrum = h5["eigs"][i]  # Only this layer's spectrum is read.
```

The index also exposes `input_config_model_type`, `model_class`, pins and
measurement settings. To query several collected runs without copying spectra:

```python
collection = pd.concat([
    pd.read_csv(path, keep_default_na=False).assign(run=path.parent.name)
    for path in sorted(Path("analysis_runs/phase2").glob("*/summary.csv"))
], ignore_index=True)
```

Keep the run, revision, base pin and conventions when joining or comparing rows.
Do not deduplicate by model ID: retries, different revisions and measurement
settings are not interchangeable. Filter valid results for measurement queries;
retain failed rows for collection coverage. `input_candidate_source` distinguishes
curated rows from explicit coverage extensions. Indexes are rebuildable, not
completion evidence; the CSV/HDF5 pairs remain authoritative. Complete coverage
and finite fits alone do not establish scientific validity.

Current output versions are numerics **7**, loader **10**, HDF5 format **2.0**.
Layer selection **2** adds declared ViT tables without changing spectral formulas.
Earlier numerics-7/loader-10 pairs without this field remain readable as selection
**1**, but do not resume under current selection settings. Keep this convention
when comparing summaries; use a fresh run directory rather than rewriting history.
Resume requires a compatible CSV/HDF5 pair: versions, canonical identities,
aligned alpha values and requested measurement settings must match. Changing
spectrum storage, precision, filtering or model revisions requires new outputs.
Runtime hardware differences are provenance, not a CPU/GPU equivalence claim.

Numerics 7 adds declared MultiheadAttention projection matrices, measured whole,
and records their weight attributes. The loader includes restricted legacy encoder
inspection, BERT layout-based recovery of missing `model_type`, and restoration
of ALBERT's stored pooler. Loader 10 adds memory-mapped PyTorch matrix analysis
and automatic fallback for the exact legacy DeBERTa pretraining layout below.
Requested analysis policy remains separate from actual analysis source.
Earlier-loader outputs remain historical data, but are not accepted
by current resume/index checks; use a fresh directory for loader 10. Incompatible/incomplete
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

`matrix_coverage` is `complete`, `partial` or `unknown`. Complete means the
recorded matrix candidates were measured and, in model mode, the loaded-tensor
inventory has no skipped/unresolved tensors or unlinked measurements. Ordinary
vectors do not count as gaps. Missing inventories stay unknown. This is coverage
within the recorded model/checkpoint scope, not proof of architecture support or
scientific validity. Partial/unknown rows keep their scalar values but are
excluded from the console's pooled alpha statistic. `outcome=success` still
means a valid artifact pair, so filter on both fields when selecting results.

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
architecture declarations fail before analysis. The exact historical GPT2
`masked_bias` buffers are permitted as extra keys, and remain recorded.
For a declared `AlbertForMaskedLM` with only the two stored pooler tensors extra,
the loader restores `albert.pooler.weight` and `.bias` from pinned safetensors.
Shapes and floating dtypes must match the configured pooler; all other loading
checks still apply. `restored_checkpoint_keys` records this recovery. The MLM
class is retained, and the pooler reaches ordinary module coverage and ESD.

Ordinary BERT, RoBERTa, DistilBERT, ALBERT and DeBERTa (v1/v2) checkpoints without
`architectures` can be selected automatically from their complete checkpoint
key/shape layout, including shards and omitted tied
aliases. Candidate built-in classes are constructed on the meta device, without
allocating their weights. The decoder config distinguishes masked from causal LM;
multiple matches (for example some one-output classification/multiple-choice
heads) fail explicitly. Declared architectures are never silently replaced.
Inspection prefers safetensors headers. When absent, encoder selection also
accepts plain PyTorch state dicts (`pytorch_model.bin` or its shard index), using
`FakeTensorMode` and explicit `weights_only=True` without allocating weight storage.
It uses Transformers' patched-PyTorch version gate, never unrestricted pickle or
checkpoint-specific allowlists. The selected format is forced during actual
loading. Direct checkpoint-matrix analysis also accepts memory-mappable ZIP
PyTorch state dicts; unlike encoder loading, it does not read pre-ZIP files.

Old BERT exports missing both `model_type` and `architectures` additionally need
explicit BERT dimensions, BERT-prefixed tensors and one complete built-in layout
match. This inference is recorded, not based on the repository name. Serialized
position-ID buffers are accepted only when their actual values match the loaded
buffers, and are recorded as `verified_checkpoint_buffers`. This verification
can require one extra shard read for legacy `.bin`; ordinary loading already
materializes those weights. No trained keys are silently discarded.

Downloads still use the worker's pinned, ephemeral cache and load-stage bounds.
Selection evidence and input/output embedding tying remain in loading provenance.
Other families, custom and quantized missing-metadata cases are not covered by
this inference path. No architecture override exists yet. Restricted loading
is not a sandbox or protection from resource exhaustion; see the
[PyTorch serialization notes](https://docs.pytorch.org/docs/2.10/notes/serialization.html).
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

Legacy BERT adapters have one narrow task-transfer path: a base declared as
`BertForMaskedLM` must match the complete `BertForPreTraining` layout, including
the historical LayerNorm `gamma`/`beta` aliases. Rename collisions and unexplained
keys fail. The full pretraining model is loaded and checked first. A `SEQ_CLS`
adapter then uses its saved classifier weight/bias to determine the output count;
an adapter explicitly mapping to built-in `BertForMaskedLM` retains the stored
MLM head. Every retained backbone/head parameter comes from the verified base
or adapter, never a random task head. Ordinary standalone loading still honors
the declaration and rejects this conflict.

Unused pretraining-head keys are listed in `base_loading_info.bert_adapter_transfer`,
with the source/target classes and transfer policy version. `summary.csv` exposes
`base_transfer_policy` and `excluded_base_key_count` (names, not unique parameter
counts). Complete coverage here means the reconstructed **adapted task model**,
not every tensor in the pretraining file. This follows the adapter class/task and
saved payload evidence in the [PEFT checkpoint format](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/checkpoint).
Only previously rejected base-layout conflicts gain this path; accepted model
measurements and loader version 10 remain unchanged. No new mode or registry is added.

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

Built-in HF `ViTEmbeddings` tables are measured as `[tokens, hidden]`, selecting
`[0]` from their singleton broadcast axis. The attribute name, original shape
and slice are recorded. This covers positional embeddings, CLS and optional mask
tokens; one-row token tables retain a rank-one spectrum and a missing power-law
fit, not a meaningful tail estimate. Patch projections remain Conv2d matrices.
Shapes must match the declared patch count and hidden width; custom subclasses
and arbitrary 3D parameters are not reshaped. See the
[Transformers ViT implementation](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/vit/modeling_vit.py).

Full supported-matrix inclusion is now the default (`filter_type=False`) in the
runner, worker and Python estimator. Other ordinary
floating, dense 2D `.weight` tensors become eligible, and the legacy Linear
aspect-ratio skip is disabled. Standard convolution layouts remain supported;
unknown higher-dimensional layouts, packed/quantized representations and
undeclared nonstandard attributes are still skipped. This is not
architecture recovery or permission to use a partially loaded checkpoint.
Inspect coverage and weight-usage records for remaining gaps. `filter_type` is
stored in HDF5 and the summary, and must match on resume. Use `--filter_type` to
explicitly resume an older restricted run; use a fresh directory for wider
coverage. No numerical/version bump is needed: the existing recorded setting
already distinguishes these conventions and rejects mismatched resume.

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
fall back once to checkpoint matrices for these identified cases:

- A valid config declares a `model_type` that the installed library does not support.
- Resolving the config requires repository code that is disabled.
- Multiple supported encoder architectures fully match the checkpoint keys/shapes.
- The exact legacy DeBERTa pretraining export matches: complete encoder, both
  prediction heads and stored embedding tensors, with every name/shape accounted for.

The DeBERTa case is a recognized stored layout, not a reconstructed model. No
embedding alias/value relationship is assumed. Any missing key, wrong shape or
unexplained extra key prevents this fallback. Other **unmatched layouts** may
mean missing or incorrect weights and remain failures. Conflicting architecture declarations,
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

Checkpoint mode prefers pinned `model.safetensors` or its shard index; when absent,
it also accepts `pytorch_model.bin` or its shard index. All shard keys/shapes are
validated. PyTorch files must be plain tensor state dicts in the ZIP format that
supports `mmap=True`, with restricted `weights_only=True` loading and the same
patched-version check as encoder inspection. Pre-ZIP files are rejected here
instead of eagerly loading an entire shard; provide safetensors for those files.
There is no unrestricted-pickle retry, custom allowlist or arbitrary-file search.
No recognized architecture, model construction or repository Python is needed.
Missing/unknown architecture metadata is fine; malformed files still fail.
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
cleanup. PyTorch analysis maps one shard at a time and releases it before the
next, retaining accumulated spectra but no constructed model or eager full-shard
copy. Mapped pages can become resident as matrices are read; this is not a hard
RSS limit. `checkpoint.format` and `checkpoint.tensor_access` record the reader.
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

Readers use safetensors' [per-tensor API](https://huggingface.co/docs/safetensors/index)
or PyTorch's [restricted loading and memory mapping](https://docs.pytorch.org/docs/2.10/notes/serialization.html).
See also safetensors' [shared-tensor limitations](https://huggingface.co/docs/safetensors/torch_shared_tensors).

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
- The runner defaults to `stale_process_action="terminate"`. Set `log`
  explicitly to observe expired timeouts without stopping workers.
- The default is one worker. Increase `--max_concurrent_jobs` explicitly after
  reviewing memory/storage costs; per-worker caps are not an aggregate host cap.
- `--max_worker_ram_gib 16` samples summed worker-tree RSS and
  `--max_worker_cache_gib 12` samples allocated private-cache bytes every five
  seconds in the existing supervisor. Zero disables each cap. These are sampled
  safety stops, not hard OS quotas: fast allocation spikes can overshoot, and
  slow filesystem reads can delay a sample. Monitoring errors stop the affected
  worker rather than silently removing its guard. Cap breaches use existing
  group termination, status recording and cleanup, then permit the next worker
  if cleanup succeeds. Infrastructure/cleanup failures still stop dispatch.
- HF and legacy Transformers cache paths share the same private Hub cache;
  Xet and remote-module caches are also kept inside the worker directory, even
  if the parent environment points elsewhere. No shared cache is deleted.
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
Availability checks are not a reservation against other users or runners:
another job can start after a GPU passes the idle checks. Coordinate an exclusive
window for isolated performance measurements. Device-wide GPU memory includes
other processes and must not be reported as worker-only memory.

## Validation

Offline regression tests cover numerical edge cases, layer identity, coverage,
checkpoint/adapter integrity and persistence:

```bash
python -m pytest esd_experiment/tests -q
```

Install pytest separately for development; it is not needed by workers. The
[installation recipe](../../README.md#1--create-the-environment) uses one root
dependency list. `python esd_experiment/tests/test_setup.py` checks imports and
a tiny CPU spectrum without accessing tokens, downloading weights or using GPUs.

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

### Bounded pilot, 2026-09-14 (loader 8)

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

### 1. Now: ordinary bounded collection

The baseline is ready for supported families: prepare, review, run, query.
Preparation checks pinned file sizes and access, counts adapter bases, retains
holds and deletes its JSON caches. The normal supervisor enforces sampled
worker RAM/cache bounds and terminating stage/heartbeat timeouts. Held rows are
visible in the existing index without running a worker. The pilot observer is
not an operational dependency; its duplicate RAM/cache stop logic is retired.

Current defaults are one worker, 6 GiB selected-weight budget, 16 GiB worker-tree
RSS and 12 GiB private cache. These are configurable operational safeguards, not
proof of memory feasibility or scientific validity. Existing numerical
conventions and authoritative measurements are unchanged.

Continue bounded collection from curated HF metadata. Use recorded failures,
coverage and actual costs to choose the next engineering change. No additional
general-purpose numerical pilot is a prerequisite for every small step.

Remaining work is driven by evidence, not an unchecked list of features:

- **Coverage:** map remaining unsupported/custom/packed layouts and metadata
  contradictions. Preserve declared classes, heads, shared aliases and exact
  checkpoint identities. Never salvage an incomplete load or promote tensor-only
  results to model-layer measurements. Expand support only with an explicit
  interpretation of the weights and a focused control for the changed path.
- **Result access:** keep authoritative CSV/HDF5 pairs separate from rebuildable
  indexes. Query compatible conventions and full layer names, without loading
  all spectra or deduplicating by model ID. Measure real query/storage costs
  before adding a catalog service, changing formats or duplicating data.
- **Loading efficiency:** quantify repeated transfers, especially shared adapter
  bases. Ephemeral caches remain the baseline. Any reuse needs a bounded budget,
  ownership and eviction; do not add unbounded shared caches, retries or prefetch.
- **Scheduling/recovery:** distinguish dispatch/import/download/analysis costs
  and genuine capacity waits. Preserve idle-device checks, signals, failure
  records and version-aware resume. Raise concurrency or tune timeout windows
  only with workload evidence. Reuse the existing supervisor, not another
  watchdog. A blocked kernel/filesystem or unconfirmed cleanup remains an
  infrastructure problem, not permission to reuse its resources.
- **Human upkeep:** simplify confusing defaults, repeated configuration, stale
  documentation and redundant code as part of each change. Keep one operating
  guide and clear errors. Existing mechanisms must earn their complexity just
  as new ones do; preserve useful research outputs when removing disposable
  artifacts.

The real metadata check in `analysis_runs/validation/preparation_20261010/`
marked seven small checkpoints ready and held OPT-30B for its 6 GiB budget,
without downloading weights. A CPU worker control verified cap-triggered
termination, cleanup and subsequent-job progress. The ordinary all-held launch
recorded the block without creating GPU configuration or artifacts. These are
operability checks, not predictive-validity evidence.

### 2. Implementation evidence (historical pilots)

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

The loader-9 slice added restricted legacy inspection, missing BERT config-type
recovery and restoration of ALBERT's stored pooler. The bounded CPU recheck used
the same three pinned encoder checkpoints, without rewriting the loader-8 pilot.
BERT-tiny and ALBERT preserved all 49 and 31 stored tensors respectively, including
BERT's regenerated position-ID buffer and ALBERT's restored pooler. One matrix
from each agreed with an independent float64 spectrum reference: maximum
absolute error divided by the largest reference eigenvalue was below 3.3e-7.
These are loader and sampled-matrix checks, not full-model or GPU validation.
Reports are in `analysis_runs/validation/encoder_recovery_20260917/`; temporary
download caches were removed.

Loader 10 recognizes that DeBERTa-v3 checkpoint's complete legacy layout and
automatically uses stored-matrix analysis, preserving its additional embedding
tensors and LM/mask-prediction heads without inventing a replacement class.
The head layouts follow Microsoft's [masked-LM implementation](https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/deberta/bert.py)
and [replaced-token implementation](https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/replaced_token_detection_model.py);
the embedding keys are checked against the observed pinned export.

The pinned DeBERTa-xsmall CPU worker recheck accounted for all 212 stored tensors:
80 matrices measured, 132 non-matrix tensors skipped with reasons. All 80 saved
spectra matched independent SciPy float64 SVD references, with maximum absolute
error divided by the largest reference eigenvalue below 3.9e-6. Summary reading
and unchanged-artifact resume passed. This cached, two-thread CPU run took about
8 seconds and peaked at 1.31 GiB process RSS; it is not a large-model cost estimate.
Outputs and verification are in `analysis_runs/validation/deberta_checkpoint_20260917/`.
The disposable 231 MiB download cache was removed. Strict model-only loading is
still unsupported for this export; these are explicitly checkpoint-matrix results.

The metadata audit found all 8,467 Atlas-derived `Architecture` cells equal the
first character of the original string in `../data/large_nlp_data.json`, including
3,848 instances of `unknown` becoming `u`. The exporter is not in this checkout;
string indexing during export is an inference from that exact source comparison.
All 3,164 analysis-ready rows have a blank primary architecture and come from
`seed_metadata`; this subset is not a balanced cross-family sample. The audit and
source hashes are in `analysis_runs/validation/metadata_audit_20260917/audit.json`.
Do not stratify by the damaged labels, or change old results to match today's
metadata. Prepare fresh config labels, keep unknowns separate, and retain explicit
pilot family/role labels for deliberate coverage controls.

The loader-10 rerun in `analysis_runs/validation/trained_pilot_20260917/` completed
on September 29, with the same 20 model pins and adapter-base pin. Config labels
are present for 18 entries; BERT-tiny and the adapter remain explicitly unlabelled.
Three rows matched their pinned raw config JSON. The folder date is preparation,
not execution. `summary.csv` and `pilot_report.json` contain the current results:

- **19 valid pairs, one AWQ block** (`unsupported_backend`; AutoAWQ is absent).
  All six encoders completed. DeBERTa is explicitly checkpoint-matrix analysis:
  80 matrices measured, 132 non-matrix tensors skipped. Other results use model
  mode. Deliberate skips remain visible, including aspect-ratio filtering and
  two unsupported DeiT attributes; success does not mean exhaustive coverage.
- **1,405 measurements, 795,639 full-spectrum eigenvalues, 6.68 MiB** of CSV/HDF5.
  All identities and spectrum lengths aligned; three missing fits were retained.
  Recorded loaded-tensor reports have zero unresolved tensors/unmapped measurements.
  The index keeps model/checkpoint conventions separate. Selected spectrum reads
  took a median of about 0.9 ms with warm filesystem caches, not a 50k-scale benchmark.
- **GPU checks passed:** four well-conditioned controls and four sensitivity
  controls completed on L40 GPU 3. Six saved MiniLM/ResNet spectra agreed with
  pinned-weight SciPy float64 references to below 2e-7 of the largest eigenvalue.
  Workers used GPUs 5/6, one at a time. Runtime was PyTorch 2.11.0+cu128 and
  Transformers 5.9.0, not the previous PyTorch build. The pytest suite was not
  rerun: `pytest` is absent from this environment. No packages were changed.
- **Recovery passed:** an active load was interrupted and resumed; the final
  resume launched no workers and left all 38 artifacts byte-for-byte unchanged.
  Changing compute precision was rejected for all 19 pairs. Worker caches emptied;
  the four disposable cache roots, including reference-download logs, were removed.
- **Observed cost:** 23m24s wall time, about 4m34s sampled active-worker time;
  the remainder includes preflight, capacity waits, dispatch and cleanup.
  Sampled process-tree RSS peaked at 1.90 GiB and cache allocation at 1.24 GiB.
  CLIP overlapped an external GPU job: its 11.23 GiB device-memory peak is not
  worker-only usage. These shared-server timings are not isolated throughput estimates.

The installation recipe now has one root list of direct analysis dependencies
and a Python-only Conda environment, with the matched PyTorch wheel installed
explicitly. The duplicate experiment requirements and obsolete GPU/download
setup probe were removed. The small CPU setup check passes in `esd_ind` and
returns a nonzero status for missing dependencies without inspecting tokens.
The recipe was also rebuilt in a disposable Conda environment: `pip check`, the
CPU setup check and all 847 existing tests passed (44 seconds for pytest).
The environment and package caches were removed afterward; installation logs
and resolved package versions are in `cost_pilot_20260929/install_retry_*` under
the validation directory. The verifier's first attempt incorrectly required
wheels only; allowing gpustat's source distribution, as the documented recipe
already does, resolved that restriction without changing the pins.
The shared environment's existing Streamlit/protobuf conflict is outside the
analysis baseline and was not changed. Optional quantization packages remain separate.

The three-model cost slice in `analysis_runs/validation/cost_pilot_20260929/`
also ran on September 29: one worker at a time on idle L40 GPU 6, in unchanged
`esd_ind`. Limits were 12 GiB disposable cache, 16 GiB sampled process-tree RSS,
and load/analyze/save timeouts of 900/900/120 seconds. No limit was reached.
`models.csv` retains the pins; `summary.csv` and `pilot_report.json` record results.

| Checkpoint | Outcome | Sampled worker time | Peak RSS / worker GPU / cache (GiB) |
| --- | --- | --- | --- |
| RoBERTa-large | 148 measurements, 147 fits | 28 s | 1.53 / 2.69 / 1.32 |
| Qwen2.5-3B, two shards | 181 measurements, 181 fits | 61 s | 6.66 / 12.33 / 5.75 |
| Qwen2.5-1.5B-Instruct-GPTQ-Int4 | Load failed before weight download | 7 s | 0.84 / 0 / <0.01 |

The two valid pairs contain **520,707 full-spectrum eigenvalues in 2.79 MiB**.
Identities, spectrum lengths and finite/nonnegative values passed checks, with
no unresolved loaded tensors or unmapped measurements. RoBERTa's token-type
embedding remains present despite having no retained eigenvalues for fitting.
Source dtypes are float32 for RoBERTa and bfloat16 for Qwen; computation is float32.
Qwen's default aspect-ratio filter skips all 72 distinct key/value projection
matrices (256×2048). Its shared output head is also filtered as a module, but the
tied embedding weight is measured. These are recorded exclusions, not full coverage.

GPTQ passed preflight but failed importing `optimum.gptq` → `datasets`:
`importlib.metadata.version("fsspec")` returns `None` from an incomplete
`fsspec-2026.3.0.dist-info`, alongside the working 2026.2.0 module/distribution.
An independent `datasets` import reproduces the failure. This is an environment
blocker, not evidence about this checkpoint's quantized-matrix support. The
failure remains `load/model_load_error`; no shared packages were repaired.

The run took 3m18s wall time; approximately 96 seconds were sampled active-worker
time, with the rest including preflight, availability checks, dispatch and cleanup.
No other GPU compute PIDs were observed on the worker device. Peaks are sampled,
and summed process RSS can double-count shared mappings; this is not an isolated
throughput benchmark. Resume with `--skip_failed` launched no workers and preserved
all four artifacts byte-for-byte. Precision-change compatibility checks rejected
both pairs. All pilot processes exited and both disposable cache roots were removed.

The follow-up in `analysis_runs/validation/qwen_coverage_20260929/` used the same
Qwen pin and settings except `--no-filter_type`. It recovered all 72 key/value
matrices plus the tied output-head view: **254 measurements, all fitted**, with
391,168 eigenvalues and no unresolved or skipped matrix tensors. All 181 common
spectra were exactly unchanged. The head and embedding spectra are identical
views of shared weights, not independent tensors; the saved aliases retain this
distinction. The fitted-alpha mean changed from 2.5172 to 2.8106 solely through
coverage, illustrating why model aggregates need matching inclusion rules.

Sampled analysis time was 32.9 versus 32.2 seconds; peak worker GPU memory stayed
12.33 GiB and output size grew from 1.85 to 2.02 MiB. This single shared-server
comparison supports wider coverage at little observed extra cost, not a general
performance guarantee. Resume preserved both files; processes exited and caches
were removed. See `comparison_report.json`. This comparison informed the new
unrestricted default. This coverage
filter is separate from the deferred statistical aspect-ratio correction.

The GPTQ recheck in `analysis_runs/validation/gptq_isolated_20260929/` used a fresh
baseline plus the [official GPTQModel 6.0.3 wheel](https://github.com/ModelCloud/GPTQModel/releases/tag/v6.0.3)
for Python 3.10 / PyTorch 2.11 / CUDA 12.8, TorchAO 0.17.0+cu128, Optimum 2.1.0,
Datasets 4.8.2 and fsspec 2026.2.0. Dependency checks, CPU setup and backend import
passed. The exact install commands and resolved packages are retained with the
report; no optional backend was added to the root requirements.

The checkpoint loaded on idle GPU 5, but **only `model.embed_tokens` was measured**:
one fit and 1,536 eigenvalues. All 196 `MarlinQuantLinear.qweight` modules were
explicitly skipped as unsupported packed attributes; their 196 `scales` parameters
remain unresolved in tensor coverage. The tied head was filtered, and the remaining
57 candidates were non-matrix weights. Thus `outcome=success` means valid artifacts,
not model-wide quantized coverage. Do not use this embedding-only alpha as a model
quality score or mistake the load success for verified GPTQ spectral support.

The GPTQ worker took about 20 sampled seconds (58 seconds run wall time), peaking
at 1.84 GiB summed RSS, 6.17 GiB worker GPU memory and 1.08 GiB cache. No other GPU
compute PIDs were observed. Resume preserved both files with no workers launched;
all processes exited, and the environment, package caches and checkpoint caches
were removed. Shared `esd_ind` remains unchanged.

Full supported-matrix inclusion is now the default, and the summary exposes
`matrix_coverage` without changing artifact or terminal-status formats. The dense
Qwen comparison reads as complete; the GPTQ control reads as partial and its
embedding-only alpha is excluded from pooled console statistics.
Keep packed quantization explicitly unsupported until a backend-specific
dequantization path has an independent numerical check; do not analyze integer
packing or scale arrays as weights. No new database service or loader framework is
needed. These pilots are not population samples, predictive validation, or
permission to scale up.

### 3. Collection costs and progress

The first operational batch completed on October 10, 2026, using
`analysis_runs/phase2/dense_batch_001/models.csv`. Use this ordinary pinned manifest
with the existing runner; no new queue, database service or batch mode was added.
`batch_review.json` retains input hashes, selection rules, pins, file sizes and
the excluded candidates. Its one-off preparation script downloads metadata only.

The current input contains 15,000 rows, not the eventual 50,000. Loader hints split
into 7,436 standard, 4,737 adapter, 2,101 quantized and 726 multimodal rows; they are
not verified support counts. Only 2,922 rows have numeric parameter-size metadata.
The damaged architecture labels must not drive selection. After excluding tagged
adapters/quantization/MoE/multimodal, 517 rows have positive size hints at most 3B.
These hints screen candidates; pinned config/access/file metadata is then reviewed.

We selected 24 candidates across two size bins, with at most one publisher and
metadata parent per selection. **16 passed metadata review**: ten at most 1B and
six between 1B and 3B, spanning fresh config types qwen2 (6), llama (4), gpt2 (2),
phi (2), gpt_neo (1), and mistral (1). Four candidates were unavailable without
credentials, two lacked the selected safetensors metadata, one exceeded the
6.5 GB checkpoint ceiling and one was gated. They remain in the review, not the
launch manifest. Metadata acceptance is not a guarantee of successful loading.

Plan for 34.5 GB of selected weight files transferred over the batch, not all
retained simultaneously. The largest is 5.56 GB; ephemeral per-worker cleanup
remains necessary. As a rough planning allowance, 16 results at the observed
Qwen size would be about 32 MiB, but architectures and compression vary. The
observed 32–33 second Qwen analysis and roughly two-minute single-model run are
anchors, not throughput guarantees. Keep a one-hour whole-batch stop/review limit
rather than extrapolating these measurements to 50k.

The user authorized GPUs 0–3 for this run. One worker used idle GPU 0 with full
supported-matrix inclusion and saved eigenvalues. **15 valid pairs have complete
recorded matrix coverage:** 2,187 measurements/fits, 2,739,840 eigenvalues and
15.16 MiB of CSV/HDF5 artifacts. The remaining GPT-Neo-125M checkpoint failed
`load/checkpoint_weight_mismatch`: its legacy attention `bias`/`masked_bias`
buffers were rejected. The worker failure did not stall the queue. This case
is now resolved in `analysis_runs/phase2/gpt_neo_retry_001/`: **75 measurements,
57,600 eigenvalues, complete recorded matrix coverage**, in 41 seconds wall time.
The loader accepts only built-in GPT-Neo buffer names after checking stored
causal masks against regenerated masks and scalar float32 `masked_bias` against
`-1e9`. Missing/mismatched weights and arbitrary extra tensors still fail.
Accepted buffers are recorded in runtime provenance. This follows the
[historical Transformers implementation](https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/gpt_neo/modeling_gpt_neo.py).
Existing successful artifacts remain untouched. Loader version 10 is retained:
this adds verified acceptance of formerly rejected constants, without changing
the measured parameters or conventions of any previously accepted checkpoint.

Wall time was 14m16s, with 6m28s sampled active-worker time: roughly 55 seconds
dispatch/import, 122 seconds load, 199 seconds analysis and 12 seconds save.
The remaining 7m48s includes preflight, GPU availability checks, dispatch waits
and cleanup. This makes orchestration delay a concrete efficiency target; inspect
the existing idle-check setting before adding new scheduling machinery.
The default is now two consecutive checks, five seconds apart, instead of five:
five seconds of confirmation per launch rather than twenty. Checks still require
no GPU processes and memory below the threshold; query failures and config
reloads reset the confirmation count. `--max_check` remains configurable.
Sampled peaks were 5.70 GiB summed process RSS, 8.55 GiB worker GPU memory and
5.18 GiB cache, within the 16/12 GiB RAM/cache limits and 900/900/120-second stage
timeouts. No overall stop was needed, and no other GPU compute PIDs were observed
on the worker device. All worker groups exited and the temporary cache root was
removed; the result index and diagnostics remain.

This session used RTX PRO 6000 Blackwell GPUs, PyTorch 2.10.0+cu130 and
Transformers 5.9.0, recorded in the artifacts. The shared runtime differed from
the earlier L40/2.11 pilot; it was not modified. `summary.csv` is the dataset
index; `batch_result.json` records costs/outcomes and `cleanup_report.json`
records cleanup. No additional numerical pilot or synthetic tests were run.

The three prepared adapters ran in `analysis_runs/phase2/adapter_batch_001/`:
two BERT derivatives and one Qwen2-0.5B derivative, with adapter and base pins
resolved from actual adapter configs. `batch_review.json` records base weight
sizes (0.44/0.99 GB) and exclusions. A fourth candidate points to a quantized
base, contrary to its imported metadata hint, and is held out. The 88-second run
produced one valid Qwen pair: **170 measurements, 115,456 eigenvalues, complete
recorded matrix coverage**. Its 336 adapter tensors passed payload/value checks
before safe merge. Both BERT adapters failed `load/checkpoint_weight_mismatch`:
the pinned base declares `BertForMaskedLM`, but also stores pooler and NSP-head
weights. No tensors were discarded to force success. Failure did not stall the
queue. Sampled peaks were 1.90 GiB process RSS, 4.46 GiB worker GPU memory and
0.94 GiB cache; all owned groups exited and the disposable cache was removed.
`summary.csv` holds outcomes; `batch_result.json` and `cleanup_report.json` hold
costs and cleanup evidence. Existing results and the shared environment are unchanged.

The BERT gap is resolved in `analysis_runs/phase2/bert_adapter_retry_002/` using
the narrow task-transfer rule above. Both adapters succeeded in 66 seconds:
**154 measurements, 115,462 eigenvalues**, complete recorded coverage of the
adapted task models. The saved two-label classifier is included. The summary
records 9 unused pretraining-head keys for classification and 4 for MLM;
these are key counts, not unique parameters. Sampled peaks were 1.88 GiB RSS,
1.56 GiB GPU memory and 0.41 GiB cache. All worker groups exited and the temporary
cache was removed. A first retry safely rejected the historical LayerNorm names;
its cause is retained in `batch_result.json` and covered by offline controls.
The original Qwen adapter result and earlier dense results remain unchanged.

The encoder/vision slice is in `analysis_runs/phase2/encoder_vision_batch_001/`:
curated BERT multiple-choice, plus public DistilBERT classifier, ViT and ResNet-18
coverage extensions. All four produced valid pairs in 93 seconds: **212
measurements, 175,512 eigenvalues**, three complete-coverage models and one partial.
The BERT scalar classifier's missing fit remains stored. ViT's learned
`cls_token` `[1,1,768]` and `position_embeddings` `[1,197,768]` are explicitly
skipped as unsupported weight attributes; no reshaping or complete-coverage claim
was forced. Peak sampled RSS/GPU/cache were 1.74/1.48/0.41 GiB, and the owned
workers and temporary cache were cleaned up.

The query examples above were exercised on five current run indexes: 26 attempt
rows, 23 valid measurement pairs, 22 with complete recorded coverage. Reading and
filtering those indexes took 36 ms; retrieving one named 768-value DistilBERT
spectrum took 1.5 ms. These are single warm-cache observations, not 50k benchmarks.
`batch_result.json` records the outcomes, gaps and query check. No new reader
framework, database service or dependencies were needed.

The gap is resolved in `analysis_runs/phase2/vit_t5_batch_001/`: pinned ViT now
has **76 measurements** with complete recorded coverage; its original 74
measurements and spectra are exactly unchanged. Public Flan-T5-small adds **150
measurements**, retaining distinct encoder, decoder and cross-attention names
and recorded shared-embedding aliases. Both use selection 2; the CLS spectrum
is retained without a fit. Together they stored **120,530 eigenvalues** in
65 seconds, with peak sampled RSS/GPU/cache of 1.71/1.29/0.32 GiB. Owned process
groups exited and the private cache was removed. Earlier artifacts and the run
script are unchanged; existing summary indexes now explicitly label selection 1.

The proposed sharded retry was redundant: Fietje-2 already exercised a two-shard,
5.56 GB checkpoint in the first dense batch. Next, continue bounded collection
from the curated HF metadata, prioritizing underrepresented families and using
observed coverage, failures and costs to choose necessary changes. Keep the
human workflow simple: prepare pins, review the CSV, run, query the summary and
named spectra. Keep one worker and existing resource limits. Packed
quantization stays a separately blocked coverage track. The first batch is dense/decoder-heavy,
not evidence of universal support or a population sample. Use measured failure
patterns to choose the next change, retire redundant paths, and keep the existing
CSV index plus HDF5 spectra unless measured query/storage costs justify more.
Do not grow concurrency or launch 50k models implicitly.

`analysis_runs/phase2/family_batch_001/` adds six models directly from the curated
metadata: Pythia-160M, BLOOM-560M, OPT-1.3B, StableLM-2-1.6B, a Korean BERT MLM
derivative and CLIP ViT-B/16. All six succeeded with complete recorded matrix
coverage and no unresolved loaded parameters: **692 measurements, 940,820 saved
eigenvalues**, 5.31 MiB of artifacts, in 3m17s. Both checkpoint formats loaded
without remote code. Peak sampled RSS/GPU/cache were 2.13/7.71/4.90 GiB; owned
groups exited and the private cache was removed. Preparation's exact `unknown`
base placeholder handling was the only code fix needed. The indexes now contain
31 valid pairs across 30 model IDs (including the older partial ViT result);
these counts do not deduplicate revisions or establish population validity.
Continue with bounded curated batches; prioritize actual missingness and cost
evidence over additional loader machinery or repeated numerical pilots.

Predictive validity, alternative-distribution comparisons and aspect-ratio
correction remain separate research questions. None follows merely from a
working loader, matching formulas or a completed dataset run.

Inspection references: [safetensors metadata](https://huggingface.co/docs/safetensors/metadata_parsing)
supports reading tensor descriptions without full weight downloads;
[PyTorch serialization](https://docs.pytorch.org/docs/main/notes/serialization.html)
documents `FakeTensorMode` inspection and the restrictions of `weights_only`.
Verify these paths against the installed versions before choosing an implementation.
