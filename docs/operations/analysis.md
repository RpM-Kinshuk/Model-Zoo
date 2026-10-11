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
measurement settings. Rebuild a collection-wide index directly from the original
pairs, selected manifests and terminal records, without copying spectra:

```bash
python esd_experiment/analyze_results.py \
  --results_dir analysis_runs/phase2/*/ --output analysis_runs/phase2/catalog.csv
```

The multi-run output must be outside the supplied run directories. Every row
keeps `run_dir`; repeated model IDs, revisions and failed attempts remain separate.
Invalid inputs leave the previous index unchanged; invalid artifact pairs remain
visible and give a nonzero exit status. The index does not read eigenvalues or
depend on older `summary.csv` files. Query it with ordinary pandas:

```python
collection = pd.read_csv("analysis_runs/phase2/catalog.csv", keep_default_na=False)
```

Keep the run, revision, base pin and conventions when joining or comparing rows.
Do not deduplicate by model ID: retries, different revisions and measurement
settings are not interchangeable. Filter valid results for measurement queries;
retain failed rows for collection coverage. `input_candidate_source` distinguishes
curated rows from explicit coverage extensions. Indexes are rebuildable, not
completion evidence; the CSV/HDF5 pairs remain authoritative. Complete coverage
and finite fits alone do not establish scientific validity.

Current output versions are numerics **7**, loader **10**, HDF5 format **2.0**.
Layer selection **3** adds declared Swin position tables after ViT in selection
**2**, without changing spectral formulas.
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

HF's declared `SwinRelativePositionBias` is measured as its stored 2D
relative-position-by-head table, not the indexed attention bias/operator.
The row count must match the declared window; arbitrary biases, custom subclasses
and Swin V2's computed continuous-position bias do not gain this interpretation.
See the [official implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py).

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

## Collection status and remaining work

The original goal is an efficient, robust dataset/database of roughly 50,000
HF-metadata-sourced models, preserving ESD analyses and eigenvalues across many
model types. Numbers must stay attached to the correct checkpoint and weight
under recorded conventions. Task inference is not itself a requirement.

Keep the project human-readable and maintainable from the start: plain names,
small helpers, understandable commands and one clear operating workflow.
Simplify or remove redundant code, configuration and documentation before adding
machinery. Preserve useful research measurements and the local run-script
changes. A 50k-model run is **not authorized**.

### Ready for bounded collection

Use the ordinary workflow: **prepare → review → run → index → query**.

- Preparation pins model/base revisions, reviews files and access, counts adapter
  bases against the weight budget, and retains explicit holds.
- Workers preserve checkpoint integrity, canonical names, missing fits, full
  spectra and coverage. Automatic fallback is narrow; explicit checkpoint mode
  describes stored matrices, not a validated model.
- The existing supervisor handles idle-GPU checks, one worker by default, sampled
  RAM/cache caps, stage/heartbeat timeouts, signals, cleanup and compatible resume.
  Cleanup must be confirmed before reusing resources; shared caches stay untouched.
- The CSV index is rebuildable across runs. Original CSV/HDF5 pairs remain
  authoritative; spectra are read by exact name, one layer at a time.
  No database service, extra observer or separate cache manager is required.

Current defaults are a 6 GiB selected-weight budget, 16 GiB worker-tree RSS and
12 GiB private cache. Limits are configurable sampled safeguards, not OS quotas
or predictions of GPU memory. Keep cache ownership, bounded disk and shared-HPC
availability checks even when optimizing overhead.

### Evidence, not another pilot checklist

Implementation controls and historical checks live in
`analysis_runs/validation/`; collected measurements live in
`analysis_runs/phase2/`. Their manifests, reports, coverage and saved pairs retain
the detailed evidence. Use the collection catalog to inspect current outcomes
instead of maintaining a second narrative tally here.

Collected families include BERT derivatives and other encoders, dense decoders,
T5/BART encoder-decoder models, CNNs, ViT/Swin, CLIP and supported LoRA adapters.
They are small curated/family-extension batches, not a representative HF sample.
Selection 3 adds Swin's learned position tables; the original 77 Swin measurements
and spectra were unchanged when its 12 tables were added.

Wav2Vec2's strict model load remains rejected for missing
`wav2vec2.masked_spec_embed`. The separately recorded explicit checkpoint result
measures only stored floating 2D tensors; higher-dimensional convolution tensors
remain skipped. It does not repair or establish a complete speech model.

### What remains outside the completed baseline

- **Coverage:** packed quantization and unsupported/custom layouts remain explicit
  gaps. Add support when actual collection missingness justifies a declared
  interpretation; never infer packing from names or reshape unknown tensors.
- **Scale:** use actual collection costs to choose concurrency, budgets and any
  bounded adapter-base reuse. Keep ephemeral caches until reuse earns its
  complexity. Measure query/storage pressure before changing the index or HDF5.
- **Research validity:** predictive usefulness, comparisons against alternative
  distributions and deferred aspect-ratio correction are separate research work.
  A finite exponent, small KS distance, complete coverage or successful run does
  not establish heavy-tailed behavior or model quality.

Continue collection with these explicit limits. Focused verification should
protect changed contracts; more general probes are not a prerequisite for each
batch or small engineering change.
