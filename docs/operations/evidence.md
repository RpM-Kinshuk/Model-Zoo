# Weight evidence

Read checkpoint weights without loading or running a model. Compare measured
weights for lineage; use `full_weights` when you need full spectra.

## Quick start

Use the environment described in the [README](../../README.md). Run these
commands from the Model-Zoo root. Set `ZOO_DATA` to your data directory and
supply `HF_TOKEN` through the environment when access requires it.

```bash
python -m evidence.cli fingerprint org/model --measurement_mode row_blocks --out "$ZOO_DATA/example" --json
python -m evidence.cli diff org/child org/parent --json
python -m evidence.cli check org/child --null "$ZOO_DATA/runs/pilot/calibration.json"
```

A target can be `org/repo@revision` or a local directory. Hub revisions resolve
to commit pins; local directories get content pins. Named lab checkpoints use
`lab/<model_id>` under `ZOO_DATA`. Explicit local paths also work outside it.

`diff` reports measured changes, shared-view cosine and available parent deltas.
Without compatible calibration, relatedness stays `uncalibrated`.
`check` uses the first declared base: exit **0** for consistent, undetermined or
uncalibrated; **1** for inconsistent; **2** for errors. Other commands use 0/2.

## Choose the measurement scope

| Evidence scope | Reads | Stores |
| --- | --- | --- |
| `metadata_only` | Metadata, configs and tensor headers | Pinned manifest, shapes, formats and coverage |
| `row_blocks` | One deterministic row block per view, plus needed scales/factors | Sketch, typed block hash, block spectrum and optional parent delta |
| `full_weights` | Full supported tensors | Full spectra, fingerprint and requested row-block companions |

Row-block defaults: 256 rows, sketch size 64, seed 20261028. Larger sketches
contain the smaller sketches as prefixes. Bytes read for row blocks vary with
model shape and encoding; the pilot measures the cost rather than assuming a
fixed saving.

The fingerprint and batch CLI accepts `--measurement_mode row_blocks|full_weights`;
`metadata_only` describes manifests and coverage, rather than a tensor measurement
mode. Use `--also_row_block_sizes 128,256,512` to select companion heights for
full tensor measurements. Generic `tier` metadata and evidence-tier fields retain
their field names and carry these descriptive values.

## Supported views

Each view has two independent choices: how to **decode** stored values, and
which **matrix** those values describe.

| Decode | Supported storage |
| --- | --- |
| `plain` | Floating tensors, including BF16/F16/F32 |
| `fp8_scaled` | FP8 with recognized scalar, row or block scales |
| `mxfp4` | Recognized MXFP4 blocks, including supported DeepSeek layouts |
| `nvfp4` | Packed FP4 with block and global scales |

| Geometry | Matrix measured |
| --- | --- |
| `matrix` | Ordinary 2D tensor; fused projections remain fused |
| `head` | Small output/classification matrix |
| `conv_flat` | Higher-dimensional weight; full tensor measurements keep the collector's convolution convention |
| `expert_slice` | Individual expert matrix, with orientation resolved from config |
| `lora_product` | Scaled LoRA update from paired factors |

LoRA supports recognized PEFT, kohya, diffusers and MLX factor names. It measures
the update, rather than claiming the adapter is a full model. MoE defaults to
eight deterministic experts per layer; use `--all_experts` for all experts.
Diffusers view names include the component folder to prevent collisions.
GPTQ, AWQ, bitsandbytes, MLX packed quantization and GGUF remain recorded gaps.
`.bin`/ONNX-only checkpoints provide metadata only; pickle weights are never loaded.

### Coverage reasons

Every tensor keeps a coverage record. A skipped tensor is not a zero matrix.

| Skip reason | Meaning |
| --- | --- |
| `not_a_matrix` | Scalar, vector, empty tensor or auxiliary value |
| `vector_like` | Matrix width below the supported minimum |
| `non_float_buffer` | Integer or Boolean buffer |
| `unsupported_representation` | Unrecognized or unsupported weight encoding |
| `unsupported_adapter_type` | Unsupported adapter, such as DoRA/LoHa/LoKr |
| `fp8_block_size_unknown` | FP8 scale grid cannot be resolved safely |
| `expert_not_sampled` | Expert outside the deterministic sample |

`decode_scale` and `adapter_alpha` mark values consumed by another view.
`lora_scale_unknown` marks a measured LoRA product with uncertain scaling.
These are coverage notes, not additional skips.

Manifest file exclusions are `duplicate_layout`, `precision_variant`,
`bundled_adapter` and `pickle_not_read`. Batch reports also distinguish denied
access (`no_access`), missing manifests and extraction errors.

## Stored artifacts

`<safe>` is the filename-safe `repo_id@revision` identity.

| File | Contents |
| --- | --- |
| `<safe>.manifest.json` | Selected files, tensor offsets, config facts and coverage |
| `<safe>.row_blocks.fingerprint.safetensors` | Sampled row-block measurements |
| `<safe>.full_weights.fingerprint.safetensors` | Full tensor measurements |
| `<safe>.row_blocks_128.fingerprint.safetensors` | Extra row-block height; similarly for 512 |
| `<safe>.full_weights.spectra.h5` | Full per-view spectra, retained for the final database |
| `summary.json`, `failures.csv` | Current per-model results and historical failures |

Fingerprints store `sketch [n,k,k]`, `rows [n,2]`, `shape [n,2]`, delta scalars
and top singular values. Row-block fingerprints also store concatenated
`spectrum_values` and `spectrum_offsets [n+1]`. Metadata holds names, geometry, components, hashes,
coverage, pinned identity, settings and measured counters. Hashes include dtype;
derived views use decoded F32 values.

Spectra are sorted **squared singular values**. Full tensor HDF5 files store
them in `/eigs`, aligned with `/layers/longname`; `/layers` also holds fit results.
Attributes record format, measurement settings and numerics version 7. Keep the
full tensor fingerprint and spectra file together. The database's
`full_weight_spectra` field points to the full-spectrum artifact relative to its
data root.

The model table uses `row_block_fingerprint`, `full_weight_fingerprint` and
`full_weight_spectra` for artifact links. Readers translate legacy numbered
values, filenames and columns at the compatibility boundary. Historical run
paths remain unchanged; use their recorded names when rebuilding those runs.

`HF_TOKEN` stays in the environment, never in manifests, configs or reports.
Raw weight blocks and parent caches are never published. Retain fingerprints,
full spectra and provenance; keep caches disposable.

## Pilot runbook

The owner launches the large batches after reviewing inputs and storage.
Set `HUB_STATS` to the path of the pinned hub-stats Parquet file already on disk.

### 1. Draw the Hub sample

```bash
RUN="$ZOO_DATA/runs/pilot"
mkdir -p "$RUN"
python -m evidence.cli frame --hub_stats "$HUB_STATS" --out "$RUN/frame.csv"
python -m evidence.cli sample --frame "$RUN/frame.csv" --plan pilot \
  --seed 20261028 --out "$RUN/models.csv"
```

Prepare two input CSVs with `repo_id,revision`; preserve frame fields such as
`stratum`, `params` and `inclusion_prob`:

- `full_weights-models.csv`: lab models, certified independent pretrains, Hub related
  pairs, and ten models per stratum above 9B.
- `row_blocks-models.csv`: the remaining Hub pilot models.

Use the lab's extractor-input export for local paths and pinned Hub references.
Create `all-models.csv` as the unique union of these inputs. Planned lab rows
are not completed checkpoints.

### 2. Pin manifests, including direct declared bases

```bash
python -m evidence.cli manifest --models "$RUN/all-models.csv" \
  --out "$RUN/manifests" --workers 2 --include_declared_bases
```

### 3. Measure full tensors, then sample the remaining models

Use one combined input for full tensor measurements so its summary describes
the requested spectral cohort. Row-block companions are written beside the full tensor outputs.

```bash
python -m evidence.cli batch --models "$RUN/full_weights-models.csv" \
  --manifests "$RUN/manifests" --measurement_mode full_weights --out "$RUN/full_weights" --workers 2

python -m evidence.cli batch --models "$RUN/row_blocks-models.csv" \
  --manifests "$RUN/manifests" --measurement_mode row_blocks --out "$RUN/row_blocks" --workers 2 \
  --sketch_k 64
```

Batch runs compute declared-parent deltas on 256-row blocks by default. The
`full_weights` mode also writes row-block companions at 128, 256 and 512 rows;
full-matrix delta status is `not_computed`. Use `--delta_blocks 256,512` for additional measured heights,
or `--full_delta` for lab validation (full deltas are never cached).

The private parent cache defaults to `$ZOO_DATA/cache/parent-blocks`, or
`<out>/cache/parent-blocks`, with a 64 GiB LRU budget. Blocks keep their stored
dtype; decoded and LoRA blocks use fp32. Declared bases, lab and anchor rows are
cached, plus rows added by `--cache_models CSV`. Parent blocks are never published.
Set `--parent_cache PATH`, `--parent_cache_max_gib N`, or `--no-parent_cache`.
Lab and anchor rows automatically use k=128 and all experts; other rows use k=64
and eight seeded experts per layer. Explicit `--sketch_k`, `--all_experts` and
`--no-all_experts` override this choice.

To restore the former batch behavior, pass `--no-with_parent --no-parent_cache
--sketch_k 64 --no-all_experts --also_row_block_sizes none`.

Resume with the same inputs, settings and output directories. Compatible outputs
are skipped; temporary files are cleaned up. Denied identities stay skipped
unless `--retry_denied` is supplied; retries preserve resolved revision pins.

### 4. Write the G0 report and calibration

Set `LAB_TABLES` to the directory containing the lab's `edges.csv`, `models.csv`
and certified reference files. Use the main Zoo-Keeper lab tables and lab/REFERENCE-LABELS.md; no separate track checkout or merge is needed.

```bash
export RUN LAB_TABLES
python - <<'PY'
import os
from pathlib import Path
from evidence.pilot_analysis import lineage_pairs, write_calibration, write_g0_report

run = Path(os.environ["RUN"])
lab = Path(os.environ["LAB_TABLES"])
pairs = lineage_pairs(lab / "edges.csv", run / "manifests")
pairs.to_csv(run / "pairs.csv", index=False)
write_g0_report(run, run / "g0.md")
write_calibration(pairs, run / "row_blocks", run / "calibration.json", sketch_k=64)
PY
```

Keep the pair table's `source` column: it preserves missing lab mappings.
The independent null uses certified reference pairs, excluding known siblings;
`kind=independent` alone is insufficient. Hub declarations are sanity checks.

Read `g0.md` before deciding the gate. Missing measurements are unavailable,
rather than evidence of scientific failure. Incomplete inputs cannot produce
calibration. False-match results include actual pool size; a small pool cannot
certify the 5k-pool target. Sorted-spectrum rank agreement is a weak diagnostic,
not proof of spectral-shape agreement. Calibration is empirical and makes no
per-query error-control claim.

## Verify

```bash
MODEL_ZOO_NETWORK_TESTS=0 python -m pytest evidence/tests esd_experiment/tests -q
MODEL_ZOO_NETWORK_TESTS=1 python -m pytest evidence/tests/test_hub.py::test_network_smoke -q
python evidence/scripts/day_one_checks.py
```

The network smoke reads a pinned header. The last command reports HTTPS/Xet
access, CPU SVD time and free data space; it requires `HF_TOKEN` and `ZOO_DATA`.

### Evidence compute devices and worker counts

`--compute_device auto` uses only GPUs assigned through `CUDA_VISIBLE_DEVICES`;
with no assignment (or an empty assignment), it uses CPU. Explicit `cpu`, `cuda`
and `cuda:<i>` override this. Assigned or explicitly requested CUDA that does not
work is an error, never a silent CPU fallback. Device indices are relative to
the visible assignment.

`--workers auto` starts one process per assigned GPU, or on CPU
`min(8, max(1, cpu_count // 8))`. An explicit worker count is distributed
round-robin over the selected GPUs. `cuda:<i>` selects one GPU; `cuda` uses all
visible GPUs. Row-block measurement moves sketches, block spectra and parent
deltas to the selected device while hashes retain the original host bytes. CUDA row-block reads use a
two-group prefetch window, bounded by the prefetch byte budget.

For full tensor measurement, the source loader also fetches the next group's
requested parent row blocks, including cache hits, while the current group's
spectra are measured.
The per-group admission budget includes child sources and parent read/decode
workspaces. Groups exceeding that budget use on-demand reads. Explicit lab
full-matrix deltas stay on demand and are never cached.

### Full-mode sidecar API

`evidence.sidecar.EvidenceSidecar` measures evidence in a background thread using
the worker's local Hub cache. `finish` reuses a runner spectrum only for an
unambiguous whole matrix with the same stored key, shape, dtype and exact
256-row block bytes. Other views are measured in a bounded second round.
The spectra file records `runner` or `second_round` for every layer.

All outputs remain in a private staging directory until completion. Spectra
publish before the full-weight completion marker; cancellation and failed promotion
remove new outputs and restore any files from an earlier incomplete attempt.
A completed bundle is never replaced. Use one sidecar per model output directory.
Full-mode runner workers enable the sidecar by default; `--no-evidence` disables it.


New evidence writes version 3 with descriptive scope values and filenames.
Versions 1, 2 and 3 remain readable, including older numbered artifact names.
Version 2 introduced explicit unrequested/unavailable parent-delta statuses.
Sketch, block-hash and spectrum numerics remain unchanged. Candidate-pair deltas
and their CLI are covered in the [zoo guide](zoo.md#candidate-pair-delta-pass).


Parent cache operations publish complete files atomically. Hot reads do not
wait for the eviction writer; unavailable/stale cache files are misses. Writers
share a small byte-count record under the existing lock, so ordinary inserts
avoid directory scans. The record is invalidated before mutation; an interrupted
write, unreadable record or changed directory membership triggers a recount.
Block files are private: do not modify their contents in place outside the cache.
On overflow, one LRU scan reclaims about 5% headroom when block sizes permit,
amortizing scans across later inserts. A recovery/eviction scan remains linear
(and eviction sorting costs O(N log N)); writers still serialize. Use local
scratch for throughput; this adds no cross-node locking guarantee.
The Nov5 audit measured same-host cross-process locking on the actual NFS4 mount;
cross-node locking remains unverified. See the
[gap audit](../../analysis_runs/validation/gap_audit_20261105/REPORT.md).
