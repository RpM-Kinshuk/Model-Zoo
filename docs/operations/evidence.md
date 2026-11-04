# Weight evidence

Read checkpoint weights without loading or running a model. Compare measured
weights for lineage; use T2 when you need full spectra.

## Quick start

Use the environment described in the [README](../../README.md). Run these
commands from the Model-Zoo root. Set `ZOO_DATA` to your data directory and
supply `HF_TOKEN` through the environment when access requires it.

```bash
python -m evidence.cli fingerprint org/model --tier t1 --out "$ZOO_DATA/example" --json
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

## Choose a tier

| Tier | Reads | Stores |
| --- | --- | --- |
| T0 | Metadata, configs and tensor headers | Pinned manifest, shapes, formats and coverage |
| T1 | One deterministic row block per view, plus needed scales/factors | Sketch, typed block hash, block spectrum and optional parent delta |
| T2 | Full supported tensors | Full spectra, fingerprint and requested free T1 blocks |

T1 defaults: 256 rows, sketch size 64, seed 20261028. Larger sketches contain
the smaller sketches as prefixes. T1 bytes vary with model shape and encoding;
the pilot measures the cost rather than assuming a fixed saving.

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
| `conv_flat` | Higher-dimensional weight; T2 keeps the collector's convolution convention |
| `expert_slice` | Individual expert matrix, with orientation resolved from config |
| `lora_product` | Scaled LoRA update from paired factors |

LoRA supports recognized PEFT, kohya, diffusers and MLX factor names. It measures
the update, rather than claiming the adapter is a full model. MoE defaults to
eight deterministic experts per layer; use `--all_experts` for all experts.
Diffusers view names include the component folder to prevent collisions.
GPTQ, AWQ, bitsandbytes, MLX packed quantization and GGUF remain recorded gaps.
`.bin`/ONNX-only checkpoints are T0-only; pickle weights are never loaded.

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
| `<safe>.t1.fingerprint.safetensors` | T1 measurements |
| `<safe>.t2.fingerprint.safetensors` | T2 measurements |
| `<safe>.t1b128.fingerprint.safetensors` | Extra T1 block height; similarly for 512 |
| `<safe>.t2.spectra.h5` | Full per-view spectra, retained for the final database |
| `summary.json`, `failures.csv` | Current per-model results and historical failures |

Fingerprints store `sketch [n,k,k]`, `rows [n,2]`, `shape [n,2]`, delta scalars
and top singular values. T1 also stores concatenated `spectrum_values` and
`spectrum_offsets [n+1]`. Metadata holds names, geometry, components, hashes,
coverage, pinned identity, settings and measured counters. Hashes include dtype;
derived views use decoded F32 values.

Spectra are sorted **squared singular values**. T2 HDF5 stores them in `/eigs`,
aligned with `/layers/longname`; `/layers` also holds fit results. Attributes
record format, measurement settings and numerics version 7. Keep the T2
fingerprint and spectra file together. The database's `spectra_t2` field points
to the full-spectrum artifact relative to its data root.

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

- `t2-models.csv`: lab models, certified independent pretrains, Hub related
  pairs, and ten models per stratum above 9B.
- `t1-models.csv`: the remaining Hub pilot models.

Use the lab's extractor-input export for local paths and pinned Hub references.
Create `all-models.csv` as the unique union of these inputs. Planned lab rows
are not completed checkpoints.

### 2. Pin manifests, including direct declared bases

```bash
python -m evidence.cli manifest --models "$RUN/all-models.csv" \
  --out "$RUN/manifests" --workers 2 --include_declared_bases
```

### 3. Measure T2, then the remaining T1 models

Use one combined T2 input so its current summary describes the whole requested
spectral cohort. Free T1 companions are written beside T2 outputs.

```bash
python -m evidence.cli batch --models "$RUN/t2-models.csv" \
  --manifests "$RUN/manifests" --tier t2 --out "$RUN/t2" --workers 2

python -m evidence.cli batch --models "$RUN/t1-models.csv" \
  --manifests "$RUN/manifests" --tier t1 --out "$RUN/t1" --workers 2 \
  --sketch_k 64
```

Batch runs compute declared-parent deltas on 256-row blocks by default. T2 also
writes T1 companions at 128, 256 and 512 rows; full-matrix delta status is
`not_computed`. Use `--delta_blocks 256,512` for additional measured heights,
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
--sketch_k 64 --no-all_experts --also_t1_blocks none`.

Resume with the same inputs, settings and output directories. Compatible outputs
are skipped; temporary files are cleaned up. Denied identities stay skipped
unless `--retry_denied` is supplied; retries preserve resolved revision pins.

### 4. Write the G0 report and calibration

Set `LAB_TABLES` to the directory containing the lab's `edges.csv`, `models.csv`
and certified reference files. Track B's file handoff can be used before merge.

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
write_calibration(pairs, run / "t1", run / "calibration.json", sketch_k=64)
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
visible GPUs. T1 moves sketches, block spectra and parent deltas to the selected
device while hashes retain the original host bytes. CUDA row-block reads use a
two-group prefetch window, bounded by the prefetch byte budget.

For T2, the source loader also fetches the next group's requested parent row
blocks, including cache hits, while the current group's spectra are measured.
The per-group admission budget includes child sources and parent read/decode
workspaces. Groups exceeding that budget use on-demand reads. Explicit lab
full-matrix deltas stay on demand and are never cached.
