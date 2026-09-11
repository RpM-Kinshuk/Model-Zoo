# Loader v2: bounded adapter and pre-quantized pilot

11 September 2026. Adapter integrity is now checked before merging; spectral
definitions remain **numerics v5**, with HDF5 format **2.0**. Resume requires
**loader v2**, so use a fresh directory instead of mixing previously accepted
adapter results. This work did not launch the 50,000-model experiment.

## What changed

The offline audit reproduced missing LoRA tensors accepted with warnings,
ignored surplus tensors, and an extra `base_layer.weight` that overwrote an
already-verified base weight. Non-finite adapters also reached merged models.
The new gate derives the configured adapter manifest through PEFT's public
serializer, checks exact keys/shapes and finiteness before applying tensors,
verifies the loaded values after recorded dtype conversion, and uses
`merge_and_unload(safe_merge=True)`.

Base and adapter revisions are now separate, and dense adapter bases retain
their declared architecture. Conflicting `auto_mapping` classes fail. Provenance
records the checked base, adapter config/tensor hashes, counts, dtypes, installed
loading-library versions, and verification scope. Hashing streams adapter
tensors, not a cloned dense base; safe merging still needs temporary storage
for the current merged layer.

Supported integrity scope is ordinary/RSLoRA **matrix** adapters, including
GPT2's transposed Conv1D layout, configured saved heads, and `bias="all"`.
DoRA, other PEFT types, `bias="lora_only"`, embedding/convolution LoRA,
unconfigured embedding dumps, and unvalidated topology/initializer variants fail explicitly. This is a
deliberately narrower claim than arbitrary PEFT support.
Validation covers materialized CPU/GPU bases, not Accelerate CPU/disk offloading
or meta tensors. The existing quantized-to-dense upstream substitution remains
outside the adapter pilot; its requested/loaded base references are distinct in
provenance and must not be interpreted as equivalent quantized measurements.

The independent merge controls use `W + (alpha/r) B A`, transposed for
`fan_in_fan_out`; RSLoRA controls use `alpha/sqrt(r)`. The frozen-base/low-rank
update distinction follows the [original LoRA paper](https://arxiv.org/abs/2106.09685).
Adapter checkpoints are not complete base checkpoints; see the
[PEFT checkpoint format](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint).
Implementation behavior was checked against installed PEFT **0.18.1**, not
assumed from current documentation:
[tagged PEFT loader source](https://github.com/huggingface/peft/blob/v0.18.1/src/peft/peft_model.py).

## Public adapter controls

Both adapter/base pairs were pinned independently. Their selected files total
5,794,073 bytes before existing-cache reuse. No remote code was executed.

| Control | Adapter revision | Base revision | Purpose |
|---|---|---|---|
| `peft-internal-testing/tiny-OPTForCausalLM-lora` | `14e64b8ba522284138bfc22e76002ab6c0ce31e2` | `hf-internal-testing/tiny-random-OPTForCausalLM@0abea37ca0a786ba455967e799b7b3d67f86541f` | All ten LoRA updates are zero: preservation, **not** evidence of nonzero merge correctness. |
| `Teja00000000001/miniclay-3c7a9189` | `1acd0a065fda7601a212a8337a84042a8a19cf66` | `sshleifer/tiny-gpt2@5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` | Two nonzero updates to transposed GPT2 projections; community-uploaded fixture. |

The configs do not pin historical training-base revisions. These are explicit
pilot base pins, not proof that each adapter was trained on exactly that commit.

Both controls passed on CPU and physical **GPU 6**. All merged state tensors,
unchanged tensors, model classes, and ordinary shared-Parameter ties were
checked. Every adapted module had analyzed coverage. Each run saved **46
measurements** (44 finite fits, two missing); all identities and spectra
round-tripped through CSV/HDF5, including fitting/base metadata.

| Control | Maximum merge absolute error | Worst CPU spectrum error | Worst GPU spectrum error |
|---|---:|---:|---:|
| OPT zero-update | 0 | 1.07e-6 | 4.65e-7 |
| GPT2 nonzero-update | 1.14e-9 | 7.72e-7 | 2.11e-7 |

Every saved spectrum was compared with local WeightWatcher **0.8.8**
(`a5940f7`) accurate CPU SVD of the independently merged float64 matrix.
Spectrum error is `max(abs(eigenvalues - reference)) / reference_lambda_max`;
the fixed gate was `1e-4`. This is not an error bound for individual tiny
eigenvalues or alpha. WeightWatcher's legacy KS/entropy and automatic dtype
conversion were not equality targets.

## Actual pre-quantized repositories

Three pinned public uploads total **4,554,349 bytes** of config/weights. They
ran unchanged through the production loader on physical **GPU 5**, offline
after download, each in a subprocess with a 300-second timeout.

- **FP4 positive:**
  `RichardErkhov/trl-internal-testing_-_tiny-random-LlamaForCausalLM-4bits@e8f56d61771489cb76c1703c7655afb96c6b1db2`
  loaded without missing/unexpected/mismatched keys. Coverage was **21
  candidates: one embedding analyzed, 20 explicitly skipped**. All 14 packed
  `Linear4bit` projections were skipped as unsupported representations; the
  other skips were five 1D norm weights and the existing aspect-ratio rule for
  `lm_head`. Its saved embedding spectrum passed the SciPy float64 gate
  (error `4.97e-7`) and exact HDF5 roundtrip.
- **NF4 negative:**
  `amanpatkar/tiny-random-llama-2@21e8d02fd722d940454de780a30e4ce7a57d713e`
  contains PEFT `base_layer`/LoRA keys under a vanilla Llama config, without a
  corresponding adapter declaration. Loading failed while initializing a
  missing packed byte weight (`normal_kernel_cuda` unsupported for Byte).
  This upload is not a sound native-NF4 positive control; the failure does not
  establish a general bitsandbytes defect.
- **GPTQ negative / environment-blocked:**
  `yujiepan/llama-3-tiny-random-gptq-w4@3ac28aac2b279d11e8e8c34844dfd3dab25686e9`
  has empty packed tensors and incomplete projection payloads. The active
  environment fails even earlier: the GPTQ/Optimum/Datasets import chain asks
  for the installed `fsspec` version, but malformed distribution metadata
  returns `None`, causing a version-parser `TypeError`. The imported module
  reports 2026.2.0 while a 2026.3.0 dist-info entry has no Name/Version metadata.
  Fixing that environment alone would not make this checkpoint a valid
  positive control. No shared-environment repair was attempted.

The two failed uploads produced no analysis CSV/HDF5. The combined report
therefore intentionally has `all_cases_loaded=false` and a nonzero exit status;
they are not counted as successful measurements.

A **separate diagnostic**, not production coverage, reconstructed three FP4
matrices with `bitsandbytes.functional.dequantize_4bit`. Float64 SVD agreed with
SciPy on those identical reconstructed matrices (worst error `1.26e-15`).
These are spectra of reconstructed **float16 quantized weights**, not recovered
original float32 checkpoint values. No automatic quantized-weight reconstruction
was added to the production estimator.

## Verification and reproduction

The offline regression suite passed **513 tests** (18 upstream deprecation
warnings; environment/GPU smoke scripts excluded). Actual CUDA pilots ran
separately. Runtime: `esd_ind`, PyTorch `2.11.0+cu128`, Transformers `5.9.0`,
PEFT `0.18.1`, bitsandbytes `0.49.1`, SciPy `1.15.3`, NVIDIA L40. No environment
packages, local launch script, or sibling WeightWatcher files were changed.

Generated artifacts are local/ignored; scripts and this summary are tracked:

- [Adapter CPU report](../../analysis_runs/validation/adapter_v2_cpu_20260911_final/adapter_report.json)
- [Adapter GPU 6 report](../../analysis_runs/validation/adapter_v2_gpu6_20260911_final/adapter_report.json)
- [Quantized GPU 5 report](../../analysis_runs/validation/quantized-public-v5-20260911-final-v2/quantized_checkpoint_report.json)

Use fresh output paths. First runs fetch only the small, pinned manifests:

```bash
conda activate esd_ind
CUDA_VISIBLE_DEVICES=5,6,7 timeout 300s python esd_experiment/scripts/adapter_checkpoint_pilot.py \
  --device cuda:1 --weightwatcher-path ../WeightWatcher \
  --output-dir analysis_runs/validation/my_adapter_pilot

CUDA_VISIBLE_DEVICES=5 python esd_experiment/scripts/quantized_checkpoint_pilot.py \
  --device cuda:0 --output-dir analysis_runs/validation/my_quantized_pilot
```

For adapters on CPU select `--device cpu`. For a cached repeat, supply
`--offline --cache-dir <previous_cache_directory>`. Requested CUDA never silently
falls back to CPU. Final reports record source hashes to detect changes during
execution.

Next bounded work would validate an explicit quantized reconstruction policy
and a healthy GPTQ checkpoint in a consistent environment. Mixed/native loading
precision, other PEFT variants, trained-model coverage and predictive validity
remain separate questions. This pilot does not validate a heterogeneous 50k run;
aspect-ratio correction remains deferred.
