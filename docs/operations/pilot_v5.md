# Numerics v5 / loader v1 pilot: 11 September 2026

This is the historical loader v1 report. The subsequent
[loader v2 pilot](pilot_loader_v2.md) adds adapter integrity checks and actual
pre-quantized checkpoint probes; it supersedes the adapter limitation below.

The bounded follow-up passed after three narrow fixes: preserve checkpoint
architecture/weights, use precision-focused CUDA SVD, and report packed recurrent
modules as skipped. It does **not** validate an unrestricted 50,000-model run.
The final offline suite passed **439 tests** (18 upstream deprecation warnings;
environment/GPU smoke scripts excluded). Real GPU checks ran separately.

Environment: `conda activate esd_ind`, PyTorch `2.11.0+cu128`, Transformers
`5.9.0`, SciPy `1.15.3`, and NVIDIA L40 GPUs **5–7**. With
`CUDA_VISIBLE_DEVICES=5,6,7`, logical CUDA indices 0/1/2 mean physical 5/6/7.
The sandbox hid the driver; actual GPU runs required execution outside it.
No environment packages, local launch script, or sibling WeightWatcher files
were changed. No new checkpoint downloads were needed.

## Bugs found and corrected

1. The automatic loader chose `BertLMHeadModel` for a checkpoint declaring
   `BertModel`. It discarded the pooler and initialized six head tensor keys.
   Both routes still produced 34 measurements, so counts did not detect this.
   Loading now honors compatible built-in architecture declarations and rejects
   missing/mismatched weights, errors, and unexplained unexpected keys.
   Historical GPT2 `masked_bias` buffers are narrowly permitted and recorded.
   This follows the distinction between base and task-specific classes in the
   [Transformers Auto Classes documentation](https://huggingface.co/docs/transformers/v5.9.0/en/model_doc/auto).

2. Default float32 CUDA SVD exceeded the predeclared spectral-error tolerance
   on well-conditioned controls. Direct same-input driver comparisons isolated
   the difference:

   | Matrix | Default Jacobi error | QR `gesvd` error |
   |---|---:|---:|
   | 768 × 768 | 2.35e-4 | 1.49e-7 |
   | 2048 × 2048 | 4.34e-4 | 1.87e-7 |

   CUDA now explicitly uses `gesvd`, including Gram-to-SVD fallback. CPU
   dispatch is unchanged. PyTorch documents this driver choice when precision
   matters and describes Jacobi as the default.
   [PyTorch 2.11 SVD documentation](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.svd.html)

3. Actual dynamic-quantized LSTM/GRU modules were absent from coverage because
   their weights live in packed storage. They now appear once per owning
   module as explicitly skipped; accessors are not invoked and storage helpers
   are not counted as additional learned layers.

The output convention is now `numerics_version="5"`, `loader_version="1"`,
with format `2.0` unchanged. Resume rejects older artifacts without deleting
them. Use a fresh output directory; this is not an in-place migration.

## Final measurements

All errors below mean
`max(abs(eigenvalues - reference)) / reference_lambda_max`, **not** relative
error of each small eigenvalue, an alpha-error bound, or a quality score.

The independent matrix reference was SciPy float64 SVD of identical float32
source weights. Seeded cases included 768² and 2048² projections, a 3072×768
MLP matrix, a 4096×256 embedding, low-rank/ill-conditioned controls, a nearly
constant spectrum, and an absolute-filter boundary.

| Execution | Cases / variant measurements | Worst float32 SVD error | Worst any variant |
|---|---:|---:|---:|
| CPU | 8 / 39 | 7.63e-6 | 1.37e-5 |
| GPU 5 | 8 / 39 | 2.58e-7 | 8.58e-5 |
| GPU 6 | 1 / 6 | 1.49e-7 | 3.03e-7 |
| GPU 7 | 1 / 6 | 1.49e-7 | 3.03e-7 |

All applicable `1e-4` spectrum gates passed. Degenerate/ill-conditioned cases
are sensitivity diagnostics, not blanket pass/fail equality tests. The GPU 5
worst variant was Gram on the embedding matrix. GPU allocator peak was about
173 MiB; CUDA-process host RSS included about 1.34 GiB of library/runtime
overhead. These measurements are not model-scale memory or throughput estimates.

The same [four pinned public checkpoints](pilot_v4.md#checkpoints-and-coverage)
were loaded through the **production automatic loader** on CPU and GPU.
Their classes, all 349 state tensors, and ordinary shared-Parameter ties matched
explicit built-in baselines exactly. Each run saved 153 measurements: 135
finite fits and 18 missing fits. Full canonical identity and eigenvalue arrays
round-tripped through CSV/HDF5 exactly.

The GPU run loaded on physical GPU 6 and dispatched measurements to GPUs 5/7.
First/middle/last representative **saved** spectra and independent reruns were
checked against local WeightWatcher 0.8.8 (`a5940f7`) accurate SVD on explicit
CPU float64 arrays. Neither its automatic dtype adapter nor its legacy KS and
entropy definitions were used as equality targets. Across all 153 saved
CPU/GPU pairs, identities and fit statuses matched; maximum spectral discrepancy
relative to the CPU float32 maximum was 1.54e-6 and maximum relative alpha
difference was 2.55e-5. This sample does not establish backend equivalence for
arbitrary spectra. Final report source hashes matched the executed code.

## Limits that remain visible

- Float32 Gram can inflate raw numerical rank: the rank-32 repeated-row control
  reported 80 on CPU and 84 on GPU, versus 32 with SVD. Default filtering hid
  that difference in retained metrics. Keep SVD as the default; tiny-eigenvalue
  rank/entropy and nearly constant-tail alpha remain sensitive. Gram
  jitter/fallback is not yet recorded per layer.
- CPU probes exercised real torch INT8, bitsandbytes NF4, and torchao INT8
  representations, including locally quantizing the cached BERT checkpoint.
  Packed weights were safely skipped. Explicit backend dequantization yielded
  spectra of **reconstructed quantized weights**, not original float32 weights;
  no automatic dequantization was added. These probes were not pre-quantized
  HF repository or GPTQ/AWQ/adapter execution validation.
- The loader gate relies on Transformers' loading report. Adapter provenance
  explicitly verifies only base loading. Shared-Parameter tie checks do not
  cover distinct Parameters sharing storage. Mixed-dtype/native checkpoint
  preservation, remote-code policy, and process-backend GPU execution need
  separate checks.
- The small checkpoints and seeded matrices do not establish predictive
  validity across model families. A finite exponent or low fitted KS distance
  is not evidence of power-law behavior or generalization. Aspect-ratio
  correction remains deferred.

## Reports and reproduction

Generated reports are local, ignored artifacts; these scripts are tracked:

- [Automatic-loader CPU report](../../analysis_runs/validation/checkpoint_v5_auto_cpu_20260911_final/pilot_report.json)
- [Automatic-loader threaded GPU report](../../analysis_runs/validation/checkpoint_v5_auto_gpu_20260911_final/pilot_report.json)
- [Matrix CPU](../../analysis_runs/validation/backend_v5_cpu_20260911_final/backend_report.json),
  [GPU 5](../../analysis_runs/validation/backend_v5_gpu5_20260911_final/backend_report.json),
  [GPU 6](../../analysis_runs/validation/backend_v5_gpu6_20260911_final/backend_report.json),
  [GPU 7](../../analysis_runs/validation/backend_v5_gpu7_20260911_final/backend_report.json)
- [Direct driver comparison](../../analysis_runs/validation/backend_driver_comparison_20260911.json)

Use fresh output paths for each repeat:

```bash
conda activate esd_ind
CUDA_VISIBLE_DEVICES=5,6,7 python esd_experiment/scripts/measurement_pilot.py \
  --loader auto --device cuda:1 --parallel-esd --offline \
  --cache-dir analysis_runs/validation/hf-pilot-20260911-WK0XBZ/cache \
  --weightwatcher-path ../WeightWatcher \
  --output-dir analysis_runs/validation/my_checkpoint_v5

CUDA_VISIBLE_DEVICES=5,6,7 python esd_experiment/scripts/backend_pilot.py \
  --device cuda:0 --output-dir analysis_runs/validation/my_matrix_v5
```

For CPU, select `--device cpu` and omit `--parallel-esd`. Without the existing
checkpoint cache, omit `--offline`/`--cache-dir`; downloads remain bounded to the
pinned ~7.5 MB config/weight files. Requested CUDA must be accessible. The matrix
pilot runs each case in a subprocess with a default 180-second timeout.
