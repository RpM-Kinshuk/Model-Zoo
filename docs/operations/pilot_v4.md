# Numerics v4 pilot: 11 September 2026

The small CPU measurement/storage pilot passed for four pinned public Hugging
Face checkpoints. It produced 153 canonical measurements, including 18 missing
fits; every layer identity and saved spectrum round-tripped exactly through
CSV/HDF5. The automated offline suite passed 368 tests (environment/GPU smoke
scripts excluded).

This is not evidence of predictive validity or readiness for an unrestricted
50,000-model run. Three checkpoints are tiny random test models. No GPU,
quantized-checkpoint, or automatic model-loader-routing validation was performed.

## Checkpoints and coverage

| Family | Public checkpoint | Measurements | Finite fits | Skipped modules |
|---|---|---:|---:|---:|
| Encoder | `hf-internal-testing/tiny-random-BertModel` | 34 | 34 | 11 |
| Decoder | `sshleifer/tiny-gpt2` | 14 | 13 | 6 |
| Encoder–decoder | `hf-internal-testing/tiny-random-T5ForConditionalGeneration` | 85 | 68 | 28 |
| CNN | `hf-internal-testing/tiny-random-ResNetModel` | 20 | 20 | 20 |

Measurements count emitted Q/K/V slices separately; skipped counts are modules.
Skips in this pilot were one-dimensional normalization weights and two LM heads
excluded by the retained legacy aspect-ratio rule. Tiny GPT2 had one
insufficient-positive-spectrum fit; tiny T5 had 17 spectra entirely below the
absolute filter. Their canonical records and full spectra remained present.

No missing or mismatched model weights were reported during loading. GPT2's
checkpoint contained two historical `masked_bias` buffers not used by the
current model class; these are recorded in the loading report.

## Precision comparison

The reference was local WeightWatcher 0.8.8, commit `a5940f7`, using its accurate
SVD on explicitly supplied float64 arrays. Its automatic tensor adapter, legacy
KS convention, and entropy implementation were not used as equality targets.

For first/middle/last eligible layers in each checkpoint (12 measurements), the
largest spectral discrepancy was:

| Computation | Maximum error |
|---|---:|
| Float64 SVD | 1.38e-14 |
| Float32 SVD | 1.10e-6 |
| Float32 Gram | 7.40e-6 |
| Forced float16 round trip, then float32 SVD | 2.79e-4 |

Error means `max(abs(computed_eigenvalues - reference_eigenvalues)) / reference_lambda_max`;
it is not a per-eigenvalue relative error or an alpha-error bound. The sample is
too small to estimate prevalence across HF. Seeded Pareto, lognormal, Gaussian,
near-constant and low-rank controls are also included in the detailed report.

## Artifacts and reproduction

- [Detailed JSON report](../../analysis_runs/validation/hf-pilot-v4-20260911-LSENel/pilot_report.json)
- [Pilot script](../../esd_experiment/scripts/measurement_pilot.py)
- [Measurement conventions and CLI controls](analysis.md)

The JSON includes full checkpoint commit SHAs, measurement source-file hashes,
runtime versions, skip reasons, fit statuses, per-layer comparisons, and paths
to CSV/HDF5 files. The four config/weight downloads total 7,495,424 bytes. Pilot
outputs and caches are local generated artifacts, excluded from Git.

To reproduce using the existing local cache and a **new** output directory:

```bash
python esd_experiment/scripts/measurement_pilot.py \
  --output-dir analysis_runs/validation/pilot_v4_repeat \
  --cache-dir analysis_runs/validation/hf-pilot-20260911-WK0XBZ/cache \
  --weightwatcher-path ../WeightWatcher --offline
```

Without that cache, omit `--cache-dir` and `--offline`. Downloads remain bounded
to the pinned config/weight files; model loading uses explicit built-in classes,
local files only, and disabled remote code. The production loader's existing
remote-code policy is unchanged by this pilot.

The next validation step is a representative GPU pilot with realistic matrix
sizes, checkpoint-loading audits, custom/quantized layouts, and threshold/cutoff
sensitivity. Aspect-ratio correction remains deferred.
