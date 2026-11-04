# GPU fingerprint pipeline implementation plan

> Implement inline with one core implementer and one independent review, following handoff/PROMPTS.md. Leave all work uncommitted.

Goal: accelerate the authorized running R3 queue by moving matrix arithmetic to CUDA, reusing child/parent tensors across full and companion measurements, and overlapping bounded storage reads with computation.

Architecture: CPU remains responsible for authenticated reads, deterministic NumPy sketch operators, stored-byte hashes and atomic ordered writes. CUDA handles floating matrix operations with the existing precision and QR SVD driver. Keep only a small bounded source-group window, with a separate loader executor from source read threads to avoid deadlock. Default CPU callers retain their existing behavior. Full spectra remain stored.

Constraints: unchanged seeds, row bounds, storage rounding, F32 subtraction then F64 delta, 50 Sinkhorn steps, thresholds, schema, atomic publication and sorting; no randomized/truncated SVD, reduced precision, package changes, commits or reruns of completed work. No sharing mutable records/HDF5 across compute threads. Existing jobs keep their progress. GPU admissions remain paired <80% checks; GPUs6/7 remain available for B. No weight artifacts in repositories.

## Task 1 — CUDA arithmetic and bounded source prefetch
Files: evidence/delta.py, evidence/sketch.py, evidence/fingerprint.py, focused tests; optional small internal helper module.
- [x] Write and run failing focused tests for device plumbing and bounded child/parent reuse/prefetch; preserve all CPU coverage.
- [x] Implement GPU delta preprocessing/whitening/norms, GPU F64 sketches using the unchanged NumPy-generated operators, and CUDA companion spectra via the collector QR helper. Keep hashes from original stored CPU tensors. Reuse device child/parent data across all heights when practical. Keep CPU default compatible.
- [x] Prefetch the next source group while measuring the current group; reuse matching full parent sources for companions, rather than reissuing four remote reads per matrix. Bound groups/bytes and handle packed/expert/adapter layouts and exceptions cleanly. Preserve deterministic single-writer behavior.
- [x] Run focused tests, then evidence package once. Real CUDA parity and timings are performed by the lead (no hidden GPU launch by implementer).

## Task 2 — runtime deployment and evidence (lead)
- [x] Remove the runtime reader's whole-child-plus-parent <=30GiB gate where its existing 32GiB per-worker LRU already bounds file storage; stage shards independently. Verify no lock-induced deadlock or unbounded memory.
- [x] Benchmark CPU vs new CUDA arithmetic on representative full and block shapes; verify discrete results, F32 output tolerance, sketch identity and complete spectra. Measure one end-to-end local checkpoint to include loading and writing.
- [x] One independent core review, resolve actual findings; no redundant reviews/tests.
- [x] Freeze source and hashes, atomically update queued-worker rollout configuration and report; preserve active workers and completed outputs. Update live stage labels and handoff records. No merge/push.

Review focus: CPU/GPU rounding near rank thresholds, sketch operator identity, memory bounds including large expert tensors, loader shutdown on failure, unchanged parent compatibility and artifact completeness.

Validation and deviations: one final companion-device regression required a focused fix and final package rerun (266passed/1skip). GPU workspace safeguard retains CPU derived statistics for oversized full views, warns explicitly, and keeps companionsCUDA. Fair equal-warm-operator-cache end-to-end local gain2.13× supersedes initialsetup-biased3.05×; fullspectra retained. Details in runs/track-a-r3-r6-20261102.rcsy95ap/GPU_PIPELINE_REPORT.md. No commit/merge/push.
