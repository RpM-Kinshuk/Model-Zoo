# Full-Mode Evidence and Model Packages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status (2026-11-05):** Owner checkpointa796b06. Gap audit restores
supervised/parallel streaming T1 for oversized rows (one assigned GPU per job,
CPU opt-out), improves cache hot-read concurrency/scan cost and adds explicit
per-change STATUS rows. Three-checkpoint fresh CPU/GPU parity passes, including
26nonzero-rank delta views. Same-card GPU evidence contention is measured;
CPU default retained.20-model/two-GPU throughput remains unmeasured because
no pair of idle GPUs passed bounded admission; RAM/timeouts remain initial
heuristics. Details: `analysis_runs/validation/gap_audit_20261105/REPORT.md`.
Progress ledger: `.superpowers/sdd/2026-11-04-full-mode-evidence-packages/progress.md`.

**Agents: never run commit steps.** Leave changes uncommitted and record each
task's suggested message in `Zoo-Keeper/handoff/STATUS.md` (DIRECTIVES, "Git:
agents never commit"; D10.1).

**Goal:** Make each full-mode runner job write one complete model package
(pinned manifest, T2 and T1 fingerprints, full per-layer eigenvalues and alpha
metrics, declared-parent block deltas) while keeping GPUs busy across and within
models, and build the zoo database from those packages, with defaults that
already do the right thing.

**Architecture:** gputracker and `net_esd` keep their roles: whole GPUs per
model, and one thread per GPU over a model's layers. A CPU evidence sidecar in
each worker runs the existing evidence code on the worker's own downloaded
checkpoint files while the GPUs compute spectra; the two halves are paired per
matrix when results are written. Each finished model is published into a
sharded `packages/` tree, and the zoo build reads it, adding a `layers` table
for alpha and the other fit metrics.

**Tech Stack:** Python 3.10, torch 2.11, numpy, safetensors, h5py, pandas,
pyarrow, huggingface-hub 1.11, gpustat, psutil, pytest.

**Spec:**
- The decisions and the defaults table below (agreed 2026-11-04).
- `Zoo-Keeper/docs/research/2026-10-27-merged-research-plan.md` §5.3, §6.1–6.4, §13.
- `docs/superpowers/plans/2026-10-28-weight-evidence-extractor.md` (Task 6
  fingerprint layout), `2026-10-31-zoo-store-and-queries.md`,
  `2026-11-03-gpu-fingerprint-pipeline.md`.

## Decisions (2026-11-04)

1. Every model eventually goes through full mode (load, GPU spectra, alpha).
   The row-block (T1) streaming path serves preliminary studies and models too
   large to load.
2. Full-mode runs write T2 plus the free T1 blocks at 128, 256 and 512 rows.
3. Parent deltas use row blocks only (§6.3). The default height is 256; 512 is
   for lab models if the lab plan asks for it, because a b-row block cannot
   resolve LoRA rank ≥ b and the lab has rank-256 LoRAs. A full-matrix delta is
   a lab-only option, computed on the fly and never cached.
4. Deltas for declared parents are computed during the run. Deltas for parents
   found by similarity come from a separate pass after the first zoo build.
5. The parent-block cache stores blocks in their stored dtype, only for likely
   parents (declared bases, lab, anchors) and blocks read as a parent, with
   least-recently-used eviction. Parent blocks are never published.
6. The runner's checkpoint mode is test-only; evidence turns off there
   automatically, with a printed notice.
7. **Defaults are the target behavior.** A plain run produces complete packages
   with no extra flags. Every control keeps an explicit opt-out, and each
   opt-out that restores today's behavior is tested to reproduce today's
   outputs. Existing defaults that would make a full-scale run miss models
   become automatic too.

## Defaults

This table is the single reference for defaults. Tasks implement and test the
rows they own.

| Control | Default | Override or opt-out | Task |
| --- | --- | --- | --- |
| `--evidence` (runner, worker) | on | `--no-evidence` | 4 |
| `--evidence_device` | `cpu` (overlaps the GPU spectra) | `gpu` | 4 |
| `--evidence_max_inflight_gib` | 2 | number | 4 |
| Declared-parent deltas (`--with_parent` in `evidence.cli batch`) | on | `--no-with_parent` | 1 |
| `--delta_blocks` | `256` | comma list, such as `256,512` | 1 |
| `--full_delta` | off (lab validation only) | `--full_delta` | 1 |
| Parent cache | on, at `$ZOO_DATA/cache/parent-blocks`, else `<out>/cache/parent-blocks`; 64 GiB | `--parent_cache PATH`, `--parent_cache_max_gib N`, `--no-parent_cache` | 1 |
| Likely parents cached | declared bases of the run, plus rows whose `stratum` is `lab` or `anchor` | `--cache_models CSV` adds rows | 1 |
| `--sketch_k` | `auto`: 128 when `stratum` is `lab` or `anchor`, else 64 (§6.3) | integer | 1 |
| `--all_experts` | `auto`: every expert for `lab` and `anchor`, else 8 seeded per layer (§5.3) | `--all_experts`, `--no-all_experts` | 1 |
| `--also_t1_blocks` (T2 only) | `128,256,512` | `none` or a comma list | 1 |
| `--compute_device` (`evidence.cli`) | `auto`: the GPUs in `CUDA_VISIBLE_DEVICES` when it is set, else CPU | `cpu`, `cuda`, `cuda:<i>` | 2 |
| `--workers` (`evidence.cli`) | `auto`: one per assigned GPU on CUDA; on CPU `min(8, max(1, cpu_count // 8))` | integer | 2 |
| `cpu_threads_per_gpu` (gputracker) | `auto`: `cpu_count // physical GPUs on the node`, at least 1 | integer; `0` leaves thread variables unset | 5 |
| `--publish` | on, to `--packages_root` = `$ZOO_DATA/packages`, else `<output_dir>/packages` | `--no-publish` | 6 |
| `zoo build` inputs | `--packages $ZOO_DATA/packages`, current conventions | `--manifests`/`--fingerprints`, `--conventions ID` | 6 |
| `zoo deltas` inputs | `--zoo $ZOO_DATA/builds/current`, `--packages $ZOO_DATA/packages`, the parent cache | explicit paths | 7 |
| GPUs per job | `auto`: from weight bytes and per-GPU memory read from gpustat, factor 1.5, up to the pool size | `--num_gpus_per_job N` fixes it | 8 |
| Download ahead | on: one model per pool GPU, at most 64 GiB staged | `--prefetch_models 0`, `--prefetch_max_gib N` | 8 |
| Models too large for the pool | T1-only package through the row-block path after the GPU queue | `--no-oversized_t1` | 8 |
| `--max_weight_gib` (today 6) | `auto`: the largest model the pool can load | number; `0` disables | 8 |
| `--max_worker_cache_gib` (today 12) | `auto` per job: selected weight bytes × 1.1 + 2 GiB | number; `0` disables | 8 |
| `--max_worker_ram_gib` (today 16) | `auto` per job: `max(16, 0.5 × weight GiB + 4)` (initial rule; tune in Task 9) | number; `0` disables | 8 |
| `--stage_timeout_seconds` | `auto`: `analyze` and `fingerprint` at `max(28800, 600 × weight GiB)`; other stages as today | explicit list | 8 |
| `--worker_cache_root` (today `/scratch/kinshuk/hf_worker_cache`) | `$MODEL_ZOO_WORKER_CACHE_ROOT`, else `/scratch/$USER/hf_worker_cache` | path | 8 |
Already at target and unchanged: `--save_eigs`, `--parallel_esd`, `--use_svd`
and `--filter_zeros` (on); `--fix_fingers xmin_mid`; `--load_dtype auto`;
`--compute_dtype float32`; `--analysis_source auto`.

Deliberately unchanged (owner decision, 2026-11-04): `--gpus [0]` and
`--max_concurrent_jobs 1`. GPU nodes are shared, so the pool and concurrency
are always passed explicitly.

## XF practical-purpose audit (2026-11-04)

Owner approved XF subject to checking fit with the practical 50k–100k+ model zoo
and querying goal. Keep the frozen identities, stored-dtype hashes, nested
sketches, reproducible row coordinates, explicit coverage and separate measured
scores/predictions. Accept additive packages, layers and pair-delta addenda.
Bump evidence1→2 and zoo0→1; numerical algorithms, tensor layouts and existing
table columns stay unchanged. New pair files have their own version1 marker.
Legacy readers remain available; current query clients require a rebuilt zoo1.

This is a suitable evidence foundation, not proof of 100k query scalability or
performance prediction. [Model Zoos (NeurIPS2022)](https://arxiv.org/abs/2209.14764)
already released50,360 models; [modelDNA](https://arxiv.org/abs/2607.10617) already
uses sampled weight fingerprints for lineage. Our research target should be
measured practical utility on heterogeneous checkpoints: model-selection quality
under hardware/license/task constraints, lineage error and abstention, score
prediction error on held-out families, and cost per successful query. Use exact
commit joins, family-separated evaluation and explicitly labeled predictions;
no venue/novelty claim follows from artifact count alone.

The present exact retrieval and shared-tensor lineage construction still have
quadratic work within large groups, and dense pair outputs can themselves be
quadratic. Before a50k/100k launch, benchmark build memory/time, query p50/p95,
bytes/model, candidate recall, and held-out calibration. A candidate/index
optimization belongs to a follow-up driven by those measurements. This pass
removes repeated whole-run manifest scans during pinned package publication;
it does not introduce an unvalidated learned index or new inference claims.

Owner-requested workflow amendment: consolidate equivalent tests and use one
focused regression run per substantial batch. One final batch review, no
per-task review loop or repeated full-suite runs. Task8 uses the existing
fingerprint API directly for oversized T1 instead of wrapping evidence.cli.

## Global Constraints

- `net_esd/` is not modified. `compute_esd_for_weight` numerics stay at
  `NUMERICS_VERSION` 7; `FORMAT_VERSION` is unchanged.
- Sketch spec unchanged (§6.3): seed 20261028; k = 64 (128 for lab and
  anchors); T1 block b = 256 starting at `min((rows−256)//2, 1920)`; 8 seeded
  experts per layer at T1; sketches stored in fp32.
- Delta procedure unchanged (§6.3): `identical` when every element is
  bit-identical; `below_floor` when more than half are; otherwise Sinkhorn
  biwhitening (50 steps), threshold `1.1·(√rows + √cols)`, gap confidence at a
  ratio ≥ 1.5.
- Frozen names: `<safe>.manifest.json`, `<safe>.<tier>.fingerprint.safetensors`
  and `<safe>.t2.spectra.h5`, with `safe = safe_filename(f"{repo_id}@{revision}")`.
  The T2 fingerprint is published last as the completion marker.
- Devices: `auto` uses GPUs only when `CUDA_VISIBLE_DEVICES` assigns them, so a
  plain command never takes a shared node's GPUs. An explicit or assigned CUDA
  device that does not work is an error, never a silent CPU fallback.
- Defaults follow the Defaults table. Every opt-out is tested, and the opt-outs
  that restore today's behavior reproduce today's outputs byte for byte.
- Evidence on by default must not break runs without `HF_TOKEN`: the worker then
  builds the manifest from its local download, and declared-parent deltas that
  need remote reads get status `parent_unavailable`.
- Runs (DIRECTIVES D4.1): agents may run jobs that read at most 100 models'
  weights and finish within 2 hours; the person launches anything larger,
  listed under "Runs waiting for a person" in `STATUS.md`. Use the collector's
  idle-GPU checks, keep caches bounded and delete T2 scratch after each model
  (D4.4).
- These interface changes need approval at the fingerprint-format freeze (XF,
  due 2026-11-06) and a `STATUS.md` change-log row each: the package layout; the
  per-pair delta file; the `layers` table; the `not_computed` and
  `parent_unavailable` delta statuses; the `/layers/spectrum_source` dataset in
  T2 spectra files.

## Review Focus

1. **Loaded weights that differ from stored bytes** (keys renamed, tensors fused
   or converted at load, dtype casts). A runner spectrum is reused for an
   evidence view only after the view's 256-row block matches the loaded weight
   bit for bit; otherwise the view is measured in the second round. Test: Task 3
   `test_finish_remeasures_views_whose_loaded_block_differs`.
2. **Evidence failing after a long ESD.** ESD outputs are still saved, the
   failure is recorded, and the next run retries evidence without recomputing
   ESD. Test: Task 4
   `test_evidence_failure_keeps_esd_outputs_and_retries_evidence_only`.
3. **Interrupts and timeouts during the sidecar.** No partial evidence or
   package is published, temporary files are removed, and the worker group exits
   so gputracker releases the GPUs. Test: Task 4
   `test_interrupt_in_fingerprint_stage_publishes_nothing`.
4. **Several workers sharing the parent cache** (two siblings miss at once;
   eviction during a read). A duplicate fetch is allowed; a torn or wrong-sized
   read is impossible. Test: Task 1 `test_block_cache_concurrent_put_get_evict`.
5. **Automatic defaults on unusual machines** (CPU-only, no `HF_TOKEN`, one GPU,
   a shared node). A plain run still completes and never takes unassigned GPUs.
   Tests: Task 2 `test_auto_without_visible_devices_uses_cpu`; Task 4
   `test_cpu_only_worker_runs_sidecar_after_esd` and
   `test_evidence_without_token_uses_local_download`; Task 5
   `test_cpu_thread_budget_auto_divides_cores_by_physical_gpus`.

## Execution Order

- **Gate 0 (person, by Nov 6):** bring the XF items above to the freeze.
- Task 1, then Task 3, Task 4, Task 6, and Task 7 (Task 7 after XF approves the
  per-pair delta file).
- Task 2 after Task 1; it is independent of Tasks 3–8.
- Task 5 before Task 9's throughput run. Task 8 after Tasks 4, 5 and 6.
- Task 9 runs after the tasks it validates.

---

### Task 1: Row-block-only parent deltas, a bounded parent cache and target batch defaults

**Files:**
- Create: `evidence/block_cache.py`
- Modify: `evidence/fingerprint.py` (`fingerprint_model`, `_measure_prefetched`,
  `_parent_delta`; replaces `_cache_path` and `_cache_write`), `evidence/cli.py`
  (`_parser`, `main`, `_batch_job`)
- Test: `evidence/tests/test_block_cache.py`, `evidence/tests/test_fingerprint.py`,
  `evidence/tests/test_cli.py`
- Docs: `docs/operations/evidence.md` (the "Measure T2" command becomes the
  plain default command; cache notes)

**Interfaces:**
- Produces: `BlockCache(root: Path, max_bytes: int)` with
  `get(manifest: dict, view_name: str, bounds: tuple[int, int], dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor | None`
  and `put(manifest: dict, view_name: str, bounds: tuple[int, int], block: torch.Tensor) -> None`.
  Plain views are stored in their stored dtype; decoded and LoRA views in
  float32. Writes are atomic. Eviction is least-recently-used by access time.
  A missing, evicted or wrong-sized entry is a miss, never an error.
- Produces: `fingerprint_model(..., parent_cache: BlockCache | Path | None = None, delta_blocks: tuple[int, ...] = (ROW_BLOCK,), full_delta: bool = False, cache_self: bool = False)`.
  A `Path` is wrapped in a `BlockCache` with the 64 GiB default budget, so
  existing callers keep working. Deltas are computed only for measured heights
  listed in `delta_blocks`. The full-matrix T2 record gets delta status
  `not_computed` unless `full_delta`. `cache_self` writes this model's
  `delta_blocks` rows to the cache.
- Produces: the `batch` defaults and opt-outs in the Defaults table rows owned
  by Task 1 (`--with_parent/--no-with_parent`, `--delta_blocks`, `--full_delta`,
  `--parent_cache`, `--parent_cache_max_gib`, `--no-parent_cache`,
  `--cache_models`, `--sketch_k auto`, `--all_experts/--no-all_experts` with
  `auto`, `--also_t1_blocks`). The `auto` rules read the models CSV `stratum`
  column when present; without it, rows get the census values.

- [ ] **Step 1: Write the failing tests**
  - `test_t2_with_parent_has_no_full_matrix_delta`: every T2 record's
    `delta_status` is `not_computed`; the 256-row T1 file has `ok`,
    `below_floor` or `identical`.
  - `test_t2_parent_reads_only_row_blocks`: with a counting fake reader, parent
    bytes read ≤ the sum over views of `256 × cols × itemsize`, plus headers.
  - `test_parent_cache_holds_stored_dtype_row_blocks_only`: for a bf16 parent,
    cache bytes equal its 256-row blocks at 2 bytes per element.
  - `test_stored_dtype_cache_gives_identical_delta`: deltas from a cached parent
    equal those from an uncached run exactly.
  - `test_delta_blocks_512_adds_512_row_deltas` and
    `test_full_delta_is_computed_but_never_cached`.
  - `test_block_cache_evicts_lru_within_budget` and
    `test_block_cache_concurrent_put_get_evict` (threads; no exception; every
    `get` returns the correct block or `None`).
  - `test_batch_defaults_match_target`: parsed defaults give `with_parent`
    true, `delta_blocks == (256,)`, the cache under `$ZOO_DATA`, and
    `also_t1_blocks == (128, 256, 512)` for T2 and `()` for T1.
  - `test_batch_auto_sketch_k_and_experts_follow_stratum`: `lab` and `anchor`
    rows get k = 128 and every expert; other rows get 64 and 8 seeded experts.
  - `test_opt_outs_reproduce_previous_batch_outputs`: `--no-with_parent
    --no-parent_cache --sketch_k 64 --also_t1_blocks none` on the existing
    fixtures gives byte-identical files to today's defaults.
- [ ] **Step 2:** Run `python -m pytest evidence/tests -q`. Expected: the new
  tests FAIL; existing tests pass.
- [ ] **Step 3:** Implement. In both the prefetched and sequential T2 paths,
  stop calling `_parent_delta` and writing the cache at the full-matrix height;
  parent prefetch loads only the parent's row blocks.
- [ ] **Step 4:** Run `python -m pytest evidence/tests -q`. Expected: all PASS.
- [ ] **Step 5:** Update `docs/operations/evidence.md`. Record the suggested
  message `fix(evidence): row-block-only parent deltas, bounded cache, target defaults`.

### Task 2: Row-block measurement on GPUs with automatic devices

**Files:**
- Modify: `evidence/fingerprint.py` (T1 path of `fingerprint_model`),
  `evidence/delta.py` (`_compute_device` accepts `auto`), `evidence/cli.py`
  (`_init_worker`, `_run_many`, `--compute_device` and `--workers` defaults)
- Test: `evidence/tests/test_fingerprint.py`, `evidence/tests/test_cli.py`,
  `evidence/tests/test_delta.py`
- Docs: `docs/operations/evidence.md`

**Interfaces:**
- Consumes: Task 1's `fingerprint_model` signature.
- Produces: T1 with a CUDA device computes sketches, block spectra and deltas
  on that device; block hashes still come from host bytes. On CUDA, T1 reads
  the next row blocks ahead with `SourcePrefetch` (window 2, bounded by
  `prefetch_bytes`).
- Produces: `--compute_device auto` (default) resolves to the GPUs in
  `CUDA_VISIBLE_DEVICES` when it is set, else CPU. With several GPUs, `--workers`
  processes are assigned to them round-robin. `cuda` uses every visible GPU,
  `cuda:<i>` one device, and `cpu` the CPU. `--workers auto` follows the
  Defaults table.

- [ ] **Step 1: Write the failing tests**
  - `test_t1_cuda_moves_sketch_spectrum_and_delta_to_device` (monkeypatched
    CUDA, as in the existing device tests).
  - `test_t1_cpu_is_byte_identical` (existing fixtures with `--compute_device cpu`).
  - `test_auto_uses_assigned_gpus_only` and
    `test_auto_without_visible_devices_uses_cpu` (Review Focus 5).
  - `test_assigned_devices_without_working_cuda_is_an_error`.
  - `test_workers_auto_spreads_over_assigned_gpus` (two visible devices: four
    workers, two per device) and `test_workers_auto_on_cpu`.
  - `test_t1_prefetch_never_holds_more_than_window_groups`.
- [ ] **Step 2:** Run `python -m pytest evidence/tests -q`. Expected: the new
  tests FAIL.
- [ ] **Step 3:** Implement: move each row block and its factors to the compute
  device before `_Records.add`, as the T2 companions already do.
- [ ] **Step 4:** Run `python -m pytest evidence/tests -q`. Expected: all PASS.
- [ ] **Step 5:** Document the device rule. Record the suggested message
  `perf(evidence): GPU row blocks and automatic devices`.

### Task 3: Evidence sidecar API

**Files:**
- Create: `evidence/sidecar.py`
- Modify: `evidence/hub.py` (`HubReader`, `RemoteFile`), `evidence/fingerprint.py`
  (`fingerprint_model`, `_Spectra`), `evidence/manifest.py` (`build_manifest`
  accepts a local-cache reader)
- Test: `evidence/tests/test_sidecar.py`, `evidence/tests/test_hub.py`,
  `evidence/tests/test_manifest.py`

**Interfaces:**
- Consumes: Task 1 (`BlockCache`, delta options); Task 2 (device handling).
- Produces: `HubReader(token=None, transport=None, local_cache: bool = False)`.
  With `local_cache`, `resolve` checks
  `huggingface_hub.try_to_load_from_cache(repo_id, filename, revision=revision)`
  and reads from disk when the file is present (new field
  `RemoteFile.local_path: str | None = None`); a local hit needs no token.
  Missing files use today's HTTP path. Pins stay Hub commits.
- Produces: `fingerprint_model(..., write_spectra: bool = True)`. With `False`,
  T2 writes its T1 companions and stages the T2 fingerprint unpublished.
  `publish_t2(out_dir: Path, manifest: dict, results: list[tuple[str, dict]], measurement: dict) -> Path`
  writes `<safe>.t2.spectra.h5` in the collector layout, with a
  `/layers/spectrum_source` string dataset (`runner` or `second_round`) and the
  worker's measurement config, then publishes the T2 fingerprint.
- Produces: `EvidenceSidecar(manifest: dict, out_dir: Path, *, parent: dict | None = None, block_cache: BlockCache | None = None, cache_self: bool = False, device: str = "cpu", max_inflight_bytes: int = 2 * 1024**3, sketch_k: int = 64, all_experts: bool = False, also_t1_blocks: tuple[int, ...] = (128, 256, 512))`:
  - `start() -> None` runs the T2 pass (`write_spectra=False`,
    `HubReader(local_cache=True)`) on one background thread.
  - `finish(esd: dict[str, dict], loaded_block: Callable[[str, tuple[int, int]], torch.Tensor | None], measure: Callable[[list[tuple[str, torch.Tensor]]], dict[str, dict]], measurement: dict) -> dict`
    waits for the pass. It maps each evidence view to a runner result by stored
    tensor key (`module_name.weight_attribute`, whole-weight results with equal
    shape only). It reuses that result only if `loaded_block` equals the stored
    256-row block bit for bit, sends the remaining views to `measure`, calls
    `publish_t2`, and returns the counts `reused`, `measured` and `gaps`.
  - `cancel() -> None` stops the pass and removes staged files.

- [x] **Step 1: Write the failing tests**
  - `test_local_cache_reads_make_no_network_calls`,
    `test_local_cache_hit_needs_no_token` and
    `test_local_cache_falls_back_to_http_for_missing_files`.
  - `test_manifest_from_local_download_matches_remote_manifest`.
  - `test_sidecar_outputs_match_standalone_t2`: on the tiny checkpoint, a
    sidecar whose `measure` calls `compute_esd_for_weight` produces the same
    fingerprints and spectra as today's `fingerprint_model`.
  - `test_finish_reuses_runner_spectra_for_matching_views`: `reused` equals the
    number of views and `measure` is not called.
  - `test_finish_remeasures_views_whose_loaded_block_differs` (Review Focus 1).
  - `test_t2_marker_published_after_spectra` and `test_cancel_publishes_nothing`.
- [x] **Step 2:** Run `python -m pytest evidence/tests -q`. Expected: the new
  tests FAIL.
- [x] **Step 3:** Implement `evidence/sidecar.py` and the reader, manifest and
  spectra options.
- [x] **Step 4:** Run `python -m pytest evidence/tests -q`. Expected: all PASS.
- [x] **Step 5:** Record the suggested message
  `feat(evidence): sidecar API for full-mode workers`.

### Task 4: Worker and runner integration, evidence on by default

**Files:**
- Create: `esd_experiment/src/evidence_bridge.py`
- Modify: `esd_experiment/src/worker.py` (`parse_args`, `main`),
  `esd_experiment/src/run_experiment.py` (`parse_args`, `generate_commands`,
  preparation after `pin_model_revisions`, `_completed_models_from_artifacts`)
- Test: `esd_experiment/tests/test_worker.py`, `esd_experiment/tests/test_run_experiment.py`
- Docs: `README.md` ("Start a run" shows the plain command and what it
  produces), `docs/operations/analysis.md` (outputs)

**Interfaces:**
- Consumes: Task 3 (`EvidenceSidecar`, `HubReader(local_cache=True)`); Task 1
  (`BlockCache` and its defaults).
- Produces: runner and worker flags from the Defaults table rows owned by
  Task 4 (`--evidence/--no-evidence`, `--evidence_device`,
  `--evidence_max_inflight_gib`), plus Task 1's parent-cache and stratum
  options passed through. `gpu` evidence uses visible device 0. With
  `--analysis_source checkpoint`, evidence turns off with a printed notice.
- Produces: preparation writes evidence manifests with
  `evidence.manifest.build_manifest` to
  `<output_dir>/evidence/<safe>/<safe>.manifest.json`, including manifests for
  declared bases, when `HF_TOKEN` is available. Without a token, the worker
  builds the manifest from its local download, and declared-parent deltas that
  need remote reads get `parent_unavailable`.
- Produces: `evidence_bridge.measure_tensors(items: list[tuple[str, torch.Tensor]], measurement: dict) -> dict[str, dict]`,
  which calls `compute_esd_for_weight` with one thread per visible GPU, or
  sequentially on CPU; and
  `evidence_bridge.loaded_block(model) -> Callable[[str, tuple[int, int]], torch.Tensor | None]`.
- Worker flow: load (as today), then `sidecar.start()`, then
  `net_esd_estimator` (unchanged), then stage `fingerprint` running
  `sidecar.finish(...)`, then stage `save` (as today). In CPU-only runs (no CUDA,
  or `--device_map cpu`), the sidecar starts after ESD. Models that fall back to
  checkpoint analysis record `evidence: skipped_checkpoint_fallback`.
- Completion requires the CSV/H5 pair and `<safe>.t2.fingerprint.safetensors`
  (with `--no-evidence`, the pair alone, as today). A model with the pair but no
  evidence reruns with `--evidence_only`: it loads the model, skips ESD and
  reuses `metrics/*.h5`.

- [x] **Step 1: Write the failing tests**
  - `test_runner_defaults_enable_evidence` and
    `test_plain_worker_run_writes_package_files`: with no extra flags, a tiny
    model gets the manifest; t2, t1, t1b128 and t1b512 fingerprints; and the
    spectra file, with `reused` equal to the matched views.
  - `test_no_evidence_outputs_unchanged`: with `--no-evidence`, the CSV/H5
    bytes equal today's.
  - `test_cpu_only_worker_runs_sidecar_after_esd` and
    `test_evidence_without_token_uses_local_download` (Review Focus 5).
  - `test_evidence_failure_keeps_esd_outputs_and_retries_evidence_only`
    (Review Focus 2).
  - `test_interrupt_in_fingerprint_stage_publishes_nothing` (Review Focus 3).
  - `test_checkpoint_mode_turns_evidence_off_with_notice`,
    `test_completion_requires_t2_marker_with_evidence` and
    `test_prepare_writes_manifests_for_models_and_declared_bases`.
- [x] **Step 2:** Run `python -m pytest esd_experiment/tests -q`. Expected: the
  new tests FAIL; existing tests pass after updating any that assumed evidence
  off.
- [x] **Step 3:** Implement the bridge, the worker stages and the runner flags.
- [x] **Step 4:** Run `python -m pytest esd_experiment/tests evidence/tests -q`.
  Expected: all PASS.
- [x] **Step 5:** Update the README and analysis guide. Record the suggested
  message `feat(runner): evidence sidecar on by default in full-mode workers`.

### Task 5: Automatic CPU thread budget per job

**Files:**
- Modify: `esd_experiment/gputracker/gputracker.py`
  (`GPUDispatcher._normalize_config`, `ChildThread.run`),
  `esd_experiment/src/run_experiment.py` (`parse_args`, `create_runtime_config`)
- Test: `esd_experiment/tests/test_gputracker_scheduler.py`,
  `esd_experiment/tests/test_run_experiment.py`
- Docs: `README.md` ("Tune the parallelism")

**Interfaces:**
- Produces: `gpu_config.json` field `cpu_threads_per_gpu: int | "auto"` (default
  `"auto"`, live-reloadable like the other fields) and runner flag
  `--cpu_threads_per_gpu`. `auto` is `os.cpu_count() // physical GPUs` (from
  gpustat), at least 1. For a positive value, `ChildThread.run` sets
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `NUMEXPR_NUM_THREADS` to `value × len(cuda_devices)`. With `0`, the
  environment is untouched, as today.

- [x] **Step 1: Write the failing tests:**
  `test_cpu_thread_budget_auto_divides_cores_by_physical_gpus`,
  `test_cpu_thread_budget_env_scales_with_assigned_gpus`,
  `test_cpu_thread_budget_zero_leaves_env_unchanged`,
  `test_reload_changes_budget_for_new_jobs_only` and
  `test_invalid_budget_on_reload_keeps_previous`.
- [x] **Step 2:** Run `python -m pytest esd_experiment/tests/test_gputracker_scheduler.py -q`.
  Expected: the new tests FAIL.
- [x] **Step 3:** Implement.
- [x] **Step 4:** Run `python -m pytest esd_experiment/tests -q`. Expected: all PASS.
- [x] **Step 5:** Document the setting. Record the suggested message
  `feat(gputracker): automatic CPU thread budget per job`.

### Task 6: Model packages, automatic publishing and the `layers` table

**Files:**
- Create: `zoo/publish.py`, `zoo/layers.py`
- Modify: `zoo/schema.py` (`LAYERS_COLUMNS`), `zoo/build.py` (`_COLUMNS`,
  `build_zoo`), `zoo/tables.py` (`_fingerprint_files`, model spectra links),
  `zoo/cli.py` (`publish`; `build` defaults), `esd_experiment/src/worker.py`
  (publish after save), `esd_experiment/src/run_experiment.py`
  (`--publish/--no-publish`, `--packages_root`)
- Test: `zoo/tests/test_publish.py`, `zoo/tests/test_layers.py`,
  `zoo/tests/test_build.py`, `zoo/tests/test_cli.py`,
  `esd_experiment/tests/test_worker.py`
- Docs: `docs/operations/zoo.md`, `README.md`

**Interfaces:**
- Consumes: Task 4's run-folder layout (`stats/`, `metrics/`, `evidence/<safe>/`).
- Produces: package path `<packages_root>/<shard>/<safe>/<conventions_id>/`.
  `shard` is the first two hex digits of `sha256(safe)`. `conventions_id` is the
  first 12 hex digits of the sha256 of the sorted measurement settings with
  `NUMERICS_VERSION`, `FORMAT_VERSION` and `EVIDENCE_VERSION`. Contents: the
  frozen evidence files, `esd/stats.csv`, `esd/metrics.h5`, an optional
  `deltas/` folder, and `package.json` with `model_key`, `repo_id`, `revision`,
  `kind` (`full` or `t1_only`), `conventions_id`, `conventions`, `files`
  (name → sha256), `complete`, `evidence` (reused, measured, gaps),
  `source_run` and `created`. A `t1_only` package (oversized models,
  preliminary studies) holds the manifest and T1 fingerprints only and is
  complete without `esd/` or T2 files.
- Produces: `publish_package(run_dir: Path, model_id: str, packages_root: Path, data_root: Path) -> Path | None`
  (used by the worker after `save`) and
  `publish_packages(runs: list[Path], packages_root: Path, data_root: Path) -> list[Path]`
  (the `zoo.cli publish` backfill). Both publish only complete models, by
  hardlink (copy across filesystems), through a staging directory and one
  rename. An identical existing package is skipped; a package is never
  overwritten.
- Produces: `current_conventions_id() -> str`, computed from the default
  measurement settings and versions.
- Produces: `LAYERS_COLUMNS = ["model_key", "longname", "module_name", "weight_attribute", "slice", "view_name", "alpha", "alpha_weighted", "D", "xmin", "xmax", "n_tail", "fit_status", "spectral_norm", "log_spectral_norm", "stable_rank", "entropy", "matrix_rank", "num_evals", "h5_index"]`,
  read from `/layers` in `esd/metrics.h5`; `view_name` is the mapped evidence
  view or empty. Eigenvalues stay in HDF5.
- Produces: `build_zoo(..., packages: Path | None = None, conventions_id: str | None = None)`.
  `python -m zoo.cli build` with no inputs reads `$ZOO_DATA/packages` at
  `current_conventions_id()` and writes `$ZOO_DATA/builds/current`. Packages
  under other conventions are skipped and counted in the build summary.

- [x] **Step 1: Write the failing tests:**
  `test_publish_creates_sharded_package_with_hashes`,
  `test_publish_is_idempotent`,
  `test_publish_never_overwrites_and_separates_conventions`,
  `test_publish_skips_incomplete_models`,
  `test_publish_t1_only_package_without_esd_files`,
  `test_worker_publishes_package_by_default`,
  `test_no_publish_keeps_run_folder_only`,
  `test_build_without_arguments_uses_packages_and_current_conventions`,
  `test_build_reports_skipped_conventions`,
  `test_build_from_packages_matches_build_from_directories` (the six existing
  tables are equal) and `test_layers_rows_match_metrics_h5`.
- [x] **Step 2:** Run `python -m pytest zoo/tests esd_experiment/tests/test_worker.py -q`.
  Expected: the new tests FAIL.
- [x] **Step 3:** Implement.
- [x] **Step 4:** Run `python -m pytest zoo/tests evidence/tests esd_experiment/tests -q`.
  Expected: all PASS.
- [x] **Step 5:** Update `docs/operations/zoo.md` and the README. Record the
  suggested message `feat(zoo): model packages, automatic publishing and per-layer metrics`.

### Task 7: Delta pass for parents found by similarity

**Gate cleared (2026-11-04):** owner approval plus practical-purpose audit above.

**Files:**
- Create: `evidence/pair_delta.py`
- Modify: `zoo/cli.py` (`deltas` command), `evidence/compare.py` (`diff` reads
  pair deltas), `zoo/tables.py` and `zoo/lineage.py` (read pair deltas)
- Test: `evidence/tests/test_pair_delta.py`, `zoo/tests/test_cli.py`,
  `zoo/tests/test_lineage.py`
- Docs: `docs/operations/zoo.md`

**Interfaces:**
- Consumes: Task 1 (`BlockCache`); Task 6 (package paths, `open_tables`).
- Produces: `pair_deltas(child: dict, parent: dict, reader: HubReader, package: Path, *, block_cache: BlockCache | None = None, heights: tuple[int, ...] = (ROW_BLOCK,)) -> Path`,
  which writes `deltas/<parent safe>.delta.safetensors` with tensors
  `rows [n,2]`, `delta_scalars [n,3]` and `delta_top_sv [n,16]`, and metadata
  `names`, `delta_status`, `delta_confidence`, the child and parent identities
  and `evidence_version`. The child's 256-row blocks are range-read; the
  parent's come from the cache or one range read per view.
- Produces: `python -m zoo.cli deltas` with the Defaults table inputs. It reads
  the detected edges that have no delta for that pair from the zoo, computes
  them with `--workers` and `--compute_device` (same rules as Task 2), and
  prints the rebuild command.

- [x] **Step 1: Write the failing tests:**
  `test_pair_delta_equals_in_run_delta_for_same_parent`,
  `test_pair_delta_reads_only_row_blocks`,
  `test_siblings_read_parent_once_through_cache`,
  `test_incompatible_shapes_recorded_not_failed`,
  `test_deltas_without_arguments_uses_default_zoo_and_packages`,
  `test_build_reads_pair_deltas` and
  `test_diff_uses_pair_delta_for_requested_parent`.
- [x] **Step 2:** Run `python -m pytest evidence/tests zoo/tests -q`. Expected:
  the new tests FAIL.
- [x] **Step 3:** Implement.
- [x] **Step 4:** Run `python -m pytest evidence/tests zoo/tests -q`. Expected:
  all PASS.
- [x] **Step 5:** Update the zoo guide. Record the suggested message
  `feat(zoo): delta pass for found parents`.

### Task 8: Automatic sizing, caps and download-ahead for full-scale runs

**Files:**
- Create: `esd_experiment/src/prefetch.py`, `esd_experiment/src/sizing.py`
- Modify: `esd_experiment/gputracker/supervision.py` (`WorkerJob`,
  `WorkerStateTracker.start_worker`), `esd_experiment/gputracker/gputracker.py`
  (`DispatchThread.run`, `ChildThread` limits), `esd_experiment/src/run_experiment.py`
  (`parse_args`, `pin_model_revisions`, `generate_worker_jobs`, `main`)
- Test: `esd_experiment/tests/test_gputracker_scheduler.py`,
  `esd_experiment/tests/test_run_experiment.py`, `esd_experiment/tests/test_sizing.py`
- Docs: `README.md` (parallelism, limits and cache sections)

**Interfaces:**
- Produces: `WorkerJob` fields `num_gpus: int = 0`, `cache_path: str = ""`,
  `ram_limit_gib: float = 0`, `cache_limit_gib: float = 0` and
  `stage_timeout_seconds: dict[str, int] | None = None`. Zero or `None` means the
  dispatcher's global value; `start_worker` adopts a pre-staged `cache_path`.
  `DispatchThread` requests `job.num_gpus or self.num_gpus_needed`.
- Produces: `sizing.job_resources(weight_bytes: int, gpu_memory_bytes: int, pool_size: int, evidence_inflight_bytes: int) -> dict`
  returning `num_gpus`, `ram_limit_gib`, `cache_limit_gib` and
  `stage_timeout_seconds` by the Defaults table rules, or `oversized: True`.
  Per-GPU memory comes from gpustat `memory.total`; the factor is 1.5.
  Checkpoint mode always gets one GPU.
- Produces: the runner rows owned by Task 8 in the Defaults table.
  `--num_gpus_per_job`, `--max_weight_gib`, `--max_worker_ram_gib`,
  `--max_worker_cache_gib` and `--stage_timeout_seconds` default to `auto`; an
  explicit value applies to every job, as today.
- Produces: `prefetch.Prefetcher` downloads the next pinned models' selected
  files into their future private caches before their GPUs are reserved,
  within `--prefetch_models` (default: pool size) and `--prefetch_max_gib`
  (default 64). Cleanup is unchanged: it happens after confirmed group exit. A
  failed prefetch leaves the worker to download, as today.
- Produces: oversized models go to `<output_dir>/oversized_models.csv` with
  reason `too_large_for_full_mode`. After the GPU queue drains, they are
  measured with `evidence.cli batch --tier t1` and published with Task 6's
  `publish_package` as `t1_only` packages, unless `--no-oversized_t1`.

- [x] **Step 1: Write the failing tests:**
  `test_job_resources_auto_from_weight_bytes`,
  `test_explicit_limits_apply_to_every_job`,
  `test_checkpoint_mode_sized_to_one_gpu`,
  `test_per_job_limits_override_global_config`,
  `test_oversized_models_listed_and_get_t1_packages`,
  `test_prefetch_on_by_default_within_byte_budget`,
  `test_prefetch_zero_downloads_in_worker_as_today`,
  `test_prefetched_cache_owned_and_removed_after_exit`,
  `test_failed_prefetch_falls_back_to_worker_download` and
  `test_worker_cache_root_defaults_to_user_scratch`.
- [x] **Step 2:** Run `python -m pytest esd_experiment/tests -q`. Expected: the
  new tests FAIL.
- [x] **Step 3:** Implement.
- [x] **Step 4:** Run `python -m pytest esd_experiment/tests -q`. Expected: all PASS.
- [x] **Step 5:** Update the README: the plain command, what `auto` resolves to
  and how to set explicit values. Record the suggested message
  `feat(runner): automatic sizing, limits and download-ahead`.

### Task 9: Validation runs (lead)

**Quick subset (2026-11-04):** [Measured report](../../../analysis_runs/validation/full_mode_quick_20261104/REPORT.md).
CPU/GPU correctness passed on local sources; warm tiny-model overhead target
missed. For direct CPU/GPU math, hashes/statuses/ranks must be exact; sketches
use rtol/atol1e-4 and spectra error must be below1e-4times the view's largest
eigenvalue. CPU sidecar/standalone equality remains exact. This subset does not
check off the larger acceptance runs below.

Each run stays within D4.1 or is handed to the person through `STATUS.md`.

- [ ] **Plain run:** a 5-model run with only `--model_list`, `--output_dir` and
  the GPU pool produces published packages, and `python -m zoo.cli build` with
  no arguments builds the database from them.
- [ ] **Parity on real models:** `openai-community/gpt2`,
  `HuggingFaceTB/SmolLM2-135M` and the Smol135 dialogue LoRA r8 from the zoo
  guide's worked queries, through the runner and through
  `python -m evidence.cli batch --tier t2`. Pass: identical block hashes,
  sketches, T1 spectra and delta statuses and ranks; reused T2 spectra within a
  relative 1e-4 of each view's largest eigenvalue.
- [ ] **Throughput:** the same 20 small models on 2 GPUs, with evidence on and
  with `--no-evidence`, sampling `nvidia-smi` every 5 s. Proposed pass (the
  owner may adjust): evidence adds at most 10% wall time, and the GPU busy
  fraction is no lower than without evidence. Use the measurements to confirm
  or adjust the `auto` RAM, cache and timeout rules.
- [ ] **Row blocks, CPU against GPU:** the 97-model T1 pilot with
  `--compute_device cpu` and with the pilot GPUs assigned through
  `CUDA_VISIBLE_DEVICES`. Report wall time, bytes read and GPU busy fraction;
  pass on the parity criteria above.
- [ ] Record the results in `STATUS.md`.


### Conditional small optimization — bulk sidecar spectrum writes

Owner authorized only if worthwhile (2026-11-04). Profile identified0.33–0.43s
for26row-at-a-time HDF5 spectrum writes. Preserve the existing streaming writer
and add an empty-file bulk writer used only by publish_t2 after complete-bundle
validation. Formats/numerics/coverage/atomic publication unchanged. Verify exact
streaming/bulk dataset parity (including ragged spectra and missing metrics),
sidecar validation/rollback tests, and before/after timings. Retain only with a
measurable benefit; no pipeline redesign or weakened integrity checks.


**Outcome:** Retained. Five paired real-row trials: spectrum publication median
0.491→0.073s (85%less). Three cold-operator sidecar pairs:2.083→1.685s (19%less).
Exact stored-value/fingerprint parity;80focused tests pass. No claim that this
alone meets the10%GPU-overhead target. See the benchmark report for raw samples.


### Nov5 observed-gap corrections

- Oversized rows use independent WorkerJobs with the existing process supervisor,
  configured concurrency, RAM/cache caps, heartbeat/stage limits and private-cache
  ownership. Streaming GPU by default, `--oversized_device cpu` opt-out. No full
  checkpoint prefetch. The earlier inline-CPU amendment was an implementation
  simplification that lost parallelism/supervision; this corrects it.
- T1 package settings record the resolved T1 device, independent of the full-mode
  `--evidence_device` flag. Default build accepts current full CPU-sidecar plus
  CPU/GPU T1 profiles; it prefers full, then CPU T1 when both T1 variants exist.
- Parent-cache reads use complete atomic inodes without the writer lock; stale
  filesystem errors are cache misses. Writers use one stat snapshot and avoid
  sorting below the cap. Global byte cap/LRU eviction remain; insertions still
  scan O(N) and need further indexing/sharding for large deployments.
- Exact nearest retrieval is O(N) per query within its architecture group;
  all-model similarity construction is quadratic across large groups.
  Shared-tensor lineage currently considers all model pairs globally. No
 100k-readiness claim or new approximate index is introduced.
- Evidence-only retries retain full-model loading: durable metrics have no
  loaded-block hashes to prove safe reuse from a later range read. Publication
  retries already skip loading. A new provenance protocol is outside the
  owner's minimal-change condition. The worker's no-op exception wrapper was
  removed; cancellation/rollback remains covered.
- No numerical/schema version change in these fixes; existing formats are logged
  separately in the handoff. Native five-model plain run and larger planned
  throughput/public-base runs remain pending; three real local CPU/GPU cases
  replace a97-model run for this quick parity check.


### Readability pass (owner request, Nov5)

Operational names describe purpose: `row_block_worker.py`,
`run_oversized_evidence_jobs` (supervision), and `extract_row_block_evidence`
(extraction/publication). Request/result paths, model options and outcomes have
explicit local names. Existing artifact tiers, CLI flags and CSV fields retain
their spelling. Root AGENTS.md records these clarity conventions. Rename and
subprocess integration tests cover all updated call sites; no numerical or
format changes.

### Bounded scaling follow-up (owner request, Nov5)

Keep exact retrieval/lineage semantics and existing artifacts. Three focused
changes, executed in the current authorized checkout without commits:

1. `evidence/block_cache.py`: share a small byte-count record under the existing
   writer lock. Invalidate it before mutation; missing/partial records or directory
   membership changes trigger a recovery scan. Scan for LRU eviction only when
   full; reclaim about 5% headroom when entry sizes permit. Preserve lock-free
   reads, atomic block publication and the byte cap. Test shared accounting,
   interrupted publication, external additions, batching and existing races.
2. `zoo/lineage.py`: index nonempty hashes with full view alignment, and compare
   only pairs sharing an aligned hash. Exclude duplicate classes before pair
   expansion. Preserve denominators, tiers, scores and deterministic orientation.
   Test sparse candidates and existing compatibility cases.
3. `zoo/retrieval.py`, `zoo/lineage.py`: reuse validated metadata/memory maps once
   per architecture during a build; pre-group model view rows during index
   assembly. Preserve the public nearest API and scoring. Test index reuse and
   existing exact score/mask/tie/calibration cases. Exact all-model scoring still
   has quadratic arithmetic; ANN/recall policy and dense edge-output limits are
   deferred as larger work. No dependencies, GPUs or format/version changes.

Run the changed cache/retrieval/lineage/build suites once after RED→GREEN;
benchmark local synthetic cache and sparse-lineage workloads, document measured
scope and remaining limits. No repeated full suites or review loops.
