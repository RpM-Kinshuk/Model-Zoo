# Curated Table Ingestion And Loader Hardening Design

Date: 2026-04-19

## Goal

Make the Model-Zoo data-gathering layer work directly from the curated phase-2 model table, while improving model loading, retry behavior, and failure reporting without a drastic rewrite.

This phase should:

- read the canonical curated table directly
- route models using the curated loader/type fields already present
- improve load-time compatibility across the main model categories we care about
- stop producing misleading empty result files
- record failures with clearer stage and reason information

This phase should not:

- redesign the ESD math itself
- build a large plugin framework
- replace the experiment runner architecture
- force unsupported model formats to appear supported

## Why This Change Is Needed

The current Model-Zoo experiment path assumes a narrow input shape:

- `model_id`
- `base_model_relation`
- `source_model`

That was acceptable for earlier playground work, but it is no longer enough for the curated dataset. The curated table now contains information that should directly influence loading and triage, such as:

- `revision_norm`
- `primary_type_bucket`
- `loader_scenario`
- `lineage_status`
- `analysis_artifact_present`

The current worker and loader path also have two practical weaknesses:

1. placeholder CSV creation can make failures look like partial success
2. failure recording is too loose for later recovery and retry work

## Design Principles

1. Keep the current structure.
   - `run_experiment.py`, `worker.py`, and `model_loader.py` remain the core path.

2. Read the curated table directly.
   - Do not introduce a second compatibility CSV format as the main path.

3. Separate support from triage.
   - A model can be curated and intentionally skipped if its loader scenario is not supported yet.

4. Be explicit about failure stage and reason.
   - `load`, `analyze`, and `save` should not collapse into one generic failure string.

5. Retry only when it makes sense.
   - transient failures should retry
   - clearly unsupported or structurally broken cases should fail once and be logged clearly

## Supported Input Shape

`run_experiment.py` should accept the curated canonical table directly.

Minimum required column:

- `model_id`

Preferred columns when available:

- `revision_norm`
- `source_model`
- `base_model_relation`
- `primary_type_bucket`
- `loader_scenario`
- `lineage_status`
- `candidate_source`

The experiment runner should normalize each selected row into one internal job record with only the fields the worker actually needs.

## Loading Policy

The loader should use `loader_scenario` when available and fall back to light inference only when necessary.

### Supported Now

- `standard_transformers`
- `adapter_requires_base`
- `quantized_transformers_native`
- MoE models that still load through standard Transformers classes
- some multimodal models, only when the underlying model class can be loaded through the normal transformers path without extra custom processing

### Explicitly Unsupported For Now

- `quantized_alt_format` such as GGUF-only repos
- adapter repos without a resolvable base model
- repos missing required config or weight-manifest structure
- loader scenarios that cannot expose analyzable PyTorch weights through the current stack

Unsupported rows should be logged cleanly, not repeatedly retried.

## File-Level Changes

### `esd_experiment/src/run_experiment.py`

Add direct curated-table ingestion.

Changes:

- allow reading the wider curated table directly
- preserve the current simple three-column path for backward compatibility
- normalize rows into a small internal job spec
- pass through `revision_norm`, `loader_scenario`, and `primary_type_bucket` when present
- improve completed/failed model checks so they are based on the actual per-model stats directory, not only root-level files

### `esd_experiment/src/model_loader.py`

Keep the current loader shape, but harden it.

Changes:

- use `loader_scenario` as the main dispatch hint when present
- keep adapter merge support, but make base-resolution failure explicit
- support revision-aware loading consistently
- explicitly reject unsupported scenarios like GGUF-only repos with a stable failure reason
- broaden the standard load path so non-CausalLM text models and compatible multimodal/MoE architectures do not fail unnecessarily when `AutoModelForCausalLM` is too narrow

The loader should return either:

- a loaded model plus metadata about how it was loaded
- or a structured failure classification

without creating result files itself.

### `esd_experiment/src/worker.py`

This is the most important hardening point.

Changes:

- stop treating an empty touched CSV as the work-in-progress signal
- write to a temporary output file and only finalize on successful save
- separate failures into:
  - `load`
  - `analyze`
  - `save`
- classify reason into a stable small set
- keep `1-2` retries only for transient classes
- write structured failure rows to a machine-readable artifact

Recommended failure artifact:

- `logs/failure_records.jsonl`

Keep the old text file if helpful, but treat it as secondary.

### `README` / docs

Update the Model-Zoo documentation so the main documented input is now the curated table, while still noting that the old minimal CSV remains accepted.

## Failure Classification

Use a small stable vocabulary.

Suggested classes:

- `repo_missing_or_private`
- `repo_gated`
- `adapter_base_unresolved`
- `unsupported_loader_scenario`
- `model_load_error`
- `cuda_oom`
- `analysis_empty`
- `analysis_exception`
- `save_error`

Suggested stage values:

- `load`
- `analyze`
- `save`

This is enough for later triage and recovery without creating a large error taxonomy.

## Retry Policy

Retry only for likely transient failures:

- hub/network fetch issues
- some generic model load errors
- some CUDA OOM cases after cleanup

Do not retry for clearly permanent cases:

- unsupported scenario
- unresolved adapter base
- missing/gated/private repo
- structurally incomplete repo

Default retry count should remain small.

## Empty Output Prevention

The worker should only leave a result CSV or HDF5 behind if:

- loading succeeded
- ESD analysis returned usable metrics
- final save succeeded

If any of those fail, temporary files should be removed and the failure should be recorded instead.

This is the key fix for the “empty result file” problem.

## Verification Plan

Keep verification practical and local.

1. Unit-test the curated-table parsing path.
2. Unit-test loader dispatch and unsupported-scenario classification.
3. Unit-test failure recording and temp-file cleanup.
4. Smoke-test a few small models covering:
   - standard text model
   - adapter
   - native quantized model if available
   - one MoE or multimodal-compatible case if small enough

These smoke artifacts should be temporary and not retained as experiment data.

## Success Criteria

This phase is successful if:

- Model-Zoo can read the curated table directly
- supported rows flow through without needing an external compatibility conversion
- unsupported rows fail fast with stable recorded reasons
- empty per-model result files stop appearing as false positives
- retry behavior is narrower and more informative
- the ESD path itself remains mostly unchanged
