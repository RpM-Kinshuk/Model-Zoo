# Descriptive Evidence Names Implementation Plan

> **For agentic workers:** Use executing-plans for implementation. Keep changes uncommitted.

**Goal:** Replace numbered evidence shorthand in controls, code and documentation
with metadata_only, row_blocks and full_weights; name the oversized-model path
explicitly. Owner explicitly requests migration of existing public names.

**Architecture:** New CLI uses --measurement_mode and --also_row_block_sizes;
runner uses --oversized_models_evidence and --oversized_models_device. New
filenames/metadata use descriptive measurement names; database columns become
row_block_fingerprint, full_weight_fingerprint and full_weight_spectra. One
compatibility module translates historical labels/filenames/table columns on
read, without rewriting historical data. Evidence3/database2 distinguish the
new contracts; numerical versions and algorithms stay unchanged.

**Constraints:** Keep Bash cleanup/exit choices; preserve current run_script.sh
edits. No GPU jobs, downloads, installs or commits. Read-only historical data
and old immutable packages remain untouched. Old names occur only in explicit
legacy-compatibility code/tests and historical input paths. Update the sibling
query consumer as required so new and old database builds remain usable.

1. Add focused regression tests for descriptive CLI/artifacts and legacy reads;
   watch them fail before implementing the new contract.
2. Migrate fingerprint/sidecar/CLI/database implementation and tests, centralizing
   legacy translation. Canonical files are .row_blocks.fingerprint.safetensors,
   .row_blocks_128/.row_blocks_512 companions, .full_weights.fingerprint.safetensors
   and .full_weights.spectra.h5. Keep generic tier fields where they describe
   evidence levels; their numbered values change to descriptive names.
3. Independently migrate runner controls/worker/status fields and active Bash
   preset; update docs and code guidance using the same vocabulary.
4. Update sibling query consumers and validate legacy table/artifact reads.
5. Run focused extraction/CLI/sidecar/database/runner/query checks, inspect
   remaining references, and log schema/name migration in the handoff. Avoid
   repeated broad test/review loops. Record real failures and fix required edges.

## Completion

Implemented in the current checkout; changes left uncommitted. Legacy-read,
runner/worker, artifact publication, database and sibling query checks pass.
One focused review found two legacy lookup gaps, both fixed with RED→GREEN
regressions. Detailed commands/results and migration costs are in the plan ledger.
Numbered names remain only in explicit compatibility tests/readers and marked
historical paths/records. Source data and historical package inventories stay
untouched; numerical conventions remain unchanged.
