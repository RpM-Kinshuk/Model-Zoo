# Build and query a zoo

A zoo combines pinned model metadata, fingerprints, measured scores and lineage
in one snapshot. Building uses existing artifacts; querying needs no server.

## Build

Use the existing environment and set `ZOO_DATA` to the artifact root. From
Model-Zoo:

```bash
python -m zoo.cli build
```

This reads `$ZOO_DATA/packages` under current measurement conventions and writes
`$ZOO_DATA/builds/current`. `--packages PATH` selects another package tree;
`--conventions ID` selects other settings. The build summary counts packages
skipped for different conventions. Default row-block-only and full packages use distinct
tier convention IDs; the default build accepts full CPU-sidecar packages and
CPU/GPU streaming row-block packages, and prefers full evidence
when both exist for the same model. When both CPU and GPU row-block-only packages exist, it selects
the CPU package deterministically. This preserves preliminary packages during
later full-mode upgrades.

Full-mode workers publish packages automatically. Set `--packages_root PATH`,
or `--no-publish` to keep run-folder outputs only. Backfill existing completed
runs without loading any weights:

```bash
python -m zoo.cli publish --runs /path/to/run-a /path/to/run-b
```

A package lives at `<packages_root>/<shard>/<safe>/<conventions_id>/`. It contains
only derived evidence files and, for full mode, `esd/stats.csv` and
`esd/metrics.h5`; raw checkpoint and cached parent blocks are never included.
`package.json` records pinned identity, kind, conventions, artifact SHA256 hashes,
reuse counts and source-run provenance. Sources outside `data_root` have an
absolute provenance-only `source_run`; stored database artifact links remain
relative to `data_root`.

Publication uses hardlinks (copies across filesystems), a private staging
directory and one rename. An identical existing package is reused; different
contents never overwrite it. Producers must replace run artifacts atomically,
as the worker does, rather than modifying hardlinked files in place. Convention
hashes exclude model identity and runtime observations, include algorithm/format
versions and evidence tier, and preserve the default per-stratum sketch/expert
policy. Explicit evidence overrides receive separate conventions. When old runs
lack a worker summary, sketch/expert settings are inferred from fingerprints.

For legacy explicit inputs:

```bash
python -m zoo.cli build --manifests "$MANIFESTS" \
  --fingerprints "$FINGERPRINTS" --data_root "$ZOO_DATA" \
  --out "$ZOO_DATA/builds/current"
```

`MANIFESTS` contains one manifest per pinned model. `FINGERPRINTS` contains
canonical row-block and full tensor files; extra row-block heights remain
alongside them. Multiple fingerprint directories may follow `--fingerprints`.

Size/modality filters and download ordering use the selection frame's metadata.
Supply `--frame` when building a catalog intended for those queries. Missing
fields stay unknown and do not match a numeric or modality filter; a database
can contain a measured model that such a filter cannot shortlist. A frame adds
metadata, not benchmark scores or predictions.

Optional inputs:

| Option | Input |
| --- | --- |
| `--frame` | Selection CSV, including sampling probabilities |
| `--leaderboard_v2` | Leaderboard export with Raw scores and available evaluation SHAs |
| `--scores_csv` and `--scores_mapping` | Measured scores and their column mapping |
| `--lineage_edges` | Lineage Parquet using the frozen edge schema |
| `--null` | An applicable calibration JSON from the analysis pipeline |

A score mapping names the CSV columns, for example:

```json
{"repo_id":"model","revision":"commit","benchmark":"benchmark","value":"value","skill":"skill","source":"our_eval"}
```

`source` may also be a constant label. Missing or mismatched evaluation SHAs
produce unverified score rows. Default score filters use revision-verified rows;
`--include_unverified` opts into the others.

The output defaults to `<data_root>/builds/current`; `data_root` defaults to
`ZOO_DATA`. Choose a fresh output or an existing zoo-managed publication path.
The builder publishes atomically, and open readers keep their original snapshot.
Older generations remain available until explicitly cleaned up.

## What is stored

| Table | Contents |
| --- | --- |
| `models` | Pinned identity, metadata, coverage and artifact links |
| `views` | Measured matrices and their block hashes |
| `layers` | Per-layer alpha, fit quality, spectral metrics and HDF5 row indices |
| `coverage` | Measured/skipped tensors and reasons |
| `lineage` | Candidate relationships with evidence tiers and statistics |
| `scores` | Measured scores and revision verification |
| `predictions` | Separate prediction records; empty in v0 |

Eigenvalues remain in HDF5. For full packages, `models.full_weight_spectra` links to
`esd/metrics.h5`; each `layers.h5_index` indexes that file's aligned `/layers`
and `/eigs` records. `view_name` maps exact stored identities when possible and
is empty otherwise. Stored-view spectra remain beside their full tensor
fingerprint under the descriptive filename, including second-round views absent from the
original runner metrics. Legacy direct-input builds keep their original
stored-view spectrum links; their `layers` table is empty. Older six-table zoo
generations remain readable with an empty `layers` table. Artifact
paths are relative to `ZOO_DATA`. Retrieval compares sampled row-block fingerprints
within the same architecture signature, giving comparable coordinates and
shapes. Full tensor sketches have a different measurement scope.

## Read the evidence

- **Declared:** supplied relationship metadata.
- **Detected:** a method-derived candidate, with its method and statistics.
- **Verified:** evidence checked under the producing method's rules.

An **uncalibrated** verdict means no applicable calibrated null supports a
relationship decision. Cosine remains a measured similarity statistic.
Duplicate labels mean "identical on every measured block."

## Query

From Zoo-Keeper, use `python -m zoo_keeper.cli`. The commands are `find`, `info`,
`similar`, `family`, `diff`, and `check`; each supports `--zoo PATH` and `--json`.
The default zoo is `$ZOO_DATA/builds/current`. Use a complete `repo@revision` key
when a repository has several revisions.

See Zoo-Keeper's README for a worked command per query. Without `--json`, output
is a table or indented card. With it, stdout is one JSON document. Errors go to
stderr.

Exit codes: **0** completed (including uncalibrated or base-not-in-zoo checks),
**1** an inconsistent `check`, **2** invalid input or unavailable/ambiguous
comparison. Read the verdict as well as the exit code.

## Build scaling

Shared-tensor lineage first indexes hashes with their view names and alignment,
then compares only pairs sharing an aligned hash. Duplicate classes are excluded
before pair expansion. Exact shared fractions and edge orientation are unchanged;
large families sharing a hash can still require quadratic pair/output work.

Retrieval assembly groups view rows once per model. During lineage detection,
each architecture index is loaded once and reused, with alignment comparisons
encoded as integer IDs. This removes repeated metadata parsing and Python
alignment checks. Scores, missing-view masks, deterministic ties and the on-disk
index format are unchanged. Exact scoring is linear per query and quadratic when
a build queries every model in a large group. Approximate search needs a separate
candidate-recall policy before it can replace this path at census scale.

Local synthetic timings and exact before/after parity are recorded in the
[scaling follow-up](../../analysis_runs/validation/scaling_followup_20261105/REPORT.md).

## Validation status

The 2026-11-02 scoped CLI smoke passed on 85 models: 43 measured checkpoints and
42 metadata-only entries, retaining all 43 full-spectrum links. The following
rebuild uses the original historical path (`t2-k128`); keep that path unchanged:

```bash
python -m zoo.cli build \
  --manifests "$ZOO_DATA/runs/track-a-z7-query-20261102.hhji0j6t/manifests" \
  --fingerprints "$ZOO_DATA/runs/track-a-l12-20261102.mnbqfy_y/t2-k128/fingerprints" \
  --data_root "$ZOO_DATA" --out "$ZOO_DATA/builds/example" \
  --leaderboard_v2 "$ZOO_DATA/Zoo-Keeper/data/heritability/sampled_leaderboard.csv"
```

The validated output is `runs/track-a-z8-cli-20261102.c45u9pog/zoo` under the
data root. It contains 18 leaderboard score rows, six revision-verified.
All six query commands passed with JSON output. For the dialogue LoRA r8 and
its SmolLM2-135M base: `diff` found 151 views identical on their measured blocks
and 60 changed views;
`diff` and `check` reported `uncalibrated`. `similar -k 3` returned three
neighbors; `family --depth 2` returned one ancestor and no descendants.
Zoo-Keeper's README records the worked queries. Exact argv, outputs and timings
are saved beside the snapshot in `commands.json`, `*.stdout` and
`cli-smoke-report.json`.

The full R6 pilot smoke also needs the R3/R4 pilot outputs and the R5 G0
calibration. Those prerequisites remain separate from this scoped smoke; no
calibration is fabricated for it.

### Finished pilot artifact smoke (2026-11-03)

The finished pilot has97/97 new row-block successes and241/287 new full-weight successes;46
full-weight failures remain explicit. Six previous full-weight bundles are reused, giving247
complete pilot full-weight bundles with full spectra and all128/256/512 companions.

The artifact-only snapshot at
`runs/track-a-r3-r6-20261102.rcsy95ap/zoo` contains556 model records,89,286
measured views,790 lineage rows and150 score rows. Its284 full-spectrum HDF5
links include reused lab evidence; all links exist. No weights were re-extracted
for this build. Hub gated states `auto`/`manual` are normalized to true only
when assembling the boolean database column; raw manifests remain unchanged.

All six query commands passed on the Smol135 dialogue LoRA r8/base example.
`find` and `similar -k3` returned three rows; `info`, `family`, `diff` and
`check` returned valid JSON. `check` remains `uncalibrated`. Exact commands
and outputs are beside the snapshot in `pilot-query-commands.json` and
`query-*.stdout`; `FINAL_REPORT.md` records failures and next priorities.
Both Model-Zoo and Zoo-Keeper must be on PYTHONPATH for source-tree queries.
Pass `--data_root` explicitly when ZOO_DATA is unset.

This closes the uncalibrated artifact/query smoke. Full R6 and G0 still need
complete formal calibration and required measurement coverage.

### Recovery checkpoint (2026-11-04)

The Hub run closed with **284/287 new full-weight successes**, six earlier full-weight bundles
reused, and three explicit gaps: one NaN source and two timeout omissions.
The owner stopped the optional retries; original sample rows and inclusion
weights remain unchanged. Row-block extraction is 97/97 complete.

The44-case formal delivery is complete. All87 current formal full-weight identities
have verified manifest/fingerprint headers and HDF5 paths. This includes the
three earlier Smol360 chain endpoints stored outside the original L12 pool.

The current artifact-only zoo has628 models,129,623 measured views,399 valid
full-spectrum links,1,500 lineage rows and150 scores. All six CLI queries
passed; `check` remains uncalibrated. Reports and exact commands are under
`runs/track-a-r3-r6-20261102.rcsy95ap/recovery-20261104T042716Z/finalization/`.
`scope-closure.json` records the accepted omissions and stopped retries.

The merged recovery code retains successful CUDA/CPU SVD behavior. Only an
observed rectangular CUDA INVALID_VALUE error uses reduced QR before the
same full SVD. Packed shapes loaded from JSON are normalized locally. Larger
bounded operator caches retain the exact seeded NumPy operators; precision,
rank rules, stored spectra and artifact schemas remain unchanged. The separate
pilot runtime's hot-file cache fix is archived with its regression/review in
the recovery run; it is not a new library cache backend.

Formal calibration remains blocked. The42 external references are pickle-only
at their frozen pins; they need an explicit safe-source decision. The other37
formal models are not generated. Safe conversion aliases are separate evidence
until an approved, recorded source change; never silently credit a new revision
as an original formal pin.


### Candidate-pair delta pass

After building, measure row-block updates for detected candidates and rebuild:

```bash
python -m zoo.cli deltas --workers auto --compute_device cpu
python -m zoo.cli build
```

Both commands default to `$ZOO_DATA`; the delta command prints an explicit rebuild
command. CUDA follows the evidence CLI's assigned-device policy. Parent blocks
share `$ZOO_DATA/cache/parent-blocks`; `--no-parent_cache` disables this cache.

Each child package gets append-only `deltas/<parent safe>.delta.safetensors`
addenda (`pair_delta_version=1`, evidence version 3). Tensors are `rows[n,2]`,
`delta_scalars[n,3]` (identical fraction, relative update norm, rank) and
`delta_top_sv[n,16]`. Metadata pins both identities and records per-view names,
block heights, statuses and confidence. Missing/incompatible views remain gaps.
The API also accepts lab block heights; default is 256. Rank is limited by block
height. Existing addenda are reused, never silently overwritten; settings
conflicts raise an error. Package core hashes and files stay unchanged; rebuild
provenance hashes the addenda separately.

`diff` uses an available matching pair measurement at the requested row-block
height. Rebuilds tag existing lineage evidence with `+row_block_delta`, retaining
its estimand, direction, tier, score and calibration status. A measured update
does not verify ancestry or predict skill changes.

New artifacts use `EVIDENCE_VERSION=3` and builds `ZOO_VERSION=2`; numerical
conventions remain NUMERICS7/FORMAT2.0. Evidence readers accept versions 1, 2 and 3;
zoo readers accept versions 0, 1 and 2, translating legacy numbered columns to
`row_block_fingerprint`, `full_weight_fingerprint` and `full_weight_spectra`.
Old six-table builds have an empty `layers` table. Current default packages use
evidence version 3; select older conventions explicitly or regenerate evidence.
Rebuild older generations for query clients that require the current `ZOO_VERSION`.


### Approved formal roster calibration smoke (2026-11-05)

The approved `formal166-public-constructions-v2`166-model build is recorded
under `$ZOO_DATA/runs/track-a-roster-v2-calibration-20261105/`. `R5-report.md`
and `verification.json` bind the existing calibration writer and all six CLI
query results. `zoo/` is an immutable build pointer; `calibration.json` uses
k64/block256/seed20261028.165 independent/157 parent fitting pairs exclude
the prior two held-out roots;15 independent evaluation pairs had0threshold
exceedances, with no held-out positive recall estimate.

Example query (existing environment and editable packages):

```bash
python -m zoo_keeper.cli find --zoo "$ZOO_DATA/runs/track-a-roster-v2-calibration-20261105/zoo" --json
```

6/18 imported Raw leaderboard associations verify their revision; default
skill queries exclude unverified associations. Explicit standalone evidence
retains full spectra and has no collector ESD layer table. This internal
snapshot does not decide G0/G1 or establish the5000-candidate retrieval gate.


### Expanded calibrated pilot/formal snapshot (2026-11-05)

`$ZOO_DATA/runs/track-a-expanded-r6-20261105/zoo` combines703distinct
identities from the available628-model historical snapshot and approved
formal166roster.474models have full-weight evidence; corrected current
artifacts take precedence for identical keys. All six CLI commands pass and
the frozen calibration is reused.24/150Raw leaderboard associations verify
their revisions; default skill queries exclude unverified associations.

`REPORT.md` and `G0-development-report.md` record the negative diagnostics:
immediate-parent false matches8.276786/query in the small/incompletely labelled
truth pool; median alpha difference0.427726 exceeds0.1 over24025matrix fits.
Retain full-weight spectra. No5000-candidate,held-out parent recall,G0/G1 or
per-query error-control claim. No new weights or measurements were needed.
