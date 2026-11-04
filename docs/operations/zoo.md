# Build and query a zoo

A zoo combines pinned model metadata, fingerprints, measured scores and lineage
in one snapshot. Building uses existing artifacts; querying needs no server.

## Build

Use the existing environment and set `ZOO_DATA` to the artifact root. From
Model-Zoo:

```bash
python -m zoo.cli build \
  --manifests "$MANIFESTS" \
  --fingerprints "$FINGERPRINTS" \
  --data_root "$ZOO_DATA" \
  --out "$ZOO_DATA/builds/current"
```

`MANIFESTS` contains one manifest per pinned model. `FINGERPRINTS` contains the
canonical T1/T2 files; extra T1 block sizes are kept alongside them. Multiple
fingerprint directories may follow `--fingerprints`.

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
| `coverage` | Measured/skipped tensors and reasons |
| `lineage` | Candidate relationships with evidence tiers and statistics |
| `scores` | Measured scores and revision verification |
| `predictions` | Separate prediction records; empty in v0 |

Full per-view spectra remain in HDF5 files linked by the model table. Artifact
paths are relative to `ZOO_DATA`. Retrieval compares T1 row-block fingerprints
within the same architecture signature, giving comparable coordinates and
shapes. Full-matrix T2 sketches have a different measurement scope.

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

## Validation status

The 2026-11-02 scoped CLI smoke passed on 85 models: 43 measured checkpoints and
42 metadata-only entries, retaining all 43 full-spectrum links. Rebuild it with:

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

The finished pilot has97/97 new T1 successes and241/287 new T2 successes;46
T2 failures remain explicit. Six previous T2 bundles are reused, giving247
complete pilot T2 bundles with full spectra and all128/256/512 companions.

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

The Hub run closed with **284/287 new T2 successes**, six earlier T2 bundles
reused, and three explicit gaps: one NaN source and two timeout omissions.
The owner stopped the optional retries; original sample rows and inclusion
weights remain unchanged. T1 is97/97 complete.

The44-case formal delivery is complete. All87 current formal T2 identities
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
