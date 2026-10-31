# Zoo Store and v0 Queries Implementation Plan

Revised 2026-10-31: interface fixes found in the pre-handoff cross-check (T1-only
retrieval, calibrated statistic, revisions, fixtures). Each is listed in
`Zoo-Keeper/handoff/STATUS.md`, "Plan and interface change log".

**Agents: never run this plan's "Commit" steps.** Leave changes uncommitted
and record the suggested message instead (`Zoo-Keeper/handoff/DIRECTIVES.md`,
D10.1).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the extractor's outputs into the **Model Zoo v0**: a
rebuildable, queryable database of models. A Python library and CLI answer six
queries:

- `find`: filter models;
- `info`: one model's card;
- `similar`: nearest models by weights;
- `family`: lineage and license chain;
- `diff`: what changed between two models;
- `check`: is the declared base consistent?

It must be ready for internal use at gate G0 (2026-12-04).

**Architecture:** two packages in two repositories.
- **Build, in Model-Zoo** (`Model-Zoo/zoo/`, Tasks 1–6 and the `build`
  command):
  - `python -m zoo.cli build` reads the extractor's manifests and
    fingerprints, an optional selection frame, score sources, and optional
    lineage edges.
  - It writes Parquet tables plus per-architecture retrieval matrices into a
    zoo directory, swapped in atomically.
  - Small zoos may be kept in the Model-Zoo repository by the person; the big
    database lives parallel to the repositories, under `$ZOO_DATA`.
- **Query, in Zoo-Keeper** (`Zoo-Keeper/zoo_keeper/`, Tasks 7–8):
  - `Zoo(path)` opens the tables in an in-memory DuckDB session and answers
    queries. There is no database server.
  - Zoo-Keeper imports `zoo` and `evidence` from Model-Zoo after
    `pip install -e Model-Zoo`.
- **Reuse:** `diff` and `check` reuse the extractor's
  `evidence.compare.diff_fingerprints`, and weight similarity reuses its sketch
  conventions.
- **Later:** `recommend`, score predictions, and the MCP wrapper come in zoo v1
  (a later plan).

**Tech Stack:** Python 3.10, pandas, pyarrow, duckdb, numpy, and the `evidence`
package from the extractor plan.

**Spec:** `Zoo-Keeper/docs/research/2026-10-27-merged-research-plan.md`,
section 6.7 (the zoo and its query framework), with sections 0.1, 1, and 12.
**Depends on:** `Model-Zoo/docs/superpowers/plans/2026-10-28-weight-evidence-extractor.md`.
Tasks 1–4 here need extractor Tasks 5 (manifest), 6 (fingerprints,
`load_fingerprint`, `fingerprint_tensors`), and 7 (the sampled `models.csv`).
Tasks 5–8 here also need extractor Task 8 (`diff_fingerprints`,
`pair_similarity`, `check_verdict`). Task 9's calibration file is optional
input. File names follow the extractor's "Artifact names" rule.

## Global Constraints

- **Environment.** The existing `esd_ind` conda environment on the work
  machine. The build needs nothing new (pyarrow comes from the extractor
  plan). The queries add `duckdb`, pinned in Zoo-Keeper's `requirements.txt`
  to the installed version.
- **Model key.** `"<repo_id>@<revision>"`. The revision is a 40-character SHA,
  or `local:<sha256>` for local models.
- **No guessing between revisions.** A bare `repo_id` that matches several
  revisions raises `AmbiguousModel` and lists them. The zoo never guesses a
  revision.
- **Zoo directory layout:**

  ```
  <zoo>/zoo.json               # ZOO_VERSION, build time, inputs with sha256, row counts
  <zoo>/models.parquet  views.parquet  coverage.parquet
  <zoo>/lineage.parquet scores.parquet predictions.parquet
  <zoo>/retrieval/<arch_signature>.npy    # float16 [n_models, RETRIEVAL_VIEWS * 1024]
  <zoo>/retrieval/<arch_signature>.json   # {"models": [...keys in row order], "views": [...names]}
  ```

- **Atomic build.** The build writes `<zoo>.building/`, renames any existing
  `<zoo>` to `<zoo>.previous/`, renames the new directory into place, then
  deletes `.previous`. An interrupted build leaves the old zoo intact.
- **Deterministic and rebuildable.** The same inputs give identical table
  contents: rows are sorted by `model_key`, then `view_index`. Fingerprint and
  spectra paths are stored relative to `--data_root`.
- **Measured and predicted are never mixed.**
  - `scores` holds measurements only. `revision_verified` is true only when
    the source's commit equals the zoo revision; unverified scores are kept and
    labeled.
  - `predictions` holds predictions only. It is empty in v0.
  - `info` reports the two separately.
- **SQL is always parameterized.** User strings never enter SQL text.
- **Retrieval constants.**
  - `RETRIEVAL_VIEWS = 32`. Retrieval uses the top-left 32×32 block of each
    view's sketch, flattened, unit-normalized per view, and stored as float16.
  - The cosine of two concatenated rows, divided by `RETRIEVAL_VIEWS`, equals
    the mean per-view cosine.
  - **Weight similarity compares only models with the same architecture
    signature.** Other models are never ranked by weights.
  - **Retrieval uses T1 fingerprints only.** A T2 sketch covers the full
    matrix and is not comparable with a row-block sketch. T2 passes write a
    standard T1 file for free, so the lab and anchors still take part.
- **Architecture signature.**
  - **Config-based:** when the manifest config has `hidden_size` and
    `num_hidden_layers`, the signature is `"cfg:" + sha256(json(sorted subset))[:12]`
    over `model_type`, `hidden_size`, `num_hidden_layers`,
    `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, and
    `num_experts`. Vocabulary size is excluded, so a fine-tune that extends the
    vocabulary keeps its parent's signature.
  - **Shape-based otherwise:** `"shape:" + sha256(sorted matrix shapes)[:12]`,
    excluding views whose row count exceeds 8× the median.
- `ZOO_VERSION = "0"`.

## Review Focus

1. **Models without fingerprints** (T0-only, `no_access`, `.bin`-only, GGUF-only)
   must appear in `find` and `info` with their coverage status. `similar`
   raises `NotComparable` with the reason, and never crashes or silently
   returns nothing. Tested in Tasks 3 and 7.
2. **A fine-tune with an extended vocabulary** must share its parent's signature,
   so `similar` finds the parent. Tested in Tasks 1 and 4.
3. **A repository with several pinned revisions** must not have its scores
   merged across revisions: a bare `repo_id` raises `AmbiguousModel`, and
   unverified scores are labeled. Tested in Tasks 2 and 7.
4. **User strings containing quotes or SQL fragments,** in `license`, `text`,
   or `modality` filters, must be treated as literals. Tested in Task 7.
5. **A build interrupted mid-way** must leave the previous zoo readable and
   unchanged. Tested in Task 6.

---

### Task 1: Package skeleton, schema, keys, and signatures

**Files:**
- Create: `zoo/__init__.py` (exports `ZOO_VERSION = "0"`)
- Create: `zoo/schema.py`
- Create: `zoo/testing.py` with `make_universe(root: Path) -> SimpleNamespace`,
  the synthetic-zoo builder below. It lives in the package so that Zoo-Keeper's
  tests can reuse it (Task 7).
- Create: `zoo/tests/conftest.py`
  - Adds the repository root to `sys.path`.
  - Defines the fixture `universe(tmp_path)` as `make_universe(tmp_path)`. It
    writes manifests and
    fingerprints for a synthetic zoo. It returns
    `SimpleNamespace(root, manifests, fingerprints)`: the data root, the
    manifest directory, and the fingerprint directory.
    - Fingerprints come from `evidence.fingerprint.fingerprint_tensors`, with
      `identity = {"repo_id": ..., "revision": ...}`, written to
      `<safe>.t1.fingerprint.safetensors`.
    - Manifests are JSON dicts with the extractor's Task 5 keys, written to
      `<safe>.manifest.json`.
    - All "llama" models use `hidden_size = 512`, `num_hidden_layers = 2`, and
      matrices of 512 or more rows.
    - The constants are `SHA1` … `SHA9 = str(i) * 40`, `SHA10 = "a" * 40`,
      `SHA11 = "b" * 40`, and `SHA12 = "c" * 40`.
  - The universe contains these models:

    | Key | What it is |
    | --- | --- |
    | `org/parent@<sha1>` | the base |
    | `org/child@<sha2>` | parent plus a rank-4 update; declares the parent |
    | `org/child-vocab@<sha3>` | like the child, with 8 extra embedding rows |
    | `org/sibling@<sha4>` | parent plus a different rank-4 update |
    | `org/dup@<sha5>` | a byte copy of the child |
    | `org/unrelated@<sha6>` | the same config, independent weights; declares the base `other/foundation`, which is not in the zoo |
    | `org/other-arch@<sha7>` | hidden size 384 |
    | `org/t0only@<sha8>` | `access = "t0_only"`, `weight_format = "bin"`, no fingerprint |
    | `org/gated@<sha9>` | `access = "no_access"` |
    | `org/child@<sha10>` | a second revision of the child with byte-identical weights (a README-only commit), so it shares the child's `dup_group` |
    | `org/decider@<sha11>` | a System One model: the parent plus a classification head `score.weight [2, 512]`; tagged `system-one`, with labels `["no", "yes"]`; declares the parent |
    | `org/parent-GGUF@<sha12>` | GGUF-only (`weight_format = "gguf"`, `access = "t0_only"`, GGUF architecture `llama`); declares the parent |

  - Licenses: the parent is `apache-2.0`, both child revisions are `mit`, the
    duplicate is `mit`, and other-arch's license is the string `it's "odd"`
    (for the injection test). Every other model is `apache-2.0`.
  - Also defines the fixture `zoo_models(universe)`: a small DataFrame with
    `model_key`, `repo_id`, and `revision` read straight from the universe's
    manifests, for Task 2.
- Test: `zoo/tests/test_schema.py`

**Interfaces:**
- Produces:
  - `model_key(repo_id: str, revision: str) -> str`.
  - `split_key(key: str) -> tuple[str, str]`.
  - `arch_signature(config: dict, view_shapes: list[tuple[int, int]]) -> str`.
  - The column lists `MODELS_COLUMNS`, `VIEWS_COLUMNS`, `COVERAGE_COLUMNS`,
    `LINEAGE_COLUMNS`, `SCORES_COLUMNS`, and `PREDICTIONS_COLUMNS`, plus
    `empty_table(columns) -> pd.DataFrame`. The columns are:
    - `models`: `model_key`, `repo_id`, `revision`, `access`, `weight_format`,
      `is_adapter`, `license`, `gated`, `declared_bases`,
      `base_model_relation`, `architecture` (the config's `model_type`, or
      the GGUF architecture), `is_decision`, `system_one`, `decision_type`,
      `num_labels`, `labels` (from the manifest's `decision` block),
      `arch_signature`, `n_views`, `modality`,
      `pipeline_tag`, `downloads`, `params`, `stratum`, `inclusion_prob`,
      `dup_group`, `fingerprint_t1`, `fingerprint_t2`, `spectra_t2`,
      `coverage_status` (`measured`, `partial`, `not_measured`, `t0_only`, or
      `no_access`), and `skipped_tensors`.
    - `views`: `model_key`, `view_index`, `name`, `geometry`, `component`,
      `n_rows`, `n_cols`, `r0`, `r1`, `block_sha256`, `tier`.
    - `coverage`: `model_key`, `tensor_name`, `decode`, `geometry`, `status`,
      `reason`.
    - `lineage`: `child_key`, `parent_key` (a zoo key, or `"<repo_id>@external"`),
      `estimand` (`same_root`, `ancestor`, or `parent`), `edge_type`
      (`finetune`, `adapter`, `merge`, `quantized`, `abliteration`, or
      `unknown`; the first four match the Hub's `base_model_relation`),
      `evidence_tier` (`verified`, `detected`, or `declared`), `evidence`,
      `score`, `p_value`, `calibrated`, `method_version`.
    - `scores`: `model_key`, `repo_id`, `benchmark`, `skill`, `value`,
      `metric`, `source`, `source_revision`, `revision_verified`, `harness`,
      `n_items`.
    - `predictions`: `model_key`, `skill`, `value`, `lower`, `upper`, `method`,
      `inputs`.

- [ ] **Step 1: Write the failing tests**

```python
def test_keys_roundtrip():
    k = model_key("org/m", "a" * 40)
    assert k == "org/m@" + "a" * 40 and split_key(k) == ("org/m", "a" * 40)
    assert split_key("local/path@local:" + "b" * 64)[1].startswith("local:")

def test_signature_ignores_vocab_and_uses_config():
    cfg = {"model_type": "llama", "hidden_size": 512, "num_hidden_layers": 2, "vocab_size": 1000}
    assert arch_signature(cfg, []) == arch_signature({**cfg, "vocab_size": 1008}, [])
    assert arch_signature(cfg, []).startswith("cfg:")
    assert arch_signature({**cfg, "hidden_size": 384}, []) != arch_signature(cfg, [])

def test_shape_signature_fallback_excludes_tall_views():
    shapes = [(512, 512)] * 10 + [(50000, 512)]
    assert arch_signature({}, shapes) == arch_signature({}, [(512, 512)] * 10 + [(50008, 512)])
    assert arch_signature({}, shapes).startswith("shape:")
```

- [ ] **Step 2:** Run `python -m pytest zoo/tests/test_schema.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `zoo/schema.py` with the column lists and rules in Global Constraints.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git add zoo && git commit -m "zoo: schema, model keys and architecture signatures"`

### Task 2: Score sources with revision verification

**Files:**
- Create: `zoo/scores.py`
- Test: `zoo/tests/test_scores.py`

**Interfaces:**
- Consumes: Task 1 `model_key` and `SCORES_COLUMNS`.
- Produces:
  - `LEADERBOARD_V2_SKILLS`: the benchmarks
    `IFEval Raw` → `instruction_following`, `BBH Raw` → `reasoning`,
    `MATH Lvl 5 Raw` → `math`, `GPQA Raw` → `science_qa`,
    `MUSR Raw` → `reasoning`, and `MMLU-PRO Raw` → `knowledge`. These are the
    raw accuracies in [0, 1], not the floor-clipped normalized scores.
  - `load_leaderboard_v2(path: Path, zoo_models: pd.DataFrame) -> pd.DataFrame`.
    It reads the Open LLM Leaderboard v2 export: `fullname` gives the repo,
    `Model sha` the commit, and the `* Raw` columns the values.
    - It emits one row per zoo model and benchmark for every zoo revision of
      that repository, with `source = "open_llm_leaderboard_v2"`,
      `metric = "raw_accuracy"`, `source_revision = Model sha`, and
      `revision_verified = (Model sha == zoo revision)`.
    - Repositories match case-insensitively. An empty `Model sha` gives
      `revision_verified = False`.
    - Some repositories were evaluated at both bfloat16 and float16, sometimes
      with different results at the same commit. Both rows are kept, with
      `harness = "open_llm_leaderboard_v2:<Precision>"`.
    - Rows for repositories not in the zoo are dropped.
  - `load_scores_csv(path: Path, mapping: dict, zoo_models: pd.DataFrame) -> pd.DataFrame`.
    This is a generic adapter, for example for Every Eval Ever. `mapping`
    names the source columns for `repo_id`, `revision` (optional), `benchmark`,
    `value`, `skill` (optional), and `source`. A missing revision gives
    `revision_verified = False`.

- [ ] **Step 1: Write the failing tests** (a 3-row CSV fixture with the real column names from `Model-Research/data/sampled_leaderboard.csv`)

```python
def test_leaderboard_verified_join(zoo_models, lb_csv):
    s = load_leaderboard_v2(lb_csv, zoo_models)
    assert set(s.columns) == set(SCORES_COLUMNS)
    row = s[(s.model_key == "org/child@" + SHA2) & (s.benchmark == "IFEval Raw")].iloc[0]
    assert row.skill == "instruction_following" and 0 <= row.value <= 1 and bool(row.revision_verified)

def test_other_revision_unverified_not_merged(zoo_models, lb_csv):
    s = load_leaderboard_v2(lb_csv, zoo_models)
    other = s[s.model_key == "org/child@" + SHA10]
    assert len(other) and not other.revision_verified.any()

def test_generic_csv_without_revision(zoo_models, tmp_path):
    p = tmp_path / "eee.csv"; p.write_text("model,bench,score\norg/parent,arc,0.5\n")
    s = load_scores_csv(p, {"repo_id": "model", "benchmark": "bench", "value": "score", "source": "every_eval_ever"}, zoo_models)
    assert (~s.revision_verified).all() and s.source.eq("every_eval_ever").all()
```

- [ ] **Step 2:** Run `python -m pytest zoo/tests/test_scores.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `zoo/scores.py`. When there is no explicit `source`
  column, the mapping's `source` value is used as a constant.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "zoo: score sources with revision-verified joins"`

### Task 3: Build the tables in memory (models, views, coverage, duplicates)

**Files:**
- Create: `zoo/tables.py`
- Modify: `zoo/tests/conftest.py`, adding the fixture
  `tables(universe) = build_tables(...)` on the universe.
- Test: `zoo/tests/test_tables.py`

**Interfaces:**
- Consumes:
  - Task 1 (schema, `arch_signature`) and Task 2 (score tables);
  - `evidence.fingerprint.load_fingerprint`;
  - manifest JSON files (`*.manifest.json`, extractor Task 5);
  - fingerprint files (`*.t1.fingerprint.safetensors` and
    `*.t2.fingerprint.safetensors`) and `*.t2.spectra.h5`;
  - the sampled `models.csv` (extractor Task 7, `sample`), which carries the
    frame columns plus `inclusion_prob`: `modality`, `pipeline_tag`,
    `downloads`, `params`, `stratum`, and `inclusion_prob`. Models not in it,
    such as lab models, get the stratum `lab` and empty frame fields.
- Produces:
  - `build_tables(manifests: Path, fingerprints: list[Path], data_root: Path, frame: Path | None = None, score_tables: Sequence[pd.DataFrame] = ()) -> dict[str, pd.DataFrame]`.
    It is pure: it writes nothing. It returns `models`, `views`, `coverage`,
    `scores`, and `predictions` (empty), sorted as in Global Constraints.
  - `duplicate_groups(views: pd.DataFrame) -> dict[str, str]`, mapping each
    `model_key` to its `dup_group`. Models whose T1 view names and block
    hashes are all identical share a group, named after the smallest key.
    This means "identical on every measured block": strong evidence of a copy,
    not proof of identical files. `info` says so.
  - Joining rules:
    - fingerprints join to manifests on `(repo_id, revision)` from the
      fingerprint metadata;
    - `views` come from the T1 fingerprint when there is one, otherwise from
      T2; `tier` records which;
    - two fingerprints of the same tier for one model (for example, from two
      `--fingerprints` directories) are an error that names both files,
      never a silent pick;
    - `coverage_status` uses the extractor's closed coverage vocabulary
      (Task 4b):
      - `measured` when no tensor is `skipped` with a reason other than
        `not_a_matrix`, `non_float_buffer`, `vector_like`, or
        `expert_not_sampled` (`consumed` and `duplicate` never count against
        it);
      - `partial` otherwise;
      - `not_measured` when access was granted but no fingerprint exists yet;
      - the manifest's `access` value (`t0_only` or `no_access`) when there is
        no fingerprint for those reasons.

- [ ] **Step 1: Write the failing tests**

```python
def test_models_and_coverage(tables):
    m = tables["models"].set_index("model_key")
    assert m.loc["org/t0only@" + SHA8, "coverage_status"] == "t0_only"
    assert m.loc["org/gated@" + SHA9, "coverage_status"] == "no_access"
    assert m.loc["org/child@" + SHA2, "n_views"] > 0 and not Path(m.loc["org/child@" + SHA2, "fingerprint_t1"]).is_absolute()
    assert m.loc["org/child-vocab@" + SHA3, "arch_signature"] == m.loc["org/parent@" + SHA1, "arch_signature"]
    assert list(tables["models"].model_key) == sorted(tables["models"].model_key)
    assert m.loc["org/decider@" + SHA11, "system_one"] and m.loc["org/decider@" + SHA11, "num_labels"] == 2
    assert m.loc["org/parent-GGUF@" + SHA12, "coverage_status"] == "t0_only"

def test_duplicates_grouped(tables):
    m = tables["models"].set_index("model_key")
    assert m.loc["org/dup@" + SHA5, "dup_group"] == m.loc["org/child@" + SHA2, "dup_group"]
    assert m.loc["org/sibling@" + SHA4, "dup_group"] != m.loc["org/child@" + SHA2, "dup_group"]
```

- [ ] **Step 2:** Run `python -m pytest zoo/tests/test_tables.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `zoo/tables.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "zoo: in-memory tables for models, views, coverage and duplicates"`

### Task 4: Retrieval matrices and weight similarity

**Files:**
- Create: `zoo/retrieval.py`
- Modify: `zoo/tests/conftest.py`, adding the fixture `staged_zoo(tables, tmp_path)`:
  a directory where `build_retrieval` has been run on `tables["models"]` and
  `tables["views"]`.
  Tests here and in Task 5 use it in place of `built_zoo`.
- Test: `zoo/tests/test_retrieval.py`

**Interfaces:**
- Consumes: Task 3 tables, `evidence.fingerprint.load_fingerprint`.
- Produces:
  - `select_retrieval_views(views: pd.DataFrame) -> list[str]`, given the
    `views` rows of one signature's models.
    - Candidates are the view names present in at least 90% of those models,
      excluding views whose `n_rows` exceeds 8× the median (Task 1's rule).
    - Take `RETRIEVAL_VIEWS` of them evenly spaced over the sorted list, or all
      of them when fewer exist.
  - `build_retrieval(zoo_dir: Path, models: pd.DataFrame, views: pd.DataFrame, data_root: Path) -> None`.
    It writes `retrieval/<sig>.npy` and `.json` for every signature that has at
    least 2 models with a T1 fingerprint. T2-only models are left out (Global
    Constraints). Views a model lacks become zero blocks, and their count is
    recorded in the `.json`.
  - `nearest(zoo_dir: Path, key: str, sig: str, k: int) -> list[tuple[str, float]]`.
    It returns other models with their mean per-view cosine (concatenated dot
    product divided by the number of views present in both), using exact
    search. Results are sorted by cosine descending, then by key. The query
    model itself is excluded.

- [ ] **Step 1: Write the failing tests**

```python
def test_parent_is_nearest_to_child_and_vocab_child(tables, staged_zoo):
    sig = tables["models"].set_index("model_key").loc["org/child@" + SHA2, "arch_signature"]
    top = nearest(staged_zoo, "org/child-vocab@" + SHA3, sig, k=3)
    assert {"org/parent@" + SHA1, "org/child@" + SHA2} & {k for k, _ in top[:2]}
    scores = dict(nearest(staged_zoo, "org/child@" + SHA2, sig, k=10))
    assert scores["org/parent@" + SHA1] > scores["org/unrelated@" + SHA6] + 0.5

def test_other_arch_not_in_same_matrix(tables, staged_zoo):
    sig = tables["models"].set_index("model_key").loc["org/child@" + SHA2, "arch_signature"]
    keys = json.loads((staged_zoo / "retrieval" / f"{sig}.json").read_text())["models"]
    assert "org/other-arch@" + SHA7 not in keys and "org/t0only@" + SHA8 not in keys
```

- [ ] **Step 2:** Run `python -m pytest zoo/tests/test_retrieval.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `zoo/retrieval.py`. Load each `.npy` with
  `np.load(..., mmap_mode="r")`, and upcast to float32 for the dot products.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "zoo: per-architecture retrieval matrices and exact weight similarity"`

### Task 5: v0 lineage edges (declared, shared tensors, sketch similarity)

**Files:**
- Create: `zoo/lineage.py`
- Modify: `zoo/tests/conftest.py`, adding the fixture `null_dict`: a
  calibration dict in the extractor's format (Task 8), with `sketch_k = 64`,
  `row_block = 256`, `tier = "t1"`, and
  `null_quantiles = np.linspace(-0.1, 0.3, 101)`.
- Test: `zoo/tests/test_lineage.py`

**Interfaces:**
- Consumes: Tasks 1, 3, and 4, and the extractor's calibration JSON (Task 9
  `write_calibration`), which is optional.
- Produces:
  - `declared_edges(models: pd.DataFrame) -> pd.DataFrame`.
    - `estimand = "parent"`, `evidence_tier = "declared"`,
      `evidence = "declared_metadata"`.
    - The parent is the zoo key when exactly one revision of the base repo is
      in the zoo, otherwise `"<repo>@external"`.
  - `shared_tensor_edges(views: pd.DataFrame, dup_groups: dict) -> pd.DataFrame`.
    - Pairs with identical block hashes on more than 0 but fewer than all
      shared views get `estimand = "same_root"`, `evidence_tier = "verified"`,
      `evidence = "shared_tensors"`, and `score` equal to the fraction of
      shared identical views.
    - The pair is stored once, with `child_key` < `parent_key`, because the
      relation is undirected.
    - Exact duplicates are excluded (they are `dup_group`).
  - `similarity_edges(zoo_dir: Path, models: pd.DataFrame, data_root: Path, k: int = 5, null: dict | None = None) -> pd.DataFrame`.
    - For each model, take its top-k neighbors from Task 4 as candidates.
    - For each candidate pair, `score` is
      `evidence.compare.pair_similarity(a, b, sketch_k)["median_cosine"]` on
      the two T1 fingerprints, with `sketch_k` taken from `null` (64 without
      one). This is the statistic the calibration file describes; the
      retrieval cosine only picks the candidates.
    - `estimand = "same_root"`, `evidence_tier = "detected"`, and
      `evidence = "sketch_similarity"`.
    - With `null`, `p_value` comes from the null quantiles and
      `calibrated = True`. Without it, `p_value` is NaN and `calibrated = False`.
    - These p-values are for display. Their floor, `1 / (n + 1)`, cannot
      support per-query error control over a large pool; the P1 engine's
      calibrated edges replace them.
  - `merge_edges(*tables: pd.DataFrame, external: Path | None = None) -> pd.DataFrame`.
    - It concatenates the tables, appends the P1 engine's Parquet if given, and
      drops exact duplicates of
      `(child_key, parent_key, estimand, evidence)`.
    - `method_version` is `"zoo-v0"` for the edges built here.

- [ ] **Step 1: Write the failing tests**

```python
def test_declared_and_external(tables):
    e = declared_edges(tables["models"])
    child = e[e.child_key == "org/child@" + SHA2].iloc[0]
    assert child.parent_key == "org/parent@" + SHA1 and child.evidence_tier == "declared"
    assert (e.parent_key.str.endswith("@external")).any()      # a base not in the zoo

def test_shared_tensor_edges_exclude_duplicates(tables):
    e = shared_tensor_edges(tables["views"], duplicate_groups(tables["views"]))
    pairs = set(zip(e.child_key, e.parent_key))
    assert ("org/child@" + SHA2, "org/dup@" + SHA5) not in pairs and ("org/dup@" + SHA5, "org/child@" + SHA2) not in pairs
    assert e.estimand.eq("same_root").all() and e.evidence_tier.eq("verified").all()

def test_similarity_edges_uncalibrated_then_calibrated(staged_zoo, tables, universe, null_dict):
    e = similarity_edges(staged_zoo, tables["models"], universe.root, k=2)
    assert (~e.calibrated).all() and e.p_value.isna().all()
    e2 = similarity_edges(staged_zoo, tables["models"], universe.root, k=2, null=null_dict)
    assert e2.calibrated.all() and e2.p_value.between(0, 1).all()
```

- [ ] **Step 2:** Run `python -m pytest zoo/tests/test_lineage.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `zoo/lineage.py`. The p-value is the upper-tail
  fraction of `null_quantiles` at or above the observed cosine, floored at
  `1 / (len(null_quantiles) + 1)`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "zoo: v0 lineage edges from metadata, shared tensors and sketch similarity"`

### Task 6: Assemble and write the zoo atomically

**Files:**
- Create: `zoo/build.py`
- Modify: `zoo/tests/conftest.py`, adding the fixture
  `built_zoo(universe, tmp_path)`: the result of `build_zoo` on the universe.
- Test: `zoo/tests/test_build.py`

**Interfaces:**
- Consumes: Task 3 `build_tables`, Task 4 `build_retrieval`, and Task 5
  `declared_edges`, `shared_tensor_edges`, `similarity_edges`, and
  `merge_edges`.
- Produces:
  - `build_zoo(manifests: Path, fingerprints: list[Path], out: Path, data_root: Path, frame: Path | None = None, score_tables: Sequence[pd.DataFrame] = (), lineage_edges: Path | None = None, null: Path | None = None) -> Path`.
    It runs these steps into `<out>.building/`, then does the atomic swap from
    Global Constraints:
    1. tables;
    2. retrieval;
    3. lineage;
    4. Parquet;
    5. `zoo.json`.

    It returns `out`.
  - `open_tables(zoo: Path) -> dict[str, pd.DataFrame]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_writes_all_tables(built_zoo):
    t = open_tables(built_zoo)
    assert set(t) == {"models", "views", "coverage", "lineage", "scores", "predictions"}
    assert json.loads((built_zoo / "zoo.json").read_text())["zoo_version"] == "0"
    assert list((built_zoo / "retrieval").glob("*.npy"))

def test_deterministic_rebuild(universe, tmp_path):
    a = open_tables(build_zoo(universe.manifests, [universe.fingerprints], tmp_path / "a", data_root=universe.root))
    b = open_tables(build_zoo(universe.manifests, [universe.fingerprints], tmp_path / "b", data_root=universe.root))
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])

def test_interrupted_build_keeps_previous(universe, tmp_path, monkeypatch):
    zoo = build_zoo(universe.manifests, [universe.fingerprints], tmp_path / "zoo", data_root=universe.root)
    before = (zoo / "models.parquet").read_bytes()
    monkeypatch.setattr("zoo.build.build_retrieval", Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        build_zoo(universe.manifests, [universe.fingerprints], zoo, data_root=universe.root)
    assert (zoo / "models.parquet").read_bytes() == before and not (tmp_path / "zoo.building").exists()
```

- [ ] **Step 2:** Run `python -m pytest zoo/tests/test_build.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `zoo/build.py`.
  - Write tables with `pyarrow.parquet.write_table`.
  - `zoo.json` records `zoo_version`, the input file sha256 values, row counts,
    and the build time. Time is the only non-deterministic field, and it lives
    only in `zoo.json`.
  - On any exception, delete `.building` and re-raise.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "zoo: atomic deterministic zoo build"`

### Task 7: The query API (`Zoo`), in Zoo-Keeper

This task and Task 8's query commands are built in the Zoo-Keeper repository.
Its paths below are relative to `Zoo-Keeper/`.

**Files:**
- Create: `zoo_keeper/__init__.py` and `zoo_keeper/query.py`
- Create: `requirements.txt` pinning `duckdb` (and anything else the queries
  import), plus a note that Model-Zoo is installed with
  `pip install -e ../Model-Zoo`.
- Create: `tests/conftest.py`, with the fixtures `universe` (from
  `zoo.testing.make_universe`), `built_zoo` (from `zoo.build.build_zoo`), and
  `z`.
- Test: `tests/test_query.py`

**Interfaces:**
- Consumes: Tasks 1–6 (`zoo.schema`, `zoo.build.open_tables`,
  `zoo.retrieval.nearest`), `evidence.compare.diff_fingerprints`, and
  `evidence.compare.check_verdict`.
- Test fixture: `z = Zoo(built_zoo, data_root=universe.root)`, in Zoo-Keeper's
  `tests/conftest.py`.
- Produces:
  - Exceptions: `AmbiguousModel(ValueError)` (its message lists the keys),
    `UnknownModel(KeyError)`, and `NotComparable(ValueError)` (with a
    `.reason`).
  - `Zoo(path: Path, data_root: Path | None = None)`. It opens an in-memory
    DuckDB connection with views over each Parquet file. Methods:
    - `resolve(model: str) -> str`, which accepts a key, or a `repo_id` with
      exactly one revision.
    - `find(modality=None, license=None, params_min=None, params_max=None, weight_format=None, access=None, gated=None, text=None, skill=None, min_score=None, verified_only=True, decision=None, system_one=None, architecture=None, limit=50) -> pd.DataFrame`.
      - The columns are `model_key`, `repo_id`, `modality`, `params`,
        `license`, `downloads`, `coverage_status`, and `dup_group`, sorted by
        `downloads` descending, then `model_key`.
      - `license` accepts a string or a list; `text` is a case-insensitive
        substring of `repo_id`.
      - `decision` and `system_one` filter on the decision columns;
        `architecture` matches the `architecture` column exactly.
      - `skill` and `min_score` filter on measured scores. Per model, the
        mean of its rows for that skill must be at least `min_score`; only
        revision-verified rows count when `verified_only` is set.
    - `info(model) -> dict`, with these keys:
      - `identity`, `metadata`, `coverage` (status, skipped-tensor reasons with
        counts, skipped files);
      - `measured_scores` (a list of dicts with `revision_verified`);
      - `predictions` (an empty list in v0) and `predictions_note`
        ("predictions arrive in zoo v1");
      - `lineage`, holding `parents`, `children`, and `same_root`, each a list
        of `{key, estimand, evidence_tier, evidence, score, p_value}`;
      - `duplicates`, labeled "identical on every measured block";
      - `decision`: `is_decision`, `system_one`, `decision_type`,
        `num_labels`, `labels`, and the names of the model's `head` views;
      - `fingerprints` (relative paths).
    - `similar(model, k=10, collapse_duplicates=True) -> pd.DataFrame`.
      - The columns are `model_key`, `cosine`, `dup_group`, and `license`.
      - Members of the query model's own `dup_group` are never returned. They
        are listed in `info(...)["duplicates"]` instead.
      - With `collapse_duplicates`, each other `dup_group` keeps its
        highest-downloads member.
      - It raises `NotComparable` when the model has no fingerprint (reason =
        its coverage status), has only a T2 fingerprint (`no_t1_fingerprint`),
        or is alone in its signature (`no_same_architecture_models`).
    - `family(model, depth=2) -> dict`, with keys `ancestors`, `descendants`,
      `same_root`, `license_chain` (`[(key, license), ...]` from the model up
      through the best-evidence parents), and `license_differs`.
      - Ancestors and descendants follow directed edges only (`estimand` of
        `parent` or `ancestor`). Among a child's parent edges, the strongest
        evidence tier wins: verified, then detected, then declared.
      - Undirected `same_root` edges are listed under `same_root` and never
        enter the chain.
    - `diff(a, b, null: Path | None = None) -> dict`. It loads both models'
      fingerprints at the same tier: T1 if both have it, otherwise T2. If they
      share no tier, it raises `NotComparable("no_common_tier")`. It returns
      `evidence.compare.diff_fingerprints`; `null` is a calibration JSON path.
    - `check(model, null: Path | None = None) -> dict`: the diff against the
      declared base when it is in the zoo, plus a top-level `verdict` from
      `evidence.compare.check_verdict`. It returns
      `{"verdict": "base_not_in_zoo"}` otherwise.

- [ ] **Step 1: Write the failing tests**

```python
def test_find_filters_and_literal_strings(z):
    df = z.find(text="child")
    assert set(df.model_key) >= {"org/child@" + SHA2, "org/child-vocab@" + SHA3}
    assert "org/child@" + SHA2 in set(z.find(license="mit").model_key)
    assert set(z.find(license=["mit"]).license) == {"mit"}
    assert len(z.find(license='it\'s "odd"')) == 1
    assert len(z.find(text="'; DROP TABLE models; --")) == 0 and len(z.find()) > 0

def test_resolve_ambiguous_and_unknown(z):
    with pytest.raises(AmbiguousModel) as err:
        z.resolve("org/child")
    assert SHA2 in str(err.value) and SHA10 in str(err.value)
    with pytest.raises(UnknownModel):
        z.resolve("org/missing")

def test_info_separates_measured_and_predicted(z):
    card = z.info("org/child@" + SHA2)
    assert card["predictions"] == [] and all("revision_verified" in s for s in card["measured_scores"])
    assert card["coverage"]["status"] in {"measured", "partial"}
    assert z.info("org/t0only@" + SHA8)["coverage"]["status"] == "t0_only"

def test_similar_and_not_comparable(z):
    df = z.similar("org/child@" + SHA2, k=5)
    assert df.iloc[0].model_key in {"org/parent@" + SHA1, "org/sibling@" + SHA4, "org/child-vocab@" + SHA3,
                                    "org/decider@" + SHA11}
    assert "org/dup@" + SHA5 not in set(df.model_key)               # own duplicate group excluded
    assert "org/dup@" + SHA5 in z.info("org/child@" + SHA2)["duplicates"]
    with pytest.raises(NotComparable) as err:
        z.similar("org/t0only@" + SHA8)
    assert err.value.reason == "t0_only"
    with pytest.raises(NotComparable):
        z.similar("org/other-arch@" + SHA7)

def test_family_license_chain_and_diff(z):
    fam = z.family("org/child@" + SHA2)
    assert fam["license_chain"][:2] == [("org/child@" + SHA2, "mit"), ("org/parent@" + SHA1, "apache-2.0")]
    assert fam["license_differs"] is True
    d = z.diff("org/child@" + SHA2, "org/parent@" + SHA1)
    assert d["relatedness"]["verdict"] == "uncalibrated" and d["summary"]["counts"]["changed"] > 0
    assert z.check("org/child@" + SHA2)["verdict"] == "uncalibrated"

def test_decision_models_are_queryable(z):
    assert set(z.find(system_one=True).model_key) == {"org/decider@" + SHA11}
    card = z.info("org/decider@" + SHA11)
    assert card["decision"]["decision_type"] == "binary" and card["decision"]["labels"] == ["no", "yes"]
    d = z.diff("org/decider@" + SHA11, "org/parent@" + SHA1)
    assert [h["name"] for h in d["summary"]["heads"]] == ["score.weight"] and d["summary"]["counts"]["added"] == 1
    assert "org/parent-GGUF@" + SHA12 in set(z.find(architecture="llama").model_key)
```

- [ ] **Step 2:** Run `python -m pytest tests/test_query.py -q` in Zoo-Keeper. Expected: FAIL.
- [ ] **Step 3:** Implement `zoo_keeper/query.py`.
  - Every filter becomes a `?` parameter in DuckDB.
  - `license` lists use `list_contains(?, license)`.
  - `text` uses `contains(lower(repo_id), lower(?))`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "zoo: query API with find, info, similar, family, diff and check"`

### Task 8: CLIs, documentation, and a real-data smoke run

**Files:**
- In Model-Zoo:
  - Create `zoo/cli.py` (`python -m zoo.cli build ...`) and
    `zoo/tests/test_cli.py`.
  - Create `docs/operations/zoo.md` (the tables and how to build).
  - Modify `README.md`, adding a short "Zoo database (build)" section.
- In Zoo-Keeper:
  - Create `zoo_keeper/cli.py` (`python -m zoo_keeper.cli find ...`) and
    `tests/test_cli.py`.
  - Modify `README.md`, adding the query commands with one example each.

**Interfaces:**
- Consumes: Tasks 1–7.
- Model-Zoo's `zoo.cli` has one subcommand:
  - `build --manifests DIR --fingerprints DIR [DIR ...] --data_root DIR --out ZOO [--frame CSV] [--leaderboard_v2 CSV] [--scores_csv CSV --scores_mapping JSON] [--lineage_edges PARQUET] [--null JSON]`,
    where `--frame` is the sampled `models.csv`, and `--out` defaults to
    `$ZOO_DATA/builds/current`.
- Zoo-Keeper's `zoo_keeper.cli` has the query subcommands, each with a
  `--zoo PATH` option (default `$ZOO_DATA/builds/current`):
  - `find [--modality M] [--license L ...] [--params_min N] [--params_max N] [--format F] [--access A] [--text S] [--skill K --min_score X] [--include_unverified] [--decision] [--system_one] [--architecture A] [--limit N] [--json]`
  - `info MODEL [--json]`
  - `similar MODEL [-k N] [--keep_duplicates] [--json]`
  - `family MODEL [--depth N] [--json]`
  - `diff A B [--null JSON] [--json]`
  - `check MODEL [--null JSON] [--json]`

  Output:
  - Without `--json`, the output is a readable table (`DataFrame.to_string`)
    or an indented card.
  - With `--json`, it is one JSON document on stdout.
  - Exit codes: 0 on success, 2 for `UnknownModel`, `AmbiguousModel`, or
    `NotComparable` (message on stderr), and 1 for `check` with verdict
    `inconsistent`.

- [ ] **Step 1: Write the failing tests**

```python
# Model-Zoo: zoo/tests/test_cli.py
def test_cli_build(universe, tmp_path):
    assert main(["build", "--manifests", str(universe.manifests), "--fingerprints", str(universe.fingerprints),
                 "--data_root", str(universe.root), "--out", str(tmp_path / "zoo")]) == 0
    assert (tmp_path / "zoo" / "zoo.json").exists()

# Zoo-Keeper: tests/test_cli.py
def test_cli_find(built_zoo, capsys):
    assert main(["find", "--zoo", str(built_zoo), "--text", "child", "--json"]) == 0
    assert any(r["model_key"].startswith("org/child") for r in json.loads(capsys.readouterr().out))

def test_cli_error_exit_codes(built_zoo, capsys):
    assert main(["info", "--zoo", str(built_zoo), "org/child"]) == 2           # ambiguous
    assert "org/child@" in capsys.readouterr().err
    assert main(["similar", "--zoo", str(built_zoo), "org/t0only@" + SHA8]) == 2
```

- [ ] **Step 2:** Run `python -m pytest zoo/tests/test_cli.py -q` in Model-Zoo and
  `python -m pytest tests/test_cli.py -q` in Zoo-Keeper. Expected: FAIL.
- [ ] **Step 3:** Implement both CLIs, and write the docs. Together they cover:
  - what each table holds;
  - the evidence tiers and what "uncalibrated" means;
  - why weight similarity compares only same-architecture models;
  - how to rebuild;
  - one worked example per query on the pilot zoo.
- [ ] **Step 4:** Run `python -m pytest zoo/tests evidence/tests -q` in Model-Zoo and
  `python -m pytest tests -q` in Zoo-Keeper. Expected: all pass.
- [ ] **Step 5:** Run a smoke build on real pilot outputs. It reads
  fingerprints, not weights, so an agent may run it (handoff D4.1). Use `python -m zoo.cli build ...` on the extractor pilot's
  manifests and fingerprints, with `--leaderboard_v2` and the G0 calibration
  file. Then run each `python -m zoo_keeper.cli` query once on a known
  derivative pair, and record the outputs in Zoo-Keeper's README.
- [ ] **Step 6:** Commit: `git commit -m "zoo: CLI, docs and v0 smoke run"`

## After this plan

Zoo v0 is ready for G0. The next plan, zoo v1 (February–March), adds:

- `recommend`: constraints, then measured and predicted ranking with intervals,
  then near-duplicate collapse;
- the first score predictions (a parent's score plus per-skill shifts from the
  heritability study, plus fingerprint neighbors, with abstention);
- decision-model scores (System1-Arena and similar) through
  `load_scores_csv`, with accuracy, calibration error, and latency as metrics,
  so `recommend` can weigh a System One model against an LLM for a decision
  task;
- pip packaging, the demo Space, and the thin MCP wrapper;
- GGUF lineage (every type `gguf-py` can dequantize, including IQ types) and
  MLX quantization, the derivative-watch script, and the training callback.
