# Weight Evidence Extractor Implementation Plan (v2, 2026-10-29)

Revised 2026-10-31: interface fixes found in the pre-handoff cross-check (artifact
names, coverage vocabulary, calibration statistic, config subset, fixtures). Each is
listed in `Zoo-Keeper/handoff/STATUS.md`, "Plan and interface change log".

**Agents: never run this plan's "Commit" steps.** Leave changes uncommitted
and record the suggested message instead (`Zoo-Keeper/handoff/DIRECTIVES.md`,
D10.1).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fingerprint any Hugging Face checkpoint, or any local or private one,
by reading only what is needed over HTTP range requests. The fingerprints feed
the team's lineage experiments and a small, useful tool. The tool has three
commands, each with JSON output and exit codes:

- `fingerprint`;
- `diff`, which compares two models, or one against its declared base;
- `check`, which tests whether a model's declared base is consistent with its
  weights.

Measurement comes in three tiers:

- **T0 manifest:** pinned commit, de-duplicated file set, hashes, tensor
  headers, and config facts.
- **T1 signature:** one row block per matrix, giving an orthonormal k×k sketch,
  a block hash, a block spectrum, and the exact delta against a known parent.
- **T2 measurement:** full tensors, giving a sketch, a hash, and the existing
  collector's full spectra. A T2 pass also emits T1 blocks at several block
  heights at no extra read cost.

**One tool for every model type.** The extractor works on *stored tensors*, not
architectures. A small table of two independent choices, *decode* × *geometry*,
turns each stored tensor into the matrices that are measured (Task 4b), and
every decision is recorded in coverage. It covers:

- dense LLMs, and mixture-of-experts LLMs whether experts are fused or not;
- vision-language models;
- vision models (ViTs, CNNs), speech models, and embedding models;
- diffusion models (UNet or DiT weights in subfolders);
- merges;
- PEFT, kohya, diffusers, and MLX LoRA adapters, including PEFT factors
  stored inside a full checkpoint;
- decision ("System One") models and other classifiers, including their small
  output heads;
- FP8 (with F32, BF16, or E8M0 scales), MXFP4, NVFP4, and FP4 packed in int8
  (DeepSeek-V4.x) checkpoints.

GPTQ, AWQ, bitsandbytes, MLX quantization, and GGUF stay recorded gaps until
zoo v1 (see the plan sequence). `.bin`-only and ONNX-only repositories are
T0-only. Section 6.8 of the merged plan records which trending model types were
checked against these rules on 2026-10-31.

**Simplicity rule.** One reader, one decode × geometry table, one fingerprint
format. A new model type means a new table row, never a per-architecture code
path. Nothing is added unless an experiment or a user needs it.

**Architecture:** A self-contained package `Model-Zoo/evidence/`, CPU-first,
which never builds a model.
- It reuses two pieces of the existing collector:
  `net_esd.core.compute_esd_for_weight` for T2 spectra (numerics version 7),
  and `measurement_config` (pins, `safe_filename`, `NUMERICS_VERSION`, and
  `FORMAT_VERSION`). See "Imports" in Global Constraints.
- **Hub access:** each file is resolved once, then its redirected CDN URL is
  range-read. Verified 2026-10-28: CDN responses are `206` with no `RateLimit`
  header, so only `/resolve/` lookups count against the limit.
- **Fingerprints:** written as `<safe>.<tier>.fingerprint.safetensors` (see
  "Artifact names" in Global Constraints). Sketches are stored as tensors and
  the identity as JSON metadata, so a user can share one file and load it with
  any safetensors reader.
- **T2 spectra:** written in addition as `<safe>.t2.spectra.h5`, using the
  collector's `/layers` and `/eigs` conventions (Task 6).
- **Runner integration:** stays separate from the GPU worker runner until gate
  G0 decides on adoption.

**Tech Stack:** Python 3.10, torch 2.11 (CPU), numpy, safetensors, h5py,
pandas, huggingface-hub 1.11, pytest; plus pyarrow for the hub-stats parquet.

**Spec:** `Zoo-Keeper/docs/research/2026-10-27-merged-research-plan.md`.
The governing sections are 0, 0.1 (re-centered on the zoo, 2026-10-31), 1.1,
5.3, 6.1–6.7, 12, and 15.2.

## Plan sequence

The zoo and its query framework are the core product (merged plan, section
6.7). This extractor is the zoo's measurement layer.

1. **This plan:** the extractor, `diff`/`check`, and the pilot harness. It
   produces the fingerprints and manifests the zoo stores.
2. **Zoo store and v0 queries** (`2026-10-31-zoo-store-and-queries.md`; due for G0, 2026-12-04):
   - Parquet and DuckDB tables `models`, `views`, `lineage`, `scores`,
     `predictions`, and `coverage`, rebuilt from this plan's outputs and from
     score sources with verified revision joins;
   - a Python library and CLI for `find`, `info`, `similar` (exact
     common-block cosine search), and `family`, alongside this plan's `diff`
     and `check`.

   The build lives in Model-Zoo (`zoo/`); the queries live in the Zoo-Keeper
   repository (`zoo_keeper/`), which also holds items 3–5 below.
3. **Genealogy lab.** Build the minimum lab of about 120 models (merged plan,
   section 12) with an exact edge list. Release it as a benchmark with a scorer.
4. **Heritability and exchange-rate pass** on existing leaderboard and Atlas
   data. Independent of plans 1–3; person C starts it in week 1. Its per-skill
   parent-to-child shifts become the zoo's first score predictions.
5. **P1 lineage engine,** which writes into the zoo's `lineage` table:
   - nested estimands;
   - retrieval;
   - merge regression;
   - edge typing;
   - empirical calibration;
   - arborescence assembly;
   - baselines.
6. **Zoo v1, after the ICML submission:**
   - `recommend` without the quick test;
   - pip packaging and a demo Space;
   - GGUF dequantization for lineage only: every type `gguf-py` can
     dequantize, including the IQ types common on trending repositories, and
     MLX affine quantization in the same way;
   - System1-Arena and similar decision-model scores, so `recommend` can
     compare System One models with LLMs on decision tasks;
   - a derivative-watch script;
   - a training callback for checkpoint diffs.

## Global Constraints

- **Environment.** The existing `esd_ind` conda environment on the work
  machine (Python 3.10 assumed; handoff item P0-4 records the actual
  versions), with the pins in `Model-Zoo/requirements.txt`. One new direct
  dependency, `pyarrow`, pinned to the installed version. Installed packages
  are never upgraded or downgraded (handoff D7.5).
- **Imports.** `evidence/__init__.py` puts the repository root and
  `esd_experiment/src` on `sys.path` (the same path the collector's tests
  use). So `import measurement_config` and `net_esd` work from the CLI, from
  tests, and from Zoo-Keeper after `pip install -e Model-Zoo`. Never import `esd_experiment.src` as a
  package: its `__init__` loads transformers.
- **Artifact names.**
  `<safe> = measurement_config.safe_filename(f"{repo_id}@{revision}")`, so two
  revisions of one repository never collide. The files are
  `<safe>.manifest.json`, `<safe>.<tier>.fingerprint.safetensors` (tier `t1`,
  `t2`, or `t1b<b>` for extra block heights), and `<safe>.t2.spectra.h5`.
- **Pinned inputs.**
  - Hub reads use a 40-character SHA (`measurement_config.is_commit_sha`).
    `main` resolves once, in T0.
  - Local directories are pinned by content as
    `local:<sha256 over the sorted per-file sha256 values>`.
- **Tokens.**
  - One token, `HF_TOKEN` (the owner's), covers ungated and granted gated
    repositories. It is never logged, printed, or written to disk.
  - Honor `HF_ENDPOINT` (default `https://huggingface.co`), proxies from the
    standard environment variables, and `HF_HUB_OFFLINE=1` (Hub reads then
    raise `OfflineError`).
- **One lookup per file.** Resolve each file at most once per process.
  Re-resolve only after a CDN `403`, at most once per file per read.
- **Sketch operators: orthonormal, nested, side-specific.**
  - `SKETCH_SEED = 20261028`, `SKETCH_K_MAX = 256`, `SKETCH_K_DEFAULT = 64`.
  - For dimension `d` and side `s` (`0` for rows, `1` for columns):
    1. `G = np.random.default_rng([SKETCH_SEED, d, s]).standard_normal((256, d))`
       in float64;
    2. `Q, R = np.linalg.qr(G.T)`;
    3. `Q *= np.sign(np.diag(R))`;
    4. the operator of size `k` is `Q.T[:k]`, cast to float32.
  - Rows are orthonormal. `Q.T[:k]` depends only on `G[:k]`, so a size-64
    operator is exactly the first 64 rows of the size-256 one.
  - When `d < 256`, use `min(256, d)` rows.
  - Any two sketches compare on their common top-left `min(k_a, k_b)` block, by
    cosine similarity.
  - The defaults are `k = 64` for the census and `k = 128` for the lab and
    anchors (`--sketch_k`).
- **Row block.**
  - `ROW_BLOCK = 256`.
  - The start row is `min((rows - 256) // 2, 1920)` when `rows > 256`;
    otherwise the whole matrix is read. For matrices of 4,096 rows or more,
    such as embeddings, the start is therefore fixed and does not shift with
    vocabulary size.
  - A parent delta is computed whenever the child and parent views share
    `(r0, r1, cols)`, even if their row counts differ.
- **Mixture-of-experts subset at T1.**
  - Read `EXPERTS_PER_LAYER = 8` routed experts per layer, chosen as
    `sorted(np.random.default_rng([SKETCH_SEED, E]).choice(E, 8, replace=False))`;
    when `E <= 8`, read every expert.
  - The subset applies to fused 3D expert tensors and to unfused per-expert 2D
    tensors (names containing `experts.<i>.`, as in Mixtral, DeepSeek, and
    Qwen3 MoE). Experts outside it are recorded as `expert_not_sampled`.
  - Always read shared experts and routers.
  - `--all_experts` reads every expert. It is the default for the lab and
    anchors.
- **Nothing is dropped silently.** Every stored tensor appears in coverage with
  its decode, its geometry, and either its views or a skip reason.
- **Privacy of blocks.** Raw row blocks and cached parent blocks live only in
  `--parent_cache` and are never written into fingerprints.
- **Tests are offline.** One network smoke test is skipped unless
  `MODEL_ZOO_NETWORK_TESTS=1`. Local mode makes zero network calls, and a test
  proves it.

## Review Focus

1. **A parent delta whose update is near the bf16 noise floor** (relative update
   ρ < ~3e-3). The output must say `below_floor` rather than report a confident,
   wrong rank. Tested in Task 4.
2. **Quantization side tensors** must never be measured as weight matrices:
   NVFP4 `weight_scale` (F8 2D), GPTQ/AWQ `scales` (F16 2D), kohya `.alpha`, and
   the `_scale_inv` grids. Tested in Task 4b.
3. **3D tensors that are not expert stacks**, such as Whisper `encoder.conv1`
   `[768, 80, 3]` and linear-attention `conv1d` `[8192, 1, 4]`, must not be
   sliced into hundreds of junk views. Tested in Task 4b.
4. **Repositories that store the same weights more than once:**
   - SDXL root single-files next to `unet/`;
   - `.fp16.` variants;
   - Mistral `consolidated.safetensors` next to HF shards;
   - gpt-oss `original/`.

   They must be read once and the skipped files recorded. Tested in Task 5.
5. **An interrupted batch run** must resume:
   - leftover `.tmp` files are ignored and overwritten;
   - finished files are skipped;
   - `no_access` is recorded once and not retried.

   Tested in Task 8.
6. **Decision heads and integer buffers.**
   - A System One model's head (for example `classifier.weight [2, 384]`) must
     be measured as a `head` view, not skipped.
   - Integer buffers such as `relative_position_index` must not block their
     layer.

   Tested in Task 4b.

---

### Task 1: Package skeleton, safetensors headers, plain decoding

**Files:**
- Create: `evidence/__init__.py` (exports `EVIDENCE_VERSION = "1"`)
- Create: `pyproject.toml`, minimal (setuptools; packages `evidence` and,
  later, `zoo`), so other repositories can `pip install -e Model-Zoo`.
- Create: `evidence/safetensors_header.py`
- Create: `evidence/tests/conftest.py`
  - Adds `esd_experiment/src` and the repository root to `sys.path`, as
    `esd_experiment/tests` does.
  - Defines the `tiny_checkpoint(tmp_path)` fixture, which writes two
    safetensors shards (with `metadata={"format": "pt"}`) plus
    `model.safetensors.index.json` and a `config.json`.
    The shards hold BF16, F16, and F32 2D tensors, a 2D tensor with **512 rows**
    (so that T1 actually reads a block), a 1D norm, and an I64
    `position_ids`.
- Test: `evidence/tests/test_safetensors_header.py`

**Interfaces:**
- Produces:
  - `TensorEntry(name: str, dtype: str, shape: tuple[int, ...], file: str, start: int, end: int)`:
    a frozen dataclass with absolute, end-exclusive byte offsets in `file`.
  - `header_length(first8: bytes) -> int`, which raises `ValueError` above
    `MAX_HEADER_BYTES = 100 * 1024 * 1024`.
  - `parse_header(prefix: bytes, file: str) -> tuple[dict[str, TensorEntry], dict]`,
    returning entries and metadata. It ignores `__metadata__`.
  - `decode_plain(raw: bytes, dtype: str, shape: tuple[int, ...]) -> torch.Tensor`
    for F64, F32, F16, BF16, F8_E4M3 (`torch.float8_e4m3fn`), F8_E5M2, and
    F8_E8M0 (block scales: the value is 2^(e−127)). It raises
    `UnsupportedDtype` (a `ValueError` subclass) otherwise.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_header_absolute_offsets_and_metadata(tiny_checkpoint):
    path = tiny_checkpoint / "model-00001-of-00002.safetensors"
    data = path.read_bytes(); n = header_length(data[:8])
    entries, metadata = parse_header(data[:8 + n], path.name)
    assert "__metadata__" not in entries and metadata.get("format") == "pt"
    e = entries["model.layers.0.self_attn.q_proj.weight"]
    assert e.start >= 8 + n and e.end - e.start == 2 * 32 * 16   # BF16 32x16

def test_decode_matches_safetensors(tiny_checkpoint):
    from safetensors.torch import load_file
    path = tiny_checkpoint / "model-00001-of-00002.safetensors"
    data = path.read_bytes(); n = header_length(data[:8])
    entries, _ = parse_header(data[:8 + n], path.name)
    ref = load_file(path)
    for name, e in entries.items():
        if e.dtype in {"BF16", "F16", "F32"}:
            assert torch.equal(decode_plain(data[e.start:e.end], e.dtype, e.shape), ref[name])

def test_decode_fp8_roundtrip_and_rejects_int():
    x = torch.tensor([[0.5, -1.0], [2.0, 0.0]]).to(torch.float8_e4m3fn)
    got = decode_plain(x.view(torch.uint8).numpy().tobytes(), "F8_E4M3", (2, 2))
    assert torch.equal(got.float(), x.float())
    e8 = decode_plain(bytes([127, 128, 126]), "F8_E8M0", (3,))
    assert torch.equal(e8.float(), torch.tensor([1.0, 2.0, 0.5]))
    with pytest.raises(UnsupportedDtype):
        decode_plain(b"\x00" * 8, "I64", (1,))

def test_header_length_guard():
    with pytest.raises(ValueError):
        header_length((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_safetensors_header.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement. The header is a little-endian u64 `N` followed by
  `N` bytes of JSON, and offsets are relative to byte `8 + N`. Decode with
  `torch.frombuffer(bytearray(raw), dtype=...)`, then reshape.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git add evidence && git commit -m "evidence: safetensors headers and plain decoding"`

### Task 2: Hub reader (resolve once, CDN range reads, offline-safe)

**Files:**
- Create: `evidence/hub.py`
- Modify: `evidence/tests/conftest.py`. Add the constant `SHA = "a" * 40` and
  the `fake_transport` fixture: an in-memory Hub serving a 1,000-byte blob for
  `org/m` (`x-linked-etag: "abc"`, `x-linked-size: 1000`, and a CDN
  `Location`). It provides `expire_next_range_read()`,
  `gate(repo, allowed_token)`, and the recorded `last_url` and `last_headers`.
- Test: `evidence/tests/test_hub.py`

**Interfaces:**
- Consumes: Task 1 `header_length`, `parse_header`.
- Produces:
  - `RemoteFile(repo_id, filename, revision, url, size: int, sha256: str | None, xet_hash: str | None)`.
  - `HubReader(token: str | None = None, transport: Callable | None = None)`.
    The token defaults to the `HF_TOKEN` environment variable. Methods:
    - `resolve(repo_id, filename, revision) -> RemoteFile`. It is cached. A
      local directory `repo_id` returns a file path and makes no network call.
    - `read(remote, start, end) -> bytes`, over `[start, end)`.
    - `read_header(remote) -> dict[str, TensorEntry]`.
    - `stats -> dict` with keys `resolves`, `range_reads`, `bytes_read`,
      `refreshes`, `retries`.
  - Exceptions: `AccessDenied` (a lookup `401`/`403`) and `OfflineError`.
  - `transport(method, url, headers) -> (status, headers, body)` is injectable.
    The default uses urllib, which honors proxy environment variables, and does
    not follow the lookup redirect.

- [ ] **Step 1: Write the failing tests** (fake transport; no network)

```python
def test_resolve_once_and_range_read_cdn(fake_transport):
    reader = HubReader(token="t", transport=fake_transport)
    f1 = reader.resolve("org/m", "model.safetensors", SHA)
    assert reader.resolve("org/m", "model.safetensors", SHA) is f1 and reader.stats["resolves"] == 1
    assert f1.sha256 == "abc" and f1.size == 1000            # x-linked-etag / x-linked-size
    assert reader.read(f1, 10, 20) == fake_transport.blob[10:20]
    assert fake_transport.last_headers["Range"] == "bytes=10-19"
    assert "Authorization" not in fake_transport.last_headers

def test_expired_cdn_url_refreshes_once(fake_transport):
    fake_transport.expire_next_range_read()
    reader = HubReader(token="t", transport=fake_transport)
    f = reader.resolve("org/m", "model.safetensors", SHA)
    assert reader.read(f, 0, 8) == fake_transport.blob[:8]
    assert reader.stats["refreshes"] == 1 and reader.stats["resolves"] == 2

def test_gated_allowed_or_denied(fake_transport):
    fake_transport.gate("org/gated", allowed_token="owner")
    assert HubReader(token="owner", transport=fake_transport).resolve("org/gated", "model.safetensors", SHA)
    with pytest.raises(AccessDenied):
        HubReader(token="other", transport=fake_transport).resolve("org/gated", "model.safetensors", SHA)

def test_endpoint_and_offline(fake_transport, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    HubReader(token="t", transport=fake_transport).resolve("org/m", "model.safetensors", SHA)
    assert fake_transport.last_url.startswith("https://mirror.example/org/m/resolve/")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(OfflineError):
        HubReader(token="t", transport=fake_transport).resolve("org/m", "model.safetensors", SHA)

def test_local_mode_makes_no_network_calls(tiny_checkpoint, monkeypatch):
    monkeypatch.setattr("socket.socket.connect", Mock(side_effect=AssertionError("network")))
    reader = HubReader()
    f = reader.resolve(str(tiny_checkpoint), "model-00001-of-00002.safetensors", "local")
    assert "model.layers.0.self_attn.q_proj.weight" in reader.read_header(f)
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_hub.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  - **Lookup:** `HEAD {HF_ENDPOINT}/{repo}/resolve/{sha}/{filename}` with a
    bearer token. A `401`/`403` raises `AccessDenied`.
  - **Headers:** `Location` gives the CDN URL; `x-linked-size`, `x-linked-etag`
    (unquoted), and `x-xet-hash` give the size, sha256, and Xet hash. A `200`
    without a redirect (a small non-LFS file) uses the lookup URL itself.
  - **Range read:** `GET` with `Range: bytes=a-(b-1)` and no Authorization
    header; expect `206`.
  - **Retries:** retry `5xx` and connection errors 3 times, with backoff of 1,
    4, and 16 seconds.
  - **Rate limit:** on a lookup `429`, sleep for the `t=` seconds in the
    `RateLimit` header.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Add `test_network_smoke`, skipped unless
  `MODEL_ZOO_NETWORK_TESTS=1`. It resolves `HuggingFaceTB/SmolLM2-135M` at a
  pinned commit and asserts that `model.embed_tokens.weight` is in the header,
  that `resolves == 1`, and that `range_reads == 2`.
- [ ] **Step 6:** Commit: `git commit -m "evidence: hub reader with single resolve, CDN range reads, offline safety"`

### Task 3: Orthonormal nested sketches and row-block bounds

**Files:**
- Create: `evidence/sketch.py`
- Test: `evidence/tests/test_sketch.py`

**Interfaces:**
- Produces:
  - Constants: `SKETCH_SEED`, `SKETCH_K_MAX`, `SKETCH_K_DEFAULT`, `ROW_BLOCK`,
    and `ROW_START_CAP = 1920`.
  - `sketch_operator(dim: int, side: int, k: int = SKETCH_K_DEFAULT) -> np.ndarray`,
    shape `(min(k, dim), dim)`, float32, with orthonormal rows. The cache is
    capped at `OPERATOR_CACHE_BYTES = 2 * 1024**3` (least recently used
    entries are evicted by bytes).
  - `two_sided_sketch(w: torch.Tensor, k: int = SKETCH_K_DEFAULT) -> np.ndarray`.
    It computes `S(rows, 0) @ w @ S(cols, 1).T` in float64 and returns float32.
  - `factored_sketch(b: torch.Tensor, a: torch.Tensor, k: int = SKETCH_K_DEFAULT) -> np.ndarray`:
    the sketch of `b @ a` without forming it.
  - `sketch_cosine(x: np.ndarray, y: np.ndarray) -> float`, computed on the
    common top-left block.
  - `row_block_bounds(rows: int, b: int = ROW_BLOCK) -> tuple[int, int]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_operator_orthonormal_nested_and_side_specific():
    a64, a256 = sketch_operator(4096, 0, 64), sketch_operator(4096, 0, 256)
    assert np.allclose(a256 @ a256.T, np.eye(256), atol=1e-5)
    assert np.array_equal(a64, a256[:64])
    assert not np.allclose(sketch_operator(4096, 0, 64), sketch_operator(4096, 1, 64))

def test_sketch_is_linear():
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(300, 128, generator=g), torch.randn(300, 128, generator=g)
    lhs = two_sided_sketch(0.3 * x + 0.7 * y)
    rhs = 0.3 * two_sided_sketch(x) + 0.7 * two_sided_sketch(y)
    assert np.allclose(lhs, rhs, rtol=0, atol=1e-5 * np.abs(lhs).max())

def test_sizes_compare_on_common_block():
    w = torch.randn(512, 384)
    s64, s128 = two_sided_sketch(w, 64), two_sided_sketch(w, 128)
    assert np.allclose(s128[:64, :64], s64, atol=1e-5 * np.abs(s64).max())
    assert sketch_cosine(s64, s128) > 0.9999

def test_factored_sketch_matches_dense():
    b, a = torch.randn(300, 8), torch.randn(8, 200)
    d = two_sided_sketch(b @ a)
    assert np.allclose(factored_sketch(b, a), d, atol=1e-4 * np.abs(d).max())

def test_unrelated_sketches_near_orthogonal():
    g = torch.Generator().manual_seed(1)
    a, b = (two_sided_sketch(torch.randn(512, 512, generator=g)) for _ in range(2))
    assert abs(sketch_cosine(a, b)) < 4 / 64

@pytest.mark.parametrize("rows,expected", [
    (4096, (1920, 2176)), (151936, (1920, 2176)), (1024, (384, 640)), (256, (0, 256)), (128, (0, 128))])
def test_row_block_bounds(rows, expected):
    assert row_block_bounds(rows) == expected
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_sketch.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement using the operator recipe in Global Constraints.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "evidence: orthonormal nested side-specific sketches and row blocks"`

### Task 4: Block delta and rank with abstention

The September review measured that Gavish–Donoho on raw bf16 deltas over-counts
rank when the relative update ρ is below about 3e-3, and on heteroscedastic
weights. This task implements the corrected procedure.

**Files:**
- Create: `evidence/delta.py`
- Test: `evidence/tests/test_delta.py`

**Interfaces:**
- Produces:
  - `gavish_donoho_rank(singular_values: np.ndarray, rows: int, cols: int) -> int`.
    This is the unknown-noise rule: with `β = min/max` and
    `ω(β) = 0.56β³ − 0.95β² + 1.82β + 1.43`, it counts the `σ_i > ω(β) · median(σ)`.
    It is used only when the parent is not stored in a 16-bit or 8-bit dtype.
  - `block_delta(child: torch.Tensor, parent: torch.Tensor, storage_dtype: torch.dtype) -> dict`.
    It returns these keys:
    - `frac_identical: float`;
    - `rel_update_norm: float`;
    - `rank: int` (−1 when `status != "ok"`);
    - `rank_confidence: str` (`"high"` or `"low"`);
    - `status: str` (`"ok"`, `"identical"`, or `"below_floor"`);
    - `top_singular_values`: shape `(16,)`, float32, NaN-padded.

    It raises `ValueError` on a shape mismatch.
  - `FLOOR_FRAC_IDENTICAL = 0.5`.

- [ ] **Step 1: Write the failing tests**

```python
def lora_case(r, rho, rows=256, cols=4096, hetero=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = 0.02 * torch.randn(rows, cols, generator=g)
    if hetero:  # lognormal row/col scales plus 4 outlier columns
        w = w * torch.exp(0.5 * torch.randn(rows, 1, generator=g)) * torch.exp(0.5 * torch.randn(1, cols, generator=g))
        w[:, :4] *= 20
    upd = torch.randn(rows, r, generator=g) @ torch.randn(r, cols, generator=g)
    upd = upd * (rho * w.norm() / upd.norm())
    parent = w.bfloat16()
    return (parent.float() + upd).bfloat16(), parent

@pytest.mark.parametrize("r", [8, 64])
def test_rank_recovered_in_working_regime(r):
    out = block_delta(*lora_case(r, 1e-2), torch.bfloat16)
    assert out["status"] == "ok" and out["rank"] == r

def test_heteroscedastic_parent():
    out = block_delta(*lora_case(8, 1e-2, hetero=True), torch.bfloat16)
    assert out["status"] == "ok" and out["rank"] == 8

def test_below_floor_abstains():
    out = block_delta(*lora_case(8, 1e-3), torch.bfloat16)
    assert out["status"] == "below_floor" and out["rank"] == -1 and out["frac_identical"] > 0.5

def test_rank_one_never_confident_unless_exact():
    out = block_delta(*lora_case(1, 1e-2), torch.bfloat16)
    assert out["rank"] == 1 or out["rank_confidence"] == "low"

def test_identical_blocks():
    p = torch.randn(64, 64).bfloat16()
    out = block_delta(p.clone(), p, torch.bfloat16)
    assert out["status"] == "identical" and out["frac_identical"] == 1.0 and out["rel_update_norm"] == 0.0

def test_gd_rule_on_float32_noise():
    g = torch.Generator().manual_seed(0)
    low = torch.randn(256, 3, generator=g) @ torch.randn(3, 2048, generator=g)
    sv = torch.linalg.svdvals(low + 0.01 * torch.randn(256, 2048, generator=g)).numpy()
    assert gavish_donoho_rank(sv, 256, 2048) == 3
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_delta.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  1. **Identical check.** Compute `frac_identical` bitwise on the stored dtype.
     If it is 1, return status `identical`.
  2. **Floor check.** If `frac_identical > FLOOR_FRAC_IDENTICAL`, return status
     `below_floor`. This corresponds to ρ ≲ 3e-3 in bf16.
  3. **Otherwise, biwhiten and threshold.**
     - With `D = child.float() − parent.float()`, set
       `V_ij = ulp(parent_ij)² / 12`, where `ulp` is the storage-dtype spacing
       at `|parent_ij|`.
     - Run 50 Sinkhorn iterations to find positive vectors `u` and `v` that
       make the row and column means of `u_i V_ij v_j` equal to 1.
     - Form `D̃ = diag(√u) D diag(√v)` and threshold it at
       `τ = 1.1 · (√rows + √cols)`.
     - The rank is the count of `σ_i(D̃) > τ`.
     - `rank_confidence` is `"high"` if `σ_rank / σ_{rank+1} ≥ 1.5`, otherwise
       `"low"`.
- [ ] **Step 4:** Run the tests. Expected: PASS. If the heteroscedastic test fails, report the measured ranks instead of loosening the test.
- [ ] **Step 5:** Commit: `git commit -m "evidence: block delta with floor abstention and biwhitened rank"`

### Task 4b: Decode × geometry table (every model type)

**Files:**
- Create: `evidence/views.py`
- Test: `evidence/tests/test_views.py`

**Interfaces:**
- Consumes: Task 1 `TensorEntry`, `decode_plain`.
- Produces:
  - `View(name: str, decode: str, geometry: str, sources: tuple[str, ...], slice_index: int | None, matrix_shape: tuple[int, int], component: str, fused: bool, scale: float)`.
  - **`decode`** is one of:
    - `plain`;
    - `fp8_scaled`: per-tensor, per-channel, block, or per-expert block scale,
      stored as F32, BF16, or F8_E8M0 (`scale_fmt: ue8m0`);
    - `mxfp4`: U8 `_blocks` plus U8 E8M0 `_scales` (gpt-oss). Also FP4 packed
      two per byte in an I8 or U8 `.weight`, with an F8_E8M0 `.scale` on
      blocks of 32 inputs, when `quantization_config.expert_dtype == "fp4"`
      (DeepSeek-V4.x). The logical width is twice the stored width;
    - `nvfp4`: U8 packed weights plus an F8 `weight_scale` block and an F32
      `weight_scale_2`.
  - **`geometry`** is one of:
    - `matrix`: 2D, after squeezing leading 1-dims;
    - `head`: a 2D tensor with fewer than 8 rows but at least 8 columns, such
      as a classification or decision head (`classifier.weight [2, 384]`,
      `score.weight [1, 2560]`). It is read whole and sketched like any matrix;
      the operators shrink to `min(k, d)` rows;
    - `expert_slice`: a 3D tensor whose name contains `experts`, or whose
      `shape[0] == config["num_experts"]`. One view per selected expert,
      oriented as (out, in) using `config["hidden_size"]`, and named
      `"<tensor>[i]"`;
    - `conv_flat`: other 3D tensors, and 4D/5D tensors, as
      `shape[0] × prod(shape[1:])`;
    - `lora_product`.
  - `plan_views(entries: dict[str, TensorEntry], config: dict, adapter_config: dict | None = None, all_experts: bool = False) -> tuple[list[View], list[dict]]`,
    returning views plus coverage records for every stored tensor
    `{name, status, reason, decode, geometry}`. `config` is the manifest's
    normalized config (Task 5), so `num_experts` and `hidden_size` are at the
    top level.
  - **Coverage vocabulary.** It is closed, and the zoo relies on it.
    - `status = "measured"`: the tensor forms or contributes to a view
      (weights, LoRA factors). `reason` is empty, or `lora_scale_unknown` for
      LoRA factors whose scale could not be read.
    - `status = "consumed"`: used only to decode or scale another tensor's
      view. `reason` is `decode_scale` (FP8, MXFP4, and NVFP4 scales) or
      `adapter_alpha` (kohya `.alpha`).
    - `status = "skipped"`: `reason` is one of `not_a_matrix`,
      `non_float_buffer`, `vector_like`, `unsupported_representation`,
      `unsupported_adapter_type`, `fp8_block_size_unknown`, or
      `expert_not_sampled`.
    - `status = "duplicate"` (set in Task 6): `reason` is
      `duplicate_of:<view name>`.
  - `decode_view(view: View, read: Callable[[str, slice | None], torch.Tensor], config: dict) -> torch.Tensor`,
    which returns the float32 matrix of the view, or its row block when given a
    row slice.
  - `NEVER_MATRIX_SUFFIXES = ("_scale", "_scale_inv", "_scale_2", ".scale", "scales", "qzeros", "g_idx", "absmax", "quant_map", "SCB", "input_scale", "_bias", ".alpha", "position_ids", "num_batches_tracked")`.
  - `component_of(name: str, file: str) -> str`: the file's subfolder (if any)
    plus the name prefix up to the first numeric segment. For example,
    `model.visual.blocks.3.attn.qkv.weight` gives `model.visual.blocks`, and
    `unet/…/down_blocks.1…` gives `unet/down_blocks`. It is a report label only.

- [ ] **Step 1: Write the failing tests** (entries built from shapes and names seen on the Hub on 2026-10-29)

```python
def E(**s): return {n: TensorEntry(n, d, sh, "f.safetensors", 0, 0) for n, (d, sh) in s.items()}

def test_fused_experts_slice_and_orient():
    cfg = {"hidden_size": 2048, "num_experts": 128}
    v, _ = plan_views(E(**{"layers.0.mlp.experts.gate_up_proj": ("BF16", (128, 2048, 1536))}), cfg, all_experts=True)
    gate = [x for x in v if x.name.startswith("layers.0.mlp.experts.gate_proj[")]
    assert len(gate) == 128 and gate[0].matrix_shape == (768, 2048)       # split gate/up, oriented (out,in)

def test_expert_subset_default():
    cfg = {"hidden_size": 2048, "num_experts": 128}
    v, _ = plan_views(E(**{"layers.0.mlp.experts.down_proj": ("BF16", (128, 768, 2048))}), cfg)
    assert len(v) == 8

def test_unfused_expert_subset():
    cfg = {"hidden_size": 64, "num_experts": 16}
    v, cov = plan_views(E(**{f"layers.0.mlp.experts.{i}.down_proj.weight": ("BF16", (64, 32)) for i in range(16)}), cfg)
    assert len(v) == 8 and [c["reason"] for c in cov].count("expert_not_sampled") == 8

def test_conv1d_and_whisper_not_sliced():
    v, cov = plan_views(E(**{"encoder.conv1.weight": ("F32", (768, 80, 3)),
                             "layers.0.linear_attn.conv1d.weight": ("BF16", (8192, 1, 4))}), {})
    assert [x.geometry for x in v] == ["conv_flat"] and v[0].matrix_shape == (768, 240)
    assert {c["name"]: c["reason"] for c in cov}["layers.0.linear_attn.conv1d.weight"] == "vector_like"

def test_scale_tensors_never_matrices():
    v, cov = plan_views(E(**{"m.q.weight": ("U8", (512, 1024)), "m.q.weight_scale": ("F8_E4M3", (512, 128)),
                             "m.q.weight_scale_2": ("F32", ()), "g.k.qweight": ("I32", (8, 64)),
                             "g.k.scales": ("F16", (38, 896)), "g.k.qzeros": ("I32", (38, 112))}), {})
    assert [x.decode for x in v] == ["nvfp4"]
    reasons = {c["name"]: c["reason"] for c in cov}
    assert reasons["g.k.scales"] == "unsupported_representation"     # module poisoned by qweight

def test_fp8_block_size_from_config_and_unknown():
    entries = E(**{"kv_a.weight": ("F8_E4M3", (576, 7168)), "kv_a.weight_scale_inv": ("F32", (5, 56))})
    v, _ = plan_views(entries, {"quantization_config": {"weight_block_size": [128, 128]}})
    assert v[0].decode == "fp8_scaled"
    _, cov = plan_views(entries, {})
    assert {c["name"]: c["reason"] for c in cov}["kv_a.weight"] == "fp8_block_size_unknown"

def test_lora_families_and_scaling():
    kohya = E(**{"lora_unet_x_attn1_to_k.lora_down.weight": ("BF16", (32, 640)),
                 "lora_unet_x_attn1_to_k.lora_up.weight": ("BF16", (640, 32)),
                 "lora_unet_x_attn1_to_k.alpha": ("BF16", ())})
    v, cov = plan_views(kohya, {})
    assert [x.geometry for x in v] == ["lora_product"] and not [c for c in cov if c["status"] == "skipped"]
    peft = E(**{"base_model.model.q_proj.lora_A.weight": ("F32", (16, 64)),
                "base_model.model.q_proj.lora_B.weight": ("F32", (64, 16))})
    v, _ = plan_views(peft, {}, adapter_config={"r": 16, "lora_alpha": 32})
    assert v[0].scale == 2.0
    v, _ = plan_views(peft, {}, adapter_config={"r": 16, "lora_alpha": 32, "use_rslora": True})
    assert v[0].scale == pytest.approx(32 / 4)

def test_unsupported_adapter_types():
    _, cov = plan_views(E(**{"m.hada_w1_a": ("F32", (8, 64))}), {})
    assert cov[0]["reason"] == "unsupported_adapter_type"

def test_decision_head_kept_and_depthwise_conv_skipped():
    v, cov = plan_views(E(**{"classifier.weight": ("F32", (2, 384)),
                             "layers.0.conv1d.weight": ("BF16", (8192, 1, 4))}), {})
    assert [(x.name, x.geometry) for x in v] == [("classifier.weight", "head")]
    assert {c["name"]: c["reason"] for c in cov}["layers.0.conv1d.weight"] == "vector_like"

def test_integer_buffers_do_not_poison():
    v, cov = plan_views(E(**{"attn.qkv.weight": ("F16", (432, 144)),
                             "attn.relative_position_index": ("I64", (144, 144)),
                             "decoder.upsample_factor": ("I64", ()),
                             "decoder.upsample.weight": ("F32", (64, 64, 2))}), {})
    assert {x.name for x in v} == {"attn.qkv.weight", "decoder.upsample.weight"}
    assert {c["name"]: c["reason"] for c in cov}["attn.relative_position_index"] == "non_float_buffer"

def test_mlx_and_embedded_peft_lora():
    mlx = E(**{"layers.0.w1.lora_a": ("F32", (2048, 16)), "layers.0.w1.lora_b": ("F32", (16, 8192))})
    v, _ = plan_views(mlx, {}, adapter_config={"lora_parameters": {"rank": 16, "scale": 2.0}})
    assert v[0].geometry == "lora_product" and v[0].matrix_shape == (8192, 2048) and v[0].scale == 2.0
    embedded = E(**{"model.layers.0.q_proj.weight": ("BF16", (64, 64)),
                    "base_model.model.model.layers.0.q_proj.lora_A.weight": ("BF16", (8, 64)),
                    "base_model.model.model.layers.0.q_proj.lora_B.weight": ("BF16", (64, 8))})
    v, cov = plan_views(embedded, {})
    assert {x.geometry for x in v} == {"matrix", "lora_product"}
    assert "lora_scale_unknown" in {c["reason"] for c in cov}

def test_deepseek_fp4_experts_and_e8m0_fp8():
    cfg = {"hidden_size": 5120, "num_experts": 256,
           "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32],
                                   "scale_fmt": "ue8m0", "expert_dtype": "fp4"}}
    v, _ = plan_views(E(**{"layers.1.ffn.experts.0.w1.weight": ("I8", (2304, 2560)),
                           "layers.1.ffn.experts.0.w1.scale": ("F8_E8M0", (2304, 160)),
                           "layers.1.attn.wkv.weight": ("F8_E4M3", (512, 5120)),
                           "layers.1.attn.wkv.scale": ("F8_E8M0", (16, 160))}), cfg, all_experts=True)
    by = {x.name: x for x in v}
    assert by["layers.1.ffn.experts.0.w1.weight"].decode == "mxfp4"
    assert by["layers.1.ffn.experts.0.w1.weight"].matrix_shape == (2304, 5120)
    assert by["layers.1.attn.wkv.weight"].decode == "fp8_scaled"
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_views.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement as an ordered rule list where the first match wins.
  **Coverage precedence.** A *packed weight* is an integer tensor of at least
  2 dims whose name ends in `weight`, `qweight`, `weight_packed`, or
  `_blocks`.
  1. Scale tensors used by a decode rule (FP8, MXFP4, NVFP4): `consumed`,
     `decode_scale`. Kohya `.alpha` and MLX scales: `consumed`,
     `adapter_alpha`.
  2. Module poisoning (below): skipped, `unsupported_representation`. This
     includes the packed weight itself, such as GPTQ `qweight`.
  3. Every other integer or bool tensor (`position_ids`,
     `relative_position_index`, integer scalars, bool flags, tokenizer bytes):
     skipped, `non_float_buffer`.
  4. `hada_`, `lokr_`, and DoRA magnitude tensors: skipped,
     `unsupported_adapter_type`.
  5. Any other `NEVER_MATRIX_SUFFIXES` tensor (for example `_bias`, or an
     unused `input_scale`): skipped, `not_a_matrix`.
  6. Geometry rules: 1D tensors `not_a_matrix`; `head` for 2D tensors with
     fewer than 8 rows but at least 8 columns; `vector_like` when there are
     fewer than 8 columns; `fp8_block_size_unknown`; `expert_not_sampled`;
     otherwise measured.

  **Decode rules:**
  - FP8: the scale always **multiplies**. Its dtype may be F32, BF16, or
    F8_E8M0. A scale of shape `[]` becomes `(1, 1)`, `(out,)` becomes
    `(out, 1)`, and 3D grids apply per expert.
  - The block size comes from `config.quantization_config.weight_block_size`.
    Without it, the shapes must divide exactly; otherwise the tensor is
    recorded as `fp8_block_size_unknown`.
  - MXFP4 and NVFP4 use a 16-entry E2M1 lookup times the block scale (times
    the global scale for NVFP4). **Verify nibble order and scale layout against
    the reference dequantizer** (transformers' MXFP4 integration, and NVIDIA
    ModelOpt for NVFP4) on one real tensor from `openai/gpt-oss-20b` and
    `nvidia/Qwen3.6-35B-A3B-NVFP4` before writing their fixtures. Verify
    DeepSeek-V4.x's FP4 experts the same way, on one tensor from
    `deepseek-ai/DeepSeek-V4.1-Flash-1017`, against DeepSeek's published
    reference dequantization.

  **Geometry rules:**
  - **Module poisoning:** if a packed weight under a module (the name up to
    its last `.`) has no decode rule, it and every sibling are
    `unsupported_representation`. Other integer and bool tensors never
    poison.
  - **Adapters:**
    - LoRA pairs are found by `lora_A`/`lora_B`, `lora_down`/`lora_up`,
      `lora.down`/`lora.up`, and `lora_embedding_A`/`lora_embedding_B`.
    - The scale is `α / r`, or `α / √r` for rslora; `rank_pattern` and
      `alpha_pattern` are honored. Kohya α comes from `.alpha`; when it is
      absent, `α = r`.
    - Conv factors use `up.flatten(1) @ down.flatten(1)`.
    - MLX adapters use `lora_a` `[in, r]` and `lora_b` `[r, out]`. The
      product is `(a @ b)ᵀ`, and the scale is
      `adapter_config["lora_parameters"]["scale"]`.
    - PEFT factors stored next to full weights (names starting with
      `base_model.model.`, and no `adapter_config.json`, as many System One
      checkpoints ship) pair with the base tensor after stripping that prefix.
      The scale is unknown, so it is set to 1.0 and recorded as
      `lora_scale_unknown`; cosines are unaffected.
    - `hada_`, `lokr_`, and DoRA magnitude tensors are recorded as
      `unsupported_adapter_type`.
  - **Matrices:**
    - Squeeze leading 1-dims.
    - A matrix with fewer than 8 columns is `vector_like`. One with fewer
      than 8 rows but at least 8 columns is a `head` view.
    - 1D tensors are recorded as `not_a_matrix`, and integer buffers as
      `non_float_buffer`.
  - **Fused tensors:**
    - Split `gate_up_proj` into gate and up halves when its out-rows are even,
      except when `config["model_type"] == "gpt_oss"` (interleaved halves:
      `fused=True`, no split).
    - Mark `qkv|query_key_value|w13|in_proj` names as `fused=True`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "evidence: decode x geometry view table for every model type"`

### Task 5: T0 manifest with file-set selection

**Files:**
- Create: `evidence/manifest.py`
- Modify: `evidence/tests/conftest.py`. Add fakes that need no network:
  - `fake_api`: `model_info` for `org/m` (sharded; `config.json` with
    `hidden_size = 896`; declared base `Qwen/Qwen2.5-0.5B` with relation
    `finetune`) and for `org/gated`;
  - `fake_reader`: a `HubReader` over `fake_transport` that serves those files
    and headers;
  - `denying_reader`: raises `AccessDenied` on every lookup;
  - `bin_only_api`: `org/convnext`, with only `pytorch_model.bin`;
  - `kohya_api`: `org/pixel-lora`, with one kohya LoRA file and no
    `adapter_config.json`;
  - `decider_api`: `org/decider`, tagged `system-one`, with a
    `DebertaV2ForSequenceClassification` config with two labels and a
    `classifier.weight [2, 384]`.

  `fake_reader` serves the files and headers of every fake repository.
- Test: `evidence/tests/test_manifest.py`

**Interfaces:**
- Consumes: Task 2 `HubReader`, `AccessDenied`; Task 4b `plan_views` (to count
  views).
- Produces:
  - `pin_revision(api, repo_id: str, revision: str = "main") -> str`.
  - `build_manifest(repo_id: str, revision: str, reader: HubReader, api) -> dict`,
    with these keys:
    - identity and access: `repo_id`, `revision`, `gated`, `access`
      (`"granted"`, `"no_access"`, or `"t0_only"`), `license`;
    - lineage: `declared_bases`, `base_model_relation`, `is_adapter`;
    - weights: `weight_format` (`"safetensors"`, `"gguf"`, `"bin"`,
      `"onnx"`, or `"other"`), `params_by_dtype`;
    - files: `files` (with filename, size, sha256, and xet_hash),
      `skipped_files` (as `{filename, reason}`);
    - tensors and configs: `tensors`, `config` (the parsed `config.json`
      subset: `model_type`, `hidden_size`, `num_hidden_layers`,
      `intermediate_size`, `num_attention_heads`, `num_key_value_heads`,
      `vocab_size`, `num_experts`, `tie_word_embeddings`, and
      `quantization_config`), and `adapter_config`;
      - `num_experts` is normalized from `num_experts`, `num_local_experts`,
        or `n_routed_experts`;
      - a key missing at the top level is read from `text_config`, because
        VLMs such as Qwen3-VL keep the language model's fields there;
    - `view_counts` (per geometry);
    - `decision`: `{is_decision, system_one, decision_type, num_labels, labels}`.
      - `system_one` is true when the Hub tags include `system-one` or
        `decision-model`.
      - `is_decision` is true for those, for a config with `id2label` or
        `num_labels` and an architecture ending in
        `ForSequenceClassification`, `ForTokenClassification`,
        `ForMultipleChoice`, or `ForQuestionAnswering`, and whenever a `head`
        view exists.
      - `decision_type` is `score` (one output), `binary`, `multiclass`,
        `multilabel` (from `problem_type`), `span` (token classification or
        question answering), or `unknown`.
      - `labels` holds at most 64 entries of `id2label`.
    - `gguf`: for repositories with GGUF files, the Hub API's `architecture`,
      `total` parameters, and `context_length`, plus the quantization types
      parsed from the `.gguf` file names (for example `Q4_K_M` or `IQ2_S`).
      This is free T0 metadata; reading GGUF tensors waits for zoo v1;
    - `evidence_version`.
  - `select_file_set(filenames: list[str]) -> tuple[list[str], list[dict]]`.
  - `write_manifest(manifest: dict, out_dir: Path) -> Path`, an atomic write
    to `<safe>.manifest.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_file_set_selection_dedupes():
    sdxl = ["sd_xl_base_1.0.safetensors", "sd_xl_base_1.0_0.9vae.safetensors",
            "unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/model.safetensors", "vae/diffusion_pytorch_model.safetensors",
            "sd_xl_offset_example-lora_1.0.safetensors", "model_index.json"]
    keep, skipped = select_file_set(sdxl)
    assert keep == ["text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors",
                    "vae/diffusion_pytorch_model.safetensors"]
    assert {s["reason"] for s in skipped} == {"duplicate_layout", "bundled_adapter"}
    keep, _ = select_file_set(["consolidated.safetensors", "model-00001-of-00002.safetensors",
                               "model-00002-of-00002.safetensors", "model.safetensors.index.json"])
    assert "consolidated.safetensors" not in keep
    keep, _ = select_file_set(["model.safetensors", "original/model.safetensors"])
    assert keep == ["model.safetensors"]
    keep, skipped = select_file_set(["diffusion_models/ltx-22b-dev-transformer-bf16.safetensors",
                                     "diffusion_models/ltx-22b-dev-transformer-comfy-int8-convrot.safetensors",
                                     "diffusion_models/ltx-22b-dev-transformer-nvfp4.safetensors",
                                     "loras/ltx-22b-distilled-lora-450-bf16.safetensors", "vocoder.pt"])
    assert keep == ["diffusion_models/ltx-22b-dev-transformer-bf16.safetensors"]
    assert {s["reason"] for s in skipped} == {"precision_variant", "bundled_adapter", "pickle_not_read"}

def test_decision_facts(fake_reader, decider_api):
    m = build_manifest("org/decider", SHA, fake_reader, decider_api)
    assert m["decision"]["system_one"] and m["decision"]["decision_type"] == "binary"
    assert m["decision"]["num_labels"] == 2 and m["view_counts"]["head"] == 1

def test_manifest_from_sharded_repo(fake_api, fake_reader):
    m = build_manifest("org/m", SHA, fake_reader, fake_api)
    assert m["access"] == "granted" and m["config"]["hidden_size"] == 896
    assert m["declared_bases"] == ["Qwen/Qwen2.5-0.5B"] and m["base_model_relation"] == "finetune"

def test_manifest_no_access_and_bin_only(fake_api, denying_reader, bin_only_api):
    assert build_manifest("org/gated", SHA, denying_reader, fake_api)["access"] == "no_access"
    m = build_manifest("org/convnext", SHA, denying_reader, bin_only_api)
    assert m["access"] == "t0_only" and m["weight_format"] == "bin" and m["tensors"] == []

def test_kohya_adapter_detected_without_config(fake_reader, kohya_api):
    m = build_manifest("org/pixel-lora", SHA, fake_reader, kohya_api)
    assert m["is_adapter"] and m["view_counts"]["lora_product"] > 0

def test_local_directory_is_content_pinned(tiny_checkpoint):
    a = build_manifest(str(tiny_checkpoint), "local", HubReader(), api=None)
    assert a["revision"].startswith("local:") and len(a["revision"]) == 70
    assert build_manifest(str(tiny_checkpoint), "local", HubReader(), api=None)["revision"] == a["revision"]

def test_manifest_rejects_unpinned(fake_api, fake_reader):
    with pytest.raises(ValueError):
        build_manifest("org/m", "main", fake_reader, fake_api)
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_manifest.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  **`select_file_set`** considers weight files only (`*.safetensors`,
  `*.gguf`, `*.bin`, `*.pt`, `*.pth`, `*.onnx`); other files are neither kept
  nor skipped. In order:
  - an `*.index.json` defines its shard set;
  - with `model_index.json` (diffusers), keep one non-variant file per
    component subfolder (skip `.fp16.`, `ema`, `_0.9vae`, and root
    single-files);
  - drop `consolidated*` and `original/` when HF shards exist;
  - single-file repositories that ship one component in several precisions
    (ComfyUI-style `*-bf16`, `*-fp8`, `*-comfy-int8-convrot`, and `*-nvfp4`
    files in one folder) keep one variant. Two files are variants when their
    names match after removing precision and packaging tokens (`bf16`,
    `fp16`, `f16`, `fp32`, `f32`, `fp8`, `e4m3fn`, `int8`, `int4`, `nvfp4`,
    `mxfp4`, `comfy`, `convrot`, `scaled`). The preference is bf16, then fp16,
    f32, fp8, and anything else; the others are `precision_variant`;
  - LoRA files bundled in a non-adapter repository (stray root LoRAs, or a
    `loras/` folder) are `bundled_adapter`;
  - pickle weights (`*.bin`, `*.pt`, `*.pth`) next to safetensors are
    `pickle_not_read`;
  - every other dropped file is recorded with `duplicate_layout`.

  **Other rules:**
  - Read `config.json` and `adapter_config.json` from the hub at the pinned
    commit (8 MiB cap, as in `checkpoint_tensors`).
  - A repository is an adapter if `adapter_config.json` exists **or** any
    tensor name matches a LoRA family.
  - GGUF-only, `.bin`-only, and ONNX-only repositories get T0 metadata only
    (`access = "t0_only"`).
  - In local mode, `declared_bases` comes from an optional `lineage.json`
    (`{"name": "...", "bases": [...], "relation": "..."}`). When `name` is
    present, it is the manifest's `repo_id`; otherwise the directory path is.
    Lab models use `name = "lab/<model_id>"`, so their model keys do not depend
    on the host's scratch path.
  - Xet chunk-overlap analysis is out of scope; `xet_hash` is recorded for
    later.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "evidence: T0 manifest with file-set selection and config facts"`

### Task 6: Fingerprints (T1 and T2) and the Python API

**Files:**
- Create: `evidence/fingerprint.py`
- Modify: `evidence/tests/conftest.py`. Add these fixtures, all with matrices
  of more than 256 rows:
  - `local_manifest`;
  - `parent_manifest`;
  - `child_with_extended_vocab` (rank-4 update; the parent's embedding has
    4,096 rows and the child's 4,104, so both share the block 1920:2176);
  - `moe_manifest` (a fused `experts.gate_up_proj` `(4, 1024, 512)` with
    `config.num_experts = 4`, `hidden_size = 512`);
  - `lora_manifest` (rank-8 PEFT pairs with an out-dim of 512).
- Test: `evidence/tests/test_fingerprint.py`

**Interfaces:**
- Consumes:
  - Tasks 1–5;
  - `net_esd.core.compute_esd_for_weight`;
  - `measurement_config.measurement_config(SimpleNamespace(analysis_source="checkpoint"))`.
- Produces:
  - `fingerprint_model(manifest: dict, reader: HubReader, tier: str, out_dir: Path, parent: dict | None = None, parent_cache: Path | None = None, sketch_k: int = SKETCH_K_DEFAULT, all_experts: bool = False, also_t1_blocks: tuple[int, ...] = (), read_threads: int = 8) -> Path`.
    It returns the fingerprint path. T2 also writes the spectra file.
  - `fingerprint_tensors(named_tensors: Iterable[tuple[str, torch.Tensor]], out_path: Path, identity: dict, sketch_k: int = SKETCH_K_DEFAULT) -> Path`.
    This is the in-memory entry point, for private formats (torch DCP, FSDP,
    Megatron) and for training-loop callbacks.
    - It treats every input as a `plain` decode and writes a T1 fingerprint
      (row blocks, `tier = "t1"`), comparable with
      `fingerprint_model(..., "t1")`.
    - `identity` must contain `repo_id` and `revision`; otherwise it raises
      `ValueError`. The caller chooses `out_path`.
  - **Block hash:** sha256 over the row block's bytes in the stored dtype for
    `plain` views (for `fingerprint_tensors`, the tensor's own dtype), and over
    the float32 decoded block for every other decode.
  - `load_fingerprint(path: Path) -> Fingerprint`, a dataclass with fields
    `names`, `geometry`, `component`, `rows`, `shape`, `sketch`, `sha256`,
    `spectra: list[np.ndarray]`, `delta: dict[str, np.ndarray]` (keys
    `frac_identical`, `rel_update_norm`, `rank`, `top_sv`, `status`, and
    `confidence`), `coverage: list[dict]`, and `meta: dict`.
  - **Fingerprint file** `<safe>.<tier>.fingerprint.safetensors`:
    - Tensors:
      - `sketch` f32 `[n, k, k]`; a view smaller than k in either dimension is
        zero-padded to k × k;
      - `rows` i64 `[n, 2]` (the block's `r0` and `r1`);
      - `shape` i64 `[n, 2]` (the view's full matrix shape);
      - `spectrum_values` f32 and `spectrum_offsets` i64 (the concatenated
        block spectra, T1 only);
      - `delta_scalars` f32 `[n, 3]` (`frac_identical`, `rel_update_norm`,
        `rank`, with `rank = −1` when there is no delta);
      - `delta_top_sv` f32 `[n, 16]`.
    - String metadata (JSON where structured):
      - per-view: `names`, `geometry`, `component`, `sha256`, `delta_status`,
        and `delta_confidence`;
      - `coverage`;
      - identity and settings: `repo_id`, `revision`, `tier`, `parent`,
        `sketch_k`, `sketch_seed`, `row_block`, `evidence_version`;
      - counters: `bytes_read`, `range_reads`, `resolves`, `elapsed_seconds`.
  - **T2 spectra file** `<safe>.t2.spectra.h5`. It uses the collector's
    canonical `/layers` and `/eigs` conventions, so code that reads those
    (`docs/operations/analysis.md`) reads it too; the legacy `/alpha` matrix is
    not written.
    - `/layers/longname` (the view names) and `/layers/<key>` for the numeric
      keys of `compute_esd_for_weight`;
    - `/eigs` (vlen float32), aligned with `/layers/longname`;
    - the attributes `format_version` (`measurement_config.FORMAT_VERSION`),
      `measurement_config_json`, and `numerics_version`.

    The collector's own writer (`worker.save_h5`) is not imported, because
    `worker` loads transformers. `conv_flat` views pass the stored N-D tensor
    to `compute_esd_for_weight`, so the collector's convolution convention
    (`conv_norm`) is kept.
  - **Extra block heights.** An entry of `also_t1_blocks` equal to
    `ROW_BLOCK` writes the standard `<safe>.t1.fingerprint.safetensors`. Other
    heights write `<safe>.t1b<b>.fingerprint.safetensors`.

- [ ] **Step 1: Write the failing tests**

```python
def test_t1_reads_only_blocks_and_records_skips(local_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "t1", tmp_path))
    assert "model.norm.weight" not in fp.names and fp.sketch.shape[1:] == (64, 64)
    reasons = {c["name"]: c["reason"] for c in fp.coverage}
    assert reasons["model.norm.weight"] == "not_a_matrix" and reasons["model.position_ids"] == "non_float_buffer"
    tensor_bytes = sum(t["end"] - t["start"] for t in local_manifest["tensors"])
    assert fp.meta["bytes_read"] < tensor_bytes

def test_t2_spectra_and_free_t1_blocks(local_manifest, tmp_path):
    fingerprint_model(local_manifest, HubReader(), "t2", tmp_path, also_t1_blocks=(128, 256))
    safe = safe_filename(f"{local_manifest['repo_id']}@{local_manifest['revision']}")
    with h5py.File(tmp_path / f"{safe}.t2.spectra.h5") as h:
        assert h.attrs["numerics_version"] == "7" and np.isfinite(h["layers/alpha"][:]).any()
        assert len(h["eigs"]) == len(h["layers/longname"])
    assert (tmp_path / f"{safe}.t1b128.fingerprint.safetensors").exists()
    assert (tmp_path / f"{safe}.t1.fingerprint.safetensors").exists()        # b == ROW_BLOCK

def test_revisions_do_not_collide(local_manifest, tmp_path):
    other = {**local_manifest, "revision": "local:" + "0" * 64}
    a = fingerprint_model(local_manifest, HubReader(), "t1", tmp_path)
    b = fingerprint_model(other, HubReader(), "t1", tmp_path)
    assert a != b and a.exists() and b.exists()

def test_delta_on_common_rows_despite_vocab_growth(child_with_extended_vocab, parent_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(child_with_extended_vocab, HubReader(), "t1", tmp_path,
                                            parent=parent_manifest, parent_cache=tmp_path / "cache"))
    i = fp.names.index("model.embed_tokens.weight")
    assert fp.delta["status"][i] in {"ok", "below_floor"}       # rows 1920:2176 exist in both

def test_moe_subset_and_lora_product(moe_manifest, lora_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(moe_manifest, HubReader(), "t1", tmp_path, sketch_k=128, all_experts=True))
    assert fp.sketch.shape[1:] == (128, 128) and fp.geometry.count("expert_slice") == 8   # 4 experts x gate/up
    fp = load_fingerprint(fingerprint_model(lora_manifest, HubReader(), "t1", tmp_path))
    i = fp.geometry.index("lora_product")
    assert np.count_nonzero(fp.spectra[i] > 1e-6 * fp.spectra[i].max()) == 8

def test_fingerprint_tensors_matches_file_path(local_manifest, tmp_path):
    from safetensors.torch import load_file
    tensors = {}
    for f in local_manifest["files"]:
        tensors.update(load_file(Path(local_manifest["repo_id"]) / f["filename"]))
    a = load_fingerprint(fingerprint_tensors(tensors.items(), tmp_path / "a.safetensors",
                                             {"repo_id": "x", "revision": "local:" + "0" * 64}))
    b = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "t1", tmp_path))
    i, j = a.names.index("model.layers.0.self_attn.q_proj.weight"), b.names.index("model.layers.0.self_attn.q_proj.weight")
    assert sketch_cosine(a.sketch[i], b.sketch[j]) > 0.9999

def test_atomic_write_leaves_no_partial_file(local_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr("evidence.fingerprint.two_sided_sketch", Mock(side_effect=RuntimeError))
    with pytest.raises(RuntimeError):
        fingerprint_model(local_manifest, HubReader(), "t1", tmp_path)
    assert not list(tmp_path.glob("*.safetensors"))
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_fingerprint.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  - **Reads:**
    - T1 reads the row block of each view as one range, or the needed slice
      bytes for `expert_slice`. Block rows are contiguous for `matrix`,
      `expert_slice` in (out, in) storage, and `conv_flat`.
    - Transposed expert storage reads the whole slice.
    - `head` views have fewer than 8 rows and are read whole.
    - FP8 and FP4 decodes also read the covering scale rows.
    - T2 reads full tensors.
  - **LoRA:** factors are always read whole. The T1 sketch is
    `scale · B[r0:r1] @ A`, using `row_block_bounds(out_rows)`, so it is
    comparable with a merged child's block. The spectrum uses the QR shortcut
    (the singular values of `R_B R_Aᵀ`).
  - **Duplicates:** a view with the same shape and block sha256 as an earlier
    view gets `duplicate_of` in coverage and no second sketch.
  - **Concurrency:** range reads within a model run on `read_threads` threads.
    Compute is sequential, one view at a time.
  - **Parent cache:** parent blocks are cached as
    `<sha256(repo@sha|name|r0|r1)>.bin` in `parent_cache`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "evidence: T1/T2 fingerprints in safetensors, spectra in HDF5, tensor API"`

### Task 7: Selection frame and the pilot draw

**Files:**
- Create: `evidence/frame.py`
- Create: `evidence/tests/data/hub_stats_sample.json` (Step 1).
- Modify: `evidence/tests/conftest.py`, adding the `hub_stats_sample` fixture:
  that JSON file loaded as a `pd.DataFrame` with the `load_hub_stats` columns.
- Test: `evidence/tests/test_frame.py`

**Interfaces:**
- Produces:
  - `load_hub_stats(path: Path) -> pd.DataFrame`. It reads only `id`,
    `pipeline_tag`, `library_name`, `tags`, `downloads`, `safetensors`, `gguf`,
    `siblings`, `baseModels`, `createdAt`, and `cardData`.
  - `MODALITY`: a dict mapping `pipeline_tag` (and `library_name ==
    "diffusers"`) to one of `text`, `vision-language`, `vision`, `audio`,
    `image-generation`, `video`, `embedding`, `3d`, `robotics`, or `other`.
    Adapters keep their base's modality, taken from `base_model:adapter:`.
  - `build_frame(df) -> pd.DataFrame`, with these columns:
    - identity and type: `repo_id`, `pipeline_tag`, `modality`, `is_adapter`,
      `system_one` (tags `system-one` or `decision-model`), and `is_decision`
      (that, or a classification `pipeline_tag`);
    - size and format: `downloads`, `params`, `weight_format`
      (`safetensors`/`gguf`/`bin`/`onnx`), `size_stratum` (cut at 1e9, 3e9,
      9e9, 35e9, and 75e9);
    - strata: `stratum` is `decision` for System One models, then
      `"text:<size_stratum>"` for text models, and the modality alone
      otherwise, matching `PILOT_HUB_PLAN`; `t0_only` (true for gguf, bin, and
      onnx);
    - lineage and time: `declared_relation`, `declared_bases`,
      `created_quarter`.
  - `draw_stratified(frame, n_per_stratum: dict[str, int], seed: int) -> pd.DataFrame`.
    It samples uniformly without replacement within each stratum. It adds
    `inclusion_prob = n_h / N_h`, `selection_reason`, and an empty `revision`
    column. A filled `revision` (for example the leaderboard's `Model sha` for
    labeled models) makes `manifest` pin that commit instead of resolving
    `main`, so score joins can verify.
  - `PILOT_HUB_PLAN = {"text:<1B": 40, "text:1-3B": 40, "text:3-9B": 60, "text:9-35B": 30, "text:35-75B": 10, "vision-language": 30, "vision": 20, "image-generation": 20, "video": 10, "audio": 20, "embedding": 20, "decision": 20}`.
    That is 320 Hub models, as a sanity check of coverage and cost. The G0
    thresholds are measured on the lab and known-independent pretrains
    (Task 9).

- [ ] **Step 1:** Inspect the real schema. Read 20 rows of `models.parquet` with
  pyarrow. Record the actual structures of `safetensors`, `gguf`, `siblings`,
  and `baseModels` in `evidence/tests/data/hub_stats_sample.json`, with 40 rows
  covering:
  - every size stratum and every modality;
  - a diffusers repository and a kohya LoRA;
  - a System One model (tag `system-one`), a video model, and an ONNX-only
    repository;
  - a mixture-of-experts model;
  - a GGUF-only repository and a `.bin`-only repository;
  - a repository with no weights.
- [ ] **Step 2: Write the failing tests**

```python
def test_build_frame_covers_modalities_and_t0_only(hub_stats_sample):
    frame = build_frame(hub_stats_sample)
    assert {"text", "vision-language", "vision", "audio", "image-generation", "video", "embedding"} <= set(frame.modality)
    assert frame.system_one.any() and frame.loc[frame.system_one, "stratum"].eq("decision").all()
    assert frame.loc[frame.weight_format.isin(["gguf", "bin"]), "t0_only"].all()
    assert frame.loc[frame.params == 7.6e9, "size_stratum"].eq("3-9B").all()

def test_draw_records_inclusion_probability(hub_stats_sample):
    frame = build_frame(hub_stats_sample)
    s = draw_stratified(frame, {"text:<1B": 2}, seed=20261028)
    assert len(s) == 2 and np.allclose(s.inclusion_prob, 2 / frame.stratum.value_counts()["text:<1B"])
    assert s.equals(draw_stratified(frame, {"text:<1B": 2}, seed=20261028))

def test_draw_caps_at_stratum_size(hub_stats_sample):
    s = draw_stratified(build_frame(hub_stats_sample), {"audio": 10_000}, seed=1)
    assert (s.inclusion_prob == 1.0).all() and set(s.modality) == {"audio"}
```

- [ ] **Step 3:** Run `python -m pytest evidence/tests/test_frame.py -q`. Expected: FAIL.
- [ ] **Step 4:** Implement `evidence/frame.py`, and pin `pyarrow` in `requirements.txt`.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit: `git commit -m "evidence: all-modality frame and stratified draws"`

### Task 8: CLI: fingerprint, diff, check, and batch with resume

**Files:**
- Create: `evidence/cli.py` (run as `python -m evidence.cli`)
- Create: `evidence/compare.py`
- Modify: `evidence/tests/conftest.py`.
  - Add the helpers `run_cli(args) -> dict`, which parses stdout JSON, and
    `run_cli_code(args) -> int`.
  - Add the fixtures `child_dir`, `parent_dir`, and `unrelated_dir`. These are
    local checkpoints. The child is the parent plus a rank-4 update, and its
    embedding has 8 more rows than the parent's (the one `reshaped` view). The
    child and the unrelated model both declare the parent in `lineage.json`,
    so `check` on the unrelated model compares it with the parent and finds it
    inconsistent.
  - Also add `null_file`, `run_dir`, `models_csv`, and
    `models_csv_with_denied` (one gated repository).
- Test: `evidence/tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 2 and 4–7.
- Produces:
  - `pair_similarity(a: Fingerprint, b: Fingerprint, sketch_k: int | None = None) -> dict`
    (in `evidence/compare.py`), with keys `median_cosine` and `n_shared`. It
    is the median cosine over shared view names, computed on the top-left
    `sketch_k` block (by default, the smaller of the two fingerprints' k).
    This is the one statistic the calibration file describes; Task 9 and the
    zoo import it from here.
  - `diff_fingerprints(a: Fingerprint, b: Fingerprint, null: dict | None = None) -> dict`
    (in `evidence/compare.py`). It reports **measured facts only**:
    - per view: `status` (`identical`, `changed`, `added`, `removed`, or
      `reshaped`), `cosine`, and when a delta exists, `frac_identical`,
      `rel_update_norm`, `rank`, `rank_confidence`, and `delta_status`;
    - `summary`: counts by status, a per-component median `rel_update_norm`,
      `median_cosine`, and `heads`: the `head` views that were added or changed,
      which is usually the whole difference between a decision model and its
      base;
    - `relatedness`: `{"verdict": "uncalibrated", "median_cosine": ...}` when
      `null` is None. With a calibration file it reports `related`,
      `unrelated`, or `undetermined` with a p-value, computing `median_cosine`
      at the file's `sketch_k`. The verdict stays `uncalibrated`, with a
      `reason`, if either fingerprint's k is below that `sketch_k`, or if the
      tiers or row-block heights differ from the file's.
    - `reshaped` compares the stored `shape`; `rank_confidence` comes from
      `delta_confidence`.
  - `check_verdict(diff: dict) -> str` maps `related` to `consistent`,
    `unrelated` to `inconsistent`, `undetermined` to `undetermined`, and
    `uncalibrated` to `uncalibrated`. The `check` JSON is the diff plus this
    top-level `verdict`; the zoo uses the same function.
  - CLI subcommands:
    - `fingerprint TARGET [--tier t1|t2] [--out DIR] [--sketch_k K] [--all_experts] [--json]`.
      `TARGET` is `org/repo[@rev]` or a local path.
    - `diff A [B] [--null PATH] [--json|--markdown]`. `B` defaults to `A`'s
      first declared base. The JSON is written to stdout. Exit codes: 0 when it
      ran, 2 on errors.
    - `check TARGET [--null PATH]`, which diffs against the declared base. It
      exits 0 for `consistent`, `undetermined`, or `uncalibrated`; 1 for
      `inconsistent` (possible only with a calibration file); and 2 on
      errors.
    - `frame --hub_stats PATH --out frame.csv`.
    - `sample --frame frame.csv --plan pilot --seed 20261028 --out models.csv`.
    - `manifest --models models.csv --out RUN/manifests --workers N [--include_declared_bases]`.
      A filled `revision` column is pinned as given; otherwise `main` is
      resolved once.
    - `batch --models models.csv --manifests RUN/manifests --tier {t1,t2} --out RUN/<tier> --workers N [--sketch_k K] [--also_t1_blocks 128,256,512] [--with_parent] [--parent_cache DIR] [--retry_denied]`.
  - Batch steps write `failures.csv` (`repo_id, stage, reason, message`) and
    `summary.json`, with per-model `bytes_read`, `range_reads`, `resolves`,
    `elapsed_seconds`, `stratum`, `modality`, view counts by geometry, and
    `status`.

- [ ] **Step 1: Write the failing tests**

```python
def test_diff_reports_facts_and_uncalibrated(child_dir, parent_dir):
    out = run_cli(["diff", str(child_dir), str(parent_dir), "--json"])
    assert out["relatedness"]["verdict"] == "uncalibrated"
    assert out["summary"]["counts"]["reshaped"] == 1 and out["summary"]["counts"]["changed"] > 0
    assert "predicted" not in json.dumps(out)              # no capability predictions

def test_check_exit_codes(child_dir, unrelated_dir, null_file):
    assert run_cli_code(["check", str(child_dir)]) == 0                         # uncalibrated
    assert run_cli_code(["check", str(unrelated_dir), "--null", str(null_file)]) == 1

def test_resume_skips_done_and_overwrites_tmp(run_dir, models_csv):
    args = ["batch", "--models", str(models_csv), "--manifests", str(run_dir / "manifests"),
            "--tier", "t1", "--out", str(run_dir / "t1"), "--workers", "2"]
    main(args); done = sorted((run_dir / "t1").glob("*.safetensors"))
    mtimes = [p.stat().st_mtime_ns for p in done]
    (run_dir / "t1" / "extra.t1.fingerprint.safetensors.tmp").write_bytes(b"partial")
    main(args)
    assert [p.stat().st_mtime_ns for p in done] == mtimes
    assert not (run_dir / "t1" / "extra.t1.fingerprint.safetensors.tmp").exists()

def test_no_access_recorded_once(run_dir, models_csv_with_denied, denying_reader, fake_api):
    args = ["manifest", "--models", str(models_csv_with_denied), "--out", str(run_dir / "manifests")]
    main(args, reader_factory=lambda: denying_reader, api=fake_api)
    main(args, reader_factory=lambda: denying_reader, api=fake_api)
    assert (pd.read_csv(run_dir / "manifests" / "failures.csv").reason == "no_access").sum() == 1

def test_tokens_never_written(run_dir, models_csv, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_marker")
    main(["manifest", "--models", str(models_csv), "--out", str(run_dir / "manifests")])
    for p in run_dir.rglob("*"):
        if p.is_file():
            assert b"hf_secret_marker" not in p.read_bytes()
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  - `main(argv: list[str], *, reader_factory: Callable[[], HubReader] | None = None, api=None) -> int`.
    Tests inject fake or denying readers and a fake API this way. The default
    builds a `HubReader` from the environment and uses `HfApi`. With
    `--workers 1` (the default), everything runs in-process.
  - Use `ProcessPoolExecutor` with one `HubReader` per worker.
  - A `no_access` manifest is written once and skipped on re-runs unless
    `--retry_denied` is given.
  - Delete `*.tmp` in the output directory at startup.
  - Single-target `fingerprint`, `diff`, and `check` build a T0 manifest on
    the fly, at the pinned revision `main` → SHA.
  - The calibration file format is JSON:
    `{"statistic": "median_cosine", "sketch_k": 64, "row_block": 256, "tier": "t1", "n_pairs": n, "null_quantiles": [...], "related_threshold": x}`,
    written by Task 9.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "evidence: fingerprint/diff/check CLI with JSON, exit codes and resumable batch"`

### Task 9: Pilot analysis and the G0 report

**Files:**
- Create: `evidence/pilot_analysis.py`
- Modify: `evidence/tests/conftest.py`, adding the `synthetic_run` fixture
  described in Step 1: a namespace with `root`, `t1`, `t2`, and `pairs`.
- Test: `evidence/tests/test_pilot_analysis.py`

**Interfaces:**
- Consumes: the fingerprints (Task 6), and the summaries and failures (Task 8).
- Produces:
  - `pair_similarity` lives in `evidence/compare.py` (Task 8); this task uses
    it.
  - `lineage_pairs(lab_edges: Path, manifests_dir: Path) -> pd.DataFrame`, with
    columns `a`, `b`, `label`, and `source`.
    - `lab_edges` is `lab/edges.csv`. `models.csv` is read from the same
      directory to map lab IDs to manifests (format:
      `Zoo-Keeper/handoff/GUIDANCE.md`, section 4.4).
    - Labels are `parent`, `sibling`, `grandparent`, and `independent`. They
      come from:
    - the lab's exact edge list, for parents, siblings, and grandparents;
    - known-independent pretrains with identical shapes, such as the
      PolyPythias seed variants (the `kind = independent` rows of
      `models.csv`);
    - Hub declared-base pairs, as a sanity-check source only.
  - `tier_agreement(pairs, t1_dir, t2_dir) -> dict`, with these keys:
    - `auroc_t1` and `auroc_t2`, for parent against independent;
    - `auroc_parent_vs_sibling_t1/t2` and `auroc_parent_vs_grandparent_t1/t2`;
    - `spearman_within_related`, between T1 and T2 cosines over the related
      pairs only;
    - `decision_agreement`, at the threshold that gives TPR 0.9 on T2;
    - `false_matches_per_query`, at TPR 0.9 against the full candidate set;
    - `null_mean`, `null_sd`, and quantiles of independent-pair similarity;
    - `byte_curve`, the AUROC for block heights 128, 256, and 512.
  - `spectra_agreement(t1_dir, t2_dir) -> dict`, with
    `median_abs_alpha_diff` and `median_per_matrix_spearman` between block and
    full spectra. Block alpha is computed with the same estimator as T2.
  - `throughput_report(run_dir) -> pd.DataFrame`, per stratum: the median and
    p90 of `bytes_read` and `elapsed_seconds`, `resolves` per model,
    `fraction_of_model_bytes`, and failure rate by reason.
  - `write_calibration(pairs, t1_dir, out: Path, sketch_k: int = 64) -> None`.
    It writes the `--null` file used by `diff` and `check` (format in Task 8),
    from `pair_similarity` at `sketch_k` over the independent pairs.
    - At k = 64 it applies to census and lab fingerprints alike, because
      sketches are nested.
    - Its p-values are empirical, floored at `1 / (n_pairs + 1)`. Tail
      modeling and per-query error control belong to the P1 engine.
  - `write_g0_report(run_dir: Path, out: Path) -> None`, writing Markdown that
    compares each metric with the G0 thresholds in the merged plan (section
    12) and prints PASS or FAIL per line.

- [ ] **Step 1: Write the failing tests.** Use synthetic fingerprints from
  Task 6 on derived tiny checkpoints, all with more than 256 rows: a parent,
  2 children (rank-4 and rank-32 updates), a grandchild, 2 siblings, and 3
  independents.

```python
def test_tier_agreement_metrics(synthetic_run):
    out = tier_agreement(synthetic_run.pairs, synthetic_run.t1, synthetic_run.t2)
    assert out["auroc_t1"] == 1.0 and out["auroc_t2"] == 1.0
    assert out["spearman_within_related"] > 0.8 and out["decision_agreement"] >= 0.9
    assert set(out["byte_curve"]) == {128, 256, 512}

def test_g0_report_marks_pass_fail(synthetic_run, tmp_path):
    write_g0_report(synthetic_run.root, tmp_path / "g0.md")
    text = (tmp_path / "g0.md").read_text()
    assert ("PASS" in text or "FAIL" in text) and "false_matches_per_query" in text
```

- [ ] **Step 2:** Run `python -m pytest evidence/tests/test_pilot_analysis.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. AUROC is computed from ranks (Mann–Whitney);
  there is no sklearn dependency.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -m "evidence: G0 pilot analysis with nested-estimand AUROCs and calibration file"`

### Task 10: Documentation, runbook, and day-one checks

**Files:**
- Create: `docs/operations/evidence.md`
- Modify: `README.md`, adding a short "Weight evidence (range reads)" section
  after "Measurement scope" with a link.
- Create: `evidence/scripts/day_one_checks.py`. It depends on no other task,
  so write and run it in week 1 (handoff checklist item A0). It is a
  diagnostic script, not library code, so it has no unit test. It takes
  `--data PATH` (default `$ZOO_DATA`), prints one JSON object to stdout with
  the keys `host`, `outbound_https`, `svdvals_seconds`, `data_path`,
  `data_free_gb`, and `xet_reachable`, and is committed as
  `evidence: day-one infrastructure checks`. Xet is reachable when an
  authenticated `HEAD` on a `/resolve/` URL of a public repository returns an
  `x-xet-hash` header. It reports:
  - outbound HTTPS from the current node;
  - CPU `svdvals` time for a 3584×18944 float32 matrix;
  - free space under the data path;
  - whether Xet reconstruction metadata is reachable for one public repository.

- [ ] **Step 1:** Write `docs/operations/evidence.md`. It covers:
  - what each tier reads and stores;
  - the decode × geometry table and every skip reason;
  - the fingerprint and spectra layouts;
  - the token rule (`HF_TOKEN` from the environment, never in files);
  - the statement that blocks and parent caches are never published;
  - the pilot runbook in order: `frame`, `sample`,
    `manifest --include_declared_bases`, then
    `batch --tier t2 --also_t1_blocks 128,256,512 --with_parent` on the lab,
    the independents, and the T2 subset, then `batch --tier t1` on the Hub
    pilot, then `write_g0_report` and `write_calibration`.
- [ ] **Step 2:** Run the full suite with `python -m pytest evidence/tests esd_experiment/tests -q`. Expected: all pass, with the network test skipped.
- [ ] **Step 3:** With `HF_TOKEN` exported, run `MODEL_ZOO_NETWORK_TESTS=1 python -m pytest evidence/tests/test_hub.py::test_network_smoke -q` and `python evidence/scripts/day_one_checks.py`. Expected: PASS, and a printed report.
- [ ] **Step 4:** Commit: `git commit -m "docs: weight evidence tiers, view table and pilot runbook"`

## After this plan (the person launches the large runs; handoff D4.1)

1. **Day-one checks first.** Their CPU SVD timing replaces the T2
   compute estimate in the merged plan's section 13.
2. **Lab and independents.**
   - Fingerprint the lab and the known-independent pretrains at T2, with free
     T1 blocks.
   - Run T2 on the Hub related pairs, plus 10 models per stratum above 9B.
   - Run T1 on the remaining Hub pilot models.
3. **Gate G0.** Generate the G0 report and decide on 2026-12-04 against the
   merged plan's section 12 thresholds. Record the decisions there.
