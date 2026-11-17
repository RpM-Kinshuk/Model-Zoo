# Bounded evidence collection, contract v1

`bounded-evidence-v1` is an additive retention contract. It preserves the existing measurement formats, numerics, geometry, sampling and calibration. It does not promise that a collected database can answer every future research question.

For a new list of Hugging Face repositories, use the existing pinned collector and publisher, then `python -m zoo.cli build --data_root "$ZOO_DATA" --packages "$ZOO_DATA/packages"`. A database generation includes `collection.json`, `score-coverage.json`, the normal tables and indexes, and durable imported score sources. Keep that generation **and** the files listed in its `collection.json` under the same relative paths in `data_root`. Parquet files alone are not the dataset.

The package builder selects complete supported measurement packages. Gated, inaccessible, unsupported and incomplete attempts can remain outside that selection: retain their normal collector attempt manifests/failure records as acquisition provenance. The receipt describes the selected catalog's references; it is not an inventory of every attempted repository. An excluded or unscored model has unknown performance, not a score of zero.

## Default retention

* Preserve exact repository revisions, normalized manifests, measurements and coverage reasons. Existing full collection retains eigenvalues for measured full-weight views and row-block fingerprints at the existing heights. Row-only fallback and sampled mixture-of-experts coverage remain explicit; a full-tier label does not mean every expert was measured.
* Write a deterministic gzip source-metadata sidecar beside each newly written manifest. Retain structured card data and model-info API objects plus a fixed allowlist of small source files: card text, model/generation/preprocessor configuration, tokenizer configuration/special-token declarations, adapter configuration and supported merge/recipe/lineage files. File content is exact UTF-8 with a hash; API objects are explicitly canonical object representations, not claimed HTTP response bytes.
* Limit each captured document to 256 KiB and the entire serialized uncompressed sidecar to 1 MiB, including JSON overhead. Record absent, denied, offline, unsupported, oversized and failed captures explicitly. Do not truncate a document and call it complete. Existing valid captures for a pinned identity survive resume and later manifest rewrites.
* Import supported structured `cardData` `model-index` scores by default. Preserve declared benchmark, metric and numeric value, including zero, without guessing units or task equivalence. A model-card commit is not proof of which weights were evaluated: these rows remain revision-unverified.
* Read configured local evaluation sources from `<data_root>/score-sources.json` by default. Sources require explicit adapters, mappings and content hashes; builders do not scan arbitrary CSV files or contact evaluation services. See [database operations](zoo.md) for the configuration and explicit import options.
* On a normal rebuild of the same managed output, retain existing score rows for selected identities and their complete source contexts, including unbound rows and within-source duplicates. Repeated identical sources are idempotent. Context-only sources retain that role. `--reset_scores` deliberately disables inheritance; current configured/card sources still apply.

An explicit hash-pinned source configuration or import is authority to activate that source against selected identities. Without that instruction, retained context-only files never gain score bindings merely because more models are selected. Rebuilding a different output has no old generation to inherit; carry its source configuration or use the explicit import operations.

## Storage and portability

No extra raw checkpoint copies, tokenizer vocabularies, datasets or executable remote code are archived by this contract. Metadata is bounded; score files are explicitly selected and stored once per distinct source context, with immutable snapshots reused through hardlinks when supported. Cross-filesystem copies can consume additional space. Separate generations still have their own database tables and indexes.

`collection.json` lists the generation's external measurement/metadata references, available sizes and already computed hashes. It copies no measurement data and does not rehash full weights. `reference_closure_complete` means those listed references exist within `data_root` and every model has a supplied manifest. `reference_hash_binding_complete` separately reports whether every reference has a recorded hash; unbound paths are listed. Per-model metadata/measurement/score coverage must also be inspected. These are not scientific readiness flags or a complete repository archive. Legacy inputs cannot acquire previously discarded metadata through an offline rebuild.

Retain one usable database generation with its referenced immutable packages and evaluation sources before considering old build cleanup. Do not remove scientific inputs, reports, splits, checkpoints or useful reproduction code through an automatic retention rule. Storage reports should distinguish logical path bytes from bytes shared by hardlinks.

## What still needs new inputs

The retained evidence supports queries over captured metadata, reported evaluations, stored fingerprints and spectra under their recorded measurement settings. Eigenvalues support compatible spectrum-based analyses; they do not reconstruct tensors or singular vectors. New tensor views, different geometry, unmeasured experts, raw-tensor methods, new behavioral evaluations, absent benchmark settings and metadata outside the capture allowlist can require another source read or evaluation. Reported scores are useful evidence, not automatically comparable performance or a model-choice guarantee.

Freeze this concrete collection contract and preserve additive compatibility. Keep query methods and research development open. A larger collection still requires the normal sampling/quality design and launch rules.
