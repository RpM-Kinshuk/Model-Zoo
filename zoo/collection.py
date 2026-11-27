"""Describe the evidence closure of a catalog without copying its inputs."""

from collections import Counter, defaultdict
import json
from pathlib import Path


def _measurement_file(path):
    name = Path(path).name
    return (name.endswith((".manifest.json", ".source-metadata.json.gz",
                           ".fingerprint.safetensors", ".delta.safetensors", ".spectra.h5"))
            or name in {"package.json", "evidence.json", "stats.csv", "metrics.h5"})


def write_collection_receipt(directory, tables, data_root, *, manifest_paths=(), provenance=(),
                             loaded_manifests=()):
    """Write a versioned inventory; completeness refers only to listed references.

    Hashes come from the builder's existing provenance pass. No checkpoint is
    archived, no measurements are recomputed and no large files are rehashed.
    `loaded_manifests` ((path, JSON) pairs the builder already parsed) avoids
    reading those manifests again.
    """
    parsed = {Path(path).resolve(): manifest for path, manifest in loaded_manifests}
    root = Path(data_root).resolve()
    directory = Path(directory)
    references = {}
    hashes = {}
    for entry in provenance:
        if entry.get("kind") == "file":
            path = Path(entry["path"])
            absolute = path if path.is_absolute() else root / path
            hashes[str(absolute.resolve())] = entry.get("sha256")

    def add(path, kind):
        path = Path(path)
        absolute = path if path.is_absolute() else root / path
        resolved = absolute.resolve()
        inside = resolved.is_relative_to(root)
        label = resolved.relative_to(root).as_posix() if inside else str(absolute)
        if label not in references:
            status = "available" if inside and resolved.is_file() else "missing" if inside else "outside_data_root"
            record = dict(path=label, kind=kind, status=status)
            if status == "available":
                record["bytes"] = resolved.stat().st_size
                if hashes.get(str(resolved)):
                    record["sha256"] = hashes[str(resolved)]
            references[label] = record
        return references[label]

    for entry in provenance:
        if entry.get("kind") == "file" and _measurement_file(entry["path"]):
            add(entry["path"], "measurement_or_metadata")

    metadata = {}
    manifests = set()
    for path in manifest_paths:
        path = Path(path)
        absolute = path if path.is_absolute() else root / path
        add(absolute, "manifest")
        if not absolute.resolve().is_relative_to(root) or not absolute.is_file():
            continue
        manifest = parsed.get(absolute.resolve()) or json.loads(absolute.read_text())
        key = manifest["repo_id"] + "@" + manifest["revision"]
        manifests.add(key)
        sibling = absolute.with_name(absolute.name.removesuffix(".manifest.json") + ".source-metadata.json.gz")
        metadata[key] = "retained" if sibling.is_file() else "not_captured"
        if sibling.is_file():
            add(sibling, "source_metadata")

    coverage_path = directory / "score-coverage.json"
    score_coverage = json.loads(coverage_path.read_text()).get("records", []) if coverage_path.is_file() else []
    score_statuses = defaultdict(set)
    for item in score_coverage:
        score_statuses[item.get("model_key")].add(item["status"])
    score_rows = tables.get("scores")
    counts = Counter(score_rows.model_key) if score_rows is not None and "model_key" in score_rows else Counter()
    coverage = tables.get("coverage")
    sampled = set()
    if coverage is not None and {"model_key", "reason"}.issubset(coverage.columns):
        sampled = set(coverage.loc[coverage.reason == "expert_not_sampled", "model_key"])
    models = []
    for row in tables["models"].to_dict("records"):
        key = row["model_key"]
        for field in ("row_block_fingerprint", "full_weight_fingerprint", "full_weight_spectra"):
            value = row.get(field)
            if isinstance(value, str) and value:
                add(value, field)
        statuses = sorted(score_statuses[key])
        tier = ("full_spectra_measured_views" if row.get("full_weight_spectra")
                else "row_blocks" if row.get("row_block_fingerprint") else "metadata_only")
        models.append(dict(model_key=key, measurement_tier=tier, sampled_experts=key in sampled,
                           manifest="available" if key in manifests else "not_supplied",
                           measurement_coverage=row.get("coverage_status", "unknown"),
                           raw_metadata=metadata.get(key, "not_captured"), score_rows=counts[key],
                           score_coverage=",".join(statuses) if statuses else "not_attempted"))
    records = [references[key] for key in sorted(references)]
    unbound = [item["path"] for item in records if not item.get("sha256")]
    receipt = dict(schema_version="1", collection_contract="bounded-evidence-v1",
                   raw_checkpoint_archival=False,
                   reference_closure_complete=(all(item["status"] == "available" for item in records)
                                               and all(item["manifest"] == "available" for item in models)),
                   reference_hash_binding_complete=not unbound,
                   references_without_hashes=unbound,
                   required_data_root_files=records, models=models,
                   requires_additional_inputs=["raw_tensor_methods", "new_tensor_views_or_singular_vectors",
                                               "new_behavioral_evaluations", "uncaptured_or_changed_metadata"],
                   limits="Closure covers listed retained evidence, not every future experiment. Metadata presence is not completeness or verification.")
    path = directory / "collection.json"
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return path
