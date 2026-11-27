"""Build complete immutable generations, then atomically publish a pointer.

The output is a relative symlink to a sibling managed generation. Rebuilds
retain prior generations for readers that resolved the old pointer. Interrupted
private staging/orphan generations can be inspected and manually removed when
no build or reader uses them; builds never sweep another call's artifacts.
Once publication is attempted, its complete generation is retained even on an
exception: the pointer may already have switched when an interrupt arrives.
Existing real directories and unrelated symlinks require manual migration.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Sequence
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import ZOO_VERSION, READABLE_ZOO_VERSIONS
from evidence.compatibility import is_baseline_fingerprint, normalize_model_table, normalize_view_table
from .lineage import declared_edges, merge_edges, shared_tensor_edges, similarity_edges, with_pair_delta_evidence
from .retrieval import build_retrieval
from .schema import (COVERAGE_COLUMNS, LINEAGE_COLUMNS, MODELS_COLUMNS,
                     PREDICTIONS_COLUMNS, SCORES_COLUMNS, VIEWS_COLUMNS, LAYERS_COLUMNS, empty_table)
from .tables import _fingerprint_files, build_tables, duplicate_groups, load_manifests, pair_delta_files
from .publish import package_inputs, current_conventions_id
from .layers import build_layers
from .collection import write_collection_receipt
from .scores import (_canonical_scores, check_score_sources, prepare_score_inputs,
                     score_source_paths, write_score_sources)
from measurement_config import safe_filename


_COLUMNS = dict(models=MODELS_COLUMNS, views=VIEWS_COLUMNS, coverage=COVERAGE_COLUMNS,
                lineage=LINEAGE_COLUMNS, scores=SCORES_COLUMNS, predictions=PREDICTIONS_COLUMNS, layers=LAYERS_COLUMNS)
_PROTOCOL = "immutable-generation-v1"
# Parquet tables use zstd: about half the bytes of snappy for these hash/name
# heavy tables, read transparently by Arrow and DuckDB. Level fixed for stable bytes.
_PARQUET_OPTIONS = dict(compression="zstd", compression_level=3)
# Zoo version <=2 columns whose data has one home elsewhere in version 3
# (skipped tensors: coverage rows with status 'skipped').
_RETIRED_COLUMNS = {"models": frozenset({"skipped_tensors"})}
# Generation files rewritten by a score extension; everything else is inherited.
_SCORE_EXTENSION_REWRITES = frozenset({"scores.parquet", "zoo.json", "collection.json"})
_FLOATS = {"score", "p_value", "value", "lower", "upper", "inclusion_prob", "downloads", "params",
           "delta_frac_identical", "delta_rel_update_norm"}
_LAYER_FLOATS = set(LAYERS_COLUMNS) - {"model_key", "longname", "module_name", "weight_attribute", "slice", "view_name", "fit_status", "h5_index", "num_evals", "matrix_rank", "n_tail"}
_FLOATS |= _LAYER_FLOATS
_INTS = {"view_index", "n_rows", "n_cols", "r0", "r1", "n_views", "num_labels", "n_items", "h5_index", "num_evals", "matrix_rank", "n_tail", "delta_rank"}
_BOOLS = {"is_adapter", "gated", "is_decision", "system_one", "calibrated", "revision_verified"}
_NESTED = {
    "labels": pa.map_(pa.string(), pa.string()),
    "declared_bases": pa.list_(pa.string()),
    "inputs": pa.list_(pa.string()),  # v0 predictions are always empty; no future format claim.
}


def _arrow(name, table, *, empty_text_as_null=True):
    """Explicit types also support all-empty maps, lists and whole tables.

    Every missing value is written as null: NaN floats and empty strings
    included, so readers need one missing-value rule for every table.
    `empty_text_as_null=False` reproduces zoo version <=2 rows, only to
    recognize identities those builds recorded.
    """
    columns = _COLUMNS[name]
    if set(table.columns) != set(columns):
        raise ValueError(f"Invalid {name} table columns")
    arrays = []
    for column in columns:
        kind = _NESTED.get(column, pa.float64() if column in _FLOATS else
                           pa.int64() if column in _INTS else
                           pa.bool_() if column in _BOOLS else pa.string())
        values = table[column].tolist()
        if column == "labels":
            values = [None if value is None else sorted(value.items()) for value in values]
        elif kind == pa.string() and empty_text_as_null:
            values = [None if value == "" else value for value in values]
        arrays.append(pa.array(values, type=kind, from_pandas=True))
    return pa.Table.from_arrays(arrays, names=columns)


def _write_table(table, path):
    pq.write_table(table, path, **_PARQUET_OPTIONS)


def _link_or_copy(source, destination):
    """Share an immutable generation file; copy only across filesystems."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(paths, score_tables, root, verified=None):
    """Hash every input once; `verified` supplies digests this build already checked."""
    verified = verified or {}
    entries = []
    for path in sorted({Path(path).resolve() for path in paths}):
        label = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        digest = verified.get(path) or _hash_file(path)
        entries.append(dict(kind="file", path=label, sha256=digest))
    for index, table in enumerate(score_tables):
        rows = _arrow("scores", table).to_pylist()
        encoded = sorted(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                         for row in rows)
        payload = "[" + ",".join(encoded) + "]"
        entries.append(dict(kind="score_table", path=f"score_table:{index}",
                            sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest()))
    return sorted(entries, key=lambda entry: (entry["kind"], entry["path"]))


def _check_locations(locations, root, sources):
    """Output may sit below data_root, beside evidence, but cannot cover inputs."""
    for location in locations:
        for path in (location, location.resolve()):
            if root.is_relative_to(path):
                raise ValueError("Output overlaps data_root or contains it")
            for source in sources:
                if path.is_relative_to(source) or source.is_relative_to(path):
                    raise ValueError(f"Output overlaps source input: {source}")


def _check_output(out):
    if not out.is_symlink():
        if out.exists():
            raise ValueError("Output is not an owned managed zoo symlink")
        return
    target = Path(os.readlink(out))
    if (target.parent != Path(".") or re.fullmatch(
            re.escape(f".{out.name}.generation-") + r"[0-9a-f]{32}", target.name) is None):
        raise ValueError("Output is not an owned managed zoo symlink")
    generation = out.parent / target
    try:
        metadata = json.loads((generation / "zoo.json").read_text())
        if (generation.is_symlink() or metadata.get("publication_protocol") != _PROTOCOL
                or metadata.get("zoo_version") not in READABLE_ZOO_VERSIONS):
            raise ValueError("Unowned generation")
    except (OSError, ValueError) as error:
        raise ValueError("Output does not point to an owned complete generation") from error


def build_zoo(manifests: Path | None = None, fingerprints: list[Path] | None = None,
              out: Path | None = None, data_root: Path | None = None,
              frame: Path | None = None, score_tables: Sequence[pd.DataFrame] = (),
              lineage_edges: Path | None = None, null: Path | None = None,
              packages: Path | None = None, conventions_id: str | None = None,
              retain_scores: bool = True, score_sources: Path | None = None) -> Path:
    """Read evidence artifacts, build privately, and publish with one os.replace.

    No checkpoint weights are read. Pre-commit failure leaves the old pointer;
    a post-commit interrupt may leave the complete new snapshot published.
    Concurrent builds use private names; last successful publication wins.
    Old generations and legacy .previous paths are never automatically deleted.
    """
    if data_root is None:
        if not os.environ.get("ZOO_DATA"):
            raise ValueError("Set data_root or ZOO_DATA")
        data_root = Path(os.environ["ZOO_DATA"])
    root = Path(data_root).resolve()
    requested_out = Path(out) if out is not None else root / "builds/current"
    out = requested_out.absolute().parent.resolve() / requested_out.name
    selected_packages, skipped = [], 0
    verified_hashes = {}
    package_mode = packages is not None or (manifests is None and fingerprints is None)
    package_sources = []
    if package_mode:
        if manifests is not None or fingerprints is not None:
            raise ValueError("Choose packages or explicit manifests/fingerprints")
        packages = Path(packages or root / "packages").resolve()
        if not packages.is_dir():
            raise ValueError("Packages input must be an existing directory")
        conventions_id = conventions_id or current_conventions_id()
        selected_packages, skipped = package_inputs(packages, conventions_id)
        manifests, fingerprints = [], []
        for folder, package in selected_packages:
            # package_inputs has just checked every listed artifact against these digests.
            verified_hashes.update({(folder / name).resolve(): digest
                                    for name, digest in package["files"].items()})
            safe = safe_filename(package["model_key"])
            manifests.append(folder / f"{safe}.manifest.json")
            fingerprints.extend(folder / name for name in package["files"]
                                if is_baseline_fingerprint(name))
            package_sources.append(folder / "package.json")
            package_sources.extend(folder / name for name in package["files"])
        sources = [packages]
    else:
        if manifests is None or fingerprints is None:
            raise ValueError("Explicit manifests and fingerprints must be supplied together")
        manifests = Path(manifests).resolve()
        fingerprints = [Path(path).resolve() for path in fingerprints]
        sources = [manifests, *fingerprints]
    _check_locations([out], root, sources)
    _check_output(out)
    prior = out.resolve(strict=True) if retain_scores and out.is_symlink() else None
    prior_files = ([prior / "scores.parquet", prior / "models.parquet",
                    *sorted((prior / "score-sources").glob("*"))] if prior else [])
    prior_hashes = {path: _hash_file(path) for path in prior_files if path.is_file()}
    loaded_manifests = load_manifests(manifests)
    manifest_paths = [path for path, _ in loaded_manifests]
    models = pd.DataFrame([dict(model_key=r["repo_id"] + "@" + r["revision"],
        repo_id=r["repo_id"], revision=r["revision"]) for _, r in loaded_manifests],
        columns=["model_key", "repo_id", "revision"])
    card_paths = [path.with_name(path.name.replace(".manifest.json", ".source-metadata.json.gz"))
                  for path in manifest_paths]
    card_paths = [path for path in card_paths if path.is_file()]
    config = Path(score_sources).resolve() if score_sources is not None else root / "score-sources.json"
    if score_sources is not None and not config.is_file():
        raise ValueError("Explicit score source configuration is missing")
    final_scores, context_tables, score_coverage, score_optional = prepare_score_inputs(
        models, tuple(score_tables), prior=prior, config=config, card_paths=card_paths)
    score_tables = (final_scores,) if len(final_scores) or context_tables else ()
    score_inputs = score_source_paths(context_tables)
    optional = [Path(path).resolve() for path in (frame, lineage_edges, null) if path is not None]
    sources += optional + [path for path in score_inputs if prior is None or not path.is_relative_to(prior)] + score_optional
    token = uuid4().hex
    stage = out.with_name(f".{out.name}.building-{token}")
    generation = out.with_name(f".{out.name}.generation-{token}")
    pointer = out.with_name(f".{out.name}.publish-{token}")
    _check_locations([out, stage, generation, pointer], root, sources)
    _check_output(out)
    inputs = manifest_paths + list(_fingerprint_files(fingerprints)) + optional + package_sources + score_inputs + score_optional
    out.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    moved = False
    publication_attempted = False
    try:
        write_score_sources(stage, context_tables)
        (stage / "score-coverage.json").write_text(json.dumps(score_coverage, sort_keys=True, indent=2) + "\n")
        if config.exists():
            shutil.copyfile(config, stage / "score-configuration.json")
        tables = build_tables(manifests, fingerprints, root, frame=frame, score_tables=score_tables,
                              loaded_manifests=loaded_manifests)
        tables["layers"] = build_layers(selected_packages)
        for relative in tables["models"].full_weight_spectra:
            if relative:
                path = (root / relative).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Spectra input escapes data_root")
                inputs.append(path)
        _check_locations([out, stage, generation, pointer], root,
                         sources + [Path(path).resolve() for path in inputs
                                    if prior is None or not Path(path).resolve().is_relative_to(prior)])
        build_retrieval(stage, tables["models"], tables["views"], root)
        calibration = json.loads(Path(null).read_text()) if null is not None else None
        tables["lineage"] = merge_edges(
            declared_edges(tables["models"]),
            shared_tensor_edges(tables["views"], duplicate_groups(tables["views"])),
            similarity_edges(stage, tables["models"], root, null=calibration),
            external=lineage_edges)
        supplements = pair_delta_files(tables["models"], root)
        inputs.extend(path for path, _ in supplements)
        tables["lineage"] = with_pair_delta_evidence(tables["lineage"], supplements)
        for name in _COLUMNS:
            _write_table(_arrow(name, tables[name]), stage / f"{name}.parquet")
        metadata = dict(zoo_version=ZOO_VERSION, build_time=datetime.now(timezone.utc).isoformat(),
                        publication_protocol=_PROTOCOL, conventions_id=conventions_id,
                        skipped_conventions=skipped,
                        inputs=_provenance(inputs, score_tables, root, verified_hashes),
                        row_counts={name: len(tables[name]) for name in _COLUMNS})
        write_collection_receipt(stage, tables, root, manifest_paths=manifest_paths,
                                 provenance=metadata["inputs"], loaded_manifests=loaded_manifests)
        (stage / "zoo.json").write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n")
        check_score_sources(context_tables)
        config_pin = score_coverage.get("configuration")
        if config_pin and any(_hash_file(path) != config_pin["sha256"] for path in
                              [Path(config_pin["path"]), stage / "score-configuration.json"]):
            raise ValueError("Score source configuration changed during build")
        if any(_hash_file(path) != digest for path, digest in prior_hashes.items()):
            raise ValueError("Retained score generation changed during rebuild")
        stage.rename(generation)
        moved = True
        pointer.symlink_to(generation.name, target_is_directory=True)
        _check_output(out)
        publication_attempted = True
        os.replace(pointer, out)
    except BaseException:
        pointer.unlink(missing_ok=True)
        if not publication_attempted:
            shutil.rmtree(generation if moved else stage)
        raise
    return requested_out


def extend_scores(zoo: Path, score_tables: Sequence[pd.DataFrame], out: Path,
                  data_root: Path, source_tables: Sequence[pd.DataFrame] = ()) -> Path:
    """Append scores/source context without recomputing any inherited evidence.

    Context-only source tables must bind to an exact multiset subset of inherited
    scores; empty tables can retain explicitly unbound source records. Every
    inherited file other than scores.parquet/zoo.json/collection.json is shared
    byte-for-byte (hardlinked when possible, otherwise copied). The new
    generation keeps its parent's zoo version and table schema; existing
    revision-verification meanings are unchanged.
    """
    from .tables import _score_tables
    parent, root = Path(zoo).resolve(strict=True), Path(data_root).resolve(strict=True)
    requested_out = Path(out)
    out = requested_out.absolute().parent.resolve() / requested_out.name
    score_tables, source_tables = tuple(score_tables), tuple(source_tables)
    all_tables = score_tables + source_tables
    sources = [parent, *score_source_paths(all_tables)]
    token = uuid4().hex
    stage = out.with_name(f".{out.name}.building-{token}")
    generation = out.with_name(f".{out.name}.generation-{token}")
    pointer = out.with_name(f".{out.name}.publish-{token}")
    _check_locations([out, stage, generation, pointer], root, sources)
    _check_output(out)
    inventory = {p.relative_to(parent).as_posix(): _hash_file(p)
                 for p in sorted(parent.rglob("*")) if p.is_file()}
    metadata = json.loads((parent / "zoo.json").read_text())
    if metadata.get("zoo_version") not in READABLE_ZOO_VERSIONS:
        raise ValueError("Unsupported inherited zoo version")
    tables = open_tables(parent)
    manifests = dict.fromkeys(tables["models"].model_key)
    additions = _score_tables(score_tables, manifests)
    inherited_rows = _arrow("scores", tables["scores"]).to_pylist()
    inherited_counts = _canonical_scores(inherited_rows)
    for table in source_tables:
        _score_tables([table], manifests)
        counts = _canonical_scores(_arrow("scores", table).to_pylist())
        if counts - inherited_counts:
            raise ValueError("Context-only score bindings are absent from inherited scores")
    # Every retained name counts (descriptors and snapshots), so a new index can
    # never land on an inherited, hardlinked file.
    existing_indices = [int(p.name.split(".", 1)[0]) for p in (parent / "score-sources").glob("*")
                        if p.name.split(".", 1)[0].isdigit()]
    start = max(existing_indices, default=-1) + 1
    # A score extension keeps its parent's table conventions: version <=2
    # generations keep their inherited "" text exactly.
    empty_text_as_null = metadata.get("zoo_version") not in {"0", "1", "2"}
    out.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    moved, publication_attempted = False, False
    try:
        for name in inventory:
            if name in _SCORE_EXTENSION_REWRITES:
                continue  # Rewritten below; a hardlink would let that write reach the parent.
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(parent / name, destination)
        write_score_sources(stage, all_tables, start_index=start,
            roles=["appended_scores"] * len(score_tables) + ["inherited_context_only"] * len(source_tables))
        combined = pa.concat_tables([_arrow("scores", table, empty_text_as_null=empty_text_as_null)
                                     for table in (tables["scores"], additions)])
        _write_table(combined, stage / "scores.parquet")
        metadata["build_time"] = datetime.now(timezone.utc).isoformat()
        metadata["publication_protocol"] = _PROTOCOL
        metadata["row_counts"]["scores"] = len(combined)
        metadata["inherited_generation"] = dict(path=str(parent),
            zoo_json_sha256=inventory["zoo.json"], file_sha256=inventory,
            status="byte_inherited_except_scores_and_metadata")
        new_inputs = _provenance(score_source_paths(all_tables), score_tables, root)
        existing_table_ids = [int(r["path"].split(":", 1)[1]) for r in metadata["inputs"]
                              if r["kind"] == "score_table" and r["path"].startswith("score_table:")]
        first_table = max(existing_table_ids, default=-1) + 1
        for row in new_inputs:
            if row["kind"] == "score_table":
                row["path"] = "score_table:" + str(first_table + int(row["path"].split(":")[1]))
        metadata["inputs"] = sorted(metadata["inputs"] + new_inputs,
                                    key=lambda row: (row["kind"], row["path"]))
        tables["scores"] = combined.to_pandas()
        retained_manifests = [root / row["path"] for row in metadata["inputs"]
                              if row["kind"] == "file" and row["path"].endswith(".manifest.json")]
        write_collection_receipt(stage, tables, root, manifest_paths=retained_manifests,
                                 provenance=metadata["inputs"])
        (stage / "zoo.json").write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n")
        check_score_sources(all_tables)
        for name, digest in inventory.items():
            if _hash_file(parent / name) != digest:
                raise ValueError("Inherited generation changed during score extension")
            staged = stage / name
            if (name not in _SCORE_EXTENSION_REWRITES and not staged.samefile(parent / name)
                    and _hash_file(staged) != digest):
                raise ValueError("Inherited evidence bytes changed during score extension")
        stage.rename(generation)
        moved = True
        pointer.symlink_to(generation.name, target_is_directory=True)
        _check_output(out)
        publication_attempted = True
        os.replace(pointer, out)
    except BaseException:
        pointer.unlink(missing_ok=True)
        if not publication_attempted:
            shutil.rmtree(generation if moved else stage)
        raise
    return requested_out


def open_tables(zoo: Path) -> dict[str, pd.DataFrame]:
    """Resolve one generation once, preserving native list/dict user fields.

    Older generations are presented in the current columns: columns whose data
    now has a single home elsewhere are not read, and new columns are null.
    """
    generation = Path(zoo).resolve(strict=True)
    tables = {}
    for name, columns in _COLUMNS.items():
        path = generation / f"{name}.parquet"
        if name == "layers" and not path.is_file():
            tables[name] = empty_table(columns)
            continue
        selected = [column for column in pq.read_schema(path).names
                    if column not in _RETIRED_COLUMNS.get(name, ())]
        data = pq.read_table(path, columns=selected)
        if name not in ("models", "scores", "predictions"):  # These hold nested/nullable-int columns.
            # Flat tables: Arrow's columnar conversion, no per-row Python objects.
            table = data.to_pandas()
        else:
            records = data.to_pylist()
            if name == "models":
                for row in records:
                    if row["labels"] is not None:
                        row["labels"] = dict(row["labels"])
            table = pd.DataFrame(records, columns=selected)
            if name == "scores" and records:
                table["n_items"] = pd.array([row["n_items"] for row in records], dtype="Int64")
        if name == "models":
            table = normalize_model_table(table)
        elif name == "views":
            table = normalize_view_table(table)
        tables[name] = table.reindex(columns=columns)
    return tables
