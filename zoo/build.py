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
from .lineage import declared_edges, merge_edges, shared_tensor_edges, similarity_edges, with_pair_delta_evidence
from .retrieval import build_retrieval
from .schema import (COVERAGE_COLUMNS, LINEAGE_COLUMNS, MODELS_COLUMNS,
                     PREDICTIONS_COLUMNS, SCORES_COLUMNS, VIEWS_COLUMNS, LAYERS_COLUMNS, empty_table)
from .tables import _fingerprint_files, _manifest_files, build_tables, duplicate_groups, pair_delta_files
from .publish import package_inputs, current_conventions_id
from .layers import build_layers
from measurement_config import safe_filename


_COLUMNS = dict(models=MODELS_COLUMNS, views=VIEWS_COLUMNS, coverage=COVERAGE_COLUMNS,
                lineage=LINEAGE_COLUMNS, scores=SCORES_COLUMNS, predictions=PREDICTIONS_COLUMNS, layers=LAYERS_COLUMNS)
_PROTOCOL = "immutable-generation-v1"
_FLOATS = {"score", "p_value", "value", "lower", "upper", "inclusion_prob", "downloads", "params"}
_LAYER_FLOATS = set(LAYERS_COLUMNS) - {"model_key", "longname", "module_name", "weight_attribute", "slice", "view_name", "fit_status", "h5_index", "num_evals", "matrix_rank", "n_tail"}
_FLOATS |= _LAYER_FLOATS
_INTS = {"view_index", "n_rows", "n_cols", "r0", "r1", "n_views", "num_labels", "n_items", "h5_index", "num_evals", "matrix_rank", "n_tail"}
_BOOLS = {"is_adapter", "gated", "is_decision", "system_one", "calibrated", "revision_verified"}
_NESTED = {
    "labels": pa.map_(pa.string(), pa.string()),
    "declared_bases": pa.list_(pa.string()),
    "skipped_tensors": pa.list_(pa.struct([("name", pa.string()), ("reason", pa.string())])),
    "inputs": pa.list_(pa.string()),  # v0 predictions are always empty; no future format claim.
}


def _arrow(name, table):
    """Explicit types also support all-empty maps, lists and whole tables."""
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
        arrays.append(pa.array(values, type=kind, from_pandas=True))
    return pa.Table.from_arrays(arrays, names=columns)


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(paths, score_tables, root):
    entries = []
    for path in sorted({Path(path).resolve() for path in paths}):
        label = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        entries.append(dict(kind="file", path=label, sha256=_hash_file(path)))
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
              packages: Path | None = None, conventions_id: str | None = None) -> Path:
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
            safe = safe_filename(package["model_key"])
            manifests.append(folder / f"{safe}.manifest.json")
            fingerprints.extend(folder / name for name in package["files"]
                                if name.endswith((".t1.fingerprint.safetensors", ".t2.fingerprint.safetensors")))
            package_sources.append(folder / "package.json")
            package_sources.extend(folder / name for name in package["files"])
        sources = [packages]
    else:
        if manifests is None or fingerprints is None:
            raise ValueError("Explicit manifests and fingerprints must be supplied together")
        manifests = Path(manifests).resolve()
        fingerprints = [Path(path).resolve() for path in fingerprints]
        sources = [manifests, *fingerprints]
    score_tables = tuple(score_tables)
    optional = [Path(path).resolve() for path in (frame, lineage_edges, null) if path is not None]
    sources += optional
    token = uuid4().hex
    stage = out.with_name(f".{out.name}.building-{token}")
    generation = out.with_name(f".{out.name}.generation-{token}")
    pointer = out.with_name(f".{out.name}.publish-{token}")
    _check_locations([out, stage, generation, pointer], root, sources)
    _check_output(out)
    inputs = list(_manifest_files(manifests)) + list(_fingerprint_files(fingerprints)) + optional + package_sources
    out.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    moved = False
    publication_attempted = False
    try:
        tables = build_tables(manifests, fingerprints, root, frame=frame, score_tables=score_tables)
        tables["layers"] = build_layers(selected_packages)
        for relative in tables["models"].spectra_t2:
            if relative:
                path = (root / relative).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Spectra input escapes data_root")
                inputs.append(path)
        _check_locations([out, stage, generation, pointer], root,
                         sources + [Path(path).resolve() for path in inputs])
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
            pq.write_table(_arrow(name, tables[name]), stage / f"{name}.parquet")
        metadata = dict(zoo_version=ZOO_VERSION, build_time=datetime.now(timezone.utc).isoformat(),
                        publication_protocol=_PROTOCOL, conventions_id=conventions_id,
                        skipped_conventions=skipped, inputs=_provenance(inputs, score_tables, root),
                        row_counts={name: len(tables[name]) for name in _COLUMNS})
        (stage / "zoo.json").write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n")
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
    """Resolve one generation once, preserving native list/dict user fields."""
    generation = Path(zoo).resolve(strict=True)
    tables = {}
    for name, columns in _COLUMNS.items():
        if name == "layers" and not (generation / "layers.parquet").is_file():
            tables[name] = empty_table(columns)
            continue
        records = pq.read_table(generation / f"{name}.parquet").to_pylist()
        if name == "models":
            for row in records:
                if row["labels"] is not None:
                    row["labels"] = dict(row["labels"])
        tables[name] = pd.DataFrame(records, columns=columns)
    return tables
