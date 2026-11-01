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

from . import ZOO_VERSION
from .lineage import declared_edges, merge_edges, shared_tensor_edges, similarity_edges
from .retrieval import build_retrieval
from .schema import (COVERAGE_COLUMNS, LINEAGE_COLUMNS, MODELS_COLUMNS,
                     PREDICTIONS_COLUMNS, SCORES_COLUMNS, VIEWS_COLUMNS)
from .tables import _fingerprint_files, build_tables, duplicate_groups


_COLUMNS = dict(models=MODELS_COLUMNS, views=VIEWS_COLUMNS, coverage=COVERAGE_COLUMNS,
                lineage=LINEAGE_COLUMNS, scores=SCORES_COLUMNS, predictions=PREDICTIONS_COLUMNS)
_PROTOCOL = "immutable-generation-v1"
_FLOATS = {"score", "p_value", "value", "lower", "upper", "inclusion_prob", "downloads", "params"}
_INTS = {"view_index", "n_rows", "n_cols", "r0", "r1", "n_views", "num_labels", "n_items"}
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
                or metadata.get("zoo_version") != ZOO_VERSION):
            raise ValueError("Unowned generation")
    except (OSError, ValueError) as error:
        raise ValueError("Output does not point to an owned complete generation") from error


def build_zoo(manifests: Path, fingerprints: list[Path], out: Path, data_root: Path,
              frame: Path | None = None, score_tables: Sequence[pd.DataFrame] = (),
              lineage_edges: Path | None = None, null: Path | None = None) -> Path:
    """Read evidence artifacts, build privately, and publish with one os.replace.

    No checkpoint weights are read. Pre-commit failure leaves the old pointer;
    a post-commit interrupt may leave the complete new snapshot published.
    Concurrent builds use private names; last successful publication wins.
    Old generations and legacy .previous paths are never automatically deleted.
    """
    requested_out = Path(out)
    out = requested_out.absolute().parent.resolve() / requested_out.name
    root = Path(data_root).resolve()
    manifests = Path(manifests).resolve()
    fingerprints = [Path(path).resolve() for path in fingerprints]
    score_tables = tuple(score_tables)
    optional = [Path(path).resolve() for path in (frame, lineage_edges, null) if path is not None]
    sources = [manifests, *fingerprints, *optional]
    token = uuid4().hex
    stage = out.with_name(f".{out.name}.building-{token}")
    generation = out.with_name(f".{out.name}.generation-{token}")
    pointer = out.with_name(f".{out.name}.publish-{token}")
    _check_locations([out, stage, generation, pointer], root, sources)
    _check_output(out)
    inputs = list(manifests.glob("*.manifest.json")) + list(_fingerprint_files(fingerprints)) + optional
    out.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    moved = False
    publication_attempted = False
    try:
        tables = build_tables(manifests, fingerprints, root, frame=frame, score_tables=score_tables)
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
        for name in _COLUMNS:
            pq.write_table(_arrow(name, tables[name]), stage / f"{name}.parquet")
        metadata = dict(zoo_version=ZOO_VERSION, build_time=datetime.now(timezone.utc).isoformat(),
                        publication_protocol=_PROTOCOL, inputs=_provenance(inputs, score_tables, root),
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
        records = pq.read_table(generation / f"{name}.parquet").to_pylist()
        if name == "models":
            for row in records:
                if row["labels"] is not None:
                    row["labels"] = dict(row["labels"])
        tables[name] = pd.DataFrame(records, columns=columns)
    return tables
