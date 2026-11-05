"""Pure assembly of pinned models, measured views, coverage and scores."""

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from evidence import EVIDENCE_VERSION, READABLE_EVIDENCE_VERSIONS
from evidence.fingerprint import load_fingerprint
from evidence.compatibility import companion_spectra_path, is_baseline_fingerprint, normalize_manifest
from .schema import (COVERAGE_COLUMNS, MODELS_COLUMNS, PREDICTIONS_COLUMNS,
                     SCORES_COLUMNS, VIEWS_COLUMNS, arch_signature, empty_table,
                     model_key, split_key)


_BENIGN_SKIPS = {"not_a_matrix", "non_float_buffer", "vector_like", "expert_not_sampled"}
_FRAME_FIELDS = ("modality", "pipeline_tag", "downloads", "params", "stratum", "inclusion_prob")
_NUMERIC_FRAME_FIELDS = {"downloads", "params", "inclusion_prob"}


def _gated(value):
    """Normalize Hub gating metadata for the nullable boolean database field."""
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str) and value in ("auto", "manual"):
        return True
    raise ValueError(f"Unknown manifest gated value: {value!r}")


def _relative(path, root):
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Artifact outside data_root: {path}") from error


def _manifest_files(inputs):
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        return sorted(path.glob("*.manifest.json")) if path.is_dir() else [path]
    return sorted(Path(path) for path in inputs)


def _fingerprint_files(inputs):
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            if (path / "package.json").is_file():
                metadata = json.loads((path / "package.json").read_text())
                yield from sorted(path / name for name in metadata["files"]
                                  if is_baseline_fingerprint(name))
            else:
                yield from sorted(item for item in path.glob("*.fingerprint.safetensors") if is_baseline_fingerprint(item))
        elif path.is_file() and is_baseline_fingerprint(path):
            yield path
        else:
            raise ValueError(f"Expected a fingerprint directory or row-block/full-weight fingerprint file: {path}")


def _rows(records, columns, sort):
    if not records:
        return empty_table(columns)
    return pd.DataFrame(records, columns=columns).sort_values(sort, kind="stable").reset_index(drop=True)


def duplicate_groups(views: pd.DataFrame) -> dict[str, str]:
    """Group identical compatible row-block blocks, never claim full-file identity."""
    if views.empty:
        return {}
    result = {key: key for key in sorted(views.model_key.unique())}
    signatures = defaultdict(list)
    first = views[views.tier == "row_blocks"]
    for key, records in first.groupby("model_key", sort=True):
        if records.name.duplicated().any():
            raise ValueError(f"Duplicate row-block view names for {key}")
        signature = tuple(sorted((r.name, r.geometry, int(r.n_rows), int(r.n_cols),
                                  int(r.r0), int(r.r1), r.block_sha256)
                                 for r in records.itertuples()))
        signatures[signature].append(key)
    for keys in signatures.values():
        representative = min(keys)
        result.update({key: representative for key in keys})
    return result


def _frame_index(frame):
    if frame is None:
        return None
    rows = pd.read_csv(frame, dtype=str, keep_default_na=False)
    if "repo_id" not in rows:
        raise ValueError("Selection frame requires repo_id")
    if "revision" not in rows:
        rows["revision"] = ""
    if rows.duplicated(["repo_id", "revision"]).any():
        raise ValueError("Duplicate repository/revision in selection frame")
    return {(row["repo_id"], row["revision"]): row for row in rows.to_dict("records")}


def _frame_values(frame, repo_id, revision):
    match = None if frame is None else frame.get((repo_id, revision), frame.get((repo_id, "")))
    values = {name: None for name in _FRAME_FIELDS}
    values["stratum"] = "lab"
    if match is not None:
        for name in _FRAME_FIELDS:
            value = match.get(name)
            if value is None or value == "":
                continue
            if name in _NUMERIC_FRAME_FIELDS:
                try:
                    value = float(value)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid selection frame {name}: {value}") from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid selection frame {name}: {value}")
            values[name] = value
    return values


def _coverage(manifest, fp):
    return fp.coverage if fp is not None else [dict(name=t["name"], decode=t.get("decode", "plain"),
                geometry=t.get("geometry", ""), status=t.get("status", "skipped"),
                reason=t.get("reason", "")) for t in manifest.get("tensors", [])]


def _score_tables(tables, manifests):
    if not tables:
        return empty_table(SCORES_COLUMNS)
    combined = []
    for table in tables:
        if set(table) != set(SCORES_COLUMNS):
            raise ValueError("Score table does not match SCORES_COLUMNS")
        for row in table.to_dict("records"):
            key = row["model_key"]
            if key not in manifests:
                raise ValueError(f"Score model has no manifest: {key}")
            repo_id, revision = split_key(key)
            if row["repo_id"] != repo_id:
                raise ValueError(f"Score repository disagrees with pinned model: {key}")
            if row["revision_verified"] and row["source_revision"] != revision:
                raise ValueError(f"Verified score has a mismatched source revision: {key}")
            combined.append(row)
    return _rows(combined, SCORES_COLUMNS, ["model_key", "benchmark", "source", "source_revision", "harness"])


def build_tables(manifests: Path, fingerprints: list[Path], data_root: Path,
                 frame: Path | None = None,
                 score_tables: Sequence[pd.DataFrame] = ()) -> dict[str, pd.DataFrame]:
    """Read extractor outputs without writing or modifying any input artifacts.

    Fingerprint inputs are directories or canonical row-block/full-weight files. Paths stored
    in the tables are resolved relative to data_root, including full-weight's complete
    HDF5 spectrum. Extra row-block block-height outputs are not database baselines.
    """
    root = Path(data_root).resolve()
    records, manifest_paths = {}, {}
    for path in _manifest_files(manifests):
        manifest = normalize_manifest(json.loads(path.read_text()))
        key = model_key(manifest["repo_id"], manifest["revision"])
        if key in records:
            raise ValueError(f"Duplicate manifest identity {key}: {manifest_paths[key]} and {path}")
        records[key], manifest_paths[key] = manifest, path
    measured = defaultdict(dict)
    for path in _fingerprint_files(fingerprints):
        relative = _relative(path, root)
        fp = load_fingerprint(path)
        key = model_key(fp.meta["repo_id"], fp.meta["revision"])
        tier = fp.meta["tier"]
        if tier not in ("row_blocks", "full_weights") or fp.meta.get("evidence_version") not in READABLE_EVIDENCE_VERSIONS:
            raise ValueError(f"Unsupported fingerprint settings: {path}")
        if key not in records:
            raise ValueError(f"Fingerprint identity has no manifest: {key}: {path}")
        if tier in measured[key]:
            raise ValueError(f"Duplicate {tier} fingerprints for {key}: {measured[key][tier][0]} and {path}")
        measured[key][tier] = (path, relative, fp)
    frame_rows = _frame_index(frame)
    model_rows, view_rows, coverage_rows = [], [], []
    for key, manifest in sorted(records.items()):
        pair = measured.get(key, {})
        chosen = pair.get("row_blocks", pair.get("full_weights"))
        fp = chosen[2] if chosen else None
        coverage = _coverage(manifest, fp)
        skipped = [dict(name=r["name"], reason=r.get("reason", ""))
                   for r in coverage if r.get("status") == "skipped"]
        access = manifest["access"]
        if access not in ("granted", "no_access", "metadata_only"):
            raise ValueError(f"Unknown manifest access state: {access}")
        status = ("partial" if any(r["reason"] not in _BENIGN_SKIPS for r in skipped) else "measured") if fp else (
            access if access in ("metadata_only", "no_access") else "not_measured")
        spectra = ""
        if "full_weights" in pair:
            path = pair["full_weights"][0]
            spectrum_path = companion_spectra_path(path)
            package_path = path.parent / "package.json"
            if package_path.is_file():
                package = json.loads(package_path.read_text())
                if package.get("kind") == "full" and "esd/metrics.h5" in package.get("files", {}):
                    spectrum_path = path.parent / "esd/metrics.h5"
            spectra = _relative(spectrum_path, root)
        config, decision = manifest.get("config", {}), manifest.get("decision", {})
        shapes = [tuple(shape) for shape in fp.shape] if fp else []
        model_rows.append(dict(model_key=key, repo_id=manifest["repo_id"], revision=manifest["revision"],
            access=access, weight_format=manifest.get("weight_format"), is_adapter=manifest.get("is_adapter", False),
            license=manifest.get("license"), gated=_gated(manifest.get("gated", False)), declared_bases=manifest.get("declared_bases", []),
            base_model_relation=manifest.get("base_model_relation"), architecture=config.get("model_type") or manifest.get("gguf", {}).get("architecture"),
            **{field: decision.get(field) for field in ("is_decision", "system_one", "decision_type", "num_labels", "labels")},
            arch_signature=arch_signature(config, shapes), n_views=len(fp.names) if fp else 0,
            **_frame_values(frame_rows, manifest["repo_id"], manifest["revision"]),
            dup_group=key, row_block_fingerprint=pair["row_blocks"][1] if "row_blocks" in pair else "",
            full_weight_fingerprint=pair["full_weights"][1] if "full_weights" in pair else "", full_weight_spectra=spectra,
            coverage_status=status, skipped_tensors=skipped))
        if fp is not None:
            for index, name in enumerate(fp.names):
                view_rows.append(dict(model_key=key, view_index=index, name=name,
                    geometry=fp.geometry[index], component=fp.component[index],
                    n_rows=int(fp.shape[index, 0]), n_cols=int(fp.shape[index, 1]),
                    r0=int(fp.rows[index, 0]), r1=int(fp.rows[index, 1]),
                    block_sha256=fp.sha256[index], tier=fp.meta["tier"]))
        coverage_rows.extend(dict(model_key=key, tensor_name=row["name"],
            **{field: row.get(field, "") for field in ("decode", "geometry", "status", "reason")})
            for row in coverage)
    views = _rows(view_rows, VIEWS_COLUMNS, ["model_key", "view_index"])
    groups = duplicate_groups(views)
    for row in model_rows:
        row["dup_group"] = groups.get(row["model_key"], row["model_key"])
    return dict(models=_rows(model_rows, MODELS_COLUMNS, ["model_key"]), views=views,
                coverage=_rows(coverage_rows, COVERAGE_COLUMNS, ["model_key", "tensor_name"]),
                scores=_score_tables(score_tables, records), predictions=empty_table(PREDICTIONS_COLUMNS))


def pair_delta_files(models, data_root):
    """Validate supplemental identities without changing frozen view columns."""
    from evidence.pair_delta import load_pair_delta
    root = Path(data_root).resolve()
    result = []
    for row in models.itertuples():
        if not row.row_block_fingerprint:
            continue
        folder = (root / row.row_block_fingerprint).resolve().parent
        if not folder.is_relative_to(root):
            raise ValueError("Pair delta input escapes data_root")
        for path in sorted((folder / "deltas").glob("*.delta.safetensors")):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Pair delta escapes data_root")
            meta, _ = load_pair_delta(path)
            if model_key(**meta["child"]) != row.model_key:
                raise ValueError("Pair delta child disagrees with its package")
            result.append((path, meta))
    return result
