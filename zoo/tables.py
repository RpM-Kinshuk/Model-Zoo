"""Pure assembly of pinned models, measured views, coverage and scores."""

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from evidence import EVIDENCE_VERSION
from evidence.fingerprint import load_fingerprint
from .schema import (COVERAGE_COLUMNS, MODELS_COLUMNS, PREDICTIONS_COLUMNS,
                     SCORES_COLUMNS, VIEWS_COLUMNS, arch_signature, empty_table,
                     model_key, split_key)


_BENIGN_SKIPS = {"not_a_matrix", "non_float_buffer", "vector_like", "expert_not_sampled"}
_FRAME_FIELDS = ("modality", "pipeline_tag", "downloads", "params", "stratum", "inclusion_prob")
_NUMERIC_FRAME_FIELDS = {"downloads", "params", "inclusion_prob"}


def _relative(path, root):
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Artifact outside data_root: {path}") from error


def _fingerprint_files(inputs):
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            yield from sorted(path.glob("*.t[12].fingerprint.safetensors"))
        elif path.is_file() and path.name.endswith((".t1.fingerprint.safetensors", ".t2.fingerprint.safetensors")):
            yield path
        else:
            raise ValueError(f"Expected a fingerprint directory or T1/T2 fingerprint file: {path}")


def _rows(records, columns, sort):
    if not records:
        return empty_table(columns)
    return pd.DataFrame(records, columns=columns).sort_values(sort, kind="stable").reset_index(drop=True)


def duplicate_groups(views: pd.DataFrame) -> dict[str, str]:
    """Group identical compatible T1 blocks, never claim full-file identity."""
    if views.empty:
        return {}
    result = {key: key for key in sorted(views.model_key.unique())}
    signatures = defaultdict(list)
    first = views[views.tier == "t1"]
    for key, records in first.groupby("model_key", sort=True):
        if records.name.duplicated().any():
            raise ValueError(f"Duplicate T1 view names for {key}")
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

    Fingerprint inputs are directories or canonical T1/T2 files. Paths stored
    in the tables are resolved relative to data_root, including T2's complete
    HDF5 spectrum. Extra T1 block-height outputs are not database baselines.
    """
    root = Path(data_root).resolve()
    records, manifest_paths = {}, {}
    for path in sorted(Path(manifests).glob("*.manifest.json")):
        manifest = json.loads(path.read_text())
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
        if tier not in ("t1", "t2") or fp.meta.get("evidence_version") != EVIDENCE_VERSION:
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
        chosen = pair.get("t1", pair.get("t2"))
        fp = chosen[2] if chosen else None
        coverage = _coverage(manifest, fp)
        skipped = [dict(name=r["name"], reason=r.get("reason", ""))
                   for r in coverage if r.get("status") == "skipped"]
        access = manifest["access"]
        if access not in ("granted", "no_access", "t0_only"):
            raise ValueError(f"Unknown manifest access state: {access}")
        status = ("partial" if any(r["reason"] not in _BENIGN_SKIPS for r in skipped) else "measured") if fp else (
            access if access in ("t0_only", "no_access") else "not_measured")
        spectra = ""
        if "t2" in pair:
            path = pair["t2"][0]
            spectra = _relative(path.with_name(path.name.removesuffix(".t2.fingerprint.safetensors") + ".t2.spectra.h5"), root)
        config, decision = manifest.get("config", {}), manifest.get("decision", {})
        shapes = [tuple(shape) for shape in fp.shape] if fp else []
        model_rows.append(dict(model_key=key, repo_id=manifest["repo_id"], revision=manifest["revision"],
            access=access, weight_format=manifest.get("weight_format"), is_adapter=manifest.get("is_adapter", False),
            license=manifest.get("license"), gated=manifest.get("gated", False), declared_bases=manifest.get("declared_bases", []),
            base_model_relation=manifest.get("base_model_relation"), architecture=config.get("model_type") or manifest.get("gguf", {}).get("architecture"),
            **{field: decision.get(field) for field in ("is_decision", "system_one", "decision_type", "num_labels", "labels")},
            arch_signature=arch_signature(config, shapes), n_views=len(fp.names) if fp else 0,
            **_frame_values(frame_rows, manifest["repo_id"], manifest["revision"]),
            dup_group=key, fingerprint_t1=pair["t1"][1] if "t1" in pair else "",
            fingerprint_t2=pair["t2"][1] if "t2" in pair else "", spectra_t2=spectra,
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
