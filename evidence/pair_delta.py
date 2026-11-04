"""Append-only row-block measurements for a pinned candidate pair.

These files supplement immutable package cores. They establish measured updates,
not parentage, direction, skill changes or calibrated probabilities.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from . import EVIDENCE_VERSION, READABLE_EVIDENCE_VERSIONS
from .fingerprint import (_Sources, _atomic_path, _no_delta, _parent_delta,
                          _plan_nonempty_views)
from .sketch import ROW_BLOCK, row_block_bounds
from measurement_config import safe_filename
from zoo.schema import model_key

PAIR_DELTA_VERSION = "1"


def pair_path(package, parent):
    return Path(package) / "deltas" / (safe_filename(model_key(parent["repo_id"], parent["revision"])) + ".delta.safetensors")


def _identity(manifest):
    model_key(manifest["repo_id"], manifest["revision"])
    return {key: manifest[key] for key in ("repo_id", "revision")}


def load_pair_delta(path):
    with safe_open(path, framework="numpy") as handle:
        raw = handle.metadata() or {}
        if raw.get("pair_delta_version") != PAIR_DELTA_VERSION or raw.get("evidence_version") not in READABLE_EVIDENCE_VERSIONS:
            raise ValueError("Unsupported pair-delta version")
        meta = {key: json.loads(value) if key not in ("pair_delta_version", "evidence_version") else value
                for key, value in raw.items()}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    n = len(meta["names"])
    for key, shape in (("rows", (n, 2)), ("delta_scalars", (n, 3)), ("delta_top_sv", (n, 16))):
        if tensors.get(key, np.empty(0)).shape != shape:
            raise ValueError("Invalid pair-delta tensors")
    if any(len(meta[key]) != n for key in ("row_blocks", "delta_status", "delta_confidence")):
        raise ValueError("Invalid pair-delta metadata")
    _identity(meta["child"])
    _identity(meta["parent"])
    return meta, tensors


def pair_deltas(child, parent, reader, package, *, block_cache=None, heights=(ROW_BLOCK,),
                compute_device="cpu", all_experts=False):
    heights = tuple(sorted(set(heights)))
    if not heights or any(type(height) is not int or height < 1 for height in heights):
        raise ValueError("Block heights must be positive integers")
    child_id, parent_id = _identity(child), _identity(parent)
    path = pair_path(package, parent)
    if path.exists():
        meta, _ = load_pair_delta(path)
        if (meta["child"], meta["parent"], meta["heights"], meta["all_experts"]) != (child_id, parent_id, list(heights), all_experts):
            raise FileExistsError("Existing pair delta has different identities or selection settings")
        return path
    names, rows, row_blocks, results = [], [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        sources, parents = _Sources(child, reader, pool), _Sources(parent, reader, pool)
        views, _ = _plan_nonempty_views(sources.entries, child.get("config", {}), child.get("adapter_config"), all_experts)
        planned, _ = _plan_nonempty_views(parents.entries, parent.get("config", {}), parent.get("adapter_config"), all_experts)
        parent_views = {view.name: view for view in planned}
        for height in heights:
            for view in views:
                bounds = row_block_bounds(view.matrix_shape[0], height)
                other = parent_views.get(view.name)
                if other is None:
                    delta = _no_delta()
                elif (row_block_bounds(other.matrix_shape[0], height) != bounds or
                      other.matrix_shape[1] != view.matrix_shape[1]):
                    delta = dict(_no_delta(), status="incompatible_shape")
                else:
                    block, _, _ = sources.block(view, bounds)
                    delta = _parent_delta(block, view, bounds, parents, parent_views, block_cache, height, compute_device)
                names.append(view.name)
                rows.append(bounds)
                row_blocks.append(height)
                results.append(delta)
    n = len(names)
    tensors = dict(rows=np.asarray(rows, dtype=np.int64).reshape(n, 2),
                   delta_scalars=np.asarray([[r[k] for k in ("frac_identical", "rel_update_norm", "rank")] for r in results], dtype=np.float32).reshape(n, 3),
                   delta_top_sv=np.asarray([r["top_singular_values"] for r in results], dtype=np.float32).reshape(n, 16))
    meta = dict(names=names, row_blocks=row_blocks, heights=list(heights), all_experts=all_experts,
                child=child_id, parent=parent_id, delta_status=[r["status"] for r in results],
                delta_confidence=[r["rank_confidence"] for r in results])
    meta = {key: json.dumps(value) for key, value in meta.items()}
    meta.update(evidence_version=EVIDENCE_VERSION, pair_delta_version=PAIR_DELTA_VERSION)
    temporary = _atomic_path(path)
    try:
        save_file(tensors, temporary, metadata=meta)
        try:
            # Link publishes atomically without overwriting an existing measurement.
            os.link(temporary, path)
        except FileExistsError:
            existing, _ = load_pair_delta(path)
            if (existing["child"], existing["parent"], existing["heights"], existing["all_experts"]) != (child_id, parent_id, list(heights), all_experts):
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return path


def for_fingerprints(child, parent):
    """Find a matching supplemental delta at the child fingerprint's package."""
    path = getattr(child, "path", None)
    if path is None:
        return {}
    candidate = pair_path(Path(path).parent, _identity(parent.meta))
    if not candidate.is_file():
        return {}
    meta, tensors = load_pair_delta(candidate)
    if meta["child"] != _identity(child.meta) or meta["parent"] != _identity(parent.meta):
        raise ValueError("Pair delta disagrees with requested fingerprint identities")
    height = child.meta.get("row_block")
    indices = {name: i for i, name in enumerate(child.names)}
    result = {}
    for j, name in enumerate(meta["names"]):
        i = indices.get(name)
        if i is not None and meta["row_blocks"][j] == height and np.array_equal(tensors["rows"][j], child.rows[i]):
            frac, norm, rank = tensors["delta_scalars"][j]
            result[name] = dict(frac_identical=float(frac), rel_update_norm=float(norm), rank=int(rank),
                                rank_confidence=meta["delta_confidence"][j], delta_status=meta["delta_status"][j])
    return result
