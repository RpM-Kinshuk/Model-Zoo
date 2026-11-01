"""Measured checkpoint differences and the shared calibration statistic."""

from collections import Counter, defaultdict
import math

import numpy as np

from .fingerprint import Fingerprint
from .sketch import SKETCH_SEED, sketch_cosine


def _settings_reason(a, b):
    for key in ("tier", "row_block", "sketch_seed"):
        if a.meta.get(key) != b.meta.get(key) or key not in a.meta:
            return f"incompatible {key}"
    return None


def _view_reason(a, i, b, j):
    if a.geometry[i] != b.geometry[j]:
        return "incompatible geometry"
    if not np.array_equal(a.rows[i], b.rows[j]):
        return "incompatible row coordinates"
    if a.shape[i, 1] != b.shape[j, 1]:
        return "incompatible column dimensions"
    return None


def _scores(a, b, k):
    right = {name: j for j, name in enumerate(b.names)}
    return {name: sketch_cosine(a.sketch[i, :k, :k], b.sketch[right[name], :k, :k])
            for i, name in enumerate(a.names) if name in right
            and _view_reason(a, i, b, right[name]) is None}


def pair_similarity(a: Fingerprint, b: Fingerprint, sketch_k: int | None = None) -> dict:
    """Median cosine on compatible shared views, using nested sketch prefixes.

    T1 vocabulary growth can preserve row coordinates and column dimensions;
    its shared block is comparable even though the full matrix was reshaped.
    """
    available = min(a.meta["sketch_k"], b.meta["sketch_k"])
    k = available if sketch_k is None else sketch_k
    if type(k) is not int or not 1 <= k <= available:
        raise ValueError("Requested sketch size is unavailable")
    reason = _settings_reason(a, b)
    scores = {} if reason else _scores(a, b, k)
    result = {"median_cosine": float(np.median(list(scores.values()))) if scores else None,
              "n_shared": len(scores)}
    if reason:
        result["reason"] = reason
    return result


def _relatedness(a, b, null):
    result = {"verdict": "uncalibrated", **pair_similarity(a, b)}
    if null is None:
        return result
    try:
        k, height, n = null["sketch_k"], null["row_block"], null["n_pairs"]
        quantiles = np.asarray(null["null_quantiles"], dtype=float)
        threshold = float(null["related_threshold"])
        if (null["statistic"] != "median_cosine" or type(k) is not int or k < 1
                or type(height) is not int or height < 1 or type(n) is not int or n < 1
                or quantiles.ndim != 1 or len(quantiles) == 0
                or not np.isfinite(quantiles).all() or (np.diff(quantiles) < 0).any()
                or (np.abs(quantiles) > 1).any() or not math.isfinite(threshold)
                or not -1 <= threshold <= 1 or null["tier"] not in ("t1", "t2")
                and not str(null["tier"]).startswith("t1b")):
            raise ValueError("invalid calibration")
    except (KeyError, ValueError, TypeError, OverflowError):
        result["reason"] = "invalid calibration file"
        return result
    for fp in (a, b):
        if fp.meta["sketch_k"] < k:
            result["reason"] = "fingerprint sketch_k below calibration sketch_k"
            return result
        if fp.meta.get("tier") != null["tier"] or fp.meta.get("row_block") != height:
            result["reason"] = "calibration tier or row_block mismatch"
            return result
        if fp.meta.get("sketch_seed") != null.get("sketch_seed", SKETCH_SEED):
            result["reason"] = "calibration sketch_seed mismatch"
            return result
    right = {name: j for j, name in enumerate(b.names)}
    for i, name in enumerate(a.names):
        if name in right and (reason := _view_reason(a, i, b, right[name])):
            result["reason"] = reason
            return result
    result.update(pair_similarity(a, b, k))
    score = result["median_cosine"]
    if score is None:
        result["reason"] = "no compatible shared views"
        return result
    # Quantiles approximate the empirical tail; n_pairs sets the smoothing and
    # floor, independently of the number of stored quantiles. Include ties.
    tail = float(np.mean(quantiles >= score))
    result["p_value"] = (1 + n * tail) / (n + 1)
    result["sketch_k"] = k
    # v0 rule: the null's upper envelope and related_threshold delimit a gap.
    # Overlap or a score in that gap is undetermined; no error-control claim.
    if score >= threshold and score > quantiles[-1]:
        result["verdict"] = "related"
    elif score < threshold and score <= quantiles[-1]:
        result["verdict"] = "unrelated"
    else:
        result["verdict"] = "undetermined"
    result.pop("reason", None)
    return result


def _identity(fp):
    return {key: fp.meta.get(key) for key in ("repo_id", "revision")}


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def diff_fingerprints(a: Fingerprint, b: Fingerprint, null: dict | None = None) -> dict:
    """Report block hashes, geometry, sketches, and recorded parent deltas."""
    left, right = ({name: i for i, name in enumerate(fp.names)} for fp in (a, b))
    compatible = _settings_reason(a, b) is None
    k = min(a.meta["sketch_k"], b.meta["sketch_k"])
    scores = _scores(a, b, k) if compatible else {}
    views, counts, updates, heads = {}, Counter(), defaultdict(list), []
    for name in sorted(left.keys() | right.keys()):
        i, j = left.get(name), right.get(name)
        if i is None:
            status = "removed"
        elif j is None:
            status = "added"
        elif not np.array_equal(a.shape[i], b.shape[j]):
            status = "reshaped"
        elif compatible and _view_reason(a, i, b, j) is None and a.sha256[i] == b.sha256[j]:
            status = "identical"
        else:
            status = "changed"
        record = {"status": status, "cosine": scores.get(name)}
        if i is not None and j is not None:
            reason = _settings_reason(a, b) or _view_reason(a, i, b, j)
            if reason:
                record["reason"] = reason
            child, index = (a, i) if a.meta.get("parent") == _identity(b) else (
                (b, j) if b.meta.get("parent") == _identity(a) else (None, None))
            if child is not None and child.delta["status"][index] != "no_parent":
                record.update({key: _finite(child.delta[key][index])
                               for key in ("frac_identical", "rel_update_norm")})
                rank = int(child.delta["rank"][index])
                record.update(rank=rank if rank >= 0 else None,
                              rank_confidence=str(child.delta["confidence"][index]),
                              delta_status=str(child.delta["status"][index]))
                if record["rel_update_norm"] is not None:
                    updates[child.component[index]].append(record["rel_update_norm"])
        fp, index = (a, i) if i is not None else (b, j)
        if fp.geometry[index] == "head" and status in ("added", "changed", "reshaped"):
            heads.append(name)
        views[name] = record
        counts[status] += 1
    similarity = pair_similarity(a, b)
    return {"a": _identity(a), "b": _identity(b), "views": views,
            "summary": {"counts": {key: counts[key] for key in ("identical", "changed", "added", "removed", "reshaped")},
                        "rel_update_norm": {key: float(np.median(values)) for key, values in sorted(updates.items())},
                        **similarity, "heads": heads},
            "relatedness": _relatedness(a, b, null)}


def check_verdict(diff: dict) -> str:
    return {"related": "consistent", "unrelated": "inconsistent",
            "undetermined": "undetermined", "uncalibrated": "uncalibrated"}[diff["relatedness"]["verdict"]]
