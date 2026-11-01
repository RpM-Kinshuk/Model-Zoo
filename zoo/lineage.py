"""Declared and measured v0 lineage, with optional P1 evidence appended."""

from dataclasses import replace
from itertools import combinations
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evidence.compare import _relatedness, _view_reason, pair_similarity
from .retrieval import _fingerprint, _paths, nearest
from .schema import LINEAGE_COLUMNS, empty_table, split_key


_EDGE_TYPES = {"finetune", "adapter", "merge", "quantized", "abliteration", "unknown"}
_RELATIONS = {"lora": "adapter", "sft": "finetune", "dpo": "finetune",
              "quantization_sim": "quantized"}
_KEYS = ["child_key", "parent_key", "estimand", "evidence"]


def _edge(child, parent, estimand, tier, evidence, score=np.nan,
          edge_type="unknown", p_value=np.nan, calibrated=False):
    return dict(child_key=child, parent_key=parent, estimand=estimand, edge_type=edge_type,
                evidence_tier=tier, evidence=evidence, score=score, p_value=p_value,
                calibrated=calibrated, method_version="zoo-v0")


def _table(rows):
    if not rows:
        return empty_table(LINEAGE_COLUMNS)
    return (pd.DataFrame(rows, columns=LINEAGE_COLUMNS).drop_duplicates(_KEYS)
            .sort_values(_KEYS, kind="stable").reset_index(drop=True))


def declared_edges(models: pd.DataFrame) -> pd.DataFrame:
    """Keep explicit parent pins; resolve bare repositories only when unique."""
    repositories = models.groupby("repo_id").model_key.agg(list).to_dict()
    rows = []
    for model in models.sort_values("model_key").itertuples():
        relation = _RELATIONS.get(model.base_model_relation, model.base_model_relation)
        edge_type = relation if relation in _EDGE_TYPES else "unknown"
        for base in model.declared_bases:
            try:
                split_key(base)
                parent = base
            except ValueError:
                matches = repositories.get(base, [])
                parent = matches[0] if len(matches) == 1 else base + "@external"
            rows.append(_edge(model.model_key, parent, "parent", "declared",
                              "declared_metadata", edge_type=edge_type))
    return _table(rows)


def shared_tensor_edges(views: pd.DataFrame, dup_groups: dict) -> pd.DataFrame:
    """Count identical hashes only among aligned T1 blocks, excluding duplicates."""
    first = views[views.tier == "t1"]
    groups = {}
    for key, group in first.groupby("model_key", sort=True):
        if group.name.duplicated().any():
            raise ValueError(f"Duplicate T1 view names: {key}")
        groups[key] = {row.name: row for row in group.itertuples()}
    rows = []
    for child, parent in combinations(sorted(groups), 2):
        if dup_groups.get(child, child) == dup_groups.get(parent, parent):
            continue
        left, right = groups[child], groups[parent]
        shared, identical = 0, 0
        for name in left.keys() & right.keys():
            a, b = left[name], right[name]
            if (a.geometry, a.r0, a.r1, a.n_cols) != (b.geometry, b.r0, b.r1, b.n_cols):
                continue
            shared += 1
            identical += bool(a.block_sha256 and a.block_sha256 == b.block_sha256)
        if 0 < identical < shared:
            rows.append(_edge(child, parent, "same_root", "verified", "shared_tensors",
                              score=identical / shared))
    return _table(rows)


def _subset(fp, indices):
    """A consistent view subset, leaving source artifacts untouched."""
    return replace(fp, names=[fp.names[i] for i in indices],
                   geometry=[fp.geometry[i] for i in indices],
                   component=[fp.component[i] for i in indices],
                   rows=fp.rows[indices], shape=fp.shape[indices], sketch=fp.sketch[indices],
                   sha256=[fp.sha256[i] for i in indices], spectra=[fp.spectra[i] for i in indices],
                   delta={key: values[indices] for key, values in fp.delta.items()})


def _compatible_pair(a, b):
    """Remove precisely the views canonical pair_similarity already omits.

    Calibration's stricter geometry check then accepts vocabulary growth's
    aligned layers. Zero sketches stay: removing them would change the median
    and the calibrated statistic. Nonfinite/unscorable pairs abstain below.
    """
    right = {name: i for i, name in enumerate(b.names)}
    pairs = [(i, right[name]) for i, name in enumerate(a.names)
             if name in right and _view_reason(a, i, b, right[name]) is None]
    return _subset(a, [i for i, _ in pairs]), _subset(b, [j for _, j in pairs])


def similarity_edges(zoo_dir: Path, models: pd.DataFrame, data_root: Path,
                     k: int = 5, null: dict | None = None) -> pd.DataFrame:
    """Use retrieval candidates, then canonical k64 (or null-k) median cosine.

    Null tails and validation come from the evidence consumer, including its
    inclusive ties and n_pairs smoothing. P-values are display evidence only.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if models.model_key.duplicated().any():
        raise ValueError("Duplicate model keys in lineage input")
    root = Path(data_root).resolve()
    records = {row.model_key: row for row in models.itertuples()
               if isinstance(row.fingerprint_t1, str) and row.fingerprint_t1}
    candidates = set()
    for key, model in sorted(records.items()):
        paths = _paths(zoo_dir, model.arch_signature)
        present = [path.exists() or path.is_symlink() for path in paths]
        if not any(present):
            continue  # Singletons and signatures without common T1 views have no index.
        if not all(present):
            raise ValueError("Incomplete retrieval index: matrix and metadata are required")
        neighbors = nearest(zoo_dir, key, model.arch_signature, k)
        for neighbor, _ in neighbors:
            if neighbor not in records or records[neighbor].arch_signature != model.arch_signature:
                raise ValueError("Retrieval candidate disagrees with lineage models")
            candidates.add(tuple(sorted((key, neighbor))))
    fingerprints, rows = {}, []
    for child, parent in sorted(candidates):
        for key in (child, parent):
            if key not in fingerprints:
                fingerprints[key] = _fingerprint(records[key], root)
        a, b = fingerprints[child], fingerprints[parent]
        left, right = _compatible_pair(a, b)
        if null is None:
            sketch_k = 64
            p_value = np.nan
        else:
            result = _relatedness(left, right, null)
            if "p_value" not in result:
                if result.get("reason") == "no compatible shared views":
                    continue
                raise ValueError(f"Invalid or incompatible calibration: {result.get('reason')}")
            sketch_k, p_value = result["sketch_k"], result["p_value"]
        if (not np.isfinite(left.sketch[:, :sketch_k, :sketch_k]).all()
                or not np.isfinite(right.sketch[:, :sketch_k, :sketch_k]).all()):
            continue
        score = pair_similarity(a, b, sketch_k)["median_cosine"]
        if score is None or not math.isfinite(score):
            continue
        rows.append(_edge(child, parent, "same_root", "detected", "sketch_similarity",
                          score=score, p_value=p_value, calibrated=null is not None))
    return _table(rows)


def merge_edges(*tables: pd.DataFrame, external: Path | None = None) -> pd.DataFrame:
    """Append external evidence without rewriting its provenance or calibration.

    External calibrated evidence replaces built display evidence on the four
    frozen keys. Otherwise the first input occurrence wins deterministically.
    """
    sources = list(tables)
    if external is not None:
        sources.insert(0, pd.read_parquet(external))
    rows = []
    for table in sources:
        if set(table.columns) != set(LINEAGE_COLUMNS):
            raise ValueError("Lineage table does not match LINEAGE_COLUMNS")
        rows.extend(table.to_dict("records"))
    return _table(rows)
