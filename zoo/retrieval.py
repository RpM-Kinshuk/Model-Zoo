"""T1-only retrieval with exact mean cosine over compatible shared views."""

import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from evidence import EVIDENCE_VERSION, READABLE_EVIDENCE_VERSIONS
from evidence.fingerprint import load_fingerprint
from evidence.sketch import ROW_BLOCK, SKETCH_SEED
from .schema import model_key


RETRIEVAL_VIEWS = 32
_SKETCH_SIZE = 32
_VIEW_WIDTH = _SKETCH_SIZE ** 2
_SEARCH_CHUNK = 256


def select_retrieval_views(views: pd.DataFrame) -> list[str]:
    """Choose at most32 common non-tall view names in stable name order."""
    return _select_retrieval_views(views)


def _select_retrieval_views(views, model_count=None):
    """The builder supplies its full population, including empty T1 models."""
    if views.empty:
        return []
    first = views[views.tier == "t1"] if "tier" in views else views
    if first.empty:
        return []
    count = first.model_key.nunique() if model_count is None else model_count
    limit = 8 * first.n_rows.median()
    usable = first[first.n_rows <= limit].drop_duplicates(["model_key", "name"])
    frequency = usable.groupby("name").model_key.nunique()
    names = sorted(frequency[frequency >= math.ceil(.9 * count)].index)
    if len(names) > RETRIEVAL_VIEWS:
        names = [names[i] for i in np.linspace(0, len(names) - 1, RETRIEVAL_VIEWS, dtype=int)]
    return names


def _paths(zoo_dir, signature):
    if not isinstance(signature, str) or re.fullmatch(r"(?:cfg|shape):[0-9a-f]{12}", signature) is None:
        raise ValueError("Invalid architecture signature")
    directory = Path(zoo_dir) / "retrieval"
    return directory / f"{signature}.npy", directory / f"{signature}.json"


def _fingerprint(row, root):
    relative = Path(row.fingerprint_t1)
    if relative.is_absolute():
        raise ValueError("T1 fingerprint path must be relative to data_root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("T1 fingerprint path escapes data_root")
    fp = load_fingerprint(path)
    if model_key(fp.meta["repo_id"], fp.meta["revision"]) != row.model_key:
        raise ValueError(f"Fingerprint identity does not match model: {row.model_key}")
    if (fp.meta.get("tier") != "t1" or fp.meta.get("evidence_version") not in READABLE_EVIDENCE_VERSIONS
            or fp.meta.get("sketch_seed") != SKETCH_SEED or fp.meta.get("row_block") != ROW_BLOCK
            or fp.meta.get("sketch_k", 0) < _SKETCH_SIZE
            or fp.sketch.shape != (len(fp.names), fp.meta["sketch_k"], fp.meta["sketch_k"])):
        raise ValueError(f"Incompatible T1 fingerprint settings (seed/row_block/sketch): {row.model_key}")
    return fp


def _view_index(fp, row):
    index = int(row.view_index)
    if (not 0 <= index < len(fp.names) or fp.names[index] != row["name"]
            or fp.geometry[index] != row.geometry or fp.sha256[index] != row.block_sha256
            or fp.shape[index].tolist() != [int(row.n_rows), int(row.n_cols)]
            or fp.rows[index].tolist() != [int(row.r0), int(row.r1)]):
        raise ValueError(f"View table disagrees with T1 fingerprint: {row['name']}")
    return index


def _remove_obsolete(destination, emitted):
    """Reconcile only owned signature artifacts after a successful build."""
    for path in sorted(destination.iterdir()):
        if (path.suffix in (".npy", ".json")
                and re.fullmatch(r"(?:cfg|shape):[0-9a-f]{12}", path.stem)
                and path.stem not in emitted and (path.is_file() or path.is_symlink())):
            path.unlink()


def build_retrieval(zoo_dir: Path, models: pd.DataFrame, views: pd.DataFrame,
                    data_root: Path) -> None:
    """Write float16 row matrices plus presence/alignment facts per signature.

    Alignment stays per model/view: an extended vocabulary can shift just its
    embedding block while leaving the model's layer views comparable.
    """
    root = Path(data_root).resolve()
    destination = Path(zoo_dir) / "retrieval"
    destination.mkdir(parents=True, exist_ok=True)
    emitted = set()
    if models.empty:
        _remove_obsolete(destination, emitted)
        return
    if models.model_key.duplicated().any():
        raise ValueError("Duplicate model keys in retrieval input")
    candidates = models[models.fingerprint_t1.map(lambda value: isinstance(value, str) and bool(value))]
    for signature, group in candidates.groupby("arch_signature", sort=True):
        group = group.sort_values("model_key")
        selected_rows = views[views.model_key.isin(group.model_key) & views.tier.eq("t1")]
        names = _select_retrieval_views(selected_rows, model_count=len(group))
        if len(group) < 2 or not names:
            continue
        matrix_path, metadata_path = _paths(zoo_dir, signature)
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float16,
                                          shape=(len(group), len(names) * _VIEW_WIDTH))
        matrix[:] = 0
        keys, presence, coordinates, missing = [], [], [], {}
        for model_index, row in enumerate(group.itertuples()):
            fp = _fingerprint(row, root)
            own = selected_rows[selected_rows.model_key == row.model_key]
            if own.name.duplicated().any():
                raise ValueError(f"Duplicate retrieval view names: {row.model_key}")
            own = own.set_index("name", drop=False)
            present, alignment = [], []
            for view_index, name in enumerate(names):
                usable, facts = False, None
                if name in own.index:
                    index = _view_index(fp, own.loc[name])
                    block = np.asarray(fp.sketch[index, :_SKETCH_SIZE, :_SKETCH_SIZE], dtype=np.float32).reshape(-1)
                    norm = float(np.linalg.norm(block.astype(np.float64))) if np.isfinite(block).all() else 0.
                    if norm > 0 and math.isfinite(norm):
                        start = view_index * _VIEW_WIDTH
                        matrix[model_index, start:start + _VIEW_WIDTH] = block / norm
                        usable = True
                        facts = [fp.geometry[index], int(fp.rows[index, 0]), int(fp.rows[index, 1]), int(fp.shape[index, 1])]
                present.append(usable)
                alignment.append(facts)
            keys.append(row.model_key)
            presence.append(present)
            coordinates.append(alignment)
            missing[row.model_key] = len(names) - sum(present)
        matrix.flush()
        del matrix
        metadata = dict(models=keys, views=names, presence=presence, coordinates=coordinates,
                        missing_counts=missing, sketch_seed=SKETCH_SEED, row_block=ROW_BLOCK,
                        evidence_version=EVIDENCE_VERSION)
        metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
        emitted.add(signature)
    _remove_obsolete(destination, emitted)


def _unit_blocks(rows, n_views):
    blocks = np.asarray(rows, dtype=np.float32).reshape(-1, n_views, _VIEW_WIDTH).copy()
    finite = np.isfinite(blocks).all(axis=2)
    blocks[~finite] = 0
    norms = np.linalg.norm(blocks, axis=2)
    valid = finite & np.isfinite(norms) & (norms > 0)
    np.divide(blocks, norms[..., None], out=blocks, where=valid[..., None])
    blocks[~valid] = 0
    return blocks, valid


def nearest(zoo_dir: Path, key: str, sig: str, k: int) -> list[tuple[str, float]]:
    """Exact float32 search, bounded in memory, with pairwise view masks."""
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if k == 0:
        return []
    matrix_path, metadata_path = _paths(zoo_dir, sig)
    metadata = json.loads(metadata_path.read_text())
    keys, names = metadata["models"], metadata["views"]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate retrieval model keys")
    if key not in keys:
        raise ValueError(f"Model has no comparable T1 retrieval row: {key}")
    if (metadata.get("sketch_seed") != SKETCH_SEED or metadata.get("row_block") != ROW_BLOCK
            or metadata.get("evidence_version") not in READABLE_EVIDENCE_VERSIONS):
        raise ValueError("Incompatible retrieval settings")
    n_views = len(names)
    presence = np.asarray(metadata["presence"], dtype=bool)
    coordinates = metadata["coordinates"]
    if (presence.shape != (len(keys), n_views) or len(coordinates) != len(keys)
            or any(len(row) != n_views for row in coordinates) or not n_views):
        raise ValueError("Invalid retrieval presence/alignment metadata")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(keys), n_views * _VIEW_WIDTH) or matrix.dtype != np.float16:
        raise ValueError("Invalid retrieval matrix shape/dtype")
    query_index = keys.index(key)
    query, query_valid = _unit_blocks(matrix[query_index:query_index + 1], n_views)
    query = query[0]
    query_present = presence[query_index] & query_valid[0]
    output = []
    for start in range(0, len(keys), _SEARCH_CHUNK):
        stop = min(start + _SEARCH_CHUNK, len(keys))
        blocks, valid = _unit_blocks(matrix[start:stop], n_views)
        aligned = np.asarray([[facts is not None and facts == coordinates[query_index][view]
                               for view, facts in enumerate(row)] for row in coordinates[start:stop]], dtype=bool)
        common = presence[start:stop] & valid & query_present & aligned
        denominator = common.sum(axis=1)
        per_view = np.einsum("mvd,vd->mv", blocks, query, dtype=np.float32, optimize=True)
        totals = np.where(common, per_view, 0).sum(axis=1, dtype=np.float32)
        for offset, count in enumerate(denominator):
            index = start + offset
            if count and index != query_index:
                score = float(np.clip(totals[offset] / count, -1., 1.))
                if math.isfinite(score):
                    output.append((keys[index], score))
    return sorted(output, key=lambda item: (-item[1], item[0]))[:k]
