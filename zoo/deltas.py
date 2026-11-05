"""Measure only discovered candidate pairs; never infer ancestry direction."""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evidence.block_cache import BlockCache
from evidence.pair_delta import pair_deltas, pair_path, load_pair_delta
from measurement_config import safe_filename
from .build import open_tables
from .schema import split_key


def _init(devices, counter):
    from evidence.cli import _init_worker
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    _init_worker(None, None, devices, counter)


def _measure(task):
    from evidence import cli
    child, parent, package, cache, cache_bytes, all_experts = task
    return str(pair_deltas(child, parent, cli._READER, package,
                          block_cache=BlockCache(cache, cache_bytes) if cache else None,
                          compute_device=cli._DEVICE, all_experts=all_experts))


def run_deltas(zoo, packages, data_root, *, workers="auto", compute_device="auto",
               parent_cache=None, parent_cache_gib=64):
    from evidence.cli import _execution_policy
    devices, count = _execution_policy(compute_device, workers)
    root, packages = Path(data_root).resolve(), Path(packages).resolve()
    tables = open_tables(zoo)
    models = {row.model_key: row for row in tables["models"].itertuples()}
    tasks, seen, skipped = [], set(), 0
    edges = tables["lineage"]
    for edge in edges[edges.evidence_tier.eq("detected")].itertuples():
        pair = (edge.child_key, edge.parent_key)
        if pair in seen:
            continue
        seen.add(pair)
        manifests, folders = [], []
        for key in pair:
            row = models.get(key)
            relative = getattr(row, "row_block_fingerprint", "")
            if not isinstance(relative, str) or not relative:
                break
            artifact = (root / relative).resolve()
            if Path(relative).is_absolute() or not artifact.is_relative_to(packages) or not artifact.is_relative_to(root):
                raise ValueError("Delta input must belong to packages inside data_root")
            folder = artifact.parent
            repo, revision = split_key(key)
            manifest = json.loads((folder / (safe_filename(key) + ".manifest.json")).read_text())
            if (manifest["repo_id"], manifest["revision"]) != (repo, revision):
                raise ValueError("Delta manifest disagrees with zoo identity")
            manifests.append(manifest)
            folders.append(folder)
        if len(manifests) != 2:
            skipped += 1
            continue
        child, parent = manifests
        path = pair_path(folders[0], parent)
        if path.is_file():
            meta, _ = load_pair_delta(path)
            if meta["child"] != {k: child[k] for k in ("repo_id", "revision")} or meta["parent"] != {k: parent[k] for k in ("repo_id", "revision")}:
                raise ValueError("Pair delta identity mismatch")
            skipped += 1
            continue
        from evidence.fingerprint import load_fingerprint
        fp = load_fingerprint(root / models[pair[0]].row_block_fingerprint)
        all_experts = not any(row.get("reason") == "expert_not_sampled" for row in fp.coverage)
        tasks.append((child, parent, folders[0], parent_cache, int(parent_cache_gib * 1024**3), all_experts))
    count = min(count, max(1, len(tasks)))
    paths = []
    if tasks and count == 1:
        _init(devices, None)
        paths = [_measure(task) for task in tasks]
    elif tasks:
        context = mp.get_context("spawn" if devices[0] != "cpu" else "fork")
        counter = context.Value("i", 0)
        with ProcessPoolExecutor(max_workers=count, mp_context=context, initializer=_init, initargs=(devices, counter)) as pool:
            paths = list(pool.map(_measure, tasks))
    return dict(measured=len(paths), skipped=skipped, paths=paths)
