from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def universe(tmp_path):
    from zoo.testing import make_universe
    return make_universe(tmp_path)


@pytest.fixture
def zoo_models(universe):
    import json
    import pandas as pd
    from zoo.schema import model_key

    rows = [json.loads(path.read_text()) for path in sorted(universe.manifests.glob("*.manifest.json"))]
    return pd.DataFrame([
        {"model_key": model_key(row["repo_id"], row["revision"]),
         "repo_id": row["repo_id"], "revision": row["revision"]}
        for row in rows
    ]).sort_values("model_key").reset_index(drop=True)


@pytest.fixture
def tables(universe):
    from zoo.tables import build_tables
    return build_tables(universe.manifests, [universe.fingerprints], universe.root)


@pytest.fixture
def staged_zoo(tables, tmp_path):
    from zoo.retrieval import build_retrieval
    destination = tmp_path / "staged-zoo"
    build_retrieval(destination, tables["models"], tables["views"], tmp_path)
    return destination


@pytest.fixture
def null_dict():
    import numpy as np
    from evidence.sketch import SKETCH_SEED
    return dict(statistic="median_cosine", sketch_k=64, row_block=256, tier="t1",
                sketch_seed=SKETCH_SEED, n_pairs=1000,
                null_quantiles=np.linspace(-0.1, 0.3, 101).tolist(), related_threshold=.9)


@pytest.fixture
def built_zoo(universe, tmp_path):
    from zoo.build import build_zoo
    return build_zoo(universe.manifests, [universe.fingerprints],
                     tmp_path / "zoo", data_root=universe.root)


@pytest.fixture
def package_run(tmp_path, monkeypatch):
    """One complete, real local model run; no network or GPUs."""
    from types import SimpleNamespace
    from safetensors.torch import save_file
    import torch
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest
    from evidence.sidecar import EvidenceSidecar
    from measurement_config import measurement_config, safe_filename
    from net_esd import net_esd_estimator
    from evidence_bridge import loaded_block, measure_tensors, metrics_results
    from worker import save_results
    monkeypatch.setenv("ZOO_DATA", str(tmp_path))
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False), torch.nn.Linear(16, 16, bias=False))
    save_file(model.state_dict(), source / "model.safetensors")
    manifest = build_manifest(str(source), "main", HubReader(), None)
    manifest.update(repo_id="lab/tiny", source=dict(zoo_data_path="source"))
    safe = safe_filename(f"{manifest['repo_id']}@{manifest['revision']}")
    folder = run / "evidence" / safe
    cfg = measurement_config(SimpleNamespace(), model_id=manifest["repo_id"], revision=manifest["revision"])
    cfg["runtime"] = dict(analysis_source="model", device_map_requested="cpu", filter_type=False)
    metrics = net_esd_estimator(model, parallel=False, save_eigs=True, fix_fingers="xmin_mid", filter_zeros=True)
    save_results(metrics, run / "stats/lab--tiny.csv", "lab/tiny", False,
                 h5_output_path=run / "metrics/lab--tiny.h5", measurement_config=cfg, coverage={"counts": {}}, save_eigs=True)
    sidecar = EvidenceSidecar(manifest, folder)
    sidecar.start()
    counts = sidecar.finish(metrics_results(metrics), loaded_block(model), lambda items: measure_tensors(items, cfg), cfg)
    import json
    (folder / "evidence.json").write_text(json.dumps(counts))
    return SimpleNamespace(root=tmp_path, run=run, manifest=manifest, folder=folder,
                           safe=safe, config=cfg, packages=tmp_path / "packages")
