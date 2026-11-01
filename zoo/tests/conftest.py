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
