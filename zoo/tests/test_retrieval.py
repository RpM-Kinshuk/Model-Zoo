import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zoo.retrieval import build_retrieval, nearest, select_retrieval_views
from zoo.testing import SHA1, SHA2, SHA3, SHA6, SHA7, SHA8, SHA9, SHA12


def test_parent_is_nearest_to_child_and_vocab_child(tables, staged_zoo):
    sig = tables["models"].set_index("model_key").loc["org/child@" + SHA2, "arch_signature"]
    top = nearest(staged_zoo, "org/child-vocab@" + SHA3, sig, k=3)
    assert {"org/parent@" + SHA1, "org/child@" + SHA2} & {k for k, _ in top[:2]}
    scores = dict(nearest(staged_zoo, "org/child@" + SHA2, sig, k=10))
    assert scores["org/parent@" + SHA1] > scores["org/unrelated@" + SHA6] + 0.5


def test_other_arch_not_in_same_matrix(tables, staged_zoo):
    sig = tables["models"].set_index("model_key").loc["org/child@" + SHA2, "arch_signature"]
    keys = json.loads((staged_zoo / "retrieval" / f"{sig}.json").read_text())["models"]
    assert "org/other-arch@" + SHA7 not in keys and "org/t0only@" + SHA8 not in keys


def test_selected_views_are_common_not_tall_and_evenly_spaced():
    rows = []
    for model in range(10):
        for view in range(40):
            rows.append(dict(model_key=str(model), name=f"layer.{view:02d}", n_rows=512, tier="row_blocks"))
        rows.append(dict(model_key=str(model), name="embedding", n_rows=50000, tier="row_blocks"))
        if model < 8:
            rows.append(dict(model_key=str(model), name="rare", n_rows=512, tier="row_blocks"))
    chosen = select_retrieval_views(pd.DataFrame(rows))
    assert len(chosen) == 32 and chosen == sorted(chosen)
    assert chosen[0] == "layer.00" and chosen[-1] == "layer.39"
    assert "embedding" not in chosen and "rare" not in chosen
    assert select_retrieval_views(pd.DataFrame()) == []


def _manual_index(root, keys, blocks, presence=None):
    sig = "cfg:" + "a" * 12
    directory = root / "retrieval"
    directory.mkdir(parents=True)
    n_models, n_views, _ = blocks.shape
    np.save(directory / f"{sig}.npy", blocks.reshape(n_models, -1).astype(np.float16))
    record = dict(models=keys, views=[f"view.{i}" for i in range(n_views)],
                  presence=presence if presence is not None else [[True] * n_views for _ in keys],
                  coordinates=[[["matrix", 128, 384, 64] for _ in range(n_views)] for _ in keys],
                  sketch_seed=20261028, row_block=256, evidence_version="1")
    (directory / f"{sig}.json").write_text(json.dumps(record))
    return sig


def test_nearest_uses_common_view_masks_and_deterministic_ties(tmp_path, monkeypatch):
    blocks = np.zeros((4, 3, 1024), dtype=np.float32)
    blocks[0, :2, 0] = 1
    blocks[1, 0, 0] = 1
    blocks[2, 1, :2] = [.6, .8]
    blocks[2, 2, 0] = 1
    blocks[3, 0, 0] = 1
    sig = _manual_index(tmp_path, ["query", "b", "c", "d"], blocks,
                        [[True, True, False], [True, False, False], [False, True, True], [True, False, False]])
    original = np.load
    modes = []
    def load(*args, **kwargs):
        modes.append(kwargs.get("mmap_mode"))
        return original(*args, **kwargs)
    monkeypatch.setattr(np, "load", load)
    result = nearest(tmp_path, "query", sig, k=10)
    assert [key for key, _ in result] == ["b", "d", "c"]
    assert dict(result)["b"] == pytest.approx(1., abs=1e-6)
    assert dict(result)["c"] == pytest.approx(.6, abs=2e-4)
    assert modes == ["r"]


def test_zero_nonfinite_and_unshared_views_do_not_create_scores(tmp_path):
    blocks = np.zeros((3, 2, 1024), dtype=np.float32)
    blocks[0, 0, 0] = 1
    blocks[1, 1, 0] = 1
    blocks[2, 0, 0] = np.nan
    sig = _manual_index(tmp_path, ["query", "unshared", "invalid"], blocks)
    assert nearest(tmp_path, "query", sig, k=10) == []
    assert nearest(tmp_path, "invalid", sig, k=10) == []


def test_empty_tables_produce_no_retrieval_matrix(tmp_path):
    from zoo.schema import MODELS_COLUMNS, VIEWS_COLUMNS, empty_table
    build_retrieval(tmp_path, empty_table(MODELS_COLUMNS), empty_table(VIEWS_COLUMNS), tmp_path)
    assert not list((tmp_path / "retrieval").glob("*.npy"))


def test_vocab_child_keeps_aligned_layers_and_masks_shifted_embedding(tables, staged_zoo):
    sig = tables["models"].set_index("model_key").loc["org/child@" + SHA2, "arch_signature"]
    record = json.loads((staged_zoo / "retrieval" / f"{sig}.json").read_text())
    parent, child = (record["models"].index(key) for key in ("org/parent@" + SHA1, "org/child-vocab@" + SHA3))
    embedding = record["views"].index("model.embed_tokens.weight")
    assert record["coordinates"][parent][embedding] != record["coordinates"][child][embedding]
    assert dict(nearest(staged_zoo, "org/child-vocab@" + SHA3, sig, 20))["org/parent@" + SHA1] > .99
    assert "org/gated@" + SHA9 not in record["models"]
    assert "org/parent-GGUF@" + SHA12 not in record["models"]


@pytest.mark.parametrize("failure", ["identity", "seed", "row_block"])
def test_retrieval_rejects_mismatched_fingerprint_settings(tables, universe, tmp_path, failure):
    from safetensors import safe_open
    from safetensors.numpy import save_file

    models = tables["models"].copy(deep=True)
    row_index = models.index[models.model_key == "org/parent@" + SHA1][0]
    source = universe.root / models.loc[row_index, "row_block_fingerprint"]
    with safe_open(source, framework="numpy") as handle:
        metadata = dict(handle.metadata())
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
    if failure == "identity":
        metadata["revision"] = "f" * 40
    elif failure == "seed":
        metadata["sketch_seed"] = json.dumps(json.loads(metadata["sketch_seed"]) + 1)
    else:
        metadata["row_block"] = json.dumps(128)
    changed = universe.root / "changed.row_blocks.fingerprint.safetensors"
    save_file(arrays, changed, metadata=metadata)
    models.loc[row_index, "row_block_fingerprint"] = changed.relative_to(universe.root).as_posix()
    with pytest.raises(ValueError, match="identity|settings|seed|row_block"):
        build_retrieval(tmp_path / "invalid", models, tables["views"], universe.root)


def test_view_prevalence_counts_empty_row_blocks_models(tables, universe, tmp_path):
    from evidence.fingerprint import fingerprint_tensors

    models = tables["models"].copy(deep=True)
    template = models.loc[models.model_key == "org/parent@" + SHA1].iloc[0].to_dict()
    extra = []
    for index, revision in enumerate(("d" * 40, "e" * 40)):
        repo = f"org/empty-{index}"
        path = universe.root / f"empty-{index}.row_blocks.fingerprint.safetensors"
        fingerprint_tensors([], path, {"repo_id": repo, "revision": revision})
        extra.append(dict(template, model_key=repo + "@" + revision, repo_id=repo,
                          revision=revision, n_views=0,
                          row_block_fingerprint=path.relative_to(universe.root).as_posix()))
    models = pd.concat([models, pd.DataFrame(extra)], ignore_index=True)
    signature = template["arch_signature"]
    destination = tmp_path / "with-empty-fingerprints"
    build_retrieval(destination, models, tables["views"], universe.root)
    # Eight measured models out of ten eligible same-signature row-block models do
    # not meet the90% rule, even though every observed view row has the name.
    assert not (destination / "retrieval" / f"{signature}.npy").exists()


@pytest.mark.parametrize("cohort", ["singleton", "empty", "removed", "no_views"])
def test_successful_rebuild_removes_only_obsolete_generated_indices(tables, staged_zoo, universe, cohort):
    signature = tables["models"].set_index("model_key").loc["org/parent@" + SHA1, "arch_signature"]
    directory = staged_zoo / "retrieval"
    unrelated = [directory / "notes.txt", directory / "custom.npy", directory / "cfg:invalid.json"]
    for path in unrelated:
        path.write_bytes(b"keep unrelated file")
    models, views = tables["models"], tables["views"]
    if cohort == "singleton":
        models = models[models.model_key == "org/parent@" + SHA1]
    elif cohort == "empty":
        models = models.iloc[:0]
    elif cohort == "removed":
        models = models[models.arch_signature != signature]
    else:
        views = views.iloc[:0]
    build_retrieval(staged_zoo, models, views, universe.root)
    assert not (directory / f"{signature}.npy").exists()
    assert not (directory / f"{signature}.json").exists()
    with pytest.raises(FileNotFoundError):
        nearest(staged_zoo, "org/parent@" + SHA1, signature, 10)
    assert all(path.read_bytes() == b"keep unrelated file" for path in unrelated)


def test_failed_rebuild_does_not_clean_existing_signature_indices(tables, staged_zoo, universe, monkeypatch):
    signature = tables["models"].set_index("model_key").loc["org/parent@" + SHA1, "arch_signature"]
    directory = staged_zoo / "retrieval"
    old = {path: path.read_bytes() for path in (directory / f"{signature}.npy", directory / f"{signature}.json")}
    models = tables["models"][tables["models"].model_key.isin(["org/parent@" + SHA1, "org/child@" + SHA2])].copy()
    models["arch_signature"] = "cfg:" + "f" * 12
    def fail(*args, **kwargs):
        raise RuntimeError("measurement unavailable")
    monkeypatch.setattr("zoo.retrieval.load_fingerprint", fail)
    with pytest.raises(RuntimeError, match="measurement unavailable"):
        build_retrieval(staged_zoo, models, tables["views"], universe.root)
    assert all(path.read_bytes() == content for path, content in old.items())
