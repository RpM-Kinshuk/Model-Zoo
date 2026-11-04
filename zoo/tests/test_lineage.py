import json

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

from zoo.lineage import declared_edges, merge_edges, shared_tensor_edges, similarity_edges
from zoo.schema import LINEAGE_COLUMNS, VIEWS_COLUMNS, empty_table
from zoo.tables import duplicate_groups
from zoo.testing import SHA1, SHA2, SHA3, SHA5


def test_declared_and_external(tables):
    e = declared_edges(tables["models"])
    child = e[e.child_key == "org/child@" + SHA2].iloc[0]
    assert child.parent_key == "org/parent@" + SHA1 and child.evidence_tier == "declared"
    assert (e.parent_key.str.endswith("@external")).any()


def test_shared_tensor_edges_exclude_duplicates(tables):
    e = shared_tensor_edges(tables["views"], duplicate_groups(tables["views"]))
    pairs = set(zip(e.child_key, e.parent_key))
    assert ("org/child@" + SHA2, "org/dup@" + SHA5) not in pairs and ("org/dup@" + SHA5, "org/child@" + SHA2) not in pairs
    assert e.estimand.eq("same_root").all() and e.evidence_tier.eq("verified").all()


def test_similarity_edges_uncalibrated_then_calibrated(staged_zoo, tables, universe, null_dict):
    e = similarity_edges(staged_zoo, tables["models"], universe.root, k=2)
    assert (~e.calibrated).all() and e.p_value.isna().all()
    e2 = similarity_edges(staged_zoo, tables["models"], universe.root, k=2, null=null_dict)
    assert e2.calibrated.all() and e2.p_value.between(0, 1).all()


def test_declared_pins_and_ambiguous_bare_bases_and_relation_mapping():
    models = pd.DataFrame([
        dict(model_key="base@" + SHA1, repo_id="base", declared_bases=[], base_model_relation=None),
        dict(model_key="base@" + SHA2, repo_id="base", declared_bases=[], base_model_relation=None),
        dict(model_key="child@" + SHA3, repo_id="child", declared_bases=["base", "base@" + SHA1, "base@" + SHA5], base_model_relation="lora"),
    ])
    e = declared_edges(models)
    assert set(e.parent_key) == {"base@external", "base@" + SHA1, "base@" + SHA5}
    assert e.edge_type.eq("adapter").all()
    assert e.estimand.eq("parent").all() and e.evidence.eq("declared_metadata").all()
    for relation, expected in [("sft", "finetune"), ("dpo", "finetune"), ("merge", "merge"),
                               ("quantization_sim", "quantized"), ("abliteration", "abliteration"),
                               ("laundering", "unknown"), (None, "unknown")]:
        models.loc[2, "base_model_relation"] = relation
        assert declared_edges(models).edge_type.eq(expected).all()


def _views():
    return pd.DataFrame([
        dict(model_key=key, view_index=i, name=f"layer.{i}", geometry="matrix", component="layer",
             n_rows=512, n_cols=64, r0=128, r1=384, block_sha256=hashes[i], tier="t1")
        for key, hashes in [("a", ["same", "left"]), ("b", ["same", "right"])]
        for i in range(2)
    ], columns=VIEWS_COLUMNS)


def test_shared_hash_fraction_and_canonical_pair():
    views = _views().iloc[::-1]
    e = shared_tensor_edges(views, {})
    assert list(zip(e.child_key, e.parent_key, e.score)) == [("a", "b", .5)]
    assert e.edge_type.eq("unknown").all()
    assert shared_tensor_edges(views, {"a": "a", "b": "a"}).empty


@pytest.mark.parametrize("field,value", [("tier", "t2"), ("geometry", "head"), ("n_cols", 128), ("r0", 127), ("r1", 383)])
def test_shared_hash_requires_compatible_t1_views(field, value):
    views = _views()
    views.loc[(views.model_key == "b") & (views.view_index == 0), field] = value
    assert shared_tensor_edges(views, {}).empty


def _staged_pair(root, failure=None):
    """Synthetic extractor artifacts; first32 sketches agree, k64 medians differ."""
    from zoo.retrieval import build_retrieval
    keys = ["a@" + SHA1, "b@" + SHA2]
    models, views = [], []
    for model_index, key in enumerate(keys):
        sketches = np.zeros((3, 64, 64), dtype=np.float32)
        sketches[:, 0, 0] = 1
        sketches[:, 63, 63] = [1, 1, 1] if model_index == 0 else [1, -1, -1]
        if model_index and failure == "zero":
            sketches[1:] = 0
        if model_index and failure == "nonfinite":
            sketches[1, 63, 63] = np.nan
        meta = dict(repo_id=key.split("@")[0], revision=key.split("@")[1], tier="t1",
                    evidence_version="1", sketch_seed=20261028, row_block=256, sketch_k=64,
                    names=[f"layer.{i}" for i in range(3)], geometry=["matrix"] * 3,
                    component=["layer"] * 3, sha256=[f"{model_index}-{i}" for i in range(3)],
                    delta_status=["no_parent"] * 3, delta_confidence=["none"] * 3, coverage=[])
        path = root / f"{model_index}.t1.fingerprint.safetensors"
        arrays = dict(sketch=sketches, rows=np.tile([128, 384], (3, 1)), shape=np.tile([512, 64], (3, 1)),
                      delta_scalars=np.zeros((3, 3), dtype=np.float32), delta_top_sv=np.zeros((3, 0), dtype=np.float32),
                      spectrum_values=np.zeros(0, dtype=np.float32), spectrum_offsets=np.zeros(4, dtype=np.int64))
        save_file(arrays, path, metadata={name: value if name in {"repo_id", "revision", "tier", "evidence_version"} else json.dumps(value) for name, value in meta.items()})
        models.append(dict(model_key=key, arch_signature="cfg:" + "a" * 12, fingerprint_t1=path.name))
        for i, name in enumerate(meta["names"]):
            views.append(dict(model_key=key, view_index=i, name=name, geometry="matrix", component="layer",
                              n_rows=512, n_cols=64, r0=128, r1=384, block_sha256=meta["sha256"][i], tier="t1"))
    models = pd.DataFrame(models)
    build_retrieval(root, models, pd.DataFrame(views), root)
    return models


def test_similarity_is_k64_median_with_inclusive_n_pairs_tail(tmp_path, null_dict):
    models = _staged_pair(tmp_path)
    null = dict(null_dict, n_pairs=19, null_quantiles=[-.1, 0., .3])
    e = similarity_edges(tmp_path, models, tmp_path, null=null)
    assert len(e) == 1 and e.score.iloc[0] == pytest.approx(0.)
    assert e.p_value.iloc[0] == pytest.approx(41 / 60)
    assert e.calibrated.all() and (e.child_key < e.parent_key).all()
    null["sketch_k"] = 32
    e32 = similarity_edges(tmp_path, models, tmp_path, null=null)
    assert e32.score.iloc[0] == pytest.approx(1.)
    assert e32.p_value.iloc[0] == pytest.approx(1 / 20)


def test_zero_sketches_keep_canonical_median(tmp_path, null_dict):
    models = _staged_pair(tmp_path, "zero")
    e = similarity_edges(tmp_path, models, tmp_path, null=null_dict)
    assert len(e) == 1 and e.score.iloc[0] == 0.


def test_nonfinite_similarity_abstains(tmp_path, null_dict):
    models = _staged_pair(tmp_path, "nonfinite")
    assert similarity_edges(tmp_path, models, tmp_path, null=null_dict).empty


def test_vocab_child_calibration_retains_aligned_layers(staged_zoo, tables, universe, null_dict):
    e = similarity_edges(staged_zoo, tables["models"], universe.root, k=20, null=null_dict)
    pair = e[(e.child_key == "org/child-vocab@" + SHA3) & (e.parent_key == "org/parent@" + SHA1)]
    assert len(pair) == 1 and pair.calibrated.all() and pair.score.iloc[0] > .99


@pytest.mark.parametrize("field,value", [("sketch_k", 128), ("sketch_k", "64"), ("row_block", 128),
    ("tier", "t2"), ("sketch_seed", 7), ("statistic", "mean_cosine"), ("n_pairs", 0),
    ("null_quantiles", [0.3, -.1]), ("null_quantiles", [np.nan]), ("related_threshold", np.inf)])
def test_invalid_or_incompatible_calibration_fails_clearly(tmp_path, null_dict, field, value):
    models = _staged_pair(tmp_path)
    with pytest.raises(ValueError, match="calibration|sketch"):
        similarity_edges(tmp_path, models, tmp_path, null=dict(null_dict, **{field: value}))


def test_merge_preserves_external_provenance_and_estimands(tmp_path):
    row = dict(child_key="a", parent_key="b", estimand="parent", edge_type="finetune",
               evidence_tier="declared", evidence="declared_metadata", score=np.nan,
               p_value=np.nan, calibrated=False, method_version="zoo-v0")
    built = pd.DataFrame([row], columns=LINEAGE_COLUMNS)
    external_rows = [dict(row, evidence="p1", evidence_tier="detected", method_version="p1-v3",
                          calibrated=True, p_value=.002, estimand=estimand)
                     for estimand in ("parent", "ancestor", "same_root")]
    path = tmp_path / "p1.parquet"
    pd.DataFrame(external_rows).to_parquet(path, index=False)
    e = merge_edges(built, built, external=path)
    assert len(e) == 4 and list(e.columns) == LINEAGE_COLUMNS
    external = e[e.evidence == "p1"]
    assert external.method_version.eq("p1-v3").all()
    assert external.calibrated.all() and external.p_value.eq(.002).all()
    assert set(external.estimand) == {"parent", "ancestor", "same_root"}
    pd.testing.assert_frame_equal(e, merge_edges(built, built, external=path))


def test_empty_producers_keep_frozen_schema(tmp_path):
    models = pd.DataFrame(columns=["model_key", "repo_id", "declared_bases", "base_model_relation", "fingerprint_t1"])
    for e in [declared_edges(models), shared_tensor_edges(empty_table(VIEWS_COLUMNS), {}),
              similarity_edges(tmp_path, models, tmp_path), merge_edges()]:
        assert e.empty and list(e.columns) == LINEAGE_COLUMNS


def test_external_calibrated_edge_replaces_matching_display_edge(tmp_path):
    row = dict(child_key="a", parent_key="b", estimand="same_root", edge_type="unknown",
               evidence_tier="detected", evidence="sketch_similarity", score=.8,
               p_value=.1, calibrated=False, method_version="zoo-v0")
    external = dict(row, score=.95, p_value=.001, calibrated=True, method_version="p1-v3")
    path = tmp_path / "replacement.parquet"
    pd.DataFrame([external], columns=LINEAGE_COLUMNS).to_parquet(path, index=False)
    result = merge_edges(pd.DataFrame([row], columns=LINEAGE_COLUMNS), external=path)
    assert result.to_dict("records") == [external]


@pytest.mark.parametrize("missing", [".npy", ".json"])
def test_partial_retrieval_index_raises(tmp_path, missing):
    models = _staged_pair(tmp_path)
    (tmp_path / "retrieval" / ("cfg:" + "a" * 12 + missing)).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="retrieval|index|No such file"):
        similarity_edges(tmp_path, models, tmp_path)


def test_complete_retrieval_absence_is_legitimate(tmp_path):
    models = _staged_pair(tmp_path)
    for path in (tmp_path / "retrieval").iterdir():
        path.unlink()
    assert similarity_edges(tmp_path, models, tmp_path).empty


def test_pair_delta_does_not_promote_detected_claim_to_verified():
    from zoo.lineage import _edge, _table, with_pair_delta_evidence
    child, parent = dict(repo_id="org/a", revision="a"*40), dict(repo_id="org/b", revision="b"*40)
    edges = _table([_edge("org/a@"+"a"*40, "org/b@"+"b"*40, "same_root", "detected", "sketch_similarity", score=.8)])
    result = with_pair_delta_evidence(edges, [(None, dict(child=child, parent=parent))])
    assert result.iloc[0].evidence == "sketch_similarity+row_block_delta"
    assert result.iloc[0].evidence_tier == "detected" and not result.iloc[0].calibrated
    assert result.iloc[0].estimand == "same_root" and result.iloc[0].score == .8
