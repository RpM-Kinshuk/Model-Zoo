import json

import numpy as np
import pytest

from zoo.schema import arch_signature, model_key, split_key


def test_keys_roundtrip():
    k = model_key("org/m", "a" * 40)
    assert k == "org/m@" + "a" * 40 and split_key(k) == ("org/m", "a" * 40)
    assert split_key("local/path@local:" + "b" * 64)[1].startswith("local:")


def test_signature_ignores_vocab_and_uses_config():
    cfg = {"model_type": "llama", "hidden_size": 512, "num_hidden_layers": 2, "vocab_size": 1000}
    assert arch_signature(cfg, []) == arch_signature({**cfg, "vocab_size": 1008}, [])
    assert arch_signature(cfg, []).startswith("cfg:")
    assert arch_signature({**cfg, "hidden_size": 384}, []) != arch_signature(cfg, [])


def test_shape_signature_fallback_excludes_tall_views():
    shapes = [(512, 512)] * 10 + [(50000, 512)]
    assert arch_signature({}, shapes) == arch_signature({}, [(512, 512)] * 10 + [(50008, 512)])
    assert arch_signature({}, shapes).startswith("shape:")


def test_keys_require_pinned_revisions_and_split_at_final_separator():
    key = model_key("/tmp/model@source", "local:" + "b" * 64)
    assert split_key(key) == ("/tmp/model@source", "local:" + "b" * 64)
    for revision in ("main", "a" * 39, "g" * 40, "local:" + "b" * 63, ""):
        with pytest.raises(ValueError):
            model_key("org/m", revision)
    for key in ("org/m", "@" + "a" * 40, "org/m@main"):
        with pytest.raises(ValueError):
            split_key(key)


def test_signature_is_order_invariant_and_preserves_topology():
    shapes = [(512, 128), (256, 512), (512, 512)]
    assert arch_signature({}, shapes) == arch_signature({}, shapes[::-1])
    assert arch_signature({}, shapes) != arch_signature({}, shapes + [(512, 128)])
    cfg = {"hidden_size": 512, "num_hidden_layers": 2, "num_experts": 8}
    assert arch_signature(cfg, []) != arch_signature({**cfg, "num_experts": 16}, [])
    assert arch_signature(cfg, []) == arch_signature(dict(reversed(list(cfg.items()))), [])
    assert arch_signature({}, []).startswith("shape:")


def test_shared_universe_preserves_pins_copies_and_coverage(universe, zoo_models):
    from evidence.fingerprint import load_fingerprint
    from zoo import ZOO_VERSION
    from zoo.schema import MODELS_COLUMNS, empty_table
    from zoo.testing import SHA1, SHA2, SHA3, SHA5, SHA8, SHA9, SHA10, SHA11, SHA12

    manifests = {model_key(m["repo_id"], m["revision"]): m for m in
                 (json.loads(p.read_text()) for p in universe.manifests.glob("*.manifest.json"))}
    fingerprints = {model_key(fp.meta["repo_id"], fp.meta["revision"]): fp for fp in
                    (load_fingerprint(p) for p in universe.fingerprints.glob("*.fingerprint.safetensors"))}
    assert ZOO_VERSION == "1" and "spectra_t2" in MODELS_COLUMNS
    assert len(manifests) == len(zoo_models) == 12 and len(fingerprints) == 9
    assert manifests["org/t0only@" + SHA8]["access"] == "t0_only"
    assert manifests["org/gated@" + SHA9]["access"] == "no_access"
    assert manifests["org/parent-GGUF@" + SHA12]["gguf"]["architecture"] == "llama"
    for key in ("org/child@" + SHA2, "org/child@" + SHA10, "org/dup@" + SHA5):
        assert manifests[key]["license"] == "mit"
        assert fingerprints[key].sha256 == fingerprints["org/child@" + SHA2].sha256
    parent, child = fingerprints["org/parent@" + SHA1], fingerprints["org/child-vocab@" + SHA3]
    embedding = "model.embed_tokens.weight"
    assert child.shape[child.names.index(embedding), 0] == parent.shape[parent.names.index(embedding), 0] + 8
    decision = manifests["org/decider@" + SHA11]["decision"]
    assert decision["system_one"] and decision["labels"] == {"0": "no", "1": "yes"}
    decider = fingerprints["org/decider@" + SHA11]
    assert decider.shape[decider.names.index("score.weight")].tolist() == [2, 512]
    assert decider.geometry[decider.names.index("score.weight")] == "head"
    assert manifests["org/child@" + SHA2]["declared_bases"] == ["org/parent@" + SHA1]
    assert empty_table(MODELS_COLUMNS).empty and list(empty_table(MODELS_COLUMNS).columns) == MODELS_COLUMNS


def test_universe_child_delta_is_rank_four(universe):
    # The shared fixture must be a derived checkpoint, not merely a synthetic
    # similarity score. Compare the source checkpoint tensors directly.
    from safetensors.torch import load_file
    from zoo.testing import SHA1, SHA2
    from measurement_config import safe_filename

    paths = [universe.root / "checkpoints" / safe_filename(f"org/{name}@{sha}") / "model.safetensors"
             for name, sha in (("parent", SHA1), ("child", SHA2))]
    tensors = [load_file(path) for path in paths]
    delta = (tensors[1]["model.layers.0.self_attn.q_proj.weight"] -
             tensors[0]["model.layers.0.self_attn.q_proj.weight"]).numpy()
    singular = np.linalg.svd(delta, compute_uv=False)
    assert np.count_nonzero(singular > singular.max() * 1e-4) == 4
