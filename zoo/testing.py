"""Small deterministic checkpoint universe shared with Zoo-Keeper tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from evidence.fingerprint import fingerprint_tensors
from evidence.hub import HubReader
from evidence.manifest import build_manifest, write_manifest
from measurement_config import safe_filename


SHA1, SHA2, SHA3, SHA4, SHA5, SHA6, SHA7, SHA8, SHA9 = (str(i) * 40 for i in range(1, 10))
SHA10, SHA11, SHA12 = "a" * 40, "b" * 40, "c" * 40


def make_universe(root: Path) -> SimpleNamespace:
    """Write twelve pinned model manifests and nine real row-block fingerprints.

    The public-looking identities and commit pins are synthetic test fixtures.
    Source checkpoints remain under root/checkpoints for independent checks;
    fingerprint measurements use the actual extractor tensor API.
    """
    root = Path(root)
    manifests, fingerprints = root / "manifests", root / "fingerprints"
    manifests.mkdir(parents=True, exist_ok=True)
    fingerprints.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(20261028)
    names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.down_proj.weight"]
    parent = {name: torch.randn(512, 512, generator=generator) for name in names}
    parent["model.embed_tokens.weight"] = torch.randn(1024, 512, generator=generator)
    parent["model.norm.weight"] = torch.ones(512)

    def updated(base):
        result = {}
        for name, weight in base.items():
            if weight.ndim == 2:
                update = torch.randn(weight.shape[0], 4, generator=generator) @ torch.randn(4, weight.shape[1], generator=generator)
                result[name] = weight + .025 * update
            else:
                result[name] = weight.clone()
        return result

    child, sibling = updated(parent), updated(parent)
    vocab_child = {name: weight.clone() for name, weight in child.items()}
    vocab_child["model.embed_tokens.weight"] = torch.cat([
        child["model.embed_tokens.weight"], torch.randn(8, 512, generator=generator),
    ])
    unrelated = {name: torch.randn(weight.shape, generator=generator) if weight.ndim == 2 else weight.clone()
                 for name, weight in parent.items()}
    other = {name: torch.randn(512, 384, generator=generator) for name in names}
    other["model.embed_tokens.weight"] = torch.randn(1024, 384, generator=generator)
    other["model.norm.weight"] = torch.ones(384)
    decider = {name: weight.clone() for name, weight in parent.items()}
    decider["score.weight"] = torch.randn(2, 512, generator=generator)
    base_key = "org/parent@" + SHA1
    configs = {"model_type": "llama", "hidden_size": 512, "num_hidden_layers": 2,
               "intermediate_size": 512, "num_attention_heads": 8, "num_key_value_heads": 8,
               "vocab_size": 1024}
    entries = [
        ("org/parent", SHA1, parent, [], "apache-2.0"),
        ("org/child", SHA2, child, [base_key], "mit"),
        ("org/child-vocab", SHA3, vocab_child, [base_key], "apache-2.0"),
        ("org/sibling", SHA4, sibling, [base_key], "apache-2.0"),
        ("org/dup", SHA5, child, [base_key], "mit"),
        ("org/unrelated", SHA6, unrelated, ["other/foundation"], "apache-2.0"),
        ("org/other-arch", SHA7, other, [], "it's \"odd\""),
        ("org/t0only", SHA8, None, [], "apache-2.0"),
        ("org/gated", SHA9, None, [], "apache-2.0"),
        ("org/child", SHA10, child, [base_key], "mit"),
        ("org/decider", SHA11, decider, [base_key], "apache-2.0"),
        ("org/parent-GGUF", SHA12, None, [base_key], "apache-2.0"),
    ]
    for repo_id, revision, tensors, bases, license_id in entries:
        config = dict(configs)
        if repo_id == "org/other-arch":
            config.update(hidden_size=384, num_attention_heads=6, num_key_value_heads=6)
        elif repo_id == "org/child-vocab":
            config["vocab_size"] = 1032
        elif repo_id == "org/decider":
            config.update(architectures=["LlamaForSequenceClassification"], num_labels=2,
                          id2label={"0": "no", "1": "yes"})
        checkpoint = root / "checkpoints" / safe_filename(f"{repo_id}@{revision}")
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "config.json").write_text(json.dumps(config, sort_keys=True))
        if tensors is not None:
            save_file(tensors, checkpoint / "model.safetensors")
        elif repo_id == "org/t0only":
            (checkpoint / "pytorch_model.bin").write_bytes(b"synthetic metadata-only-only fixture; never load")
        elif repo_id == "org/parent-GGUF":
            (checkpoint / "model.gguf").write_bytes(b"synthetic metadata-only-only fixture; never decode")
        manifest = build_manifest(str(checkpoint), "main", HubReader(), api=None)
        manifest.update(repo_id=repo_id, revision=revision, license=license_id, declared_bases=bases,
                        base_model_relation="finetune" if bases else None)
        if repo_id == "org/gated":
            manifest.update(access="no_access", gated=True)
        elif repo_id == "org/parent-GGUF":
            manifest["gguf"] = {"architecture": "llama", "total": None,
                                "context_length": None, "quantization_types": []}
        elif repo_id == "org/decider":
            manifest["tags"] = ["system-one"]
            manifest["decision"]["system_one"] = True
        write_manifest(manifest, manifests)
        if tensors is not None:
            fingerprint_tensors(tensors.items(), fingerprints / f"{safe_filename(f'{repo_id}@{revision}')}.row_blocks.fingerprint.safetensors",
                                {"repo_id": repo_id, "revision": revision})
    return SimpleNamespace(root=root, manifests=manifests, fingerprints=fingerprints)
