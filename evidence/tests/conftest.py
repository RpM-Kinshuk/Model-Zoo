"""Offline fixtures for checkpoint evidence tests."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit
import hashlib

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "esd_experiment" / "src"))

SHA = "a" * 40


@pytest.fixture
def fake_transport():
    class FakeTransport:
        blob = bytes(range(250)) * 4

        def __init__(self):
            self.last_url = None
            self.last_headers = None
            self.gates = {}
            self.expired = False
            self.lookups = 0

        def gate(self, repo, allowed_token):
            self.gates[repo] = allowed_token

        def expire_next_range_read(self):
            self.expired = True

        def __call__(self, method, url, headers):
            self.last_url = url
            self.last_headers = dict(headers)
            if method == "HEAD":
                repo = urlsplit(url).path.lstrip("/").split("/resolve/")[0]
                if repo in self.gates and headers.get("Authorization") != f"Bearer {self.gates[repo]}":
                    return 403, {}, b""
                self.lookups += 1
                return 302, {
                    "x-linked-etag": '"abc"', "x-linked-size": "1000",
                    "Location": f"https://cdn.example/blob?version={self.lookups}",
                }, b""
            if self.expired:
                self.expired = False
                return 403, {}, b""
            start, end = map(int, headers["Range"][6:].split("-"))
            return 206, {"Content-Range": f"bytes {start}-{end}/{len(self.blob)}"}, self.blob[start:end + 1]

    return FakeTransport()


@pytest.fixture
def tiny_checkpoint(tmp_path):
    shards = [
        {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(512).reshape(32, 16).to(torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.arange(256).reshape(16, 16).to(torch.float16),
            "model.layers.0.self_attn.v_proj.weight": torch.arange(256).reshape(16, 16).float(),
            "model.layers.0.input_layernorm.weight": torch.ones(16),
        },
        {
            "model.embed_tokens.weight": torch.arange(8192).reshape(512, 16).to(torch.bfloat16),
            "model.layers.0.self_attn.o_proj.weight": torch.arange(512).reshape(16, 32).float(),
            "model.position_ids": torch.arange(512, dtype=torch.int64),
        },
    ]
    weight_map = {}
    total_size = 0
    for i, tensors in enumerate(shards, 1):
        filename = f"model-{i:05d}-of-00002.safetensors"
        save_file(tensors, tmp_path / filename, metadata={"format": "pt"})
        weight_map.update({name: filename for name in tensors})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": weight_map,
    }))
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "vocab_size": 512,
    }))
    return tmp_path


@pytest.fixture
def hub_stats_sample():
    import pandas as pd
    path = Path(__file__).parent / "data" / "hub_stats_sample.json"
    return pd.DataFrame(json.loads(path.read_text()))


@pytest.fixture
def manifest_repositories(tiny_checkpoint, tmp_path):
    """Real file bytes and API metadata, isolated from the X2 transport."""
    files = {"org/m": {p.name: p.read_bytes() for p in tiny_checkpoint.iterdir()}}
    files["org/m"]["config.json"] = json.dumps({"hidden_size": 896}).encode()
    for repo, tensors in {
        "org/gated": {"linear.weight": torch.ones(16, 16)},
        "org/pixel-lora": {"lora_unet_linear.lora_down.weight": torch.ones(4, 16),
                           "lora_unet_linear.lora_up.weight": torch.ones(16, 4)},
        "org/decider": {"classifier.weight": torch.ones(2, 384)},
    }.items():
        path = tmp_path / (repo.split("/")[1] + ".safetensors")
        save_file(tensors, path)
        files[repo] = {"model.safetensors": path.read_bytes()}
    files["org/decider"]["config.json"] = json.dumps({
        "architectures": ["DebertaV2ForSequenceClassification"],
        "num_labels": 2, "id2label": {"0": "no", "1": "yes"},
    }).encode()
    files["org/convnext"] = {"pytorch_model.bin": b"never unpickle"}
    infos = {}
    for repo, blobs in files.items():
        card = {"base_model": "Qwen/Qwen2.5-0.5B", "base_model_relation": "finetune", "license": "apache-2.0"} if repo == "org/m" else {}
        infos[repo] = SimpleNamespace(
            sha=SHA, gated=repo == "org/gated", card_data=card,
            tags=["system-one"] if repo == "org/decider" else [],
            siblings=[SimpleNamespace(rfilename=name, size=len(blob),
                                      lfs={"sha256": hashlib.sha256(blob).hexdigest()},
                                      xet_hash=None) for name, blob in sorted(blobs.items())],
        )
    return files, infos


def _manifest_api(manifest_repositories, repos):
    _, infos = manifest_repositories

    class FakeApi:
        def model_info(self, repo_id, revision="main", files_metadata=False, **kwargs):
            assert repo_id in repos
            assert revision in ("main", SHA)
            return infos[repo_id]

    return FakeApi()


@pytest.fixture
def fake_api(manifest_repositories):
    return _manifest_api(manifest_repositories, {"org/m", "org/gated"})


@pytest.fixture
def bin_only_api(manifest_repositories):
    return _manifest_api(manifest_repositories, {"org/convnext"})


@pytest.fixture
def kohya_api(manifest_repositories):
    return _manifest_api(manifest_repositories, {"org/pixel-lora"})


@pytest.fixture
def decider_api(manifest_repositories):
    return _manifest_api(manifest_repositories, {"org/decider"})


@pytest.fixture
def fake_reader(manifest_repositories):
    from evidence.hub import HubReader

    files, _ = manifest_repositories
    urls = {}

    def transport(method, url, headers):
        if method == "HEAD":
            repo, rest = urlsplit(url).path.lstrip("/").split("/resolve/", 1)
            revision, filename = rest.split("/", 1)
            assert revision == SHA
            assert "Authorization" in headers
            blob = files[repo][filename]
            cdn = "https://cdn.example/" + repo + "/" + filename
            urls[cdn] = blob
            return 302, {"Location": cdn, "x-linked-size": str(len(blob)),
                         "x-linked-etag": hashlib.sha256(blob).hexdigest()}, b""
        assert "Authorization" not in headers
        blob = urls[url]
        start, end = map(int, headers["Range"][6:].split("-"))
        return 206, {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}, blob[start:end + 1]

    return HubReader(token="hf_secret_marker", transport=transport)


@pytest.fixture
def denying_reader():
    from evidence.hub import AccessDenied, HubReader

    class DenyingReader(HubReader):
        def resolve(self, repo_id, filename, revision):
            raise AccessDenied("Fixture denies every lookup")

    return DenyingReader(token="hf_secret_marker")


def _fingerprint_manifest(root, tensors, config=None, adapter_config=None):
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest

    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config or {}))
    if adapter_config is not None:
        (root / "adapter_config.json").write_text(json.dumps(adapter_config))
    return build_manifest(str(root), "main", HubReader(), api=None)


@pytest.fixture
def local_manifest(tmp_path):
    generator = torch.Generator().manual_seed(600)
    return _fingerprint_manifest(tmp_path / "local", {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 64, generator=generator).to(torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(768, 64, generator=generator),
        "model.norm.weight": torch.ones(64),
        "model.position_ids": torch.arange(768, dtype=torch.int64),
    })


@pytest.fixture
def parent_manifest(tmp_path):
    generator = torch.Generator().manual_seed(601)
    return _fingerprint_manifest(tmp_path / "parent", {
        "model.embed_tokens.weight": torch.randn(4096, 64, generator=generator),
    })


@pytest.fixture
def child_with_extended_vocab(parent_manifest, tmp_path):
    from safetensors.torch import load_file

    generator = torch.Generator().manual_seed(602)
    weight = load_file(Path(parent_manifest["repo_id"]) / "model.safetensors")["model.embed_tokens.weight"]
    update = torch.randn(4096, 4, generator=generator) @ torch.randn(4, 64, generator=generator)
    child = torch.cat((weight + .05 * update, torch.randn(8, 64, generator=generator)))
    return _fingerprint_manifest(tmp_path / "child", {"model.embed_tokens.weight": child})


@pytest.fixture
def moe_manifest(tmp_path):
    generator = torch.Generator().manual_seed(603)
    return _fingerprint_manifest(tmp_path / "moe", {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 1024, 512, generator=generator),
    }, {"num_experts": 4, "hidden_size": 512})


@pytest.fixture
def lora_manifest(tmp_path):
    generator = torch.Generator().manual_seed(604)
    return _fingerprint_manifest(tmp_path / "lora", {
        "base_model.model.model.layers.0.q_proj.lora_A.weight": torch.randn(8, 64, generator=generator),
        "base_model.model.model.layers.0.q_proj.lora_B.weight": torch.randn(512, 8, generator=generator),
    }, adapter_config={"r": 8, "lora_alpha": 8})


# X8 CLI helpers keep every checkpoint and generated artifact under ZOO_DATA.
def run_cli(args):
    import contextlib
    import io
    from evidence.cli import main

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = main(args)
    assert code == 0, stream.getvalue()
    return json.loads(stream.getvalue())


def run_cli_code(args):
    import contextlib
    import io
    from evidence.cli import main

    with contextlib.redirect_stdout(io.StringIO()):
        return main(args)


@pytest.fixture
def cli_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOO_DATA", str(tmp_path))
    generator = torch.Generator().manual_seed(608)
    parent = {"model.embed_tokens.weight": torch.randn(4096, 64, generator=generator),
              "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 64, generator=generator)}
    child = {name: weight + .05 * (torch.randn(weight.shape[0], 4, generator=generator)
                                  @ torch.randn(4, 64, generator=generator))
             for name, weight in parent.items()}
    child["model.embed_tokens.weight"] = torch.cat((child["model.embed_tokens.weight"],
                                                    torch.randn(8, 64, generator=generator)))
    unrelated = {name: torch.randn(weight.shape, generator=generator) for name, weight in parent.items()}
    paths = {}
    for label, tensors in (("parent", parent), ("child", child), ("unrelated", unrelated)):
        root = tmp_path / "lab" / label
        root.mkdir(parents=True)
        save_file(tensors, root / "model.safetensors")
        (root / "lineage.json").write_text(json.dumps({
            "name": "lab/" + label, "bases": [] if label == "parent" else ["lab/parent"],
            "relation": "finetune"}))
        paths[label] = root
    return paths


@pytest.fixture
def child_dir(cli_checkpoints):
    return cli_checkpoints["child"]


@pytest.fixture
def parent_dir(cli_checkpoints):
    return cli_checkpoints["parent"]


@pytest.fixture
def unrelated_dir(cli_checkpoints):
    return cli_checkpoints["unrelated"]


@pytest.fixture
def null_file(tmp_path):
    path = tmp_path / "null.json"
    path.write_text(json.dumps({"statistic": "median_cosine", "sketch_k": 64,
                                "row_block": 256, "tier": "t1", "n_pairs": 99,
                                "null_quantiles": [-.1, 0., .1], "related_threshold": .9}))
    return path


@pytest.fixture
def models_csv(cli_checkpoints, tmp_path):
    import pandas as pd
    path = tmp_path / "models.csv"
    pd.DataFrame([{"repo_id": "lab/" + label, "revision": "", "stratum": "text:<1B", "modality": "text"}
                  for label in ("parent", "child")]).to_csv(path, index=False)
    return path


@pytest.fixture
def models_csv_with_denied(tmp_path):
    import pandas as pd
    path = tmp_path / "denied.csv"
    pd.DataFrame([{"repo_id": "org/gated", "revision": SHA}]).to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_run(tmp_path):
    """Actual tiny T1/T2 artifacts; inventory and measurements stay distinct."""
    import itertools
    import pandas as pd
    from evidence.fingerprint import fingerprint_model, load_fingerprint
    from evidence.hub import HubReader
    from evidence.manifest import write_manifest

    root = tmp_path / "pilot"
    root.mkdir()
    t1, t2 = root / "t1", root / "t2"
    generator = torch.Generator().manual_seed(909)
    base = torch.randn(512, 64, generator=generator)

    def update(rank, scale):
        return scale * (torch.randn(512, rank, generator=generator) @
                        torch.randn(rank, 64, generator=generator))

    matrices = {"parent": base, "child4": base + update(4, .015),
                "child32": base + update(32, .025), "sibling1": base + update(4, .05),
                "sibling2": base + update(4, .08)}
    matrices["grandchild"] = matrices["child4"] + update(4, .04)
    matrices.update({f"independent{i}": torch.randn(512, 64, generator=generator) for i in range(3)})
    keys, records = {}, []
    for name, matrix in matrices.items():
        manifest = _fingerprint_manifest(root / name, {"linear.weight": matrix})
        write_manifest(manifest, root / "manifests")
        keys[name] = f"{manifest['repo_id']}@{manifest['revision']}"
        fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "t1", t1))
        fingerprint_model(manifest, HubReader(), "t2", t2, also_t1_blocks=(128, 256, 512))
        records.append(dict(repo_id=manifest["repo_id"], revision=manifest["revision"],
                            status="complete", reason="", stratum="text:<1B", modality="text",
                            params=matrix.numel(), **{k: fp.meta[k] for k in
                            ("bytes_read", "range_reads", "resolves", "elapsed_seconds")}))
    pairs = [(child, "parent", "parent") for child in ("child4", "child32", "sibling1", "sibling2")]
    pairs += [("grandchild", "child4", "parent"), ("grandchild", "parent", "grandparent"),
              ("sibling1", "sibling2", "sibling"), ("child4", "child32", "sibling")]
    pairs += [(a, b, "independent") for a, b in itertools.combinations(
        ("independent0", "independent1", "independent2"), 2)]
    pairs = pd.DataFrame([dict(a=keys[a], b=keys[b], label=label, source="lab") for a, b, label in pairs])
    pairs.to_csv(root / "pairs.csv", index=False)
    (t1 / "summary.json").write_text(json.dumps({"stage": "batch", "models": records}))
    (t2 / "summary.json").write_text(json.dumps({"stage": "batch", "models": records}))
    return SimpleNamespace(root=root, t1=t1, t2=t2, pairs=pairs)


@pytest.fixture
def run_dir(manifest_repositories, cli_checkpoints, tmp_path):
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest, write_manifest
    root = tmp_path / "run"
    for path in cli_checkpoints.values():
        write_manifest(build_manifest(str(path), "main", HubReader(), None), root / "manifests")
    return root
