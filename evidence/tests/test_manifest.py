"""Offline T0 manifests and file selection."""

import pytest
import json

from evidence.hub import HubReader
from evidence.manifest import build_manifest, pin_revision, select_file_set, write_manifest

SHA = "a" * 40


def test_file_set_selection_dedupes():
    sdxl = ["sd_xl_base_1.0.safetensors", "sd_xl_base_1.0_0.9vae.safetensors",
            "unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/model.safetensors", "vae/diffusion_pytorch_model.safetensors",
            "sd_xl_offset_example-lora_1.0.safetensors", "model_index.json"]
    keep, skipped = select_file_set(sdxl)
    assert keep == ["text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors",
                    "vae/diffusion_pytorch_model.safetensors"]
    assert {s["reason"] for s in skipped} == {"duplicate_layout", "bundled_adapter"}
    keep, _ = select_file_set(["consolidated.safetensors", "model-00001-of-00002.safetensors",
                               "model-00002-of-00002.safetensors", "model.safetensors.index.json"])
    assert "consolidated.safetensors" not in keep
    keep, _ = select_file_set(["model.safetensors", "original/model.safetensors"])
    assert keep == ["model.safetensors"]
    keep, skipped = select_file_set(["diffusion_models/ltx-22b-dev-transformer-bf16.safetensors",
                                     "diffusion_models/ltx-22b-dev-transformer-comfy-int8-convrot.safetensors",
                                     "diffusion_models/ltx-22b-dev-transformer-nvfp4.safetensors",
                                     "loras/ltx-22b-distilled-lora-450-bf16.safetensors", "vocoder.pt"])
    assert keep == ["diffusion_models/ltx-22b-dev-transformer-bf16.safetensors"]
    assert {s["reason"] for s in skipped} == {"precision_variant", "bundled_adapter", "pickle_not_read"}

def test_decision_facts(fake_reader, decider_api):
    m = build_manifest("org/decider", SHA, fake_reader, decider_api)
    assert m["decision"]["system_one"] and m["decision"]["decision_type"] == "binary"
    assert m["decision"]["num_labels"] == 2 and m["view_counts"]["head"] == 1

def test_manifest_from_sharded_repo(fake_api, fake_reader):
    m = build_manifest("org/m", SHA, fake_reader, fake_api)
    assert m["access"] == "granted" and m["config"]["hidden_size"] == 896
    assert m["declared_bases"] == ["Qwen/Qwen2.5-0.5B"] and m["base_model_relation"] == "finetune"

def test_manifest_no_access_and_bin_only(fake_api, denying_reader, bin_only_api):
    assert build_manifest("org/gated", SHA, denying_reader, fake_api)["access"] == "no_access"
    m = build_manifest("org/convnext", SHA, denying_reader, bin_only_api)
    assert m["access"] == "t0_only" and m["weight_format"] == "bin" and m["tensors"] == []

def test_kohya_adapter_detected_without_config(fake_reader, kohya_api):
    m = build_manifest("org/pixel-lora", SHA, fake_reader, kohya_api)
    assert m["is_adapter"] and m["view_counts"]["lora_product"] > 0

def test_local_directory_is_content_pinned(tiny_checkpoint):
    a = build_manifest(str(tiny_checkpoint), "local", HubReader(), api=None)
    assert a["revision"].startswith("local:") and len(a["revision"]) == 70
    assert build_manifest(str(tiny_checkpoint), "local", HubReader(), api=None)["revision"] == a["revision"]

def test_manifest_rejects_unpinned(fake_api, fake_reader):
    with pytest.raises(ValueError):
        build_manifest("org/m", "main", fake_reader, fake_api)


def test_index_defines_nonstandard_shard_set(tiny_checkpoint):
    original = tiny_checkpoint / "model-00001-of-00002.safetensors"
    chosen = tiny_checkpoint / "selected.safetensors"
    chosen.write_bytes(original.read_bytes())
    (tiny_checkpoint / "model.safetensors.index.json").unlink()
    (tiny_checkpoint / "weights.index.json").write_text(json.dumps({
        "weight_map": {"some_weight": "selected.safetensors"},
    }))
    m = build_manifest(str(tiny_checkpoint), "local", HubReader(), None)
    assert [f["filename"] for f in m["files"]] == ["selected.safetensors"]
    assert {f["filename"] for f in m["skipped_files"]} == {
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    }


def test_diffusers_keeps_one_file_per_component():
    keep, skipped = select_file_set([
        "model_index.json", "unet/diffusion_pytorch_model.safetensors",
        "unet/model.safetensors", "text_encoder/model.safetensors",
    ])
    assert keep == ["text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors"]
    assert skipped == [{"filename": "unet/model.safetensors", "reason": "duplicate_layout"}]


def test_local_identity_routing_and_config_facts(tiny_checkpoint, monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("Local T0 must never use the network")

    monkeypatch.setenv("ZOO_DATA", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tiny_checkpoint / "lineage.json").write_text(json.dumps({
        "name": "lab/tiny", "bases": ["lab/base"], "relation": "finetune",
    }))
    (tiny_checkpoint / "config.json").write_text(json.dumps({
        "text_config": {"hidden_size": 16, "num_local_experts": 8},
        "architectures": ["ExampleForTokenClassification"],
        "num_labels": 100, "id2label": {str(i): str(i) for i in range(100)},
    }))
    reader = HubReader(transport=no_network)
    m = build_manifest(str(tiny_checkpoint), "local", reader, None)
    assert m["repo_id"] == "lab/tiny" and m["declared_bases"] == ["lab/base"]
    assert m["config"]["hidden_size"] == 16 and m["config"]["num_experts"] == 8
    assert m["decision"]["decision_type"] == "span" and len(m["decision"]["labels"]) == 64
    assert m["decision"]["is_decision"] and "architectures" not in m["config"]
    if "source" in m:
        from pathlib import Path
        assert (tmp_path / m["source"]["zoo_data_path"]).resolve() == tiny_checkpoint.resolve()
        assert not Path(m["source"]["zoo_data_path"]).is_absolute()
    old_pin = m["revision"]
    (tiny_checkpoint / "lineage.json").write_text('{"name":"lab/changed"}')
    assert build_manifest(str(tiny_checkpoint), "local", reader, None)["revision"] != old_pin


def test_pin_and_atomic_manifest_name(fake_api, fake_reader, tmp_path):
    from measurement_config import safe_filename

    assert pin_revision(fake_api, "org/m") == SHA
    m = build_manifest("org/m", SHA, fake_reader, fake_api)
    out_dir = tmp_path / "manifests"
    target = write_manifest(m, out_dir)
    assert target.name == safe_filename(f"org/m@{SHA}") + ".manifest.json"
    assert json.loads(target.read_text())["revision"] == SHA
    target.write_text("interrupted")
    assert json.loads(write_manifest(m, out_dir).read_text())["access"] == "granted"
    assert len(list(out_dir.iterdir())) == 1


def test_oversized_json_is_not_read(fake_api, fake_reader, manifest_repositories):
    from evidence.hub import RemoteFile
    from evidence.manifest import MAX_JSON_BYTES

    original = fake_reader.resolve

    def resolve(repo_id, filename, revision):
        if filename == "config.json":
            return RemoteFile(repo_id, filename, revision, "unused", MAX_JSON_BYTES + 1, None, None)
        return original(repo_id, filename, revision)

    fake_reader.resolve = resolve
    with pytest.raises(ValueError, match="8 MiB"):
        build_manifest("org/m", SHA, fake_reader, fake_api)
    assert fake_reader.stats["bytes_read"] == 0


def test_diffusers_component_tensor_identities_and_pairing(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    import evidence.manifest as module

    (tmp_path / "model_index.json").write_text("{}")
    planned = []
    real_plan = module.plan_views

    def capture(*args, **kwargs):
        result = real_plan(*args, **kwargs)
        planned.extend(result[0])
        return result

    monkeypatch.setattr(module, "plan_views", capture)
    originals = {}
    for i, component in enumerate(("text_encoder", "text_encoder_2")):
        folder = tmp_path / component
        folder.mkdir()
        shards = [
            {"shared.weight": torch.ones(16, 16),
             "scaled.weight": torch.ones(16, 16).to(torch.float8_e4m3fn),
             "module.lora_A.weight": torch.ones(4, 16)},
            {"scaled.weight_scale_inv": torch.ones(1),
             "module.lora_B.weight": torch.ones(16, 4)},
        ]
        shards[i]["unpaired.lora_A.weight" if i == 0 else "unpaired.lora_B.weight"] = torch.ones(4, 16)
        mapping = {}
        for j, tensors in enumerate(shards, 1):
            filename = f"model-{j:05d}-of-00002.safetensors"
            save_file(tensors, folder / filename)
            mapping.update({name: filename for name in tensors})
            reader = HubReader()
            remote = reader.resolve(str(tmp_path), f"{component}/{filename}", "local")
            for name, entry in reader.read_header(remote).items():
                originals[f"{component}/{name}"] = entry
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    m = build_manifest(str(tmp_path), "local", HubReader(), None)
    assert m["access"] == "granted"
    assert {t["name"] for t in m["tensors"]} == set(originals)
    for tensor in m["tensors"]:
        original = originals[tensor["name"]]
        assert (tensor["file"], tensor["start"], tensor["end"]) == (original.file, original.start, original.end)
    products = [v for v in planned if v.geometry == "lora_product"]
    assert len(products) == 2
    assert all("unpaired" not in v.name for v in products)
    for view in planned:
        component = view.name.split("/", 1)[0]
        assert all(s.startswith(component + "/") for s in view.sources)
        assert not view.component.startswith(component + "/" + component + "/")
        assert view._source_shapes
    assert len([v for v in planned if v.decode == "fp8_scaled"]) == 2


def test_root_shard_names_and_lora_pairing_are_preserved(tmp_path):
    import torch
    from safetensors.torch import save_file

    tensors = [{"module.lora_A.weight": torch.ones(4, 16)},
               {"module.lora_B.weight": torch.ones(16, 4)}]
    mapping = {}
    for i, shard in enumerate(tensors, 1):
        filename = f"model-{i:05d}-of-00002.safetensors"
        save_file(shard, tmp_path / filename)
        mapping.update({name: filename for name in shard})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    m = build_manifest(str(tmp_path), "local", HubReader(), None)
    assert {t["name"] for t in m["tensors"]} == set(mapping)
    assert m["view_counts"]["lora_product"] == 1


@pytest.mark.parametrize("alternate", ["fp16", "original", "bundled"])
def test_one_index_layout_is_read(tmp_path, alternate):
    import torch
    from safetensors.torch import save_file

    chosen_files = ["part-a.safetensors", "part-b.safetensors"]
    rejected_files = (["part-c.safetensors", "part-d.safetensors"] if alternate == "fp16" else
                      [f"{'original' if alternate == 'original' else 'loras'}/part-{i}.safetensors" for i in ("c", "d")])
    for group in (chosen_files, rejected_files):
        for i, filename in enumerate(group):
            path = tmp_path / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({f"layer{i}.weight": torch.ones(16, 16)}, path)
    (tmp_path / "weights.index.json").write_text(json.dumps({
        "weight_map": {str(i): name for i, name in enumerate(chosen_files)},
    }))
    other_index = ("weights.fp16.index.json" if alternate == "fp16" else
                   ("original" if alternate == "original" else "loras") + "/weights.index.json")
    from pathlib import Path
    (tmp_path / other_index).write_text(json.dumps({
        "weight_map": {str(i): Path(name).name for i, name in enumerate(rejected_files)},
    }))

    class TrackingReader(HubReader):
        def __init__(self):
            super().__init__()
            self.headers = []

        def read_header(self, remote):
            self.headers.append(remote.filename)
            return super().read_header(remote)

    reader = TrackingReader()
    m = build_manifest(str(tmp_path), "local", reader, None)
    assert reader.headers == chosen_files
    assert [f["filename"] for f in m["files"]] == chosen_files
    assert {f["filename"] for f in m["skipped_files"]} == set(rejected_files)
    assert all(filename != other_index for _, filename, _ in reader._files)


def test_explicit_precision_precedes_untagged_but_diffusers_prefers_nonvariant():
    keep, skipped = select_file_set(["model.safetensors", "model-bf16.safetensors"])
    assert keep == ["model-bf16.safetensors"]
    assert skipped == [{"filename": "model.safetensors", "reason": "precision_variant"}]
    keep, _ = select_file_set(["model_index.json", "unet/model.safetensors", "unet/model-bf16.safetensors"])
    assert keep == ["unet/model.safetensors"]


@pytest.mark.parametrize("canonical", ["model.safetensors", "adapter_model.safetensors"])
def test_canonical_root_layout_excludes_nested_checkpoint_alternatives(canonical):
    keep, skipped = select_file_set([
        canonical, f"checkpoint-10/{canonical}", f"checkpoint-20/{canonical}",
        f"sft/{canonical}",
    ])
    assert keep == [canonical]
    assert {item["filename"] for item in skipped} == {
        f"checkpoint-10/{canonical}", f"checkpoint-20/{canonical}", f"sft/{canonical}",
    }
    assert {item["reason"] for item in skipped} == {"duplicate_layout"}


def test_canonical_root_model_excludes_unindexed_precision_shards():
    keep, skipped = select_file_set([
        "model.safetensors", "model.fp32-00001-of-00002.safetensors",
        "model.fp32-00002-of-00002.safetensors",
    ])
    assert keep == ["model.safetensors"]
    assert len(skipped) == 2
    assert {item["reason"] for item in skipped} == {"duplicate_layout"}


def test_adapter_config_identifies_bundled_adapter_folders():
    keep, skipped = select_file_set([
        "model.safetensors", "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json", "adapter_vllm/adapter_model.safetensors",
        "adapter_vllm/adapter_config.json",
    ])
    assert keep == ["model.safetensors"]
    assert {item["reason"] for item in skipped} == {"bundled_adapter"}


def test_training_state_does_not_turn_adapter_only_repo_into_bundled_adapter():
    keep, skipped = select_file_set([
        "adapter_model.safetensors", "adapter_config.json", "training_args.bin",
        "checkpoint-10/adapter_model.safetensors", "checkpoint-10/adapter_config.json",
        "checkpoint-10/optimizer.pt", "checkpoint-10/scheduler.pt",
    ])
    assert keep == ["adapter_model.safetensors"]
    reasons = {item["filename"]: item["reason"] for item in skipped}
    assert reasons["checkpoint-10/adapter_model.safetensors"] == "duplicate_layout"
    assert reasons["training_args.bin"] == "pickle_not_read"


def test_canonical_root_adapter_survives_excluded_nested_base_checkpoint():
    keep, skipped = select_file_set([
        "adapter_model.safetensors", "adapter_config.json", "training_args.bin",
        "checkpoint-10/model.safetensors",
    ])
    assert keep == ["adapter_model.safetensors"]
    assert skipped == [
        {"filename": "checkpoint-10/model.safetensors", "reason": "duplicate_layout"},
        {"filename": "training_args.bin", "reason": "pickle_not_read"},
    ]


@pytest.mark.parametrize("indexed", [False, True])
def test_root_layout_prevents_nested_index_restoring_alternate(tmp_path, indexed):
    import torch
    from safetensors.torch import save_file

    root_file = "part-a.safetensors" if indexed else "model.safetensors"
    save_file({"layer.weight": torch.ones(16, 16)}, tmp_path / root_file)
    nested = tmp_path / "checkpoint-20"
    nested.mkdir()
    save_file({"layer.weight": torch.ones(16, 16)}, nested / "part-b.safetensors")
    (nested / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.weight": "part-b.safetensors"},
    }))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"layer.weight": root_file},
        }))

    manifest = build_manifest(str(tmp_path), "local", HubReader(), None)
    assert [item["filename"] for item in manifest["files"]] == [root_file]
    assert [item["name"] for item in manifest["tensors"]] == ["layer.weight"]
    assert manifest["skipped_files"] == [{
        "filename": "checkpoint-20/part-b.safetensors", "reason": "duplicate_layout",
    }]


def test_collection_without_canonical_root_remains_explicitly_ambiguous(tmp_path):
    import torch
    from safetensors.torch import save_file

    for component in ("checkpoint-10", "checkpoint-20"):
        folder = tmp_path / component
        folder.mkdir()
        save_file({"layer.weight": torch.ones(16, 16)}, folder / "model.safetensors")
    with pytest.raises(ValueError, match="duplicate tensor names"):
        build_manifest(str(tmp_path), "local", HubReader(), None)


def test_bundled_root_adapter_does_not_disable_base_index_membership(tmp_path):
    import torch
    from safetensors.torch import save_file

    for filename in ("part-a.safetensors", "unused.safetensors"):
        save_file({"layer.weight": torch.ones(16, 16)}, tmp_path / filename)
    save_file({"layer.lora_A.weight": torch.ones(4, 16)}, tmp_path / "adapter_model.safetensors")
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.weight": "part-a.safetensors"},
    }))
    manifest = build_manifest(str(tmp_path), "local", HubReader(), None)
    assert [item["filename"] for item in manifest["files"]] == ["part-a.safetensors"]


@pytest.mark.parametrize("auxiliary", ["head", "classifier", "score"])
@pytest.mark.parametrize("indexed", [False, True])
def test_canonical_index_requires_explicit_auxiliary_head_membership(tmp_path, auxiliary, indexed):
    import torch
    from safetensors.torch import save_file

    save_file({"layer.weight": torch.ones(16, 16)}, tmp_path / "part-a.safetensors")
    save_file({"proj.weight": torch.ones(24, 16)}, tmp_path / f"{auxiliary}.safetensors")
    mapping = {"layer.weight": "part-a.safetensors"}
    if indexed:
        mapping["proj.weight"] = f"{auxiliary}.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    if indexed:
        manifest = build_manifest(str(tmp_path), "local", HubReader(), None)
        assert {item["name"] for item in manifest["tensors"]} == {"layer.weight", "proj.weight"}
        assert manifest["skipped_files"] == []
    else:
        with pytest.raises(ValueError, match="auxiliary.*ambiguous"):
            build_manifest(str(tmp_path), "local", HubReader(), None)


def test_manifest_from_local_download_matches_remote_manifest(fake_reader, fake_api, manifest_repositories, tmp_path, monkeypatch):
    files, _ = manifest_repositories
    paths = {}
    for filename, blob in files["org/m"].items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        paths[filename] = str(path)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo, filename, **kwargs: paths.get(filename))
    expected = build_manifest("org/m", SHA, fake_reader, fake_api)
    reader = HubReader(local_cache=True, transport=lambda *args: pytest.fail("network read"))
    actual = build_manifest("org/m", SHA, reader, fake_api)
    assert actual == expected


@pytest.mark.parametrize("token", [None, "hf_secret_marker"])
def test_manifest_local_snapshot_without_token_keeps_hub_pin(tiny_checkpoint, monkeypatch, token):
    if token:
        monkeypatch.setenv("HF_TOKEN", token)
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: str(tiny_checkpoint))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo, filename, **kwargs: str(tiny_checkpoint / filename))
    reader = HubReader(local_cache=True, transport=lambda *args: pytest.fail("network"))
    manifest = build_manifest("org/m", SHA, reader, None)
    assert manifest["access"] == "granted"
    assert manifest["repo_id"] == "org/m" and manifest["revision"] == SHA
    assert manifest["params_by_dtype"]["BF16"] > 0 and manifest["tensors"]
