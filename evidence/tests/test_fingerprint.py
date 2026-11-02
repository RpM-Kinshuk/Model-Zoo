from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import torch

from measurement_config import safe_filename
from evidence.fingerprint import fingerprint_model, fingerprint_tensors, load_fingerprint
from evidence.hub import HubReader
from evidence.sketch import sketch_cosine


def test_t1_reads_only_blocks_and_records_skips(local_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "t1", tmp_path))
    assert "model.norm.weight" not in fp.names and fp.sketch.shape[1:] == (64, 64)
    reasons = {c["name"]: c["reason"] for c in fp.coverage}
    assert reasons["model.norm.weight"] == "not_a_matrix" and reasons["model.position_ids"] == "non_float_buffer"
    tensor_bytes = sum(t["end"] - t["start"] for t in local_manifest["tensors"])
    assert fp.meta["bytes_read"] < tensor_bytes

def test_t2_spectra_and_free_t1_blocks(local_manifest, tmp_path):
    fingerprint_model(local_manifest, HubReader(), "t2", tmp_path, also_t1_blocks=(128, 256))
    safe = safe_filename(f"{local_manifest['repo_id']}@{local_manifest['revision']}")
    with h5py.File(tmp_path / f"{safe}.t2.spectra.h5") as h:
        assert h.attrs["numerics_version"] == "7" and np.isfinite(h["layers/alpha"][:]).any()
        assert len(h["eigs"]) == len(h["layers/longname"])
    assert (tmp_path / f"{safe}.t1b128.fingerprint.safetensors").exists()
    assert (tmp_path / f"{safe}.t1.fingerprint.safetensors").exists()        # b == ROW_BLOCK

def test_revisions_do_not_collide(local_manifest, tmp_path):
    other = {**local_manifest, "revision": "local:" + "0" * 64}
    a = fingerprint_model(local_manifest, HubReader(), "t1", tmp_path)
    b = fingerprint_model(other, HubReader(), "t1", tmp_path)
    assert a != b and a.exists() and b.exists()

def test_delta_on_common_rows_despite_vocab_growth(child_with_extended_vocab, parent_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(child_with_extended_vocab, HubReader(), "t1", tmp_path,
                                            parent=parent_manifest, parent_cache=tmp_path / "cache"))
    i = fp.names.index("model.embed_tokens.weight")
    assert fp.delta["status"][i] in {"ok", "below_floor"}       # rows 1920:2176 exist in both

def test_moe_subset_and_lora_product(moe_manifest, lora_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(moe_manifest, HubReader(), "t1", tmp_path, sketch_k=128, all_experts=True))
    assert fp.sketch.shape[1:] == (128, 128) and fp.geometry.count("expert_slice") == 8   # 4 experts x gate/up
    fp = load_fingerprint(fingerprint_model(lora_manifest, HubReader(), "t1", tmp_path))
    i = fp.geometry.index("lora_product")
    assert np.count_nonzero(fp.spectra[i] > 1e-6 * fp.spectra[i].max()) == 8

def test_fingerprint_tensors_matches_file_path(local_manifest, tmp_path):
    from safetensors.torch import load_file
    tensors = {}
    for f in local_manifest["files"]:
        tensors.update(load_file(Path(local_manifest["repo_id"]) / f["filename"]))
    a = load_fingerprint(fingerprint_tensors(tensors.items(), tmp_path / "a.safetensors",
                                             {"repo_id": "x", "revision": "local:" + "0" * 64}))
    b = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "t1", tmp_path))
    i, j = a.names.index("model.layers.0.self_attn.q_proj.weight"), b.names.index("model.layers.0.self_attn.q_proj.weight")
    assert sketch_cosine(a.sketch[i], b.sketch[j]) > 0.9999

def test_block_hash_distinguishes_stored_dtype(tmp_path):
    bits = torch.full((32, 16), 0x3F80, dtype=torch.int16)
    identity = {"repo_id": "fixture/typed-hash", "revision": "local:" + "0" * 64}
    a = load_fingerprint(fingerprint_tensors(
        [("weight", bits.view(torch.bfloat16))], tmp_path / "bf16.safetensors", identity))
    b = load_fingerprint(fingerprint_tensors(
        [("weight", bits.view(torch.float16))], tmp_path / "f16.safetensors", identity))
    assert a.names == b.names == ["weight"]
    assert a.sha256[0] != b.sha256[0]  # Equal bytes encode 1.0 versus 1.875.

def test_atomic_write_leaves_no_partial_file(local_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr("evidence.fingerprint.two_sided_sketch", Mock(side_effect=RuntimeError))
    with pytest.raises(RuntimeError):
        fingerprint_model(local_manifest, HubReader(), "t1", tmp_path)
    assert not list(tmp_path.glob("*.safetensors"))


def test_expert_blocks_read_only_selected_rows(moe_manifest, tmp_path):
    reader = HubReader()
    fp = load_fingerprint(fingerprint_model(moe_manifest, reader, "t1", tmp_path))
    assert fp.meta["bytes_read"] == 8 * 256 * 512 * 4
    assert fp.meta["range_reads"] == 8
    assert fp.rows.tolist() == [[128, 384]] * 8


def test_fp8_partial_grid_matches_full_decode(tmp_path):
    import json
    from safetensors.torch import save_file
    from evidence.manifest import build_manifest
    from evidence.sketch import row_block_bounds, two_sided_sketch

    root = tmp_path / "fp8-source"
    root.mkdir()
    generator = torch.Generator().manual_seed(605)
    weight = torch.randn(576, 128, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.arange(1, 6).float().reshape(5, 1)
    save_file({"linear.weight": weight, "linear.weight_scale_inv": scales}, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"quantization_config": {"weight_block_size": [128, 128]}}))
    manifest = build_manifest(str(root), "main", HubReader(), None)
    fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "t1", tmp_path / "out"))
    first, last = row_block_bounds(576)
    decoded = weight.float() * scales.repeat_interleave(128, 0)[:576]
    np.testing.assert_allclose(fp.sketch[0], two_sided_sketch(decoded[first:last]), rtol=1e-5, atol=1e-5)
    assert fp.meta["bytes_read"] == 256 * 128 + 3 * 4
    reference = load_fingerprint(fingerprint_tensors([("linear.weight", decoded)], tmp_path / "ref.safetensors",
                                {"repo_id": "decoded", "revision": "local:" + "0" * 64}))
    assert fp.sha256 == reference.sha256


def test_t2_reuses_source_and_writes_canonical_metadata(local_manifest, tmp_path):
    import json
    from safetensors import safe_open

    reader = HubReader()
    path = fingerprint_model(local_manifest, reader, "t2", tmp_path, also_t1_blocks=(128, 256))
    fp = load_fingerprint(path)
    expected = sum(t["end"] - t["start"] for t in local_manifest["tensors"] if t["geometry"] == "matrix")
    assert fp.meta["bytes_read"] == expected
    with safe_open(path, framework="numpy") as handle:
        assert handle.metadata()["tier"] == "t2"
        assert handle.metadata()["revision"] == local_manifest["revision"]
    t1_path = path.with_name(path.name.replace(".t2.", ".t1."))
    direct = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "t1", tmp_path / "direct"))
    t1 = load_fingerprint(t1_path)
    assert t1.names == direct.names and t1.sha256 == direct.sha256
    np.testing.assert_array_equal(t1.sketch, direct.sketch)
    with safe_open(t1_path, framework="numpy") as handle:
        offsets = handle.get_tensor("spectrum_offsets")
        assert offsets.shape == (len(t1.names) + 1,) and offsets[0] == 0
        assert offsets[-1] == len(handle.get_tensor("spectrum_values"))
        assert (np.diff(offsets) >= 0).all()
    for spectrum in t1.spectra:
        assert (np.diff(spectrum) >= 0).all()


def test_named_local_parent_cache_is_portable_and_reused(child_with_extended_vocab, parent_manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("ZOO_DATA", str(tmp_path))
    child = dict(child_with_extended_vocab, repo_id="lab/child", source={"zoo_data_path": "child"})
    parent = dict(parent_manifest, repo_id="lab/parent", source={"zoo_data_path": "parent"})
    cache = tmp_path / "cache"
    first = load_fingerprint(fingerprint_model(child, HubReader(), "t1", tmp_path / "first", parent=parent, parent_cache=cache))
    assert first.delta["status"].tolist() == ["ok"]
    assert first.rows.tolist() == [[1920, 2176]]
    assert first.meta["bytes_read"] == 2 * 256 * 64 * 4
    assert len(list(cache.glob("*.bin"))) == 2
    second = load_fingerprint(fingerprint_model(child, HubReader(), "t1", tmp_path / "second", parent=parent, parent_cache=cache))
    assert second.meta["bytes_read"] == 256 * 64 * 4
    np.testing.assert_array_equal(first.delta["top_sv"], second.delta["top_sv"])
    assert str(tmp_path) not in str(second.meta)


def test_lora_hash_is_derived_f32_and_block_spectrum_is_squared(lora_manifest, tmp_path):
    from safetensors.torch import load_file

    fp = load_fingerprint(fingerprint_model(lora_manifest, HubReader(), "t1", tmp_path))
    tensors = load_file(Path(lora_manifest["repo_id"]) / "model.safetensors")
    a = tensors["base_model.model.model.layers.0.q_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.q_proj.lora_B.weight"]
    dense = load_fingerprint(fingerprint_tensors([("model.layers.0.q_proj.weight", b @ a)], tmp_path / "dense.safetensors",
                            {"repo_id": "dense", "revision": "local:" + "0" * 64}))
    assert fp.sha256 == dense.sha256
    np.testing.assert_allclose(fp.spectra[0][-8:], dense.spectra[0][-8:], rtol=2e-6)


def test_lora_nonbinary_scale_hashes_decoded_product(lora_manifest, tmp_path):
    from safetensors.torch import load_file

    manifest = dict(lora_manifest, adapter_config={"r": 8, "lora_alpha": 7.3})
    fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "t1", tmp_path))
    tensors = load_file(Path(manifest["repo_id"]) / "model.safetensors")
    a = tensors["base_model.model.model.layers.0.q_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.q_proj.lora_B.weight"]
    decoded = (b @ a) * (7.3 / 8)
    dense = load_fingerprint(fingerprint_tensors([("model.layers.0.q_proj.weight", decoded)], tmp_path / "scaled.safetensors",
                            {"repo_id": "dense", "revision": "local:" + "0" * 64}))
    assert fp.sha256 == dense.sha256


@pytest.mark.parametrize("shape,dtype", [((1024, 64), torch.bfloat16), ((4, 1024, 64), torch.float16)])
def test_tensor_fused_geometry_matches_file(shape, dtype, tmp_path):
    import hashlib
    from safetensors.torch import save_file
    from evidence.manifest import build_manifest

    generator = torch.Generator().manual_seed(606)
    name = "model.layers.0.mlp." + ("experts." if len(shape) == 3 else "") + "gate_up_proj.weight"
    weight = torch.randn(shape, generator=generator).to(dtype)
    root = tmp_path / "fused-source"
    root.mkdir()
    save_file({name: weight}, root / "model.safetensors")
    manifest = build_manifest(str(root), "main", HubReader(), None)
    file_fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "t1", tmp_path / "file"))
    tensor_fp = load_fingerprint(fingerprint_tensors([(name, weight)], tmp_path / "tensor.safetensors", manifest))
    assert len(tensor_fp.names) == (8 if len(shape) == 3 else 2)
    assert tensor_fp.names == file_fp.names
    assert tensor_fp.geometry == file_fp.geometry
    np.testing.assert_array_equal(tensor_fp.rows, file_fp.rows)
    np.testing.assert_array_equal(tensor_fp.shape, file_fp.shape)
    np.testing.assert_array_equal(tensor_fp.sketch, file_fp.sketch)
    assert tensor_fp.sha256 == file_fp.sha256
    dtype_name = "BF16" if dtype == torch.bfloat16 else "F16"
    for index, view_name in enumerate(tensor_fp.names):
        source = weight[int(view_name.rsplit("[", 1)[1][:-1])] if len(shape) == 3 else weight
        offset = 512 if "up_proj" in view_name else 0
        first, last = tensor_fp.rows[index]
        block = source[first + offset:last + offset].contiguous()
        payload = block.view(torch.uint8).numpy().tobytes()
        expected = hashlib.sha256(b"evidence-block-v1\0" + dtype_name.encode("ascii") + b"\0" + payload).hexdigest()
        assert tensor_fp.sha256[index] == expected


@pytest.mark.parametrize("api", ["file", "tensor"])
def test_empty_matrices_keep_valid_inputs_and_existing_skip_reason(api, tmp_path):
    from safetensors.torch import save_file
    from evidence.manifest import build_manifest

    generator = torch.Generator().manual_seed(607)
    tensors = {"empty.weight": torch.empty(0, 64),
               "empty_cols.weight": torch.empty(512, 0),
               "valid.weight": torch.randn(512, 64, generator=generator)}
    root = tmp_path / "empty-source"
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    manifest = build_manifest(str(root), "main", HubReader(), None)
    if api == "file":
        path = fingerprint_model(manifest, HubReader(), "t1", tmp_path / "file")
    else:
        path = fingerprint_tensors(tensors.items(), tmp_path / "tensor.safetensors", manifest)
    fp = load_fingerprint(path)
    assert fp.names == ["valid.weight"]
    coverage = {record["name"]: record for record in fp.coverage}
    for name in ("empty.weight", "empty_cols.weight"):
        assert coverage[name]["status"] == "skipped"
        assert coverage[name]["reason"] == "not_a_matrix"
    assert coverage["valid.weight"]["status"] == "measured"
    assert coverage["valid.weight"]["views"] == ["valid.weight"]


def test_t2_records_default_cpu_spectrum_device(local_manifest, tmp_path):
    path = fingerprint_model(local_manifest, HubReader(), "t2", tmp_path)
    with h5py.File(path.with_name(path.name.replace(".fingerprint.safetensors", ".spectra.h5"))) as handle:
        assert handle.attrs["compute_device"] == "cpu"


def test_t2_forwards_cuda_device_only_to_spectrum_collector(local_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module

    collector = module.compute_esd_for_weight
    devices = []

    def simulated_cuda(*args):
        assert args[1].device.type == "cpu"
        devices.append(args[10])
        result = collector(*args[:10], None, *args[11:])
        return dict(result, compute_device=f"cuda:{args[10]}")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(module, "compute_esd_for_weight", simulated_cuda)
    path = fingerprint_model(local_manifest, HubReader(), "t2", tmp_path,
                             compute_device="cuda:1", also_t1_blocks=(256,))
    assert devices and set(devices) == {1}
    with h5py.File(path.with_name(path.name.replace(".fingerprint.safetensors", ".spectra.h5"))) as handle:
        assert handle.attrs["compute_device"] == "cuda:1"


@pytest.mark.parametrize("device", ["bogus", "mps", "cpu:1", "cuda:-1", "cuda:3"])
def test_invalid_compute_device_rejected_before_extraction(device, local_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    reader = HubReader()
    before = dict(reader.stats)
    with pytest.raises(ValueError, match="[Dd]evice|CUDA"):
        fingerprint_model(local_manifest, reader, "t2", tmp_path / "out", compute_device=device)
    assert reader.stats == before
    assert not (tmp_path / "out").exists()


def test_unavailable_cuda_rejected_before_extraction(local_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reader = HubReader()
    with pytest.raises(ValueError, match="CUDA"):
        fingerprint_model(local_manifest, reader, "t2", tmp_path / "out", compute_device="cuda:0")
    assert reader.stats == dict(resolves=0, range_reads=0, bytes_read=0, refreshes=0, retries=0)
    assert not (tmp_path / "out").exists()
