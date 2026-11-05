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


def test_row_blocks_reads_only_blocks_and_records_skips(local_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path))
    assert "model.norm.weight" not in fp.names and fp.sketch.shape[1:] == (64, 64)
    reasons = {c["name"]: c["reason"] for c in fp.coverage}
    assert reasons["model.norm.weight"] == "not_a_matrix" and reasons["model.position_ids"] == "non_float_buffer"
    tensor_bytes = sum(t["end"] - t["start"] for t in local_manifest["tensors"])
    assert fp.meta["bytes_read"] < tensor_bytes

def test_full_weights_spectra_and_free_row_blocks_blocks(local_manifest, tmp_path):
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path, also_row_block_sizes=(128, 256))
    safe = safe_filename(f"{local_manifest['repo_id']}@{local_manifest['revision']}")
    with h5py.File(tmp_path / f"{safe}.full_weights.spectra.h5") as h:
        assert h.attrs["numerics_version"] == "7" and np.isfinite(h["layers/alpha"][:]).any()
        assert len(h["eigs"]) == len(h["layers/longname"])
    assert (tmp_path / f"{safe}.row_blocks_128.fingerprint.safetensors").exists()
    assert (tmp_path / f"{safe}.row_blocks.fingerprint.safetensors").exists()        # b == ROW_BLOCK

def test_revisions_do_not_collide(local_manifest, tmp_path):
    other = {**local_manifest, "revision": "local:" + "0" * 64}
    a = fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path)
    b = fingerprint_model(other, HubReader(), "row_blocks", tmp_path)
    assert a != b and a.exists() and b.exists()

def test_delta_on_common_rows_despite_vocab_growth(child_with_extended_vocab, parent_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(child_with_extended_vocab, HubReader(), "row_blocks", tmp_path,
                                            parent=parent_manifest, parent_cache=tmp_path / "cache"))
    i = fp.names.index("model.embed_tokens.weight")
    assert fp.delta["status"][i] in {"ok", "below_floor"}       # rows 1920:2176 exist in both

def test_moe_subset_and_lora_product(moe_manifest, lora_manifest, tmp_path):
    fp = load_fingerprint(fingerprint_model(moe_manifest, HubReader(), "row_blocks", tmp_path, sketch_k=128, all_experts=True))
    assert fp.sketch.shape[1:] == (128, 128) and fp.geometry.count("expert_slice") == 8   # 4 experts x gate/up
    fp = load_fingerprint(fingerprint_model(lora_manifest, HubReader(), "row_blocks", tmp_path))
    i = fp.geometry.index("lora_product")
    assert np.count_nonzero(fp.spectra[i] > 1e-6 * fp.spectra[i].max()) == 8

def test_fingerprint_tensors_matches_file_path(local_manifest, tmp_path):
    from safetensors.torch import load_file
    tensors = {}
    for f in local_manifest["files"]:
        tensors.update(load_file(Path(local_manifest["repo_id"]) / f["filename"]))
    a = load_fingerprint(fingerprint_tensors(tensors.items(), tmp_path / "a.safetensors",
                                             {"repo_id": "x", "revision": "local:" + "0" * 64}))
    b = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path))
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
        fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path)
    assert not list(tmp_path.glob("*.safetensors"))


def test_expert_blocks_read_only_selected_rows(moe_manifest, tmp_path):
    reader = HubReader()
    fp = load_fingerprint(fingerprint_model(moe_manifest, reader, "row_blocks", tmp_path))
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
    fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "row_blocks", tmp_path / "out"))
    first, last = row_block_bounds(576)
    decoded = weight.float() * scales.repeat_interleave(128, 0)[:576]
    np.testing.assert_allclose(fp.sketch[0], two_sided_sketch(decoded[first:last]), rtol=1e-5, atol=1e-5)
    assert fp.meta["bytes_read"] == 256 * 128 + 3 * 4
    reference = load_fingerprint(fingerprint_tensors([("linear.weight", decoded)], tmp_path / "ref.safetensors",
                                {"repo_id": "decoded", "revision": "local:" + "0" * 64}))
    assert fp.sha256 == reference.sha256


def test_full_weights_reuses_source_and_writes_canonical_metadata(local_manifest, tmp_path):
    import json
    from safetensors import safe_open

    reader = HubReader()
    path = fingerprint_model(local_manifest, reader, "full_weights", tmp_path, also_row_block_sizes=(128, 256))
    fp = load_fingerprint(path)
    expected = sum(t["end"] - t["start"] for t in local_manifest["tensors"] if t["geometry"] == "matrix")
    assert fp.meta["bytes_read"] == expected
    with safe_open(path, framework="numpy") as handle:
        assert handle.metadata()["tier"] == "full_weights"
        assert handle.metadata()["revision"] == local_manifest["revision"]
    row_blocks_path = path.with_name(path.name.replace(".full_weights.", ".row_blocks."))
    direct = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path / "direct"))
    row_blocks = load_fingerprint(row_blocks_path)
    assert row_blocks.names == direct.names and row_blocks.sha256 == direct.sha256
    np.testing.assert_array_equal(row_blocks.sketch, direct.sketch)
    with safe_open(row_blocks_path, framework="numpy") as handle:
        offsets = handle.get_tensor("spectrum_offsets")
        assert offsets.shape == (len(row_blocks.names) + 1,) and offsets[0] == 0
        assert offsets[-1] == len(handle.get_tensor("spectrum_values"))
        assert (np.diff(offsets) >= 0).all()
    for spectrum in row_blocks.spectra:
        assert (np.diff(spectrum) >= 0).all()


def test_named_local_parent_cache_is_portable_and_reused(child_with_extended_vocab, parent_manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("ZOO_DATA", str(tmp_path))
    child = dict(child_with_extended_vocab, repo_id="lab/child", source={"zoo_data_path": "child"})
    parent = dict(parent_manifest, repo_id="lab/parent", source={"zoo_data_path": "parent"})
    cache = tmp_path / "cache"
    first = load_fingerprint(fingerprint_model(child, HubReader(), "row_blocks", tmp_path / "first", parent=parent, parent_cache=cache, cache_self=True))
    assert first.delta["status"].tolist() == ["ok"]
    assert first.rows.tolist() == [[1920, 2176]]
    assert first.meta["bytes_read"] == 2 * 256 * 64 * 4
    assert len(list(cache.glob("*.bin"))) == 2
    second = load_fingerprint(fingerprint_model(child, HubReader(), "row_blocks", tmp_path / "second", parent=parent, parent_cache=cache, cache_self=True))
    assert second.meta["bytes_read"] == 256 * 64 * 4
    np.testing.assert_array_equal(first.delta["top_sv"], second.delta["top_sv"])
    assert str(tmp_path) not in str(second.meta)


def test_lora_hash_is_derived_f32_and_block_spectrum_is_squared(lora_manifest, tmp_path):
    from safetensors.torch import load_file

    fp = load_fingerprint(fingerprint_model(lora_manifest, HubReader(), "row_blocks", tmp_path))
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
    fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "row_blocks", tmp_path))
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
    file_fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "row_blocks", tmp_path / "file"))
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
        path = fingerprint_model(manifest, HubReader(), "row_blocks", tmp_path / "file")
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


def test_full_weights_records_default_cpu_spectrum_device(local_manifest, tmp_path):
    path = fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path)
    with h5py.File(path.with_name(path.name.replace(".fingerprint.safetensors", ".spectra.h5"))) as handle:
        assert handle.attrs["compute_device"] == "cpu"


def test_full_weights_forwards_cuda_device_only_to_spectrum_collector(local_manifest, tmp_path, monkeypatch):
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
    monkeypatch.setattr(module, "_to_compute", lambda tensor, device: tensor)
    path = fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path,
                             compute_device="cuda:1", also_row_block_sizes=(256,))
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
        fingerprint_model(local_manifest, reader, "full_weights", tmp_path / "out", compute_device=device)
    assert reader.stats == before
    assert not (tmp_path / "out").exists()


def test_unavailable_cuda_rejected_before_extraction(local_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reader = HubReader()
    with pytest.raises(ValueError, match="CUDA"):
        fingerprint_model(local_manifest, reader, "full_weights", tmp_path / "out", compute_device="cuda:0")
    assert reader.stats == dict(resolves=0, range_reads=0, bytes_read=0, refreshes=0, retries=0)
    assert not (tmp_path / "out").exists()


def test_parent_delta_cuda_forwarded_to_full_and_all_companion_blocks(
        local_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module
    collector, original_delta = module.compute_esd_for_weight, module.block_delta
    seen = []

    def simulated_spectrum(*args):
        result = collector(*args[:10], None, *args[11:])
        return dict(result, compute_device=f"cuda:{args[10]}")

    def simulated_delta(child, parent, dtype, **kwargs):
        seen.append((child.shape[0], kwargs.get("compute_device")))
        return original_delta(child, parent, dtype)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(module, "compute_esd_for_weight", simulated_spectrum)
    monkeypatch.setattr(module, "block_delta", simulated_delta)
    monkeypatch.setattr(module, "_to_compute", lambda tensor, device: tensor)
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path,
                      parent=local_manifest, compute_device="cuda:1",
                      also_row_block_sizes=(128, 256, 512), delta_blocks=(128, 256, 512), full_delta=True)
    assert seen
    assert all(torch.device(device) == torch.device("cuda:1") for _, device in seen)
    assert {128, 256, 512} <= {rows for rows, _ in seen}
    assert 768 in {rows for rows, _ in seen}  # Full full-weight matrix, not a companion.


def test_parent_delta_default_keeps_legacy_cpu_call(
        child_with_extended_vocab, parent_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module
    original = module.block_delta
    seen = []

    def legacy_delta(child, parent, dtype):
        seen.append(child.shape)
        return original(child, parent, dtype)

    monkeypatch.setattr(module, "block_delta", legacy_delta)
    fingerprint_model(child_with_extended_vocab, HubReader(), "row_blocks", tmp_path,
                      parent=parent_manifest)
    assert seen


def test_bounded_prefetch_reuses_parent_full_reads(local_manifest, tmp_path):
    class CountingReader(HubReader):
        def __init__(self):
            super().__init__()
            self.calls = []
        def read(self, remote, start, end):
            self.calls.append((remote.filename, start, end))
            return super().read(remote, start, end)
    reader = CountingReader()
    baseline = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "baseline",
                                                 parent=local_manifest, also_row_block_sizes=(128, 256, 512)))
    optimized = load_fingerprint(fingerprint_model(local_manifest, reader, "full_weights", tmp_path / "prefetched",
                                                  parent=local_manifest, also_row_block_sizes=(128, 256, 512),
                                                  prefetch_groups=2, prefetch_bytes=1024**2))
    assert len(reader.calls) == 4  # Two child + two parent source matrices, not four parent reads each.
    assert baseline.names == optimized.names and baseline.sha256 == optimized.sha256
    np.testing.assert_array_equal(baseline.sketch, optimized.sketch)
    for suffix in ("row_blocks_128", "row_blocks", "row_blocks_512"):
        name = baseline.meta["repo_id"] + "@" + baseline.meta["revision"]
        filename = safe_filename(name) + "." + suffix + ".fingerprint.safetensors"
        a, b = (load_fingerprint(tmp_path / folder / filename) for folder in ("baseline", "prefetched"))
        assert a.sha256 == b.sha256
        np.testing.assert_array_equal(a.sketch, b.sketch)
        for x, y in zip(a.spectra, b.spectra):
            np.testing.assert_array_equal(x, y)


def test_over_budget_experts_use_selected_view_reads(moe_manifest, tmp_path):
    class CountingReader(HubReader):
        def __init__(self):
            super().__init__()
            self.sizes = []
        def read(self, remote, start, end):
            self.sizes.append(end-start)
            return super().read(remote, start, end)
    reader = CountingReader()
    fp = load_fingerprint(fingerprint_model(moe_manifest, reader, "full_weights", tmp_path,
                                           all_experts=True, prefetch_groups=2, prefetch_bytes=1024))
    assert fp.geometry.count("expert_slice") == 8
    assert max(reader.sizes) <= 1024*512*4  # Never the entire four-expert source tensor.


@pytest.mark.parametrize("options", [{"prefetch_groups": 3}, {"prefetch_groups": -1},
                                     {"prefetch_bytes": 0}, {"prefetch_bytes": -1}])
def test_invalid_prefetch_options_leave_no_output(local_manifest, tmp_path, options):
    with pytest.raises(ValueError, match="prefetch"):
        fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "invalid", **options)
    assert not (tmp_path / "invalid").exists()


def test_full_cpu_workspace_fallback_keeps_small_companions_on_cuda(
        local_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module

    collector, add = module.compute_esd_for_weight, module._Records.add
    transferred, measured = {}, []

    def simulated_transfer(tensor, device):
        transferred[id(tensor)] = torch.device(device)
        return tensor

    def record(self, view, bounds, block, *args, **kwargs):
        measured.append((self.tier, block.shape[0], transferred.get(id(block))))
        return add(self, view, bounds, block, *args, **kwargs)

    def simulated_spectrum(*args):
        result = collector(*args[:10], None, *args[11:])
        return dict(result, compute_device=f"cuda:{args[10]}")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module, "_view_device", lambda block, device: torch.device("cpu"))
    monkeypatch.setattr(module, "_to_compute", simulated_transfer)
    monkeypatch.setattr(module._Records, "add", record)
    monkeypatch.setattr(module, "compute_esd_for_weight", simulated_spectrum)
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path,
                      compute_device="cuda:0", also_row_block_sizes=(128, 256, 512))
    assert measured
    assert all(device == torch.device("cpu") for tier, _, device in measured if tier == "full_weights")
    companions = [(rows, device) for tier, rows, device in measured if tier != "full_weights"]
    assert {rows for rows, _ in companions} == {128, 256, 512}
    assert all(device == torch.device("cuda:0") for _, device in companions)


@pytest.mark.parametrize("prefetch_groups", [0, 2])
def test_full_weights_with_parent_has_no_full_matrix_delta(local_manifest, tmp_path, prefetch_groups):
    path = fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path,
                             parent=local_manifest, also_row_block_sizes=(256,),
                             prefetch_groups=prefetch_groups)
    assert set(load_fingerprint(path).delta["status"]) == {"not_computed"}
    row_blocks = load_fingerprint(path.with_name(path.name.replace(".full_weights.", ".row_blocks.")))
    assert set(row_blocks.delta["status"]) == {"identical"}


@pytest.mark.parametrize("prefetch_groups", [0, 2])
def test_full_weights_parent_reads_only_row_blocks(local_manifest, tmp_path, monkeypatch, prefetch_groups):
    from evidence.fingerprint import _Sources
    reads = []
    original = _Sources.fetch
    parent = dict(local_manifest)
    def fetch(self, requests):
        requests = list(requests)
        if self.manifest is parent:
            reads.extend(requests)
        return original(self, requests)
    monkeypatch.setattr(_Sources, "fetch", fetch)
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path, parent=parent,
                      also_row_block_sizes=(256,), prefetch_groups=prefetch_groups)
    assert reads
    assert all(rows is not None and rows.stop - rows.start <= 256
               for _, _, _, rows in reads)


def test_parent_cache_holds_stored_dtype_row_blocks_only(local_manifest, tmp_path):
    cache = tmp_path / "cache"
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "out",
                      parent=local_manifest, parent_cache=cache, also_row_block_sizes=(256,))
    # Two 256-by-64 blocks: one bf16 and one fp32.
    assert sum(p.stat().st_size for p in cache.glob("*.bin")) == (
        256 * 64 * 2 + 256 * 64 * 4)


def test_stored_dtype_cache_gives_identical_delta(local_manifest, tmp_path):
    cache = tmp_path / "cache"
    a = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path / "a",
                                          parent=local_manifest, parent_cache=cache))
    b = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path / "b",
                                          parent=local_manifest, parent_cache=cache))
    for key in a.delta:
        np.testing.assert_equal(a.delta[key], b.delta[key])
    assert b.meta["bytes_read"] < a.meta["bytes_read"]


def test_delta_blocks_512_adds_512_row_deltas(local_manifest, tmp_path):
    path = fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path,
                             parent=local_manifest, also_row_block_sizes=(256, 512), delta_blocks=(256, 512))
    fp = load_fingerprint(path.with_name(path.name.replace(".full_weights.", ".row_blocks_512.")))
    assert set(fp.delta["status"]) == {"identical"}


@pytest.mark.parametrize("prefetch_groups", [0, 2])
def test_full_delta_is_computed_but_never_cached(local_manifest, tmp_path, prefetch_groups):
    cache = tmp_path / "cache"
    path = fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "out",
                             parent=local_manifest, parent_cache=cache,
                             full_delta=True, prefetch_groups=prefetch_groups)
    assert set(load_fingerprint(path).delta["status"]) == {"identical"}
    assert not list(cache.glob("*.bin"))


def test_row_blocks_cuda_moves_sketch_spectrum_and_delta_to_device(local_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module
    transfers, deltas = [], []
    original_delta = module.block_delta
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    def transfer(tensor, device):
        transfers.append(str(device))
        return tensor
    def delta(child, parent, dtype, **kwargs):
        deltas.append(str(kwargs["compute_device"]))
        return original_delta(child, parent, dtype)
    monkeypatch.setattr(module, "_to_compute", transfer)
    monkeypatch.setattr(module, "block_delta", delta)
    fp = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path,
                                            parent=local_manifest, compute_device="cuda:0"))
    assert transfers == ["cuda:0"] * len(fp.names)
    assert deltas == ["cuda:0"] * len(fp.names)
    assert set(fp.delta["status"]) == {"identical"}


def test_row_blocks_cpu_is_byte_identical(local_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.)
    a = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path / "a"))
    b = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path / "b", compute_device="cpu"))
    # Payload digests independently generated by pre-change HEAD, seed 600.
    import hashlib
    assert hashlib.sha256(b.sketch.tobytes()).hexdigest() == "513270877427a4f67c5a602179d683dfafc3984ca4b04ce7f72323480b2f1cd5"
    assert hashlib.sha256(b"".join(s.tobytes() for s in b.spectra)).hexdigest() == "9dbd26302a9d2fb08a967e01c4614b74e3339fc994f483b6684c47fdafbc9b4a"
    assert a.meta == b.meta
    assert a.sha256 == b.sha256
    assert a.sketch.tobytes() == b.sketch.tobytes()
    assert [x.tobytes() for x in a.spectra] == [x.tobytes() for x in b.spectra]


def test_row_blocks_prefetch_never_holds_more_than_window_groups(tmp_path, monkeypatch):
    import weakref
    from safetensors.torch import save_file
    from evidence.manifest import build_manifest
    import evidence.fingerprint as module
    source = tmp_path / "source"
    source.mkdir()
    save_file({f"layer{i}.weight": torch.randn(32, 16) for i in range(6)}, source / "model.safetensors")
    manifest = build_manifest(str(source), "main", HubReader(), None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module, "_to_compute", lambda tensor, device: tensor)
    original, alive, counts = module._Sources.block, [], []
    def read(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        alive.append(weakref.ref(result[0]))
        counts.append(sum(ref() is not None for ref in alive))
        return result
    monkeypatch.setattr(module._Sources, "block", read)
    result = load_fingerprint(fingerprint_model(manifest, HubReader(), "row_blocks", tmp_path / "out",
                                               compute_device="cuda:0", prefetch_bytes=65536))
    assert len(result.names) == 6
    assert max(counts) <= 2


def test_row_blocks_prefetch_does_not_admit_transposed_full_expert_reads(tmp_path, monkeypatch):
    import json
    import threading
    from safetensors.torch import save_file
    from evidence.manifest import build_manifest
    import evidence.fingerprint as module
    source = tmp_path / "source"
    source.mkdir()
    save_file({f"layers.{i}.mlp.experts.up_proj": torch.randn(2, 16, 4096)
               for i in range(2)}, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps(dict(hidden_size=16)))
    manifest = build_manifest(str(source), "main", HubReader(), None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module, "_to_compute", lambda tensor, device: tensor)
    calls = []
    original = module._Sources.block
    def read_block(self, *args, **kwargs):
        calls.append(threading.current_thread().name)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(module._Sources, "block", read_block)
    fp = load_fingerprint(fingerprint_model(manifest, HubReader(), "row_blocks", tmp_path / "out",
                                            compute_device="cuda:0", prefetch_bytes=100000))
    assert len(fp.names) == 4
    assert len(calls) == 4
    assert all("fingerprint-loader" not in thread for thread in calls)


@pytest.mark.parametrize("cache_state", ["none", "cold", "warm"])
def test_full_weights_prefetches_parent_blocks_while_spectrum_computes(local_manifest, tmp_path, monkeypatch, cache_state):
    from threading import Event, current_thread
    import evidence.fingerprint as module
    from evidence.block_cache import BlockCache
    parent = dict(local_manifest)
    cache = BlockCache(tmp_path / "cache") if cache_state != "none" else None
    if cache_state == "warm":
        fingerprint_model(local_manifest, HubReader(), "row_blocks", tmp_path / "warm",
                          parent=parent, parent_cache=cache)
    baseline = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "baseline",
                                                  parent=parent, also_row_block_sizes=(256,)))
    next_parent_ready = Event()
    parent_reads = []
    original_block, original_get = module._Sources.block, BlockCache.get
    def observe(name):
        if "q_proj" in name and current_thread().name.startswith("fingerprint-loader"):
            next_parent_ready.set()
    def block(self, view, bounds, *args, **kwargs):
        result = original_block(self, view, bounds, *args, **kwargs)
        if self.manifest is parent:
            parent_reads.append(bounds)
            observe(view.name)
        return result
    def get(self, manifest, name, *args, **kwargs):
        result = original_get(self, manifest, name, *args, **kwargs)
        if manifest is parent and result is not None:
            observe(name)
        return result
    original_spectrum = module._Spectra.add
    first = True
    def spectrum(self, name, tensor):
        nonlocal first
        if first:
            first = False
            assert next_parent_ready.wait(3), "next parent's block did not overlap the current spectrum"
        return original_spectrum(self, name, tensor)
    monkeypatch.setattr(module._Sources, "block", block)
    monkeypatch.setattr(BlockCache, "get", get)
    monkeypatch.setattr(module._Spectra, "add", spectrum)
    result = load_fingerprint(fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "prefetched",
                                                parent=parent, parent_cache=cache, also_row_block_sizes=(256,),
                                                prefetch_groups=2, prefetch_bytes=1024**2))
    assert all(last - first <= 256 for first, last in parent_reads)
    assert len(parent_reads) == (0 if cache_state == "warm" else 2)
    assert result.names == baseline.names and result.sha256 == baseline.sha256
    np.testing.assert_array_equal(result.sketch, baseline.sketch)
    for actual, expected in zip(result.spectra, baseline.spectra):
        np.testing.assert_array_equal(actual, expected)
    companion = next((tmp_path / "prefetched").glob("*.row_blocks.fingerprint.safetensors"))
    expected = load_fingerprint(next((tmp_path / "baseline").glob("*.row_blocks.fingerprint.safetensors")))
    for key, actual in load_fingerprint(companion).delta.items():
        np.testing.assert_equal(actual, expected.delta[key])


def test_full_weights_prefetch_budget_includes_parent_blocks(local_manifest, tmp_path, monkeypatch):
    from threading import current_thread
    import evidence.fingerprint as module
    parent, calls = dict(local_manifest), []
    original = module._Sources.block
    def block(self, *args, **kwargs):
        if self.manifest is parent:
            calls.append(current_thread().name)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(module._Sources, "block", block)
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "out", parent=parent,
                      also_row_block_sizes=(256,), prefetch_groups=2, prefetch_bytes=70000)
    # q_proj's stored child (65536 bytes) alone fits; child + parent does not.
    assert len(calls) == 2
    assert all(not name.startswith("fingerprint-loader") for name in calls)
