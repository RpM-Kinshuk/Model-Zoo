"""Safetensors headers and plain decoding, with no network access."""

import json

import pytest
import torch

from evidence.safetensors_header import (
    MAX_HEADER_BYTES, UnsupportedDtype, decode_plain, header_length, parse_header,
)


def test_parse_header_absolute_offsets_and_metadata(tiny_checkpoint):
    path = tiny_checkpoint / "model-00001-of-00002.safetensors"
    data = path.read_bytes(); n = header_length(data[:8])
    entries, metadata = parse_header(data[:8 + n], path.name)
    assert "__metadata__" not in entries and metadata.get("format") == "pt"
    e = entries["model.layers.0.self_attn.q_proj.weight"]
    assert e.start >= 8 + n and e.end - e.start == 2 * 32 * 16   # BF16 32x16


def test_decode_matches_safetensors(tiny_checkpoint):
    from safetensors.torch import load_file
    path = tiny_checkpoint / "model-00001-of-00002.safetensors"
    data = path.read_bytes(); n = header_length(data[:8])
    entries, _ = parse_header(data[:8 + n], path.name)
    ref = load_file(path)
    for name, e in entries.items():
        if e.dtype in {"BF16", "F16", "F32"}:
            assert torch.equal(decode_plain(data[e.start:e.end], e.dtype, e.shape), ref[name])


def test_decode_fp8_roundtrip_and_rejects_int():
    x = torch.tensor([[0.5, -1.0], [2.0, 0.0]]).to(torch.float8_e4m3fn)
    got = decode_plain(x.view(torch.uint8).numpy().tobytes(), "F8_E4M3", (2, 2))
    assert torch.equal(got.float(), x.float())
    e8 = decode_plain(bytes([127, 128, 126]), "F8_E8M0", (3,))
    assert torch.equal(e8.float(), torch.tensor([1.0, 2.0, 0.5]))
    with pytest.raises(UnsupportedDtype):
        decode_plain(b"\x00" * 8, "I64", (1,))


def test_header_length_guard():
    with pytest.raises(ValueError):
        header_length((MAX_HEADER_BYTES + 1).to_bytes(8, "little"))


@pytest.mark.parametrize("first8", [b"", b"\x00" * 7, b"\x00" * 9])
def test_header_length_rejects_incomplete_or_extra_length_bytes(first8):
    with pytest.raises(ValueError):
        header_length(first8)


def prefix_for(header):
    raw = json.dumps(header).encode()
    return len(raw).to_bytes(8, "little") + raw


@pytest.mark.parametrize("entry", [
    {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]},
    {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]},
    {"dtype": "F32", "shape": [1.0], "data_offsets": [0, 4]},
    {"dtype": "F32", "shape": [2**63], "data_offsets": [0, 0]},
    {"dtype": "F32", "shape": [1], "data_offsets": [-1, 3]},
    {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]},
    {"dtype": "F32", "shape": [1], "data_offsets": [False, 4]},
    {"dtype": "F32", "shape": [1], "data_offsets": [0, 2**64]},
    {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]},
    {"dtype": "F32", "shape": "1", "data_offsets": [0, 4]},
    {"dtype": "F32", "shape": [1], "data_offsets": [0]},
    {},
])
def test_parse_rejects_invalid_shape_offsets_or_size(entry):
    with pytest.raises(ValueError):
        parse_header(prefix_for({"weight": entry}), "unused.safetensors")


@pytest.mark.parametrize("prefix", [
    b"", (100).to_bytes(8, "little") + b"{}",
    (1).to_bytes(8, "little") + b"{", prefix_for([]),
    prefix_for({"__metadata__": []}),
])
def test_parse_rejects_malformed_headers(prefix):
    with pytest.raises(ValueError):
        parse_header(prefix, "unused.safetensors")


def test_parse_rejects_overlapping_data():
    header = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    with pytest.raises(ValueError):
        parse_header(prefix_for(header), "unused.safetensors")


def test_parse_rejects_offsets_beyond_the_contiguous_payload():
    header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [2**60, 2**60 + 4]}}
    with pytest.raises(ValueError):
        parse_header(prefix_for(header), "unused.safetensors")


@pytest.mark.parametrize("dtype,torch_dtype", [("F64", torch.float64), ("F8_E5M2", torch.float8_e5m2)])
def test_remaining_plain_dtypes(dtype, torch_dtype):
    x = torch.tensor([0.5, -1.0, 2.0, 0.0]).to(torch_dtype)
    assert torch.equal(decode_plain(x.view(torch.uint8).numpy().tobytes(), dtype, (2, 2)).float(),
                       x.reshape(2, 2).float())


@pytest.mark.parametrize("shape,raw", [((-1,), b""), ((True,), b"\x00" * 4),
                                      ((2,), b"\x00" * 4), ((2**63,), b"")])
def test_decode_rejects_invalid_shape_or_byte_count(shape, raw):
    with pytest.raises(ValueError):
        decode_plain(raw, "F32", shape)


def test_decode_scalar_and_empty():
    assert decode_plain(b"\x00\x00\x80\x3f", "F32", ()).item() == 1.0
    assert decode_plain(b"", "F32", (0, 2)).shape == (0, 2)
