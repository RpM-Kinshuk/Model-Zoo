"""Bounded safetensors header parsing and decoding of plain floating tensors."""

from dataclasses import dataclass
import json
import sys

import torch

MAX_HEADER_BYTES = 100 * 1024 * 1024

_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
    "U32": 4, "I32": 4, "U64": 8, "I64": 8,
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}
_PLAIN_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.uint8,
}


class UnsupportedDtype(ValueError):
    """The stored dtype cannot be decoded as a plain floating tensor."""


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: str
    start: int
    end: int


def header_length(first8: bytes) -> int:
    if len(first8) != 8:
        raise ValueError("Expected exactly eight header-length bytes")
    size = int.from_bytes(first8, "little")
    if size > MAX_HEADER_BYTES:
        raise ValueError("Safetensors header exceeds MAX_HEADER_BYTES")
    return size


def _numel(shape) -> int:
    if not isinstance(shape, (tuple, list)):
        raise ValueError("Tensor shape must be a sequence of dimensions")
    count = 1
    for dim in shape:
        if type(dim) is not int or not 0 <= dim <= sys.maxsize:
            raise ValueError("Tensor dimensions must be bounded nonnegative integers")
        count *= dim
        if count > sys.maxsize:
            raise ValueError("Tensor shape exceeds addressable size")
    return count


def parse_header(prefix: bytes, file: str) -> tuple[dict[str, TensorEntry], dict]:
    size = header_length(prefix[:8])
    base = 8 + size
    if len(prefix) < base:
        raise ValueError("Truncated safetensors header")
    try:
        header = json.loads(prefix[8:base])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Safetensors metadata must be an object")
    entries = {}
    for name, value in header.items():
        if not isinstance(value, dict):
            raise ValueError("Tensor entry must be an object")
        dtype = value.get("dtype")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise UnsupportedDtype("Unknown safetensors dtype")
        shape = value.get("shape")
        count = _numel(shape)
        offsets = value.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            raise ValueError("Tensor offsets must be a pair of integers")
        start, end = offsets
        if not 0 <= start <= end <= sys.maxsize - base:
            raise ValueError("Tensor offsets are outside addressable bounds")
        if end - start != count * _DTYPE_BYTES[dtype]:
            raise ValueError("Tensor byte size does not match its shape and dtype")
        entries[name] = TensorEntry(name, dtype, tuple(shape), file, base + start, base + end)
    previous_end = base
    for entry in sorted(entries.values(), key=lambda e: (e.start, e.end)):
        if entry.start != previous_end:
            raise ValueError("Tensor data offsets must be contiguous and nonoverlapping")
        previous_end = entry.end
    return entries, metadata


def decode_plain(raw: bytes, dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    if dtype not in _PLAIN_DTYPES:
        raise UnsupportedDtype(f"Unsupported plain dtype: {dtype}")
    count = _numel(shape)
    if len(raw) != count * _DTYPE_BYTES[dtype]:
        raise ValueError("Raw byte size does not match tensor shape and dtype")
    torch_dtype = _PLAIN_DTYPES[dtype]
    if count:
        tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype)
    else:
        tensor = torch.empty(0, dtype=torch_dtype)
    if dtype == "F8_E8M0":
        tensor = torch.exp2(tensor.float() - 127)
    return tensor.reshape(shape)
