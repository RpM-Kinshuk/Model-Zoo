"""Plan measurable matrix views and decode their stored representations."""

from dataclasses import dataclass, replace
import math
from pathlib import PurePosixPath
import re
from typing import Callable

import numpy as np
import torch

from .safetensors_header import TensorEntry

NEVER_MATRIX_SUFFIXES = (
    "_scale", "_scale_inv", "_scale_2", ".scale", "scales", "qzeros", "g_idx",
    "absmax", "quant_map", "SCB", "input_scale", "_bias", ".alpha",
    "position_ids", "num_batches_tracked",
)
SKETCH_SEED = 20261028
EXPERTS_PER_LAYER = 8
_FP4_VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
_LORA_PAIRS = (
    ("lora_A", "lora_B"), ("lora_down", "lora_up"),
    ("lora.down", "lora.up"), ("lora_embedding_A", "lora_embedding_B"),
    ("lora_a", "lora_b"),
)


@dataclass(frozen=True)
class View:
    name: str
    decode: str
    geometry: str
    sources: tuple[str, ...]
    slice_index: int | None
    matrix_shape: tuple[int, int]
    component: str
    fused: bool
    scale: float


def component_of(name: str, file: str) -> str:
    folder = str(PurePosixPath(file).parent)
    if folder != ".":
        name = name.removeprefix(folder + "/")
    parts = name.split(".")
    stop = next((i for i, part in enumerate(parts) if part.isdigit()), len(parts) - 1)
    prefix = ".".join(parts[:stop]) or parts[0]
    return f"{folder}/{prefix}" if folder != "." else prefix


def _squeeze(shape):
    shape = tuple(shape)
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    return shape


def _fp8_blocks(shape, scale_shape, config):
    """Return row/column block sizes, or None for an ambiguous grid."""
    if not scale_shape or scale_shape == (shape[-2],) or math.prod(scale_shape) == 1:
        return (1, 1)
    if len(scale_shape) < 2:
        return None
    rows, cols = shape[-2:]
    sr, sc = scale_shape[-2:]
    block = config.get("quantization_config", {}).get("weight_block_size")
    if block:
        if len(block) != 2 or any(type(x) is not int or x <= 0 for x in block):
            return None
        br, bc = block
        return (br, bc) if (sr, sc) == (math.ceil(rows / br), math.ceil(cols / bc)) else None
    if sr and sc and rows % sr == 0 and cols % sc == 0:
        return rows // sr, cols // sc
    return None


def _representation(name, entries, config):
    entry = entries[name]
    shape = entry.shape
    qc = config.get("quantization_config", {})
    if name.endswith("_blocks") and entry.dtype == "U8":
        scale = name[:-7] + "_scales"
        if scale in entries and len(shape) >= 3 and entries[scale].shape == shape[:-1]:
            return "mxfp4", (name, scale), shape[:-2] + (shape[-2] * shape[-1] * 2,), ""
    if name.endswith(".weight") and entry.dtype in ("I8", "U8") and len(shape) >= 2:
        module = name[:-7]
        scale, global_scale = module + ".weight_scale", module + ".weight_scale_2"
        if scale in entries and global_scale in entries:
            return "nvfp4", (name, scale, global_scale), shape[:-1] + (shape[-1] * 2,), ""
        scale = module + ".scale"
        if qc.get("expert_dtype") == "fp4" and scale in entries and entries[scale].dtype == "F8_E8M0":
            return "mxfp4", (name, scale), shape[:-1] + (shape[-1] * 2,), ""
    if entry.dtype.startswith("F8_") and len(shape) >= 2:
        module = name.removesuffix(".weight")
        scale = next((s for s in (name + "_scale_inv", name + "_scale", module + ".scale") if s in entries), None)
        if scale is not None:
            reason = "" if _fp8_blocks(shape, entries[scale].shape, config) else "fp8_block_size_unknown"
            return "fp8_scaled", (name, scale), shape, reason
    if entry.dtype.startswith(("F", "BF")):
        return "plain", (name,), shape, ""
    return None


def _expert_transpose(name, shape, config):
    hidden = config.get("hidden_size")
    if hidden is None or len(shape) != 2:
        return False
    down = bool(re.search(r"(?:down_proj|\.w2)(?:\.|_|$)", name))
    return (shape[1] == hidden and shape[0] != hidden) if down else (shape[0] == hidden and shape[1] != hidden)


def _pattern_value(patterns, module, default):
    for key, value in patterns.items():
        if module == key or re.search(r"(?:^|\.)" + key + "$", module):
            return value
    return default


def plan_views(entries: dict[str, TensorEntry], config: dict, adapter_config: dict | None = None,
               all_experts: bool = False) -> tuple[list[View], list[dict]]:
    reps = {name: _representation(name, entries, config) for name in sorted(entries)}
    consumed = {s: "decode_scale" for rep in reps.values() if rep for s in rep[1][1:]}
    poisoned = set()
    for name, entry in entries.items():
        packed = (entry.dtype.startswith(("I", "U")) and len(entry.shape) >= 2
                  and name.endswith(("weight", "qweight", "weight_packed", "_blocks")))
        if packed and reps[name] is None:
            poisoned.add(name.rpartition(".")[0])
    views, coverage, factors = [], {}, {}
    for name in sorted(entries):
        for down, up in _LORA_PAIRS:
            if re.search(r"(?:^|[./])" + re.escape(down) + r"(?:\.|$)", name):
                partner = name.replace(down, up)
                if partner not in entries:
                    continue
                a, b = entries[name], entries[partner]
                if len(a.shape) < 2 or len(b.shape) < 2:
                    continue
                module = name.split(down)[0].rstrip(".")
                cfg = adapter_config or {}
                mlx = down == "lora_a"
                rank = a.shape[1] if mlx else a.shape[0]
                rank = _pattern_value(cfg.get("rank_pattern", {}), module, cfg.get("r", rank))
                alpha = _pattern_value(cfg.get("alpha_pattern", {}), module, cfg.get("lora_alpha", rank))
                scale = float(alpha) / (math.sqrt(rank) if cfg.get("use_rslora") else rank)
                sources = (name, partner)
                reason = ""
                if mlx:
                    scale = float(cfg.get("lora_parameters", {}).get("scale", 1.))
                    matrix_shape = (b.shape[-1], a.shape[0])
                else:
                    matrix_shape = (b.shape[0], math.prod(a.shape[1:]))
                    alpha_name = module + ".alpha"
                    if alpha_name in entries:
                        sources += (alpha_name,)
                        consumed[alpha_name] = "adapter_alpha"
                        scale = 1. / rank
                    elif adapter_config is None and down in ("lora_A", "lora_embedding_A"):
                        scale, reason = 1., "lora_scale_unknown"
                vname = module.removeprefix("base_model.model.") + ".weight"
                views.append(View(vname, "plain", "lora_product", sources, None, matrix_shape,
                                  component_of(module, a.file), False, scale))
                for factor in sources[:2]:
                    factors[factor] = reason
                break
    count = config.get("num_experts", 0)
    selected = set(range(count)) if all_experts or count <= 8 else set(np.random.default_rng([SKETCH_SEED, count]).choice(count, 8, replace=False).tolist())
    for name in sorted(entries):
        entry, rep = entries[name], reps[name]
        decode = rep[0] if rep else "plain"
        record = dict(name=name, status="skipped", reason="", decode=decode, geometry="")
        coverage[name] = record
        if name in consumed:
            record.update(status="consumed", reason=consumed[name])
            continue
        if name.rpartition(".")[0] in poisoned:
            record["reason"] = "unsupported_representation"
            continue
        if entry.dtype.startswith(("I", "U")) or entry.dtype == "BOOL":
            if rep is None:
                record["reason"] = "non_float_buffer"
                continue
        if any(token in name for token in ("hada_", "lokr_", "lora_magnitude_vector", "dora_scale")):
            record["reason"] = "unsupported_adapter_type"
            continue
        if name in factors:
            record.update(status="measured", reason=factors[name], geometry="lora_product")
            continue
        if name.endswith(NEVER_MATRIX_SUFFIXES):
            record["reason"] = "not_a_matrix"
            continue
        if rep is None:
            record["reason"] = "unsupported_representation"
            continue
        decode, sources, raw_shape, decode_reason = rep
        shape = _squeeze(raw_shape)
        if len(shape) < 2:
            record["reason"] = "not_a_matrix"
            continue
        shared = "shared_expert" in name
        expert = len(shape) == 3 and not shared and ("experts" in name or shape[0] == config.get("num_experts"))
        if expert:
            matrix = shape[1:]
            if _expert_transpose(name, matrix, config):
                matrix = matrix[::-1]
            geometry = "expert_slice"
        elif len(shape) > 2:
            matrix, geometry = (shape[0], math.prod(shape[1:])), "conv_flat"
        else:
            matrix, geometry = shape, "head" if shape[0] < 8 and shape[1] >= 8 else "matrix"
        record["geometry"] = geometry
        if matrix[1] < 8:
            record["reason"] = "vector_like"
            continue
        if decode_reason:
            record["reason"] = decode_reason
            continue
        unfused = re.search(r"(?:^|[./])experts\.(\d+)\.", name)
        if not all_experts and not shared and unfused and int(unfused[1]) not in selected:
            record["reason"] = "expert_not_sampled"
            continue
        indices = sorted(selected if count == shape[0] else (range(shape[0]) if all_experts or shape[0] <= 8 else np.random.default_rng([SKETCH_SEED, shape[0]]).choice(shape[0], 8, replace=False))) if expert else [None]
        for index in indices:
            vname = name.removesuffix("_blocks")
            suffix = f"[{index}]" if index is not None else ""
            split = "gate_up_proj" in name and matrix[0] % 2 == 0 and config.get("model_type") != "gpt_oss"
            fused = bool(re.search(r"qkv|query_key_value|w13|in_proj", name)) or ("gate_up_proj" in name and not split)
            for part in ("gate", "up") if split else (None,):
                out_name = vname.replace("gate_up_proj", part + "_proj") if part else vname
                out_shape = (matrix[0] // 2, matrix[1]) if split else matrix
                views.append(View(out_name + suffix, decode, geometry, sources, index, out_shape,
                                  component_of(name, entry.file), fused, 1.))
        record["status"] = "measured"
    occupied = {view.name for view in views if view.geometry != "lora_product"}
    for index, view in enumerate(views):
        if view.geometry == "lora_product" and view.name in occupied:
            name = view.name + "[lora]"
            while name in occupied:
                name += "[lora]"
            views[index] = replace(view, name=name)
        occupied.add(views[index].name)
    for view in views:
        # Planning metadata is private: preserve the public constructor and fields.
        object.__setattr__(view, "_source_shapes", tuple(entries[name].shape for name in view.sources))
    return sorted(views, key=lambda v: v.name), list(coverage.values())


def _float_scale(tensor):
    # A callback normally uses decode_plain; byte E8M0 grids can also be supplied directly.
    return torch.exp2(tensor.float() - 127) if tensor.dtype == torch.uint8 else tensor.float()


def decode_view(view: View, read: Callable[[str, slice | None], torch.Tensor], config: dict,
                row_slice: slice | None = None) -> torch.Tensor:
    """Decode a matrix, optionally reading only its requested output rows.

    Expert orientation, fused splits and leading singleton dimensions may require
    full physical reads; crop those views after decoding in matrix coordinates.
    """
    if view.geometry == "lora_product":
        down, up = (read(name, None).float() for name in view.sources[:2])
        matrix = (down @ up).T if ".lora_a" in view.sources[0] else up.flatten(1) @ down.flatten(1)
        scale = view.scale * (float(read(view.sources[2], None).item()) if len(view.sources) > 2 else 1.)
        matrix = matrix * scale
        return matrix if row_slice is None else matrix[row_slice]
    shapes = getattr(view, "_source_shapes", ())
    split_fused = "gate_up_proj" in view.sources[0] and "gate_up_proj" not in view.name
    partial = (row_slice is not None and row_slice.step in (None, 1)
               and view.geometry in ("matrix", "head") and shapes
               and len(shapes[0]) == 2 and not split_fused)
    weight_rows, scale_offset = None, 0
    scale_rows = [None] * (len(view.sources) - 1)
    if partial:
        start, stop, _ = row_slice.indices(view.matrix_shape[0])
        stop = max(start, stop)
        weight_rows = slice(start, stop)
        if view.decode == "fp8_scaled":
            scale_shape = shapes[1]
            if math.prod(scale_shape) != 1:
                if len(scale_shape) == 1:
                    scale_rows[0] = weight_rows
                else:
                    blocks = _fp8_blocks(shapes[0], scale_shape, config)
                    if blocks is None:
                        raise ValueError("FP8 block size is unknown")
                    block_rows = blocks[0]
                    scale_rows[0] = slice(start // block_rows, math.ceil(stop / block_rows))
                    scale_offset = start % block_rows
        elif view.decode in ("mxfp4", "nvfp4"):
            scale_rows[0] = weight_rows
    weight = read(view.sources[0], weight_rows)
    scales = [read(name, rows) for name, rows in zip(view.sources[1:], scale_rows)]
    if view.decode == "mxfp4":
        lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=weight.device)
        packed = weight.to(torch.uint8)
        matrix = lut[torch.stack((packed & 15, packed >> 4), -1).long()]
        if view.sources[0].endswith("_blocks"):
            matrix = matrix.flatten(-2) * _float_scale(scales[0]).unsqueeze(-1)
            matrix = matrix.flatten(-2)
        else:
            matrix = matrix.flatten(-2) * _float_scale(scales[0]).repeat_interleave(32, -1)
    elif view.decode == "nvfp4":
        lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=weight.device)
        packed = weight.to(torch.uint8)
        matrix = lut[torch.stack((packed & 15, packed >> 4), -1).long()].flatten(-2)
        matrix = matrix * scales[0].float().repeat_interleave(16, -1) * scales[1].float()
    elif view.decode == "fp8_scaled":
        matrix, scale = weight.float(), _float_scale(scales[0])
        if scale.numel() == 1:
            matrix = matrix * scale
        elif scale.ndim == 1:
            matrix = matrix * scale.unsqueeze(-1)
        else:
            blocks = _fp8_blocks(shapes[0], shapes[1], config) if partial else _fp8_blocks(tuple(matrix.shape), tuple(scale.shape), config)
            if blocks is None:
                raise ValueError("FP8 block size is unknown")
            scale = scale.repeat_interleave(blocks[0], -2).repeat_interleave(blocks[1], -1)
            matrix = matrix * scale[..., scale_offset:scale_offset + matrix.shape[-2], :matrix.shape[-1]]
    else:
        matrix = weight.float()
    while matrix.ndim > 2 and matrix.shape[0] == 1 and view.slice_index is None:
        matrix = matrix.squeeze(0)
    if view.geometry == "expert_slice":
        matrix = matrix[view.slice_index]
        if _expert_transpose(view.sources[0], tuple(matrix.shape), config):
            matrix = matrix.T
    elif view.geometry == "conv_flat":
        matrix = matrix.flatten(1)
    if split_fused:
        half = matrix.shape[0] // 2
        matrix = matrix[:half] if "gate_proj" in view.name else matrix[half:]
    matrix = matrix.float() * view.scale
    return matrix if partial or row_slice is None else matrix[row_slice]
