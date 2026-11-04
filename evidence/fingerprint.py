"""Bounded checkpoint fingerprints and collector-compatible full spectra."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from itertools import groupby
from pathlib import Path
import tempfile
from threading import Lock
import time
from types import SimpleNamespace
from typing import Iterable

import h5py
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import torch

from measurement_config import (FORMAT_VERSION, NUMERICS_VERSION,
                                measurement_config, safe_filename)
from net_esd.core import compute_esd_for_weight, _svdvals_for_measurement

from . import EVIDENCE_VERSION
from .delta import block_delta, _compute_device
from .hub import HubReader
from .safetensors_header import TensorEntry, _DTYPE_BYTES, _numel, decode_plain
from .sketch import (ROW_BLOCK, SKETCH_K_DEFAULT, SKETCH_K_MAX, SKETCH_SEED,
                     row_block_bounds, two_sided_sketch)
from .source_prefetch import SourcePrefetch
from .views import View, _expert_transpose, _fp8_blocks, decode_view, plan_views


@dataclass
class Fingerprint:
    names: list[str]
    geometry: list[str]
    component: list[str]
    rows: np.ndarray
    shape: np.ndarray
    sketch: np.ndarray
    sha256: list[str]
    spectra: list[np.ndarray]
    delta: dict[str, np.ndarray]
    coverage: list[dict]
    meta: dict


_TORCH_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2", torch.uint8: "U8", torch.int8: "I8",
    torch.int16: "I16", torch.int32: "I32", torch.int64: "I64", torch.bool: "BOOL",
}


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _block_hash(tensor, dtype):
    return hashlib.sha256(b"evidence-block-v1\0" + dtype.encode("ascii") + b"\0" + _bytes(tensor)).hexdigest()


def _atomic_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(temporary)


def _source(manifest):
    repo = manifest["repo_id"]
    if not manifest["revision"].startswith("local:"):
        return repo
    locator = manifest.get("source", {}).get("zoo_data_path")
    if locator is not None or not Path(repo).is_absolute():
        root = Path(os.environ["ZOO_DATA"]).resolve()
        relative = Path(locator if locator is not None else repo)
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError("Local source must stay inside ZOO_DATA")
        return str(target)
    return repo


def _entries(manifest):
    entries = {}
    for record in manifest["tensors"]:
        entry = TensorEntry(**{key: record[key] for key in ("name", "dtype", "shape", "file", "start", "end")})
        if (entry.dtype not in _DTYPE_BYTES or type(entry.start) is not int
                or type(entry.end) is not int or entry.start < 0
                or entry.end - entry.start != _numel(entry.shape) * _DTYPE_BYTES[entry.dtype]):
            raise ValueError("Invalid manifest tensor bounds or dtype")
        if entry.name in entries:
            raise ValueError("Duplicate manifest tensor identity")
        entries[entry.name] = entry
    return entries


def _plan_nonempty_views(entries, config, adapter_config=None, all_experts=False):
    views, coverage = plan_views(entries, config, adapter_config, all_experts)
    for record in coverage:
        if record["geometry"] and not all(entries[record["name"]].shape):
            record.update(status="skipped", reason="not_a_matrix")
    return [view for view in views if all(view.matrix_shape)], coverage


class _Sources:
    """Only the current view's reads are submitted; compute stays on the caller."""

    def __init__(self, manifest, reader, pool):
        self.manifest, self.reader, self.pool = manifest, reader, pool
        self.entries = _entries(manifest)
        self.source = _source(manifest)
        self.bytes_read = self.range_reads = 0
        self.lock = Lock()

    def fetch(self, requests):
        resolved = []
        for name, shape, expert, rows in requests:
            entry = self.entries[name]
            # shape is the source after leading singleton/expert axes are removed.
            size = math.prod(shape) * _DTYPE_BYTES[entry.dtype]
            start = entry.start + (expert * size if expert is not None else 0)
            selected_shape = tuple(shape)
            if rows is not None:
                first, last, step = rows.indices(shape[0])
                if step != 1 or last < first:
                    raise ValueError("Expected contiguous source rows")
                stride = math.prod(shape[1:]) * _DTYPE_BYTES[entry.dtype]
                start += first * stride
                size = (last - first) * stride
                selected_shape = (last - first, *shape[1:])
            if not entry.start <= start <= start + size <= entry.end:
                raise ValueError("View read escapes its tensor")
            remote = self.reader.resolve(self.source, entry.file, self.manifest["revision"])
            resolved.append((remote, start, start + size, entry.dtype, selected_shape))

        def read(args):
            remote, start, end, dtype, shape = args
            raw = self.reader.read(remote, start, end)
            with self.lock:
                self.bytes_read += len(raw)
                self.range_reads += 1
            return decode_plain(raw, dtype, shape) if dtype not in ("I8", "U8") else torch.frombuffer(bytearray(raw), dtype=torch.int8 if dtype == "I8" else torch.uint8).reshape(shape)

        return list(self.pool.map(read, resolved))

    def full(self, view):
        return dict(zip(view.sources, self.fetch([
            (name, self.entries[name].shape, None, None) for name in view.sources])))

    def block(self, view, bounds, full=None):
        """Return decoded block, stored block (plain only), and LoRA factors."""
        first, last = bounds
        config = self.manifest.get("config", {})
        if view.geometry == "lora_product":
            factors = full if full is not None else self.full(view)
            a, b = (factors[name].float() for name in view.sources[:2])
            if ".lora_a" in view.sources[0]:
                a, b = b.T, a.T
            else:
                a, b = a.flatten(1), b.flatten(1)
            scale = view.scale * (float(factors[view.sources[2]].item()) if len(view.sources) > 2 else 1.)
            b = b[first:last]
            return (b @ a) * scale, None, (b * scale, a)

        weight = self.entries[view.sources[0]]
        weight_shape = tuple(weight.shape)
        while len(weight_shape) > 2 and weight_shape[0] == 1 and view.slice_index is None:
            weight_shape = weight_shape[1:]
        expert = view.slice_index
        if expert is not None:
            weight_shape = weight_shape[1:]
        transpose = expert is not None and _expert_transpose(view.sources[0], weight_shape, config)
        split = "gate_up_proj" in view.sources[0] and "gate_up_proj" not in view.name
        offset = view.matrix_shape[0] if split and "up_proj" in view.name else 0
        physical_rows = None if transpose else slice(first + offset, last + offset)
        shapes, indices = [], []
        for index, name in enumerate(view.sources):
            shape = tuple(self.entries[name].shape)
            source_expert = expert if (expert is not None and shape and shape[0] == weight.shape[0]
                                      and (index == 0 or len(shape) >= len(weight.shape) - 1)) else None
            if source_expert is not None:
                shape = shape[1:]
            elif index == 0:
                shape = weight_shape
            shapes.append(shape)
            indices.append(source_expert)

        if view.decode == "plain":
            if full is None:
                stored = self.fetch([(view.sources[0], weight_shape, expert, physical_rows)])[0]
            else:
                stored = full[view.sources[0]].reshape((-1, *weight_shape))[expert] if expert is not None else full[view.sources[0]].reshape(weight_shape)
                if physical_rows is not None:
                    stored = stored[physical_rows]
            if transpose:
                stored = stored.T
                stored = stored[first + offset:last + offset]
            stored = stored.reshape(last - first, view.matrix_shape[1]).contiguous()
            return stored.float() * view.scale, stored, None

        # Decode expert slices in local matrix coordinates, preserving their full
        # source shapes so partial FP8 grids retain their original block sizes.
        local = replace(view, name=view.sources[0], geometry="matrix", slice_index=None,
                        matrix_shape=(view.matrix_shape[1], view.matrix_shape[0]) if transpose else (weight_shape[0], view.matrix_shape[1]), scale=1.)
        object.__setattr__(local, "_source_shapes", tuple(shapes))
        partial = physical_rows is not None and len(weight_shape) == 2
        requests = []
        for index, (name, shape, source_expert) in enumerate(zip(view.sources, shapes, indices)):
            rows = physical_rows if index == 0 and (partial or not transpose) else None
            if index > 0 and physical_rows is not None and shape and math.prod(shape) != 1:
                if view.decode == "fp8_scaled":
                    br = 1 if len(shape) == 1 else _fp8_blocks(weight_shape, shape, config)[0]
                    rows = slice(physical_rows.start // br, math.ceil(physical_rows.stop / br))
                elif index == 1:
                    rows = physical_rows
            requests.append((name, shape, source_expert, rows))
        if full is None:
            values = self.fetch(requests)
        else:
            values = []
            for name, shape, source_expert, rows in requests:
                value = full[name]
                value = value[source_expert] if source_expert is not None else value.reshape(shape)
                values.append(value if rows is None else value[rows])
        tensors = dict(zip(view.sources, values))
        if not partial:
            object.__setattr__(local, "_source_shapes", tuple(tuple(t.shape) for t in values))
        decoded = decode_view(local, lambda name, rows: tensors[name], config,
                              row_slice=physical_rows if partial else None)
        if transpose:
            decoded = decoded.T[first + offset:last + offset]
        return decoded.reshape(last - first, view.matrix_shape[1]) * view.scale, None, None


def _no_delta():
    return dict(frac_identical=math.nan, rel_update_norm=math.nan, rank=-1,
                status="no_parent", rank_confidence="none",
                top_singular_values=np.full(16, np.nan, np.float32))


def _cache_path(cache, manifest, name, bounds):
    text = f"{manifest['repo_id']}@{manifest['revision']}|{name}|{bounds[0]}|{bounds[1]}"
    return Path(cache) / (hashlib.sha256(text.encode()).hexdigest() + ".bin")


def _cache_write(path, block):
    temporary = _atomic_path(path)
    try:
        temporary.write_bytes(_bytes(block.float()))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parent_delta(block, view, bounds, parent_sources, parent_views, cache, height,
                  compute_device="cpu", *, full=None, parent_block=None):
    other = parent_views.get(view.name)
    if other is None:
        return _no_delta()
    parent_bounds = (0, other.matrix_shape[0]) if height is None else row_block_bounds(other.matrix_shape[0], height)
    if parent_bounds != bounds or other.matrix_shape[1] != view.matrix_shape[1]:
        result = _no_delta()
        result["status"] = "incompatible_shape"
        return result
    path = _cache_path(cache, parent_sources.manifest, view.name, bounds) if cache is not None else None
    if parent_block is not None:
        pass
    elif path is not None and path.exists():
        raw = path.read_bytes()
        if len(raw) != block.numel() * 4:
            raise ValueError("Invalid cached parent block size")
        parent_block = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(block.shape)
    else:
        parent_block, _, _ = parent_sources.block(other, bounds, full)
        if path is not None:
            _cache_write(path, parent_block)
    dtype = _TORCH_DTYPES_REVERSE.get(parent_sources.entries[other.sources[0]].dtype, torch.float32)
    if other.decode != "plain" or other.geometry == "lora_product":
        dtype = torch.float32
    if torch.device(compute_device).type == "cpu":
        return block_delta(block, parent_block, dtype)
    return block_delta(block, parent_block, dtype, compute_device=compute_device)


_TORCH_DTYPES_REVERSE = {name: dtype for dtype, name in _TORCH_DTYPES.items()}


def _to_compute(tensor, device):
    return tensor.to(device=device)


def _view_device(block, device):
    # Large single required views can exceed a device even when source reads
    # are bounded. Keep the original CPU arithmetic for that view explicitly.
    if device.type == "cuda" and block.numel()*block.element_size() > 128*1024**2:
        free, _ = torch.cuda.mem_get_info(device)
        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if block.numel()*4*20 > .8*(free + reusable):
            import warnings
            warnings.warn("Matrix exceeds CUDA workspace budget; using CPU derived statistics for this view", RuntimeWarning)
            return torch.device("cpu")
    return device


class _Records:
    def __init__(self, coverage, k, tier):
        self.coverage = [dict(record) for record in coverage]
        self.k, self.tier, self.records, self.seen = k, tier, [], {}
        self.by_source = {record["name"]: record for record in self.coverage}

    def add(self, view, bounds, block, stored, factors, dtype, delta, *, host_block=None):
        digest = _block_hash(stored if stored is not None else (
            host_block if host_block is not None else block.float()), dtype if stored is not None else "F32")
        duplicate_key = (view.matrix_shape, bounds, digest)
        if duplicate_key in self.seen:
            for source in view.sources:
                self.by_source[source].setdefault("duplicate_of", self.seen[duplicate_key])
            return False
        self.seen[duplicate_key] = view.name
        sketch = two_sided_sketch(block, self.k)
        padded = np.zeros((self.k, self.k), np.float32)
        padded[:sketch.shape[0], :sketch.shape[1]] = sketch
        spectrum = np.empty(0, np.float32)
        if self.tier != "t2":
            if factors is None:
                singular = _svdvals_for_measurement(block.float()) if block.is_cuda else torch.linalg.svdvals(block.float())
            else:
                b, a = factors
                if block.is_cuda:
                    b, a = b.to(block.device), a.to(block.device)
                rb = torch.linalg.qr(b.double(), mode="reduced").R
                ra = torch.linalg.qr(a.double().T, mode="reduced").R
                product = rb @ ra.T
                singular = _svdvals_for_measurement(product) if product.is_cuda else torch.linalg.svdvals(product)
            spectrum = torch.sort(singular.square()).values.float().cpu().numpy()
        self.records.append(dict(view=view, rows=bounds, sha256=digest, sketch=padded,
                                 spectrum=spectrum, delta=delta))
        for source in view.sources:
            self.by_source[source].setdefault("views", []).append(view.name)
        return True

    def write(self, path, meta):
        records = sorted(self.records, key=lambda record: record["view"].name)
        n = len(records)
        spectra = [record["spectrum"] for record in records]
        tensors = {
            "sketch": np.stack([record["sketch"] for record in records]) if n else np.empty((0, self.k, self.k), np.float32),
            "rows": np.asarray([record["rows"] for record in records], dtype=np.int64).reshape(n, 2),
            "shape": np.asarray([record["view"].matrix_shape for record in records], dtype=np.int64).reshape(n, 2),
            "delta_scalars": np.asarray([[record["delta"][key] for key in ("frac_identical", "rel_update_norm", "rank")] for record in records], dtype=np.float32).reshape(n, 3),
            "delta_top_sv": np.asarray([record["delta"]["top_singular_values"] for record in records], dtype=np.float32).reshape(n, 16),
        }
        if self.tier != "t2":
            tensors.update(spectrum_values=np.concatenate(spectra) if n else np.empty(0, np.float32),
                           spectrum_offsets=np.asarray([0, *np.cumsum([len(s) for s in spectra])], dtype=np.int64))
        metadata = dict(meta, tier=self.tier, coverage=self.coverage,
                        names=[record["view"].name for record in records],
                        geometry=[record["view"].geometry for record in records],
                        component=[record["view"].component for record in records],
                        sha256=[record["sha256"] for record in records],
                        delta_status=[record["delta"]["status"] for record in records],
                        delta_confidence=[record["delta"]["rank_confidence"] for record in records])
        temporary = _atomic_path(path)
        try:
            save_file(tensors, str(temporary), metadata={key: value if isinstance(value, str) else json.dumps(value) for key, value in metadata.items()})
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


class _Spectra:
    def __init__(self, path, compute_device):
        self.path, self.temporary = path, _atomic_path(path)
        self.compute_device = compute_device
        self.handle = h5py.File(self.temporary, "w")
        self.layers = self.handle.create_group("layers")
        self.layers.create_dataset("longname", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        self.eigs = self.handle.create_dataset("eigs", (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype("float32")))
        self.config = measurement_config(SimpleNamespace(analysis_source="checkpoint"))
        self.handle.attrs.update(format_version=FORMAT_VERSION, numerics_version=NUMERICS_VERSION,
                                 measurement_config_json=json.dumps(self.config, sort_keys=True),
                                 compute_device=str(compute_device))

    def add(self, name, tensor):
        cfg = self.config
        result = compute_esd_for_weight(name, tensor, cfg["evals_thresh"], cfg["bins"],
                                        cfg["fix_fingers"], cfg["xmin_pos"], cfg["conv_norm"],
                                        cfg["filter_zeros"], cfg["use_svd"], True, self.compute_device.index,
                                        tensor.numel(), cfg["compute_dtype"])
        if result is None:
            raise ValueError(f"No collector spectrum for view {name}")
        if result.get("compute_device") != str(self.compute_device):
            raise ValueError("Collector spectrum used an unexpected compute device")
        index = len(self.eigs)
        for key, value in result.items():
            if key not in ("eigs", "longname") and isinstance(value, (int, float, bool, np.number)):
                if key not in self.layers:
                    self.layers.create_dataset(key, (0,), maxshape=(None,), dtype="f8")
                dataset = self.layers[key]
                dataset.resize(index + 1, axis=0)
                dataset[index] = value
        names = self.layers["longname"]
        names.resize(index + 1, axis=0)
        names[index] = name
        self.eigs.resize(index + 1, axis=0)
        self.eigs[index] = np.asarray(result["eigs"], dtype=np.float32)

    def close(self, publish=False):
        self.handle.close()
        if publish:
            os.replace(self.temporary, self.path)
        self.temporary.unlink(missing_ok=True)


def _metadata(identity, parent, k, height, stats, elapsed):
    return dict(repo_id=identity["repo_id"], revision=identity["revision"],
                parent={key: parent[key] for key in ("repo_id", "revision")} if parent else None,
                sketch_k=k, sketch_seed=SKETCH_SEED, row_block=height,
                evidence_version=EVIDENCE_VERSION, elapsed_seconds=elapsed, **stats)


def _measure_prefetched(views, sources, parent_sources, parent_views, parent_cache,
                        heights, records, spectra, manifest, device, window, max_bytes):
    groups = [list(group) for _, group in groupby(
        sorted(views, key=lambda view: (view.sources, view.name)), key=lambda view: view.sources)]

    def parent_groups(group):
        result = {}
        if parent_sources is not None and parent_cache is None:
            for view in group:
                other = parent_views.get(view.name)
                if other is not None and other.matrix_shape == view.matrix_shape:
                    result.setdefault(other.sources, other)
        return result

    def estimate(group):
        total = sum(sources.entries[name].end-sources.entries[name].start for name in group[0].sources)
        for names in parent_groups(group):
            total += sum(parent_sources.entries[name].end-parent_sources.entries[name].start for name in names)
        return total

    def load(group):
        child = sources.full(group[0])
        parents = {names: parent_sources.full(other) for names, other in parent_groups(group).items()}
        return child, parents

    def measure_view(view, full, parents):
        plain = view.decode == "plain" and view.geometry != "lora_product"
        cpu_full, stored_full, factors_full = sources.block(view, (0, view.matrix_shape[0]), full)
        current_device = _view_device(cpu_full, device)
        gpu_full = _to_compute(cpu_full, current_device)
        gpu_factors = tuple(_to_compute(factor, current_device) for factor in factors_full) if factors_full is not None else None
        other = parent_views.get(view.name)
        parent_full = parents.get(other.sources) if other is not None else None
        parent_gpu = None
        if (parent_sources is not None and parent_cache is None and other is not None
                and other.matrix_shape == view.matrix_shape and other.decode == "plain"
                and other.geometry != "lora_product"):
            cpu_parent = parent_sources.block(other, (0, other.matrix_shape[0]), parent_full)[0]
            parent_gpu = _to_compute(cpu_parent, current_device)
            del cpu_parent
        for height, result in records.items():
            bounds = (0, view.matrix_shape[0]) if height is None else row_block_bounds(view.matrix_shape[0], height)
            # A full-matrix workspace fallback must not move its small,
            # independently bounded companion computations off CUDA.
            block_device = current_device if height is None else device
            if height is None:
                cpu_block, stored, factors = cpu_full, stored_full, factors_full
                block = gpu_full
            elif plain:
                cpu_block = cpu_full[bounds[0]:bounds[1]]
                stored = stored_full[bounds[0]:bounds[1]] if stored_full is not None else None
                factors = None
                block = (gpu_full[bounds[0]:bounds[1]] if block_device == current_device
                         else _to_compute(cpu_block, block_device))
            else:
                # Decode/hash the canonical host block for packed and
                # adapter layouts; do not change F32 product rounding.
                cpu_block, stored, factors = sources.block(view, bounds, full)
                block = _to_compute(cpu_block, block_device)
            if gpu_factors is not None:
                factors = (gpu_factors[0][bounds[0]:bounds[1]], gpu_factors[1])
            override = parent_gpu[bounds[0]:bounds[1]] if parent_gpu is not None else None
            delta = _parent_delta(block, view, bounds, parent_sources, parent_views,
                                  parent_cache, height, block_device, full=parent_full,
                                  parent_block=override) if parent_sources else _no_delta()
            kept = result.add(view, bounds, block, stored, factors,
                              sources.entries[view.sources[0]].dtype, delta, host_block=cpu_block)
            if parent_cache is not None:
                _cache_write(_cache_path(parent_cache, manifest, view.name, bounds), cpu_block)
            if kept and height is None:
                if view.geometry == "conv_flat":
                    # Collector conv normalization needs the original 4D
                    # stored weight, not the flattened view used by hashes.
                    weight = full[view.sources[0]] if full is not None else stored_full.reshape(sources.entries[view.sources[0]].shape)
                else:
                    weight = block
                spectra.add(view.name, weight)

    def measure(group, payload):
        full, parents = payload if payload is not None else (None, {})
        for view in group:
            measure_view(view, full, parents)

    loader = SourcePrefetch(groups, load, estimate, window=window, max_bytes=max_bytes)
    try:
        while True:
            try:
                group, payload = loader.next()
            except StopIteration:
                break
            measure(group, payload)
            # Release the current group's host/device references before the
            # next payload becomes current; never retain three source groups.
            del group, payload
    finally:
        loader.close()


def fingerprint_model(manifest: dict, reader: HubReader, tier: str, out_dir: Path,
                      parent: dict | None = None, parent_cache: Path | None = None,
                      sketch_k: int = SKETCH_K_DEFAULT, all_experts: bool = False,
                      also_t1_blocks: tuple[int, ...] = (), read_threads: int = 8,
                      compute_device: str = "cpu", prefetch_groups: int | None = None,
                      prefetch_bytes: int = 512*1024**2) -> Path:
    """Measure one pinned manifest, publishing the final T2 marker last."""
    if tier not in ("t1", "t2") or not 1 <= sketch_k <= SKETCH_K_MAX or read_threads < 1:
        raise ValueError("Invalid tier, sketch size, or read thread count")
    if any(type(height) is not int or height <= 0 for height in also_t1_blocks):
        raise ValueError("Block heights must be positive integers")
    device = _compute_device(compute_device)
    window = (2 if device.type == "cuda" and tier == "t2" else 0) if prefetch_groups is None else prefetch_groups
    if type(window) is not int or window not in (0, 1, 2):
        raise ValueError("prefetch_groups must be 0, 1 or 2")
    if type(prefetch_bytes) is not int or prefetch_bytes < 1:
        raise ValueError("prefetch_bytes must be positive")
    started = time.monotonic()
    initial_resolves = reader.stats["resolves"]
    safe = safe_filename(f"{manifest['repo_id']}@{manifest['revision']}")
    out_dir = Path(out_dir)
    out = out_dir / f"{safe}.{tier}.fingerprint.safetensors"
    spectra = None
    try:
        with ThreadPoolExecutor(max_workers=read_threads) as pool:
            sources = _Sources(manifest, reader, pool)
            views, coverage = _plan_nonempty_views(sources.entries, manifest.get("config", {}), manifest.get("adapter_config"), all_experts)
            parent_sources, parent_views = None, {}
            if parent is not None:
                parent_sources = _Sources(parent, reader, pool)
                planned, _ = _plan_nonempty_views(parent_sources.entries, parent.get("config", {}), parent.get("adapter_config"), all_experts)
                parent_views = {view.name: view for view in planned}
            heights = [None] if tier == "t2" else [ROW_BLOCK]
            if tier == "t2":
                heights += sorted(set(also_t1_blocks))
                spectra = _Spectra(out_dir / f"{safe}.t2.spectra.h5", device)
            records = {height: _Records(coverage, sketch_k, "t2" if height is None else ("t1" if height == ROW_BLOCK else f"t1b{height}")) for height in heights}
            if tier == "t2" and (device.type == "cuda" or window):
                _measure_prefetched(views, sources, parent_sources, parent_views, parent_cache,
                                    heights, records, spectra, manifest, device, window, prefetch_bytes)
                views = []
            previous_sources, full = None, None
            # Grouping bounds the full-source cache to one weight/factor group.
            for view in sorted(views, key=lambda view: (view.sources, view.name)):
                if tier == "t2" and previous_sources != view.sources:
                    full = None
                    full = sources.full(view)
                    previous_sources = view.sources
                for height, result in records.items():
                    bounds = (0, view.matrix_shape[0]) if height is None else row_block_bounds(view.matrix_shape[0], height)
                    block, stored, factors = sources.block(view, bounds, full)
                    delta = _parent_delta(block, view, bounds, parent_sources, parent_views,
                                          parent_cache, height, device) if parent_sources else _no_delta()
                    kept = result.add(view, bounds, block, stored, factors, sources.entries[view.sources[0]].dtype, delta)
                    if parent_cache is not None:
                        _cache_write(_cache_path(parent_cache, manifest, view.name, bounds), block)
                    if kept and height is None:
                        weight = full[view.sources[0]] if view.geometry == "conv_flat" else block
                        spectra.add(view.name, weight)
            stats = dict(bytes_read=sources.bytes_read + (parent_sources.bytes_read if parent_sources else 0),
                         range_reads=sources.range_reads + (parent_sources.range_reads if parent_sources else 0),
                         resolves=reader.stats["resolves"] - initial_resolves)
            for height in heights:
                if height is None:
                    continue
                result = records[height]
                path = out_dir / f"{safe}.{result.tier}.fingerprint.safetensors"
                result.write(path, _metadata(manifest, parent, sketch_k, height, stats, time.monotonic() - started))
            if tier == "t2":
                spectra.close(publish=True)
                spectra = None
                records[None].write(out, _metadata(manifest, parent, sketch_k, ROW_BLOCK, stats, time.monotonic() - started))
            return out
    finally:
        if spectra is not None:
            spectra.close()


def fingerprint_tensors(named_tensors: Iterable[tuple[str, torch.Tensor]], out_path: Path,
                        identity: dict, sketch_k: int = SKETCH_K_DEFAULT) -> Path:
    """Write plain T1 views directly from tensors without checkpoint I/O."""
    if not all(identity.get(key) for key in ("repo_id", "revision")):
        raise ValueError("identity requires repo_id and revision")
    if not 1 <= sketch_k <= SKETCH_K_MAX:
        raise ValueError("Invalid sketch size")
    started = time.monotonic()
    records = _Records([], sketch_k, "t1")
    names = set()
    for name, tensor in named_tensors:
        if name in names:
            raise ValueError("Duplicate input tensor identity")
        names.add(name)
        dtype = _TORCH_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError("Unsupported input dtype")
        entry = TensorEntry(name, dtype, tuple(tensor.shape), "", 0, tensor.numel() * tensor.element_size())
        # Plan each input independently: caller tensors are already decoded,
        # so adapter factors and quantization scale pairs are never synthesized.
        views, coverage = _plan_nonempty_views({name: entry}, {}, all_experts=True)
        records.coverage.extend(coverage)
        records.by_source.update({record["name"]: record for record in coverage})
        for view in views:
            bounds = row_block_bounds(view.matrix_shape[0])
            stored = tensor.detach().cpu()
            if view.slice_index is not None:
                stored = stored[view.slice_index]
            stored = stored.reshape(-1, view.matrix_shape[1])
            if "gate_up_proj" in view.sources[0] and "gate_up_proj" not in view.name:
                half = view.matrix_shape[0]
                stored = stored[:half] if "gate_proj" in view.name else stored[half:]
            stored = stored.reshape(view.matrix_shape)[bounds[0]:bounds[1]].contiguous()
            records.add(view, bounds, stored.float(), stored, None, dtype, _no_delta())
    return records.write(Path(out_path), _metadata(identity, None, sketch_k, ROW_BLOCK,
                         dict(bytes_read=0, range_reads=0, resolves=0), time.monotonic() - started))


def load_fingerprint(path: Path) -> Fingerprint:
    path = Path(path)
    with safe_open(path, framework="numpy") as handle:
        meta = {key: value if key in {"repo_id", "revision", "tier", "evidence_version"} else json.loads(value)
                for key, value in handle.metadata().items()}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    names = meta["names"]
    if meta["tier"] == "t2":
        suffix = ".t2.fingerprint.safetensors"
        spectra_path = path.with_name(path.name.removesuffix(suffix) + ".t2.spectra.h5")
        with h5py.File(spectra_path, "r") as handle:
            indices = {name: index for index, name in enumerate(handle["layers/longname"].asstr()[:])}
            spectra = [np.asarray(handle["eigs"][indices[name]], dtype=np.float32) for name in names]
    else:
        values, offsets = tensors["spectrum_values"], tensors["spectrum_offsets"]
        if (offsets.shape != (len(names) + 1,) or offsets[0] != 0
                or offsets[-1] != len(values) or (np.diff(offsets) < 0).any()):
            raise ValueError("Invalid spectrum offsets")
        spectra = [values[first:last] for first, last in zip(offsets[:-1], offsets[1:])]
    scalars = tensors["delta_scalars"]
    delta = {key: scalars[:, index] for index, key in enumerate(("frac_identical", "rel_update_norm", "rank"))}
    delta.update(top_sv=tensors["delta_top_sv"], status=np.asarray(meta["delta_status"]),
                 confidence=np.asarray(meta["delta_confidence"]))
    return Fingerprint(names, meta["geometry"], meta["component"], tensors["rows"], tensors["shape"],
                       tensors["sketch"], meta["sha256"], spectra, delta, meta["coverage"], meta)
