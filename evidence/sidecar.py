"""Private background evidence, conservative spectrum reuse, marker-last publish."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import Event, Lock, RLock

import numpy as np
from safetensors import safe_open
import torch

from measurement_config import safe_filename
from .fingerprint import (_Sources, _Spectra, _TORCH_DTYPES, _block_hash, _entries,
                          _plan_nonempty_views, fingerprint_model)
from .hub import HubReader
from .manifest import write_manifest
from .sketch import row_block_bounds
from .safetensors_header import _DTYPE_BYTES


def _paths(out_dir, manifest):
    safe = safe_filename(f"{manifest['repo_id']}@{manifest['revision']}")
    marker = Path(out_dir) / f"{safe}.t2.fingerprint.safetensors"
    return marker, marker.with_name(marker.name + ".staged"), marker.with_name(f"{safe}.t2.spectra.h5")


def _metadata(path):
    with safe_open(path, framework="numpy") as handle:
        return handle.metadata()


def publish_t2(out_dir: Path, manifest: dict, results: list[tuple[str, dict]], measurement: dict) -> Path:
    """Publish complete spectra first, then the matching staged fingerprint."""
    marker, staged, spectra_path = _paths(out_dir, manifest)
    if marker.exists():
        raise FileExistsError("Completed evidence already exists")
    names = json.loads(_metadata(staged)["names"])
    by_name = dict(results)
    if len(by_name) != len(results) or set(by_name) != set(names):
        raise ValueError("Spectra must cover every staged evidence view exactly once")
    for name in names:
        result = by_name[name]
        eigs = np.asarray(result.get("eigs"), dtype=np.float32)
        if eigs.ndim != 1 or not eigs.size or not np.isfinite(eigs).all() or (eigs < 0).any():
            raise ValueError(f"Invalid spectrum for evidence view {name}")
        if result.get("spectrum_source") not in ("runner", "second_round"):
            raise ValueError("Spectrum provenance must be runner or second_round")
    spectra = _Spectra(spectra_path, torch.device("cpu"), measurement)
    devices = {str(result["compute_device"]) for result in by_name.values() if result.get("compute_device")}
    spectra.handle.attrs["compute_device"] = next(iter(devices)) if len(devices) == 1 else ("mixed" if devices else "unknown")
    try:
        spectra.add_results((name, by_name[name]) for name in names)
        spectra.close(publish=True)
        spectra = None
        os.replace(staged, marker)
    finally:
        if spectra is not None:
            spectra.close()
    return marker


class _CancellableReader(HubReader):
    def __init__(self, cancelled):
        super().__init__(local_cache=True)
        self.cancelled = cancelled

    def read(self, *args):
        if self.cancelled.is_set():
            raise CancelledError("Evidence sidecar cancelled")
        result = super().read(*args)
        if self.cancelled.is_set():
            raise CancelledError("Evidence sidecar cancelled")
        return result


class EvidenceSidecar:
    def __init__(self, manifest: dict, out_dir: Path, *, parent: dict | None = None,
                 block_cache=None, cache_self: bool = False, device: str = "cpu",
                 max_inflight_bytes: int = 2 * 1024**3, sketch_k: int = 64,
                 all_experts: bool = False, also_t1_blocks: tuple[int, ...] = (128, 256, 512)):
        if type(max_inflight_bytes) is not int or max_inflight_bytes < 1:
            raise ValueError("max_inflight_bytes must be positive")
        self.manifest, self.out_dir = manifest, Path(out_dir)
        self.parent, self.block_cache, self.cache_self = parent, block_cache, cache_self
        self.device, self.max_inflight_bytes = device, max_inflight_bytes
        self.sketch_k, self.all_experts, self.also_t1_blocks = sketch_k, all_experts, also_t1_blocks
        self._cancelled, self._lock, self._state_lock = Event(), RLock(), Lock()
        self._pool = self._future = self._stage = None
        self._finished = False

    def _check(self):
        if self._cancelled.is_set():
            raise CancelledError("Evidence sidecar cancelled")

    def start(self) -> None:
        with self._lock:
            self._check()
            if self._future is not None:
                raise RuntimeError("Evidence sidecar already started")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._stage = Path(tempfile.mkdtemp(prefix=".evidence-sidecar-", dir=self.out_dir))
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence-sidecar")
            self._future = self._pool.submit(self._run)

    def _run(self):
        write_manifest(self.manifest, self._stage)
        return fingerprint_model(self.manifest, _CancellableReader(self._cancelled), "t2", self._stage,
                                 parent=self.parent, parent_cache=self.block_cache, cache_self=self.cache_self,
                                 compute_device=self.device, sketch_k=self.sketch_k, all_experts=self.all_experts,
                                 also_t1_blocks=self.also_t1_blocks, write_spectra=False,
                                 prefetch_groups=2, prefetch_bytes=max(1, self.max_inflight_bytes // 2))

    def finish(self, esd, loaded_block, measure, measurement) -> dict:
        if self._future is None:
            raise RuntimeError("Evidence sidecar has not started")
        if self._finished:
            raise RuntimeError("Evidence sidecar already finished")
        try:
            staged = self._future.result()
            self._check()
            metadata = _metadata(staged)
            names = json.loads(metadata["names"])
            entries = _entries(self.manifest)
            views, _ = _plan_nonempty_views(entries, self.manifest.get("config", {}),
                                             self.manifest.get("adapter_config"), self.all_experts)
            views = {view.name: view for view in views}
            # Only exact stored keys; ambiguous identities and slices never reuse.
            runner = {}
            for key, result in esd.items():
                if result.get("slice", "") != "":
                    continue
                identity = (".".join(part for part in (result["module_name"], result.get("weight_attribute", "weight")) if part)
                            if "module_name" in result else key)
                runner.setdefault(identity, []).append(result)
            hashes = {}
            for path in self._stage.glob("*.t1.fingerprint.safetensors"):
                block_metadata = _metadata(path)
                hashes.update(zip(json.loads(block_metadata["names"]), json.loads(block_metadata["sha256"])))
            results, counts = [], dict(reused=0, measured=0, gaps=0)
            batch, batch_bytes = [], 0

            def flush():
                nonlocal batch, batch_bytes
                if not batch:
                    return
                self._check()
                measured = measure(batch)
                for name, _ in batch:
                    result = measured.get(name)
                    if result is None:
                        raise ValueError(f"Second round did not measure evidence view {name}")
                    results.append((name, dict(result, spectrum_source="second_round")))
                    counts["measured"] += 1
                batch, batch_bytes = [], 0

            with ThreadPoolExecutor(max_workers=8) as reads:
                sources = _Sources(self.manifest, _CancellableReader(self._cancelled), reads)
                for name in names:
                    self._check()
                    view = views[name]
                    candidates = runner.get(view.sources[0], [])
                    result = candidates[0] if len(candidates) == 1 else None
                    plain = (view.geometry == "matrix" and view.decode == "plain" and view.scale == 1
                             and view.name == view.sources[0] and tuple(entries[view.sources[0]].shape) == view.matrix_shape)
                    reuse = False
                    if plain and result is not None and (result.get("M"), result.get("N")) == view.matrix_shape and result.get("eigs") is not None:
                        bounds = row_block_bounds(view.matrix_shape[0])
                        loaded = loaded_block(view.sources[0], bounds)
                        if loaded is not None and tuple(loaded.shape) == (bounds[1]-bounds[0], view.matrix_shape[1]):
                            dtype = entries[view.sources[0]].dtype
                            expected = hashes.get(name)
                            if expected is None:
                                _, stored, _ = sources.block(view, bounds)
                                expected = _block_hash(stored, dtype)
                            # A dtype cast changes bytes even if values round-trip.
                            reuse = _TORCH_DTYPES.get(loaded.dtype) == dtype and _block_hash(loaded, dtype) == expected
                    if reuse:
                        results.append((name, dict(result, spectrum_source="runner")))
                        counts["reused"] += 1
                        continue
                    if view.decode == "plain" and view.geometry != "lora_product":
                        estimate = (view.matrix_shape[0] * view.matrix_shape[1]
                                    * (2 * _DTYPE_BYTES[entries[view.sources[0]].dtype] + 8))
                    else:
                        estimate = (2 * sum(entries[source].end-entries[source].start for source in view.sources)
                                    + 16 * view.matrix_shape[0] * view.matrix_shape[1])
                    if batch and batch_bytes + estimate > self.max_inflight_bytes:
                        flush()
                    tensor = (sources.full(view)[view.sources[0]] if view.geometry == "conv_flat"
                              else sources.block(view, (0, view.matrix_shape[0]))[0])
                    batch.append((name, tensor))
                    batch_bytes += estimate
                    del tensor
                    # One required oversized view is processed alone.
                    if batch_bytes >= self.max_inflight_bytes:
                        flush()
                flush()
            self._check()
            publish_t2(self._stage, self.manifest, results, measurement)
            marker, _, _ = _paths(self.out_dir, self.manifest)
            promoted, backups = [], []
            with self._lock:
                self._check()
                if marker.exists():
                    raise FileExistsError("Completed evidence already exists")
                try:
                    paths = sorted(self._stage.iterdir(), key=lambda path: (path.name == marker.name, path.name))
                    previous = self._stage / ".previous"
                    previous.mkdir()
                    for path in paths:
                        self._check()
                        destination = self.out_dir / path.name
                        if destination.exists():
                            backup = previous / path.name
                            os.replace(destination, backup)
                            backups.append((backup, destination))
                        os.replace(path, destination)
                        promoted.append(destination)
                    with self._state_lock:
                        self._check()
                        self._finished = True
                except BaseException:
                    for path in promoted:
                        path.unlink(missing_ok=True)
                    for backup, destination in backups:
                        os.replace(backup, destination)
                    raise
            self._pool.shutdown(wait=True)
            shutil.rmtree(self._stage)
            return counts
        except BaseException:
            self.cancel()
            raise

    def cancel(self) -> None:
        with self._state_lock:
            if self._finished:
                return
            self._cancelled.set()
        if self._future is not None:
            self._future.cancel()
            try:
                self._future.result()
            except Exception:
                pass
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            if self._stage is not None:
                shutil.rmtree(self._stage, ignore_errors=True)
