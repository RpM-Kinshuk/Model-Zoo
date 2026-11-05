"""Publish complete derived model artifacts atomically; never publish weights."""
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import h5py
from safetensors import safe_open

from evidence import EVIDENCE_VERSION
from measurement_config import (FORMAT_VERSION, NUMERICS_VERSION, artifact_compatibility,
                                measurement_config, safe_filename)
from .schema import model_key, split_key


_IDENTITY = {"repo_id", "requested_revision", "source_model", "base_model_relation", "loader_scenario", "runtime"}


def conventions(settings=None):
    settings = measurement_config(SimpleNamespace()) if settings is None else settings
    settings = dict(settings)
    settings.setdefault("evidence_tier", "full")
    settings.setdefault("evidence", dict(sketch_k="auto", all_experts="auto", with_parent=True, device="cpu"))
    result = {key: value for key, value in settings.items() if key not in _IDENTITY}
    for key, value in dict(numerics_version=NUMERICS_VERSION, format_version=FORMAT_VERSION, evidence_version=EVIDENCE_VERSION).items():
        result.setdefault(key, value)
    return result


def conventions_id(settings=None):
    payload = json.dumps(conventions(settings), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def current_conventions_id():
    return conventions_id()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_files(run):
    return sorted(path for path in (Path(run) / "evidence").glob("*/*.manifest.json")
                  if not path.parent.name.startswith("."))


def _candidate(run, manifest_path, *, in_package=False, summary_path=None):
    manifest = json.loads(manifest_path.read_text())
    repo, revision = manifest["repo_id"], manifest["revision"]
    key = model_key(repo, revision)
    if manifest.get("access") != "granted":
        return None
    safe, folder = safe_filename(key), manifest_path.parent
    if manifest_path.name != safe + ".manifest.json":
        raise ValueError("Manifest filename does not match its pinned identity")
    marker, spectra = folder / f"{safe}.t2.fingerprint.safetensors", folder / f"{safe}.t2.spectra.h5"
    sources = {manifest_path.name: manifest_path}
    cfg, kind = None, "t1_only"
    pairs = [(Path(run) / "stats" / f"{safe_filename(label)}.csv",
              Path(run) / "metrics" / f"{safe_filename(label)}.h5") for label in (key, repo)]
    if in_package:
        pairs = [(folder / "esd/stats.csv", folder / "esd/metrics.h5")]
    has_esd = any(csv.exists() or h5.exists() for csv, h5 in pairs)
    if marker.exists():
        kind = "full"
        pair = next(((csv, h5) for csv, h5 in pairs if artifact_compatibility(csv, h5)[0]), None)
        if pair is None or not spectra.is_file():
            return None
        csv, h5 = pair
        with h5py.File(h5) as handle:
            cfg = json.loads(handle.attrs["measurement_config_json"])
        if cfg.get("repo_id") != repo or cfg.get("requested_revision") != revision:
            return None
        with h5py.File(spectra) as handle:
            spectrum_cfg = json.loads(handle.attrs["measurement_config_json"])
        if conventions(cfg) != conventions(spectrum_cfg):
            return None
        sources.update({marker.name: marker, spectra.name: spectra, "esd/stats.csv": csv, "esd/metrics.h5": h5})
    elif has_esd or spectra.exists() or list(folder.glob("*.staged")):
        return None
    tiers = ("t1", "t1b128", "t1b512") if kind == "full" else ("t1",)
    for tier in tiers:
        path = folder / f"{safe}.{tier}.fingerprint.safetensors"
        if not path.is_file():
            return None
        sources[path.name] = path
    for path in folder.glob(f"{safe}.t1b*.fingerprint.safetensors"):
        sources[path.name] = path
    fingerprint_metadata = []
    for name, path in sources.items():
        if name.endswith(".fingerprint.safetensors"):
            with safe_open(path, framework="numpy") as handle:
                meta = handle.metadata()
            fingerprint_metadata.append(meta)
            if meta.get("repo_id") != repo or meta.get("revision") != revision or meta.get("evidence_version") != EVIDENCE_VERSION:
                raise ValueError("Fingerprint identity or version disagrees with its manifest")
    for path in sorted((folder / "deltas").glob("*.delta.safetensors")):
        sources["deltas/" + path.name] = path
    summary_path = summary_path or folder / "evidence.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    scientific = conventions(cfg)
    scientific["evidence_tier"] = kind
    settings = summary.get("settings", {})
    special = summary.get("stratum") in ("lab", "anchor")
    sizes = {int(meta["sketch_k"]) for meta in fingerprint_metadata}
    if len(sizes) != 1:
        raise ValueError("Inconsistent fingerprint sketch settings")
    k = sizes.pop()
    if settings.get("sketch_k", k) != k:
        raise ValueError("Evidence summary disagrees with fingerprint sketch size")
    from evidence.fingerprint import _entries
    from evidence.views import plan_views
    _, expected_coverage = plan_views(_entries(manifest), manifest.get("config", {}), manifest.get("adapter_config"), special)
    expected_skips = {row["name"] for row in expected_coverage if row.get("reason") == "expert_not_sampled"}
    coverages = [json.loads(meta["coverage"]) for meta in fingerprint_metadata]
    actual_skips = [{row["name"] for row in coverage if row.get("reason") == "expert_not_sampled"} for coverage in coverages]
    if any(skips != actual_skips[0] for skips in actual_skips):
        raise ValueError("Inconsistent fingerprint expert selection")
    if actual_skips[0] == expected_skips:
        experts = "auto"
    elif not actual_skips[0]:
        experts = True
    else:
        _, sampled = plan_views(_entries(manifest), manifest.get("config", {}), manifest.get("adapter_config"), False)
        if actual_skips[0] != {row["name"] for row in sampled if row.get("reason") == "expert_not_sampled"}:
            raise ValueError("Unknown fingerprint expert selection")
        experts = False
    has_parent = any(json.loads(meta.get("parent", "null")) is not None for meta in fingerprint_metadata)
    with_parent = settings.get("with_parent", has_parent or not manifest.get("declared_bases"))
    if not with_parent and has_parent:
        raise ValueError("Evidence summary disagrees with fingerprint parent policy")
    scientific["evidence"] = dict(sketch_k="auto" if k == (128 if special else 64) else k,
                                  all_experts=experts, with_parent=with_parent, device=settings.get("device", "cpu"))
    metadata = dict(model_key=key, repo_id=repo, revision=revision, kind=kind,
                    conventions_id=conventions_id(scientific), conventions=scientific, complete=True,
                    evidence={name: summary.get(name, 0) for name in ("reused", "measured", "gaps")})
    return safe, sources, metadata


@contextmanager
def _lock(path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), destination)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
        shutil.copy2(source, destination)


def _select_candidate(run, model_id):
    try:
        split_key(model_id)
    except ValueError:
        paths = _manifest_files(run)
    else:
        safe = safe_filename(model_id)
        path = Path(run) / "evidence" / safe / (safe + ".manifest.json")
        paths = [path] if path.is_file() else []
    found = []
    for path in paths:
        manifest = json.loads(path.read_text())
        key = model_key(manifest["repo_id"], manifest["revision"])
        if model_id in (manifest["repo_id"], key):
            candidate = _candidate(run, path)
            if candidate is not None:
                found.append(candidate)
    if len(found) > 1:
        raise ValueError("Several complete revisions match; publish an explicit repo@revision")
    return found[0] if found else None


def _destination(packages_root, safe, metadata):
    return Path(packages_root) / hashlib.sha256(safe.encode()).hexdigest()[:2] / safe / metadata["conventions_id"]


def _matches(destination, metadata, hashes):
    existing = json.loads((destination / "package.json").read_text()) if (destination / "package.json").is_file() else {}
    return (existing.get("model_key") == metadata["model_key"] and existing.get("kind") == metadata["kind"]
            and existing.get("conventions") == metadata["conventions"] and existing.get("files") == hashes
            and existing.get("complete") is True
            and all(_hash_file(destination / name) == digest for name, digest in hashes.items()))


def package_complete(run_dir, model_id, packages_root, data_root):
    try:
        selected = _select_candidate(Path(run_dir), model_id)
        if selected is None:
            return False
        safe, sources, metadata = selected
        return _matches(_destination(packages_root, safe, metadata), metadata,
                        {name: _hash_file(path) for name, path in sources.items()})
    except (OSError, ValueError, KeyError):
        return False


def publish_package(run_dir: Path, model_id: str, packages_root: Path, data_root: Path) -> Path | None:
    run, root = Path(run_dir).resolve(), Path(data_root).resolve()
    selected = _select_candidate(run, model_id)
    if selected is None:
        return None
    safe, sources, metadata = selected
    destination = _destination(packages_root, safe, metadata)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    with _lock(parent / ".publish.lock"):
        hashes = {name: _hash_file(path) for name, path in sorted(sources.items())}
        if destination.exists():
            if _matches(destination, metadata, hashes):
                return destination
            raise FileExistsError(f"Package already exists with different contents: {destination}")
        stage = Path(tempfile.mkdtemp(prefix=".publishing-", dir=parent))
        try:
            for name, path in sources.items():
                _link_or_copy(path, stage / name)
                if _hash_file(stage / name) != hashes[name]:
                    raise ValueError("Source artifact changed during package publication")
            summary = run / "evidence" / safe / "evidence.json"
            staged_summary = stage / ".run-evidence.json"
            if summary.is_file():
                shutil.copyfile(summary, staged_summary)
            staged = _candidate(stage, stage / (safe + ".manifest.json"), in_package=True, summary_path=staged_summary)
            staged_summary.unlink(missing_ok=True)
            if staged is None or staged[0] != safe or staged[2] != metadata or set(staged[1]) != set(sources):
                raise ValueError("Source artifact settings changed during package publication")
            metadata.update(files=hashes, source_run=run.relative_to(root).as_posix() if run.is_relative_to(root) else str(run),
                            created=datetime.now(timezone.utc).isoformat())
            (stage / "package.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
            os.rename(stage, destination)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return destination


def publish_packages(runs: list[Path], packages_root: Path, data_root: Path) -> list[Path]:
    published, seen = [], set()
    for run in runs:
        keys = sorted({model_key(m["repo_id"], m["revision"]) for path in _manifest_files(run)
                       for m in [json.loads(path.read_text())]})
        for key in keys:
            path = publish_package(run, key, packages_root, data_root)
            if path is not None and path not in seen:
                published.append(path)
                seen.add(path)
    return published


def package_inputs(packages, selected=None):
    """Select complete packages and verify their immutable artifact inventory."""
    selected = selected or current_conventions_id()
    rows, skipped, identities = [], 0, {}
    accepted = {selected}
    if selected == current_conventions_id():
        t1 = conventions()
        t1["evidence_tier"] = "t1_only"
        accepted.add(conventions_id(t1))
        # Streaming T1 may use a supervisor-assigned GPU even when full-mode
        # evidence defaults to CPU. Keep this policy explicit in its package.
        t1["evidence"] = dict(t1["evidence"], device="gpu")
        accepted.add(conventions_id(t1))
    for path in sorted(Path(packages).rglob("package.json")):
        # Private publication stages are invisible to readers.
        if any(part.startswith(".") for part in path.relative_to(packages).parts):
            continue
        meta = json.loads(path.read_text())
        if meta.get("complete") is not True:
            continue
        if meta.get("conventions_id") not in accepted:
            skipped += 1
            continue
        if meta["model_key"] != model_key(meta["repo_id"], meta["revision"]):
            raise ValueError("Invalid package identity")
        if meta["conventions"] != conventions(meta["conventions"]) or conventions_id(meta["conventions"]) != meta["conventions_id"]:
            raise ValueError("Package conventions do not match their identifier")
        safe = safe_filename(meta["model_key"])
        required = {safe + ".manifest.json", safe + ".t1.fingerprint.safetensors"}
        if meta.get("kind") == "full":
            required.update({safe + ".t2.fingerprint.safetensors", safe + ".t2.spectra.h5", safe + ".t1b128.fingerprint.safetensors", safe + ".t1b512.fingerprint.safetensors", "esd/stats.csv", "esd/metrics.h5"})
        elif meta.get("kind") != "t1_only":
            raise ValueError("Unknown model package kind")
        if not required.issubset(meta["files"]):
            raise ValueError("Incomplete package artifact inventory")
        for name, digest in meta["files"].items():
            artifact = (path.parent / name).resolve()
            if Path(name).is_absolute() or not artifact.is_relative_to(path.parent.resolve()):
                raise ValueError("Package artifact escapes its directory")
            if _hash_file(artifact) != digest:
                raise ValueError(f"Package artifact checksum mismatch: {name}")
        key = meta["model_key"]
        if key in identities:
            index = identities[key]
            previous = rows[index][1]
            if previous["kind"] == meta["kind"]:
                old_device = previous["conventions"]["evidence"]["device"]
                new_device = meta["conventions"]["evidence"]["device"]
                if meta["kind"] != "t1_only" or old_device == new_device:
                    raise ValueError("Duplicate package identity and tier")
                if new_device == "cpu":
                    rows[index] = (path.parent, meta)
                continue
            if meta["kind"] == "full":
                rows[index] = (path.parent, meta)
        else:
            identities[key] = len(rows)
            rows.append((path.parent, meta))
    return rows, skipped
