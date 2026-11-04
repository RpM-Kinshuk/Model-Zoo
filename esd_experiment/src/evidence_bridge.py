"""Runner/worker evidence glue; heavy imports stay inside runtime functions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from queue import Empty, Queue

from measurement_config import safe_filename


def add_evidence_arguments(parser, *, worker=False):
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packages_root", type=Path)
    parser.add_argument("--evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidence_device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--evidence_max_inflight_gib", type=float, default=2)
    parser.add_argument("--parent_cache", type=Path)
    parser.add_argument("--parent_cache_max_gib", type=float, default=64)
    parser.add_argument("--no-parent_cache", action="store_true")
    parser.add_argument("--with_parent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_models", type=Path)
    parser.add_argument("--stratum", default="")
    parser.add_argument("--sketch_k", default="auto")
    parser.add_argument("--all_experts", action=argparse.BooleanOptionalAction, default="auto")
    if worker:
        parser.add_argument("--cache_self", action="store_true")
        parser.add_argument("--evidence_only", action="store_true")


def validate_evidence_arguments(args, parser):
    for name in ("evidence_max_inflight_gib", "parent_cache_max_gib"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "evidence_max_inflight_gib" and value == 0):
            parser.error(f"--{name} must be finite and {'positive' if name == 'evidence_max_inflight_gib' else 'nonnegative'}")
    if args.sketch_k != "auto":
        try:
            if not 1 <= int(args.sketch_k) <= 256:
                raise ValueError
        except ValueError:
            parser.error("--sketch_k must be auto or an integer between 1 and 256")
    if args.analysis_source == "checkpoint" and args.evidence:
        print("Checkpoint mode is test-only; evidence is disabled.")
        args.evidence = False


def identity(model_id, revision=""):
    repo, _, embedded = model_id.partition("@")
    return repo, revision or embedded


def evidence_dir(output_dir, repo, revision):
    return Path(output_dir) / "evidence" / safe_filename(f"{repo}@{revision}")


def evidence_settings(args, stratum=None):
    special = (getattr(args, "stratum", "") if stratum is None else stratum) in ("lab", "anchor")
    k, experts = getattr(args, "sketch_k", "auto"), getattr(args, "all_experts", "auto")
    return dict(sketch_k=(128 if special else 64) if k == "auto" else int(k),
                all_experts=special if experts == "auto" else experts,
                with_parent=getattr(args, "with_parent", True),
                device=getattr(args, "evidence_device", "cpu"))


def quarantine_evidence(output_dir, repo, revision):
    folder = evidence_dir(output_dir, repo, revision)
    marker = folder / (folder.name + ".t2.fingerprint.safetensors")
    if not marker.is_file():
        return None
    backup = folder.with_name("." + folder.name + ".previous-" + uuid.uuid4().hex)
    os.replace(folder, backup)
    folder.mkdir()
    manifest = backup / (folder.name + ".manifest.json")
    if manifest.is_file():
        shutil.copyfile(manifest, folder / manifest.name)
    return backup


def evidence_complete(output_dir, model_id, revision="", *, measurement=None, options=None, stratum=None):
    repo, revision = identity(model_id, revision)
    coverage = Path(output_dir) / "logs/coverage" / f"{safe_filename(model_id)}.json"
    if coverage.is_file():
        try:
            if json.loads(coverage.read_text()).get("evidence") == "skipped_checkpoint_fallback":
                return True
        except (ValueError, OSError):
            pass
    folders = ([evidence_dir(output_dir, repo, revision)] if revision else
               (Path(output_dir) / "evidence").glob(safe_filename(repo) + "__*"))
    for folder in folders:
        marker = folder / (folder.name + ".t2.fingerprint.safetensors")
        manifest_path = folder / (folder.name + ".manifest.json")
        if marker.is_file() and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                if manifest["repo_id"] != repo or (revision and manifest["revision"] != revision):
                    continue
                if options is not None:
                    summary = json.loads((folder / "evidence.json").read_text())
                    if summary.get("settings") != evidence_settings(options, summary.get("stratum", getattr(options, "stratum", "")) if stratum is None else stratum):
                        continue
                if measurement is not None:
                    import h5py
                    with h5py.File(folder / (folder.name + ".t2.spectra.h5")) as h:
                        actual = json.loads(h.attrs["measurement_config_json"])
                    if any(actual.get(key) != value for key, value in measurement.items() if key != "runtime"):
                        continue
                return True
            except (ValueError, OSError, KeyError):
                pass
    return False


def _manifest(output_dir, repo, revision):
    folder = evidence_dir(output_dir, repo, revision)
    path = folder / (folder.name + ".manifest.json")
    return json.loads(path.read_text()) if path.is_file() else None


def prepare_evidence_manifests(rows, output_dir):
    if not os.environ.get("HF_TOKEN"):
        return
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest, pin_revision, write_manifest
    reader, known, bases = HubReader(local_cache=True), set(), []
    for row in rows.to_dict("records"):
        if row.get("pin_status") == "error":
            continue
        repo, revision = identity(row["model_id"], row.get("revision_norm", ""))
        if not revision:
            continue
        manifest = _manifest(output_dir, repo, revision) or build_manifest(repo, revision, reader, None)
        write_manifest(manifest, evidence_dir(output_dir, repo, revision))
        known.add(repo)
        bases.extend(manifest.get("declared_bases", []))
        if row.get("source_model"):
            bases.append(row["source_model"])
    for target in dict.fromkeys(base for base in bases if isinstance(base, str)):
        if not isinstance(target, str):
            continue
        repo, revision = identity(target)
        if repo in known:
            continue
        known.add(repo)
        try:
            matches = list((Path(output_dir) / "evidence").glob(safe_filename(repo) + "__*/*.manifest.json"))
            if not revision and len(matches) == 1:
                revision = json.loads(matches[0].read_text())["revision"]
            revision = pin_revision(None, repo, revision or "main")
            manifest = _manifest(output_dir, repo, revision) or build_manifest(repo, revision, reader, None)
            write_manifest(manifest, evidence_dir(output_dir, repo, revision))
        except Exception:
            print(f"Declared-parent evidence metadata unavailable: {repo}; workers will record a delta gap")


def cached_models(path):
    if not path:
        return set()
    with Path(path).open(newline="") as stream:
        return {identity(row.get("repo_id") or row["model_id"])[0] for row in csv.DictReader(stream)}


def create_sidecar(args, output_dir, repo, revision):
    from evidence.block_cache import BlockCache
    from evidence.hub import HubReader, AccessDenied, OfflineError
    from evidence.manifest import build_manifest, pin_revision
    from evidence.sidecar import EvidenceSidecar
    reader = HubReader(local_cache=True)
    manifest = _manifest(output_dir, repo, revision) or build_manifest(repo, revision, reader, None)
    if manifest["access"] != "granted":
        raise ValueError("Local downloaded checkpoint has no readable evidence weights")
    parent = None
    bases = manifest.get("declared_bases", [])
    if getattr(args, "with_parent", True) and (bases or args.source_model):
        target = args.source_model or bases[0]
        parent_repo, parent_revision = identity(target)
        if parent_revision:
            parent = _manifest(output_dir, parent_repo, parent_revision)
        else:
            matches = list((Path(output_dir) / "evidence").glob(safe_filename(parent_repo) + "__*/*.manifest.json"))
            if len(matches) == 1:
                parent = json.loads(matches[0].read_text())
                parent_revision = parent["revision"]
        if parent is None:
            try:
                if not parent_revision and os.environ.get("HF_TOKEN") and os.environ.get("HF_HUB_OFFLINE") != "1":
                    parent_revision = pin_revision(None, parent_repo, "main")
                if parent_revision:
                    parent = _manifest(output_dir, parent_repo, parent_revision) or build_manifest(parent_repo, parent_revision, reader, None)
            except Exception:
                parent = None
            if parent is None or parent.get("access") != "granted":
                parent = dict(repo_id=parent_repo, revision=parent_revision, access="parent_unavailable", tensors=[], config={})
        if parent.get("access") != "granted":
            parent = dict(parent, access="parent_unavailable", tensors=[], config={})
    root = getattr(args, "parent_cache", None) or (Path(os.environ.get("ZOO_DATA", output_dir)) / "cache/parent-blocks")
    cache = None if getattr(args, "no_parent_cache", False) else BlockCache(root, int(args.parent_cache_max_gib * 1024**3))
    special = args.stratum in ("lab", "anchor")
    return EvidenceSidecar(manifest, evidence_dir(output_dir, repo, revision), parent=parent, block_cache=cache,
                           cache_self=args.cache_self or special or repo in cached_models(args.cache_models),
                           device="cuda:0" if args.evidence_device == "gpu" else "cpu",
                           max_inflight_bytes=int(args.evidence_max_inflight_gib * 1024**3),
                           sketch_k=(128 if special else 64) if args.sketch_k == "auto" else int(args.sketch_k),
                           all_experts=special if args.all_experts == "auto" else args.all_experts)


def _publication_paths(args, output_dir):
    root = Path(os.environ.get("ZOO_DATA", output_dir))
    return root, getattr(args, "packages_root", None) or root / "packages"


def _skip_publication(args, output_dir, model_id):
    if not getattr(args, "publish", False) or not getattr(args, "evidence", False):
        return True
    coverage = Path(output_dir) / "logs/coverage" / f"{safe_filename(model_id)}.json"
    return coverage.is_file() and json.loads(coverage.read_text()).get("evidence") == "skipped_checkpoint_fallback"


def publish_model(args, output_dir, model_id, revision=""):
    if _skip_publication(args, output_dir, model_id):
        return None
    from zoo.publish import publish_package
    root, packages = _publication_paths(args, output_dir)
    repo, pin = identity(model_id, revision)
    result = publish_package(output_dir, f"{repo}@{pin}" if pin else repo, packages, root)
    if result is None:
        raise ValueError("Full-mode artifacts are incomplete for package publication")
    return result


def publication_complete(args, output_dir, model_id, revision=""):
    if _skip_publication(args, output_dir, model_id):
        return True
    from zoo.publish import package_complete
    root, packages = _publication_paths(args, output_dir)
    repo, pin = identity(model_id, revision)
    return package_complete(output_dir, f"{repo}@{pin}" if pin else repo, packages, root)


def loaded_block(model):
    def get(name, bounds):
        import torch
        module_name, _, attribute = name.rpartition(".")
        try:
            module = model.get_submodule(module_name) if module_name else model
            tensor = getattr(module, attribute)
        except (AttributeError, KeyError):
            return None
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.device.type == "meta":
            return None
        return tensor.detach()[bounds[0]:bounds[1]]
    return get


def metrics_results(metrics):
    return {name: {key: values[i] for key, values in metrics.items()} for i, name in enumerate(metrics["longname"]) if name is not None}


def read_metrics(path):
    import h5py
    with h5py.File(path) as h:
        metrics = {key: (dataset.asstr()[:].tolist() if h5py.check_string_dtype(dataset.dtype) else dataset[:].tolist())
                   for key, dataset in h["layers"].items()}
        metrics["eigs"] = list(h["eigs"][:]) if "eigs" in h else [None] * len(metrics["longname"])
        coverage = json.loads(h["coverage"].asstr()[()]) if "coverage" in h else {"counts": {}}
    return metrics, coverage


def measure_tensors(items, measurement):
    import torch
    from net_esd.core import compute_esd_for_weight
    def one(name, tensor, device):
        with torch.no_grad():
            return compute_esd_for_weight(name, tensor, measurement["evals_thresh"], measurement["bins"],
                measurement["fix_fingers"], measurement["xmin_pos"], measurement["conv_norm"],
                measurement["filter_zeros"], measurement["use_svd"], True, device, tensor.numel(), measurement["compute_dtype"])
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cpu = measurement.get("runtime", {}).get("device_map_requested") == "cpu"
    if cpu or not assigned or assigned == "-1" or not torch.cuda.is_available():
        return {name: one(name, tensor, None) for name, tensor in items}
    queue, results = Queue(), {}
    for item in items:
        queue.put(item)
    def consume(device):
        while True:
            try:
                name, tensor = queue.get_nowait()
            except Empty:
                return
            results[name] = one(name, tensor, device)
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as pool:
        list(pool.map(consume, range(torch.cuda.device_count())))
    return results
