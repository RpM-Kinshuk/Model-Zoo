"""Offline-capable evidence commands with pinned inputs and resumable outputs."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
import tempfile
import time

from measurement_config import is_commit_sha, safe_filename

from . import EVIDENCE_VERSION
from .compare import check_verdict, diff_fingerprints
from .fingerprint import fingerprint_model, load_fingerprint, _atomic_path
from .frame import PILOT_HUB_PLAN, build_frame, draw_stratified, load_hub_stats
from .hub import AccessDenied, HubReader
from .manifest import build_manifest, pin_revision, write_manifest, _local_files
from .sketch import ROW_BLOCK, SKETCH_K_DEFAULT, SKETCH_SEED

_READER = None
_API = None


def _write(path, value):
    path = Path(path)
    temporary = _atomic_path(path)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv(path, rows, columns):
    path = Path(path)
    temporary = _atomic_path(path)
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataframe_csv(path, frame):
    temporary = _atomic_path(Path(path))
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _clean(out):
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("*.tmp"):
        path.unlink()


def _message(exc):
    message = str(exc)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[redacted]")
    message = re.sub(r"https?://\S+|hf_[A-Za-z0-9_]+", "[redacted]", message)
    return f"{type(exc).__name__}: {message}"


def _target(target, revision=""):
    """Resolve named lab paths before considering any Hub lookup."""
    raw = str(target)
    if Path(raw).is_dir():
        return str(Path(raw).resolve()), revision or "main", True
    source, marker, ref = raw.rpartition("@")
    if not marker:
        source, ref = raw, "main"
    root = os.environ.get("ZOO_DATA")
    if root and not Path(source).is_absolute():
        data_root = Path(root).resolve()
        candidate = (data_root / source).resolve()
        if not candidate.is_relative_to(data_root):
            raise ValueError("Local source escapes ZOO_DATA")
        if candidate.is_dir():
            return str(candidate), revision or ref, True
    if source.startswith(("lab/", "./", "../", "/")):
        raise FileNotFoundError("Local checkpoint directory does not exist")
    return source, revision or ref, False


def _make_manifest(target, revision, reader, api):
    source, ref, local = _target(target, revision)
    pinned = ref if local else pin_revision(api, source, ref)
    manifest = build_manifest(source, pinned, reader, api)
    if local and ref.startswith("local:") and manifest["revision"] != ref:
        raise ValueError("Local checkpoint does not match the requested content pin")
    if local and Path(manifest["repo_id"]).is_absolute() and os.environ.get("ZOO_DATA"):
        root = Path(os.environ["ZOO_DATA"]).resolve()
        if Path(source).resolve().is_relative_to(root):
            manifest["repo_id"] = Path(source).resolve().relative_to(root).as_posix()
        # Unnamed checkpoints outside ZOO_DATA retain X5's explicit-path
        # fallback; an explicit local input does not require a data root.
    return manifest


def _artifact(manifest, out, tier):
    safe = safe_filename(f"{manifest['repo_id']}@{manifest['revision']}")
    return Path(out) / f"{safe}.{tier}.fingerprint.safetensors"


def _complete(path, manifest, tier, k, height=ROW_BLOCK, parent=None):
    if not path.is_file():
        return False
    try:
        fp = load_fingerprint(path)
        expected = dict(repo_id=manifest["repo_id"], revision=manifest["revision"], tier=tier,
                        sketch_k=k, row_block=height, sketch_seed=SKETCH_SEED, parent=parent,
                        evidence_version=EVIDENCE_VERSION)
        return all(fp.meta.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def _base(manifest):
    bases = manifest.get("declared_bases", [])
    if not bases or not isinstance(bases[0], str):
        raise ValueError("Target has no declared base")
    return bases[0]


def _init_worker(factory, api):
    global _READER, _API
    _READER = (factory or HubReader)()
    _API = api


def _stats(reader):
    return {key: reader.stats[key] for key in ("bytes_read", "range_reads", "resolves")}


def _record(row, stage, reader, started, before):
    return dict(repo_id=row["repo_id"], requested_revision=row.get("requested_revision", row.get("revision", "")),
                stage=stage, stratum=row.get("stratum", ""), modality=row.get("modality", ""),
                elapsed_seconds=time.monotonic() - started,
                **{key: reader.stats[key] - before[key] for key in before})


def _manifest_job(row):
    started, before = time.monotonic(), _stats(_READER)
    manifest = None
    try:
        manifest = _make_manifest(row["repo_id"], row.get("revision", ""), _READER, _API)
        if manifest["access"] == "no_access":
            raise AccessDenied("Checkpoint access denied")
        status, reason, message = "complete", "", ""
    except Exception as exc:
        reason = "no_access" if isinstance(exc, AccessDenied) else "error"
        status, message = "failed", _message(exc)
        if manifest is None and reason == "no_access":
            manifest = dict(repo_id=row["repo_id"], revision=row.get("revision") or "main",
                            access="no_access", declared_bases=[], tensors=[])
    result = _record(row, "manifest", _READER, started, before)
    result.update(status=status, reason=reason, message=message,
                  revision=manifest["revision"] if manifest else row.get("revision", ""),
                  view_counts=manifest.get("view_counts", {}) if manifest else {})
    return result, manifest


def _batch_job(job):
    row, manifest, options, manifests = job
    started, before = time.monotonic(), _stats(_READER)
    result = None
    denied_manifest = None
    try:
        if manifest["access"] == "no_access":
            raise AccessDenied("Checkpoint access denied")
        if manifest["access"] != "granted":
            result = _record(row, "batch", _READER, started, before)
            result.update(status="skipped", reason="t0_only", message="No readable checkpoint weights",
                          revision=manifest["revision"], view_counts={})
            return result, None
        parent = None
        if options["with_parent"] and manifest.get("declared_bases"):
            target = _base(manifest)
            parent = _select_manifest(manifests, target, "")
            if parent is None:
                parent = _make_manifest(target, "", _READER, _API)
            if parent["access"] != "granted":
                raise AccessDenied("Declared parent weights unavailable")
        parent_id = {key: parent[key] for key in ("repo_id", "revision")} if parent else None
        out, tier, k = Path(options["out"]), options["tier"], options["sketch_k"]
        path = _artifact(manifest, out, tier)
        companions = [(path, tier, ROW_BLOCK)]
        if tier == "t2":
            companions += [(_artifact(manifest, out, "t1" if h == ROW_BLOCK else f"t1b{h}"),
                            "t1" if h == ROW_BLOCK else f"t1b{h}", h) for h in options["also_t1_blocks"]]
        complete = all(_complete(p, manifest, t, k, h, parent_id) for p, t, h in companions)
        # Expert selection is stored in coverage, not in the frozen metadata.
        if complete:
            fp = load_fingerprint(path)
            skipped_experts = any(c.get("reason") == "expert_not_sampled" for c in fp.coverage)
            if options["all_experts"] and skipped_experts:
                complete = False
            if not options["all_experts"]:
                from .fingerprint import _entries
                from .views import plan_views
                _, expected_coverage = plan_views(_entries(manifest), manifest.get("config", {}),
                                                  manifest.get("adapter_config"), False)
                expected = {c["name"] for c in expected_coverage if c.get("reason") == "expert_not_sampled"}
                actual = {c["name"] for c in fp.coverage if c.get("reason") == "expert_not_sampled"}
                complete = expected == actual
        if not complete:
            path = fingerprint_model(manifest, _READER, tier, out, parent=parent,
                                     parent_cache=options["parent_cache"], sketch_k=k,
                                     all_experts=options["all_experts"],
                                     also_t1_blocks=tuple(options["also_t1_blocks"]),
                                     compute_device=options.get("compute_device", "cpu"))
        fp = load_fingerprint(path)
        result = _record(row, "batch", _READER, started, before)
        result.update(status="skipped" if complete else "complete", reason="already_complete" if complete else "",
                      message="", revision=manifest["revision"], view_counts=dict(sorted(Counter(fp.geometry).items())))
        # Keep measured extraction counters when resuming a completed artifact.
        result.update({key: fp.meta[key] for key in ("bytes_read", "range_reads", "resolves", "elapsed_seconds")})
    except Exception as exc:
        result = _record(row, "batch", _READER, started, before)
        result.update(status="failed", reason="no_access" if isinstance(exc, AccessDenied) else "error",
                      message=_message(exc), revision=manifest["revision"], view_counts={})
        if isinstance(exc, AccessDenied):
            denied_manifest = dict(manifest, access="no_access")
    return result, denied_manifest


def _read_models(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(not row.get("repo_id") for row in rows):
        raise ValueError("models CSV requires repo_id")
    identities = [(row["repo_id"], row.get("revision", "")) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("models CSV contains duplicate identities")
    return sorted(rows, key=lambda row: (row["repo_id"], row.get("revision", "")))


def _manifests(directory):
    result = []
    if Path(directory).is_dir():
        for path in sorted(Path(directory).glob("*.manifest.json")):
            result.append(json.loads(path.read_text()))
    return result


def _select_manifest(manifests, target, revision):
    source, ref, local = _target(target, revision)
    requested = revision or (ref if ref != "main" else "")
    current_pin = _local_files(Path(source))[1] if local else None
    matches = []
    for manifest in manifests:
        same = manifest["repo_id"] in (target, source)
        if local and not same and manifest["revision"].startswith("local:"):
            from .fingerprint import _source
            same = Path(_source(manifest)).resolve() == Path(source).resolve()
        if (same and (not requested or manifest["revision"] == requested)
                and (not local or manifest["revision"] == current_pin)):
            matches.append(manifest)
    if len(matches) > 1:
        raise ValueError("Multiple manifests match; pin the models CSV revision")
    return matches[0] if matches else None


def _reports(out, records, stage):
    previous = []
    failures_path = out / "failures.csv"
    if failures_path.exists():
        with failures_path.open(newline="") as stream:
            previous = list(csv.DictReader(stream))
    failures = {(r["repo_id"], r["stage"], r["reason"]): r for r in previous}
    for record in records:
        if record["status"] == "failed":
            failures[(record["repo_id"], stage, record["reason"])] = record
    _csv(failures_path, sorted(failures.values(), key=lambda r: (r["repo_id"], r["stage"], r["reason"])),
         ["repo_id", "stage", "reason", "message"])
    summary = dict(stage=stage, models=sorted(records, key=lambda r: (r["repo_id"], r.get("revision", ""))),
                   counts=dict(sorted(Counter(r["status"] for r in records).items())))
    _write(out / "summary.json", summary)
    return 2 if any(r["status"] == "failed" for r in records) else 0


def _run_many(args, factory, api):
    out = Path(args.out)
    _clean(out)
    rows = _read_models(args.models)
    manifests = _manifests(args.out if args.command == "manifest" else args.manifests)
    records, jobs = [], []
    for row in rows:
        try:
            manifest = _select_manifest(manifests, row["repo_id"], row.get("revision", ""))
        except Exception as exc:
            records.append(dict(row, stage=args.command, status="failed", reason="error", message=_message(exc),
                                bytes_read=0, range_reads=0, resolves=0, elapsed_seconds=0, view_counts={}))
            continue
        if manifest and manifest["access"] == "no_access" and not args.retry_denied:
            records.append(dict(row, revision=manifest["revision"], stage=args.command, status="failed",
                                reason="no_access", message="Checkpoint access denied", bytes_read=0,
                                range_reads=0, resolves=0, elapsed_seconds=0, view_counts={}))
        elif args.command == "manifest" and manifest and manifest["access"] != "no_access":
            records.append(dict(row, revision=manifest["revision"], stage="manifest", status="skipped",
                                reason="already_complete", message="", bytes_read=0, range_reads=0, resolves=0,
                                elapsed_seconds=0, view_counts=manifest.get("view_counts", {})))
        elif args.command == "manifest":
            if manifest and manifest["access"] == "no_access":
                jobs.append(dict(row, revision=_retry_revision(manifest, row),
                                 requested_revision=row.get("revision", "")))
            else:
                jobs.append(row)
        elif manifest is None:
            records.append(dict(row, stage="batch", status="failed", reason="missing_manifest",
                                message="No manifest matches the requested identity", bytes_read=0,
                                range_reads=0, resolves=0, elapsed_seconds=0, view_counts={}))
        else:
            if manifest["access"] == "no_access" and args.retry_denied:
                # Retry metadata first; retain the original identity pin.
                jobs.append((row, manifest, dict(vars(args), retry_manifest=True), manifests))
            else:
                jobs.append((row, manifest, vars(args), manifests))
    function = _manifest_job if args.command == "manifest" else _batch_with_retry
    if args.workers == 1:
        _init_worker(factory, api)
        results = map(function, jobs)
        for record, manifest in results:
            records.append(record)
            if manifest:
                write_manifest(manifest, out if args.command == "manifest" else Path(args.manifests))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(factory, api)) as pool:
            for record, manifest in pool.map(function, jobs):
                records.append(record)
                if manifest:
                    write_manifest(manifest, out if args.command == "manifest" else Path(args.manifests))
    if args.command == "manifest" and args.include_declared_bases:
        known = {r["repo_id"] for r in rows}
        base_rows = []
        failed = {(r["repo_id"], r.get("requested_revision", r.get("revision", "")))
                  for r in records if r["status"] == "failed"}
        for selected in rows:
            if (selected["repo_id"], selected.get("revision", "")) in failed:
                continue
            try:
                manifest = _select_manifest(_manifests(out), selected["repo_id"], selected.get("revision", ""))
            except Exception as exc:
                records.append(dict(selected, stage="manifest", status="failed", reason="error", message=_message(exc),
                                    bytes_read=0, range_reads=0, resolves=0, elapsed_seconds=0, view_counts={}))
                continue
            if manifest is None:
                continue
            for base in manifest.get("declared_bases", []):
                if isinstance(base, str) and base not in known:
                    known.add(base)
                    base_rows.append(dict(repo_id=base, revision=""))
        # Only direct declared bases are included; cycles and transitive graphs
        # do not turn a selected metadata run into an unbounded traversal.
        if base_rows:
            _init_worker(factory, api)
            for row in base_rows:
                try:
                    existing = _select_manifest(_manifests(out), row["repo_id"], "")
                    if existing and (existing["access"] != "no_access" or not args.retry_denied):
                        records.append(dict(row, stage="manifest", revision=existing["revision"],
                                            status="failed" if existing["access"] == "no_access" else "skipped",
                                            reason="no_access" if existing["access"] == "no_access" else "already_complete",
                                            message="", bytes_read=0, range_reads=0, resolves=0,
                                            elapsed_seconds=0, view_counts=existing.get("view_counts", {})))
                        continue
                    if existing:
                        row = dict(row, revision=_retry_revision(existing, row),
                                   requested_revision=row.get("revision", ""))
                    record, manifest = _manifest_job(row)
                    records.append(record)
                    if manifest:
                        write_manifest(manifest, out)
                except Exception as exc:
                    records.append(dict(row, stage="manifest", status="failed", reason="error", message=_message(exc),
                                        bytes_read=0, range_reads=0, resolves=0, elapsed_seconds=0, view_counts={}))
    code = _reports(out, records, args.command)
    print(json.dumps({"counts": dict(Counter(r["status"] for r in records))}, sort_keys=True))
    return code


def _retry_revision(manifest, row):
    revision = manifest["revision"]
    if is_commit_sha(revision) or revision.startswith("local:"):
        return revision
    return row.get("revision", "")


def _batch_with_retry(job):
    row, manifest, options, manifests = job
    if options.get("retry_manifest"):
        try:
            manifest = _make_manifest(row["repo_id"], _retry_revision(manifest, row), _READER, _API)
        except Exception as exc:
            return dict(row, stage="batch", status="failed", reason="no_access" if isinstance(exc, AccessDenied) else "error",
                        message=_message(exc), bytes_read=0, range_reads=0, resolves=0,
                        elapsed_seconds=0, view_counts={}), None
        result, denied_manifest = _batch_job((row, manifest, options, manifests))
        return result, denied_manifest or manifest
    return _batch_job(job)


def _default_out():
    if not os.environ.get("ZOO_DATA"):
        raise ValueError("Set ZOO_DATA or supply --out")
    return Path(os.environ["ZOO_DATA"]) / "evidence" / "fingerprints"


def _single(args, factory, api):
    reader = (factory or HubReader)()
    if args.command == "fingerprint":
        out = Path(args.out) if args.out else _default_out()
        manifest = _make_manifest(args.target, "", reader, api)
        if manifest["access"] != "granted":
            raise ValueError("Target has no accessible checkpoint weights")
        write_manifest(manifest, out)
        path = fingerprint_model(manifest, reader, args.tier, out, sketch_k=args.sketch_k,
                                 all_experts=args.all_experts, compute_device=args.compute_device)
        fp = load_fingerprint(path)
        print(json.dumps({"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                          "fingerprint": path.name, "tier": args.tier, "n_views": len(fp.names)}, sort_keys=True))
        return 0
    first = Path(args.a)
    a = load_fingerprint(first) if first.is_file() and first.suffix == ".safetensors" else None
    ma = None if a else _make_manifest(args.a, "", reader, api)
    if a and not args.b:
        manifest_path = first.with_name(first.name.split(f".{a.meta['tier']}.fingerprint.safetensors")[0] + ".manifest.json")
        ma = json.loads(manifest_path.read_text())
    target_b = args.b or _base(ma)
    second = Path(target_b)
    b = load_fingerprint(second) if second.is_file() and second.suffix == ".safetensors" else None
    mb = None if b else _make_manifest(target_b, "", reader, api)
    null = json.loads(Path(args.null).read_text()) if args.null else None
    if a is None or b is None:
        root = _default_out().parent
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="comparison-", dir=root) as temporary:
            tier = a.meta["tier"] if a else b.meta["tier"] if b else "t1"
            if tier not in ("t1", "t2"):
                raise ValueError("Build comparison targets at t1 or t2")
            k = a.meta["sketch_k"] if a else b.meta["sketch_k"] if b else SKETCH_K_DEFAULT
            if a is None:
                if ma["access"] != "granted":
                    raise ValueError("First target has no accessible checkpoint weights")
                a = load_fingerprint(fingerprint_model(ma, reader, tier, Path(temporary), parent=mb,
                                                      sketch_k=k))
            if b is None:
                if mb["access"] != "granted":
                    raise ValueError("Base has no accessible checkpoint weights")
                b = load_fingerprint(fingerprint_model(mb, reader, tier, Path(temporary), sketch_k=k))
    result = diff_fingerprints(a, b, null)
    if args.command == "check":
        result["verdict"] = check_verdict(result)
    if getattr(args, "markdown", False):
        print(f"Measured difference: {result['relatedness']['verdict']}\n")
        print("| View | Status | Cosine |\n| --- | --- | --- |")
        for name, record in result["views"].items():
            print(f"| {name.replace('|', '&#124;')} | {record['status']} | {record['cosine']} |")
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result.get("verdict") == "inconsistent" else 0


def _parser():
    parser = argparse.ArgumentParser(prog="python -m evidence.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("fingerprint")
    single.add_argument("target")
    single.add_argument("--tier", choices=("t1", "t2"), default="t1")
    single.add_argument("--out")
    single.add_argument("--sketch_k", type=int, default=SKETCH_K_DEFAULT)
    single.add_argument("--all_experts", action="store_true")
    single.add_argument("--compute_device", default="cpu", help="T2 spectrum device: cpu or cuda:<index>")
    single.add_argument("--json", action="store_true")
    for name in ("diff", "check"):
        command = commands.add_parser(name)
        command.add_argument("a", metavar="TARGET" if name == "check" else "A")
        if name == "diff":
            command.add_argument("b", nargs="?")
        else:
            command.set_defaults(b=None)
        command.add_argument("--null")
        formatting = command.add_mutually_exclusive_group()
        formatting.add_argument("--json", action="store_true")
        if name == "diff":
            formatting.add_argument("--markdown", action="store_true")
    frame = commands.add_parser("frame")
    frame.add_argument("--hub_stats", required=True)
    frame.add_argument("--out", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--frame", required=True)
    sample.add_argument("--plan", choices=("pilot",), required=True)
    sample.add_argument("--seed", type=int, default=SKETCH_SEED)
    sample.add_argument("--out", required=True)
    for name in ("manifest", "batch"):
        command = commands.add_parser(name)
        command.add_argument("--models", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--retry_denied", action="store_true")
        if name == "manifest":
            command.add_argument("--include_declared_bases", action="store_true")
        else:
            command.add_argument("--manifests", required=True)
            command.add_argument("--tier", choices=("t1", "t2"), required=True)
            command.add_argument("--sketch_k", type=int, default=SKETCH_K_DEFAULT)
            command.add_argument("--all_experts", action="store_true")
            command.add_argument("--compute_device", default="cpu", help="T2 spectrum device: cpu or cuda:<index>")
            command.add_argument("--also_t1_blocks", default="")
            command.add_argument("--with_parent", action="store_true")
            command.add_argument("--parent_cache", type=Path)
    return parser


def main(argv: list[str], *, reader_factory=None, api=None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command in ("manifest", "batch"):
            if args.workers < 1:
                raise ValueError("workers must be positive")
            if args.command == "batch":
                args.also_t1_blocks = sorted(set(int(h) for h in args.also_t1_blocks.split(",") if h))
                if args.tier == "t1" and args.also_t1_blocks:
                    raise ValueError("Additional T1 block outputs require tier t2")
                if any(h < 1 for h in args.also_t1_blocks):
                    raise ValueError("Block heights must be positive")
            return _run_many(args, reader_factory, api)
        if args.command == "frame":
            frame = build_frame(load_hub_stats(Path(args.hub_stats)))
            _dataframe_csv(Path(args.out), frame)
            _write(Path(args.out).with_suffix(".excluded.json"), frame.attrs.get("excluded", []))
            print(json.dumps({"n_models": len(frame)}))
            return 0
        if args.command == "sample":
            import pandas as pd
            frame = pd.read_csv(args.frame)
            sample = draw_stratified(frame, PILOT_HUB_PLAN, args.seed)
            _dataframe_csv(Path(args.out), sample)
            print(json.dumps({"n_models": len(sample)}))
            return 0
        return _single(args, reader_factory, api)
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:
        print(json.dumps({"error": _message(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
