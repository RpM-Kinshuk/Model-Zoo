"""metadata-only checkpoint metadata, using pinned files without loading model weights."""

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from measurement_config import is_commit_sha, safe_filename

from . import EVIDENCE_VERSION
from .hub import AccessDenied, HubReader, OfflineError
from .views import plan_views

MAX_JSON_BYTES = 8 * 1024 * 1024
_WEIGHTS = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".onnx")
_PICKLE = (".bin", ".pt", ".pth")
_PRECISION = {"bf16", "fp16", "f16", "fp32", "f32", "fp8", "e4m3fn",
              "int8", "int4", "nvfp4", "mxfp4", "comfy", "convrot", "scaled"}
_CONFIG_FIELDS = ("model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
                  "num_attention_heads", "num_key_value_heads", "vocab_size",
                  "tie_word_embeddings", "quantization_config")


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _mapping(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return obj.to_dict() if hasattr(obj, "to_dict") else vars(obj)


def _model_info(api, repo_id, revision):
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise OfflineError("Hub reads are disabled by HF_HUB_OFFLINE")
    if api is None:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise AccessDenied("Hub metadata requires HF_TOKEN")
        api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"), token=token)
    try:
        return api.model_info(repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        if _get(_get(exc, "response"), "status_code") in (401, 403):
            raise AccessDenied("Hub metadata access denied") from None
        raise


def _cached_model_info(repo_id, revision):
    """Inspect an existing pinned HF snapshot without metadata/network access."""
    from huggingface_hub import ModelCard, snapshot_download
    root = Path(snapshot_download(repo_id, revision=revision, local_files_only=True))
    card = ModelCard.load(root / "README.md").data.to_dict() if (root / "README.md").is_file() else {}
    siblings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".env":
            continue
        blob = path.resolve().name
        digest = blob if re.fullmatch(r"[0-9a-f]{64}", blob) else None
        siblings.append(dict(rfilename=path.relative_to(root).as_posix(), size=path.stat().st_size,
                             lfs=dict(sha256=digest), xet_hash=None))
    return dict(sha=revision, siblings=siblings, card_data=card, tags=[], gated=False)


def pin_revision(api, repo_id: str, revision: str = "main") -> str:
    """Resolve a branch once; already pinned commits need no API call."""
    if is_commit_sha(revision):
        return revision
    sha = _get(_model_info(api, repo_id, revision), "sha")
    if not is_commit_sha(sha):
        raise ValueError("Hub revision did not resolve to a 40-character commit SHA")
    return sha


def _variant(filename):
    path = PurePosixPath(filename)
    tokens = re.split(r"[._-]+", path.stem.lower())
    present = set(tokens) & _PRECISION
    key = (str(path.parent), path.suffix.lower(), tuple(t for t in tokens if t not in _PRECISION))
    rank = next((i for i, options in enumerate(({"bf16"}, {"fp16", "f16"},
                                               {"fp32", "f32"}, {"fp8", "e4m3fn"}))
                 if present & options), 4)
    return key, rank


def _layout_rank(filename):
    """Canonical component/index layouts precede their precision variants."""
    tokens = set(re.split(r"[._-]+", PurePosixPath(filename).name.lower()))
    return _variant(filename)[1] if tokens & _PRECISION else -1


def _select_indexes(filenames):
    if "model_index.json" not in filenames:
        if "model.safetensors" in filenames:
            return []
        if "model.safetensors.index.json" in filenames:
            return ["model.safetensors.index.json"]
        if "adapter_model.safetensors" in filenames:
            return []
    indexes = sorted(name for name in filenames if name.endswith(".index.json"))
    if any(not name.startswith("original/") for name in indexes):
        indexes = [name for name in indexes if not name.startswith("original/")]
    is_adapter_index = lambda name: "lora" in PurePosixPath(name).name.lower() or "loras" in PurePosixPath(name).parts
    if any(not is_adapter_index(name) for name in indexes):
        indexes = [name for name in indexes if not is_adapter_index(name)]
    if "model_index.json" in filenames:
        indexes = [name for name in indexes if PurePosixPath(name).parent != PurePosixPath(".")]
    if any(name.endswith(".safetensors") for name in filenames):
        indexes = [name for name in indexes if not name.endswith((".bin.index.json", ".pt.index.json", ".pth.index.json"))]
    components = {}
    for name in indexes:
        components.setdefault(PurePosixPath(name).parent, []).append(name)
    return sorted(min(group, key=lambda name: (_layout_rank(name), name)) for group in components.values())


def select_file_set(filenames: list[str]) -> tuple[list[str], list[dict]]:
    """Choose one checkpoint layout and record every excluded weight file."""
    names = sorted(set(filenames))
    weights = [name for name in names if name.lower().endswith(_WEIGHTS)]
    safe = [name for name in weights if name.endswith(".safetensors")]
    adapters = {name for name in weights if "lora" in PurePosixPath(name).name.lower()
                or "loras" in PurePosixPath(name).parts
                or (PurePosixPath(name).name.startswith("adapter_model.")
                    and str(PurePosixPath(name).parent / "adapter_config.json") in names)}
    diffusers = "model_index.json" in names and any("/" in name for name in safe)
    hf_shards = any(re.search(r"-\d{5}-of-\d{5}\.safetensors$", name) for name in safe)
    root_safe = any("/" not in name for name in safe)
    canonical_root = not diffusers and bool({
        "model.safetensors", "adapter_model.safetensors", "model.safetensors.index.json",
    } & set(names))
    reasons = {}
    for name in weights:
        lower = name.lower()
        if safe and lower.endswith(_PICKLE):
            reasons[name] = "pickle_not_read"
        elif canonical_root and name.endswith(".safetensors") and "/" in name:
            reasons[name] = "duplicate_layout"
        elif (not diffusers and "model.safetensors" in names
              and re.search(r"-\d{5}-of-\d{5}\.safetensors$", name)
              and _variant(re.sub(r"-\d{5}-of-\d{5}(?=\.safetensors$)", "", name))[0]
              == _variant("model.safetensors")[0]):
            reasons[name] = "duplicate_layout"
        elif diffusers and ("/" not in name or ".fp16." in lower or "ema" in PurePosixPath(name).stem.lower()
                            or "_0.9vae" in lower):
            reasons[name] = "duplicate_layout"
        elif (hf_shards and PurePosixPath(name).name.startswith("consolidated")) or (
                (hf_shards or root_safe) and name.startswith("original/")):
            reasons[name] = "duplicate_layout"
    bundled = any(name not in reasons and name not in adapters for name in weights)
    if bundled:
        for name in adapters:
            reasons[name] = "bundled_adapter"
    if diffusers:
        components = {}
        for name in safe:
            if name not in reasons:
                components.setdefault(PurePosixPath(name).parent, []).append(name)
        for component in components.values():
            shards = [name for name in component if re.search(r"-\d{5}-of-\d{5}\.safetensors$", name)]
            preferred = shards or sorted(component, key=lambda name: (_layout_rank(name), name))[:1]
            for name in component:
                if name not in preferred:
                    reasons[name] = "duplicate_layout"
    groups = {}
    for name in weights:
        if name not in reasons:
            key, rank = _variant(name)
            groups.setdefault(key, []).append((rank, name))
    for group in groups.values():
        for _, name in sorted(group)[1:]:
            reasons[name] = "precision_variant"
    return ([name for name in weights if name not in reasons],
            [dict(filename=name, reason=reasons[name]) for name in sorted(reasons)])


def _read_json(reader, source, revision, filename):
    remote = reader.resolve(source, filename, revision)
    if remote.size > MAX_JSON_BYTES:
        raise ValueError("Checkpoint JSON exceeds the 8 MiB limit")
    try:
        value = json.loads(reader.read(remote, 0, remote.size))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid checkpoint JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Checkpoint JSON must be an object")
    return value


def _local_files(root):
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".env")
    files = []
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Local checkpoint file escapes its directory")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append(dict(filename=path.relative_to(root).as_posix(), size=path.stat().st_size,
                          sha256=digest.hexdigest(), xet_hash=None))
    pin = hashlib.sha256("".join(sorted(f["sha256"] for f in files)).encode()).hexdigest()
    return files, "local:" + pin


def _config_subset(config):
    text = config.get("text_config", {})
    if not isinstance(text, dict):
        text = {}
    merged = {**text, **config}
    result = {key: merged[key] for key in _CONFIG_FIELDS if key in merged}
    aliases = ("num_experts", "num_local_experts", "n_routed_experts")
    for source in (config, text):
        value = next((source[k] for k in aliases if k in source), None)
        if value is not None:
            result["num_experts"] = value
            break
    return result


def _decision(config, tags, counts):
    system_one = bool({"system-one", "decision-model"} & set(tags))
    labels = config.get("id2label", {})
    labels = labels if isinstance(labels, dict) else {}
    num = config.get("num_labels", len(labels) or None)
    architectures = config.get("architectures", []) or []
    endings = ("ForSequenceClassification", "ForTokenClassification",
               "ForMultipleChoice", "ForQuestionAnswering")
    classified = any(str(a).endswith(endings) for a in architectures)
    decision = system_one or bool(counts.get("head")) or (classified and ("num_labels" in config or "id2label" in config))
    span = any(str(a).endswith(("ForTokenClassification", "ForQuestionAnswering")) for a in architectures)
    kind = "unknown"
    if decision:
        if config.get("problem_type") == "multi_label_classification":
            kind = "multilabel"
        elif span:
            kind = "span"
        elif num == 1:
            kind = "score"
        elif num == 2:
            kind = "binary"
        elif isinstance(num, int) and num > 2:
            kind = "multiclass"
    return dict(is_decision=bool(decision), system_one=system_one, decision_type=kind,
                num_labels=num, labels=dict(sorted(labels.items(), key=lambda item: str(item[0]))[:64]))


def build_manifest(repo_id: str, revision: str, reader: HubReader, api) -> dict:
    """Inspect selected safetensors headers and free metadata at a pinned input."""
    source = str(repo_id)
    root = Path(source)
    local = root.is_dir()
    if not local and not is_commit_sha(revision):
        raise ValueError("Manifests require a pinned 40-character commit SHA")
    manifest = dict(repo_id=source, revision=revision, gated=False, access="metadata_only", license=None,
                    declared_bases=[], base_model_relation=None, is_adapter=False,
                    weight_format="other", params_by_dtype={}, files=[], skipped_files=[], tensors=[],
                    config={}, adapter_config=None, view_counts={}, gguf={}, evidence_version=EVIDENCE_VERSION)
    manifest["decision"] = _decision({}, [], {})
    try:
        if local:
            root = root.resolve()
            source = str(root)
            files, revision = _local_files(root)
            manifest["revision"] = revision
            names = [f["filename"] for f in files]
            tags = []
            if "lineage.json" in names:
                lineage = _read_json(reader, source, revision, "lineage.json")
                manifest["repo_id"] = lineage.get("name") or str(root)
                manifest["declared_bases"] = lineage.get("bases", [])
                manifest["base_model_relation"] = lineage.get("relation")
                if lineage.get("name") and os.environ.get("ZOO_DATA"):
                    data_root = Path(os.environ["ZOO_DATA"]).resolve()
                    if root.is_relative_to(data_root):
                        locator = root.relative_to(data_root).as_posix()
                        if locator != manifest["repo_id"]:
                            manifest["source"] = dict(zoo_data_path=locator)
        else:
            if (api is None and reader.local_cache
                    and (not os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_OFFLINE") == "1")):
                from huggingface_hub.errors import LocalEntryNotFoundError
                try:
                    info = _cached_model_info(repo_id, revision)
                except LocalEntryNotFoundError:
                    info = _model_info(api, repo_id, revision)
            else:
                info = _model_info(api, repo_id, revision)
            card = _mapping(_get(info, "card_data", _get(info, "cardData")))
            bases = card.get("base_model", [])
            manifest.update(gated=_get(info, "gated", False), license=card.get("license"),
                            declared_bases=[bases] if isinstance(bases, str) else bases or [],
                            base_model_relation=card.get("base_model_relation"))
            tags = _get(info, "tags", []) or []
            files = []
            for sibling in _get(info, "siblings", []) or []:
                lfs = _get(sibling, "lfs")
                files.append(dict(filename=_get(sibling, "rfilename"), size=_get(sibling, "size"),
                                  sha256=_get(lfs, "sha256"), xet_hash=_get(sibling, "xet_hash")))
            names = [f["filename"] for f in files]
            gguf = _mapping(_get(info, "gguf"))
            if any(name.endswith(".gguf") for name in names):
                manifest["gguf"] = {key: gguf.get(key) for key in ("architecture", "total", "context_length")}
                manifest["gguf"]["quantization_types"] = sorted({m.group(0).upper() for name in names if name.endswith(".gguf")
                    for m in re.finditer(r"(?i)(?:IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)(?=[.\-]|$)", PurePosixPath(name).stem)})
        selected, skipped = select_file_set(names)
        manifest["skipped_files"] = skipped
        by_name = {f["filename"]: f for f in files}
        manifest["weight_format"] = next((fmt for suffix, fmt in ((".safetensors", "safetensors"), (".gguf", "gguf"),
            (_PICKLE, "bin"), (".onnx", "onnx")) if any(name.endswith(suffix) for name in selected)), "other")
        manifest["files"] = [by_name[name] for name in selected]
        config = _read_json(reader, source, revision, "config.json") if "config.json" in names else {}
        manifest["config"] = _config_subset(config)
        if "adapter_config.json" in names:
            manifest["adapter_config"] = _read_json(reader, source, revision, "adapter_config.json")
            manifest["is_adapter"] = True
        # The index, rather than a filename heuristic, defines which shards belong.
        indexes = _select_indexes(names)
        indexed = set()
        indexed_folders = set()
        for index in sorted(indexes):
            weight_map = _read_json(reader, source, revision, index).get("weight_map")
            if not isinstance(weight_map, dict) or not all(isinstance(v, str) for v in weight_map.values()):
                raise ValueError("Safetensors index requires a string weight_map")
            folder = PurePosixPath(index).parent
            has_safetensors = False
            for shard in weight_map.values():
                path = folder / shard
                if path.is_absolute() or ".." in path.parts or path.as_posix() not in names:
                    raise ValueError("Checkpoint index references an invalid or missing shard")
                if shard.endswith(".safetensors"):
                    indexed.add(path.as_posix())
                    has_safetensors = True
            if has_safetensors:
                indexed_folders.add(folder)
        if indexes:
            if "model.safetensors.index.json" in indexes and any(
                    name not in indexed for name in ("head.safetensors", "classifier.safetensors", "score.safetensors")
                    if name in selected):
                raise ValueError("Checkpoint index omits an auxiliary head; compound layout is ambiguous")
            for name in list(selected):
                if name.endswith(".safetensors") and PurePosixPath(name).parent in indexed_folders and name not in indexed:
                    selected.remove(name)
                    skipped.append(dict(filename=name, reason="duplicate_layout"))
            # Explicit shard membership takes precedence over layout heuristics.
            for name in sorted(indexed):
                if name not in selected:
                    selected.append(name)
                    skipped[:] = [s for s in skipped if s["filename"] != name]
            selected.sort()
            skipped.sort(key=lambda item: item["filename"])
            manifest["files"] = [by_name[name] for name in selected]
        entries = {}
        for name in selected:
            if not name.endswith(".safetensors"):
                continue
            remote = reader.resolve(source, name, revision)
            if not local:
                by_name[name].update(size=remote.size,
                                     sha256=remote.sha256 or by_name[name].get("sha256"),
                                     xet_hash=remote.xet_hash or by_name[name].get("xet_hash"))
            header = reader.read_header(remote)
            folder = str(PurePosixPath(name).parent)
            if "model_index.json" in names and folder != ".":
                header = {f"{folder}/{key}": replace(entry, name=f"{folder}/{key}")
                          for key, entry in header.items()}
            if entries.keys() & header.keys():
                raise ValueError("Selected files contain duplicate tensor names")
            entries.update(header)
        views, coverage = plan_views(entries, manifest["config"], manifest["adapter_config"])
        manifest["view_counts"] = dict(sorted(Counter(v.geometry for v in views).items()))
        manifest["is_adapter"] |= any(re.search(r"(?:^|[./])(?:lora_A|lora_B|lora_down|lora_up|lora\.down|lora\.up|lora_embedding_A|lora_embedding_B|lora_a|lora_b)(?:\.|$)", name) for name in entries)
        counts = Counter()
        coverage_by_name = {item["name"]: item for item in coverage}
        for name, entry in sorted(entries.items()):
            counts[entry.dtype] += math.prod(entry.shape)
            manifest["tensors"].append({**asdict(entry), **coverage_by_name[name]})
        manifest["params_by_dtype"] = dict(sorted(counts.items()))
        manifest["decision"] = _decision(config, tags, manifest["view_counts"])
        if manifest["weight_format"] == "safetensors":
            manifest["access"] = "granted"
    except AccessDenied:
        manifest["access"] = "no_access"
    return manifest


def write_manifest(manifest: dict, out_dir: Path) -> Path:
    """Atomically replace the manifest named by its complete pinned identity."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(f"{manifest['repo_id']}@{manifest['revision']}") + ".manifest.json"
    destination = out_dir / filename
    fd, temporary = tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
