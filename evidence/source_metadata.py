"""Bounded source observations, separate from normalized scientific manifests."""
from copy import deepcopy
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from .hub import AccessDenied, OfflineError

SCHEMA_VERSION = "1"
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_SIDECAR_BYTES = 1024 * 1024
MAX_COMPRESSED_BYTES = MAX_SIDECAR_BYTES + 65536
FILE_DOCUMENTS = (
    "README.md", "config.json", "tokenizer_config.json", "special_tokens_map.json",
    "adapter_config.json", "merge.json", "recipe.json", "lineage.json",
    "mergekit_config.yml", "mergekit_config.yaml", "merge_config.yml", "merge_config.yaml",
    "generation_config.json", "preprocessor_config.json",
)
API_DOCUMENTS = ("cardData.json", "model_info.json")
STATUSES = {"captured", "missing", "oversize", "total_limit", "offline", "denied",
            "unsafe_path", "invalid_utf8", "error", "not_captured", "not_applicable"}


def canonical_json(value):
    """Canonical API-object representation; never claimed as HTTP wire bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError("Non-JSON source metadata")


def _summarize(snapshot):
    rows = list(snapshot["documents"].values())
    snapshot["status"] = "captured" if any(r["status"] == "captured" for r in rows) else "not_captured"
    snapshot.setdefault("stats", {}).update(
        captured_documents=sum(r["status"] == "captured" for r in rows),
        captured_content_bytes=sum(r.get("bytes", 0) for r in rows if r["status"] == "captured"))
    return snapshot


class SourceManifest(dict):
    """A normal JSON dict carrying unsaved metadata through worker serialization."""


class SourceMetadataCapture:
    def __init__(self, repo_id, revision, *, source_kind="unavailable"):
        self.identity = dict(repo_id=repo_id, revision=revision)
        self.source_kind = source_kind
        self.observed_at = datetime.now(timezone.utc).isoformat()
        self.documents = {}
        self.stats = dict(extra_read_bytes=0, extra_resolves=0)

    def snapshot(self):
        documents = {name: dict(status="not_captured") for name in API_DOCUMENTS + FILE_DOCUMENTS}
        documents.update(deepcopy(self.documents))
        return _summarize(dict(schema_version=SCHEMA_VERSION, identity=dict(self.identity),
                               source_kind=self.source_kind, observed_at=self.observed_at,
                               scope="bounded_allowlist_not_complete_repository",
                               limits=dict(per_document_bytes=MAX_DOCUMENT_BYTES,
                                           total_uncompressed_bytes=MAX_SIDECAR_BYTES),
                               documents=documents, stats=dict(self.stats)))

    def _record(self, name, row):
        if name not in API_DOCUMENTS + FILE_DOCUMENTS:
            raise ValueError("Source metadata document is outside the allowlist")
        self.documents[name] = dict(row, source_kind=self.source_kind,
                                    observed_at=datetime.now(timezone.utc).isoformat(),
                                    source_revision=self.identity["revision"])
        if len(canonical_json(self.snapshot())) > MAX_SIDECAR_BYTES:
            self.documents[name] = dict(status="total_limit", source_kind=self.source_kind,
                                        observed_at=self.documents[name]["observed_at"],
                                        source_revision=self.identity["revision"], bytes=row.get("bytes", 0))

    def record_status(self, name, status, *, reason=None, size=None):
        row = dict(status=status)
        if reason:
            row["reason"] = reason
        if size is not None:
            row["bytes"] = size
        self._record(name, row)

    def record_json(self, name, content):
        try:
            raw = json.dumps(content, default=_json_default, sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(raw) > MAX_DOCUMENT_BYTES:
                self.record_status(name, "oversize", size=len(raw))
                return
            value = json.loads(raw)
            representation = "canonical_cached_object" if self.source_kind == "hub_cache" else "canonical_api_object"
            self._record(name, dict(status="captured", format="json", content=value,
                                    representation=representation, bytes=len(raw),
                                    sha256=hashlib.sha256(raw).hexdigest()))
        except (TypeError, ValueError, RecursionError, UnicodeError):
            self.record_status(name, "error", reason="non_json_source_object")

    def record_bytes(self, name, raw):
        if len(raw) > MAX_DOCUMENT_BYTES:
            self.record_status(name, "oversize", size=len(raw))
            return
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self.record_status(name, "invalid_utf8", size=len(raw))
            return
        self._record(name, dict(status="captured", format="text", content=text,
                                representation="original_utf8_file", bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest()))

    def capture_files(self, reader, source, revision, files):
        """Read only fixed root-level metadata names, under a separate byte budget."""
        by_name = {row["filename"]: row for row in files}
        unsafe = {PurePosixPath(name).name for name in by_name
                  if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name}
        for name in FILE_DOCUMENTS:
            if name in self.documents:
                continue
            if name not in by_name:
                self.record_status(name, "unsafe_path" if name in unsafe else "missing")
                continue
            size = by_name[name].get("size")
            if isinstance(size, int) and size > MAX_DOCUMENT_BYTES:
                self.record_status(name, "oversize", size=size)
                continue
            try:
                remote = reader.resolve(source, name, revision)
                self.stats["extra_resolves"] += 1
                if remote.size > MAX_DOCUMENT_BYTES:
                    self.record_status(name, "oversize", size=remote.size)
                elif self.stats["extra_read_bytes"] + remote.size > MAX_SIDECAR_BYTES:
                    self.record_status(name, "total_limit", size=remote.size)
                else:
                    raw = reader.read(remote, 0, remote.size)
                    self.stats["extra_read_bytes"] += len(raw)
                    self.record_bytes(name, raw)
            except OfflineError:
                self.record_status(name, "offline")
            except AccessDenied:
                self.record_status(name, "denied")
            except Exception as exc:
                # Exception text may contain signed URLs; retain only its type.
                self.record_status(name, "error", reason=type(exc).__name__)


def source_metadata_path(manifest_path):
    path = Path(manifest_path)
    if not path.name.endswith(".manifest.json"):
        raise ValueError("Source metadata requires a manifest filename")
    return path.with_name(path.name[:-len(".manifest.json")] + ".source-metadata.json.gz")


def _validate(snapshot, identity=None):
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported source metadata version")
    actual = snapshot.get("identity")
    if (not isinstance(actual, dict) or not all(isinstance(actual.get(k), str) and actual[k] for k in ("repo_id", "revision"))
            or (identity is not None and actual != identity)):
        raise ValueError("Source metadata identity mismatch")
    documents = snapshot.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Invalid source metadata documents")
    for name, row in documents.items():
        if name not in API_DOCUMENTS + FILE_DOCUMENTS or not isinstance(row, dict) or row.get("status") not in STATUSES:
            raise ValueError("Invalid source metadata document status/name")
        if row["status"] != "captured":
            if "content" in row:
                raise ValueError("Uncaptured metadata must not contain partial content")
            continue
        representations = {"json": {"canonical_api_object", "canonical_cached_object"},
                           "text": {"original_utf8_file"}}
        if row.get("representation") not in representations.get(row.get("format"), set()):
            raise ValueError("Invalid source metadata representation")
        try:
            raw = canonical_json(row["content"]) if row.get("format") == "json" else row["content"].encode("utf-8")
        except (KeyError, TypeError, ValueError, AttributeError, UnicodeError) as exc:
            raise ValueError("Invalid source metadata content") from exc
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("Source metadata document exceeds limit")
        if row.get("bytes") != len(raw) or row.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("Source metadata content hash mismatch")
    return snapshot


def bind_source_identity(snapshot, identity):
    """Record the collector's explicit alias for an unchanged local content pin."""
    if snapshot["identity"] == identity:
        return snapshot
    original = snapshot["identity"]
    if (snapshot.get("source_kind") != "local" or original["revision"] != identity["revision"]
            or not original["revision"].startswith("local:") or not Path(original["repo_id"]).is_absolute()):
        raise ValueError("Source metadata identity mismatch")
    rebound = deepcopy(snapshot)
    rebound["source_identity"] = dict(original)
    rebound["identity"] = dict(identity)
    rebound["identity_binding"] = dict(kind="local_content_alias", basis="same_local_content_pin")
    return rebound


def read_source_metadata(path, identity=None):
    path = Path(path)
    if path.stat().st_size > MAX_COMPRESSED_BYTES:
        raise ValueError("Compressed source metadata exceeds limit")
    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read(MAX_SIDECAR_BYTES + 1)
        if len(raw) > MAX_SIDECAR_BYTES:
            raise ValueError("Decompressed source metadata exceeds limit")
        snapshot = json.loads(raw)
    except (gzip.BadGzipFile, EOFError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid source metadata gzip/JSON") from exc
    return _validate(snapshot, identity)


def write_source_metadata(snapshot, manifest_path):
    path = source_metadata_path(manifest_path)
    snapshot = deepcopy(snapshot)
    _validate(snapshot)
    if path.is_file():
        existing = read_source_metadata(path, snapshot["identity"])
        if any(row["status"] == "captured" for row in existing["documents"].values()):
            combined = deepcopy(existing)
            changed = False
            for name, row in snapshot["documents"].items():
                if row["status"] != "captured" or combined["documents"].get(name, {}).get("status") == "captured":
                    continue
                previous = combined["documents"].get(name)
                combined["documents"][name] = row
                if len(canonical_json(_summarize(combined))) <= MAX_SIDECAR_BYTES:
                    changed = True
                elif previous is not None:
                    combined["documents"][name] = previous
                else:
                    del combined["documents"][name]
            if not changed:
                return path
            snapshot = _summarize(combined)
    raw = canonical_json(snapshot)
    if len(raw) > MAX_SIDECAR_BYTES:
        raise ValueError("Source metadata exceeds total uncompressed limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(gzip.compress(raw, mtime=0))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def carry_source_metadata(source, normalized):
    if hasattr(source, "source_metadata"):
        result = SourceManifest(normalized)
        result.source_metadata = source.source_metadata
        return result
    return normalized


def load_source_manifest(manifest_path):
    path = Path(manifest_path)
    manifest = SourceManifest(json.loads(path.read_text()))
    sidecar = source_metadata_path(path)
    if sidecar.is_file():
        manifest.source_metadata = read_source_metadata(sidecar, {k: manifest[k] for k in ("repo_id", "revision")})
    return manifest


def copy_source_metadata(source_manifest_path, destination_manifest_path):
    source = source_metadata_path(source_manifest_path)
    if source.is_file():
        manifest = json.loads(Path(destination_manifest_path).read_text())
        snapshot = read_source_metadata(source, {k: manifest[k] for k in ("repo_id", "revision")})
        return write_source_metadata(snapshot, destination_manifest_path)
    return None
