"""Bounded source snapshots survive normal collection and publication paths."""
import gzip
import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest

from evidence.hub import OfflineError
from evidence.manifest import build_manifest, write_manifest

SHA = "a" * 40


def add_document(repositories, name, content):
    files, infos = repositories
    files["org/m"][name] = content
    infos["org/m"].siblings.append(SimpleNamespace(
        rfilename=name, size=len(content), lfs={"sha256": hashlib.sha256(content).hexdigest()}, xet_hash=None))


def test_build_normalize_write_retains_full_config_card_and_raw_readme(fake_reader, fake_api, manifest_repositories, tmp_path):
    from evidence.compatibility import normalize_manifest
    files, infos = manifest_repositories
    card = infos["org/m"].card_data
    card["model-index"] = [{"name": "example", "results": []}]
    config = json.loads(files["org/m"]["config.json"])
    config["unmodeled_generation_setting"] = {"µ": [1, 2, 3]}
    files["org/m"]["config.json"] = json.dumps(config, ensure_ascii=False).encode()
    text = "---\nlicense: apache-2.0\n---\nA complete µ model card.\n"
    add_document(manifest_repositories, "README.md", text.encode())
    manifest = build_manifest("org/m", SHA, fake_reader, fake_api)
    target = write_manifest(normalize_manifest(pickle.loads(pickle.dumps(manifest))), tmp_path)
    sidecar = target.with_name(target.name.replace(".manifest.json", ".source-metadata.json.gz"))
    assert sidecar.is_file(), "ordinary manifest writing must retain source metadata"
    value = json.loads(gzip.decompress(sidecar.read_bytes()))
    assert value["schema_version"] == "1" and value["identity"] == {"repo_id": "org/m", "revision": SHA}
    assert value["documents"]["cardData.json"]["content"] == card
    assert json.loads(value["documents"]["config.json"]["content"]) == config
    assert value["documents"]["README.md"]["content"] == text
    assert value["documents"]["README.md"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    canonical = json.dumps(card, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert value["documents"]["cardData.json"]["sha256"] == hashlib.sha256(canonical).hexdigest()
    serialized = json.loads(target.read_text())
    assert "unmodeled_generation_setting" not in serialized["config"]
    assert "documents" not in serialized and "source_metadata" not in serialized
    first = sidecar.read_bytes()
    write_manifest(dict(manifest), tmp_path)
    assert sidecar.read_bytes() == first, "normal dict rewrites preserve the original captured observation"
    assert first[4:8] == b"\0\0\0\0", "gzip timestamps must be deterministic"


def test_optional_oversize_unsafe_and_offline_metadata_never_blocks_weights(fake_reader, fake_api, manifest_repositories, tmp_path):
    from evidence.source_metadata import read_source_metadata, source_metadata_path
    add_document(manifest_repositories, "README.md", b"x" * (256 * 1024 + 1))
    add_document(manifest_repositories, "../tokenizer_config.json", b"unsafe")
    add_document(manifest_repositories, "special_tokens_map.json", b"{}")
    original_resolve = fake_reader.resolve

    def resolve(repo, name, revision):
        assert name != "README.md", "known oversize metadata must not even be resolved"
        assert ".." not in name
        if name == "special_tokens_map.json":
            raise OfflineError("offline")
        return original_resolve(repo, name, revision)

    fake_reader.resolve = resolve
    manifest = build_manifest("org/m", SHA, fake_reader, fake_api)
    assert manifest["access"] == "granted" and manifest["tensors"]
    value = read_source_metadata(source_metadata_path(write_manifest(manifest, tmp_path)))
    assert value["documents"]["README.md"]["status"] == "oversize"
    assert value["documents"]["tokenizer_config.json"]["status"] == "unsafe_path"
    assert value["documents"]["special_tokens_map.json"]["status"] == "offline"
    assert value["documents"]["merge.json"]["status"] == "missing"
    assert "content" not in value["documents"]["README.md"]


def test_entire_serialized_snapshot_cap_counts_escaping_and_statuses(tmp_path):
    from evidence.source_metadata import SourceMetadataCapture, write_source_metadata
    capture = SourceMetadataCapture("org/m", SHA, source_kind="hub_api")
    text = '"' * 240000
    for name in ("README.md", "tokenizer_config.json", "special_tokens_map.json", "merge.json"):
        capture.record_bytes(name, text.encode())
    path = write_source_metadata(capture.snapshot(), tmp_path / "model.manifest.json")
    raw = gzip.decompress(path.read_bytes())
    assert len(raw) <= 1024 * 1024
    documents = json.loads(raw)["documents"]
    assert any(row["status"] == "total_limit" for row in documents.values())
    assert all(row.get("content") == text for row in documents.values() if row["status"] == "captured")


def test_read_checks_identity_document_hash_and_both_gzip_limits(tmp_path):
    from evidence.source_metadata import SourceMetadataCapture, read_source_metadata, write_source_metadata
    capture = SourceMetadataCapture("org/m", SHA)
    capture.record_bytes("README.md", b"complete")
    path = write_source_metadata(capture.snapshot(), tmp_path / "model.manifest.json")
    with pytest.raises(ValueError, match="identity"):
        read_source_metadata(path, identity={"repo_id": "org/other", "revision": SHA})
    value = json.loads(gzip.decompress(path.read_bytes()))
    value["documents"]["README.md"]["content"] = "tampered"
    path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
    with pytest.raises(ValueError, match="hash"):
        read_source_metadata(path)
    path.write_bytes(gzip.compress(b"x" * (1024 * 1024 + 1), mtime=0))
    with pytest.raises(ValueError, match="limit"):
        read_source_metadata(path)
    path.write_bytes(b"x" * (1024 * 1024 + 65537))
    with pytest.raises(ValueError, match="limit"):
        read_source_metadata(path)


def test_prepared_bridge_and_staging_preserve_first_hub_capture(fake_reader, fake_api, manifest_repositories, tmp_path):
    from evidence.source_metadata import SourceMetadataCapture, source_metadata_path
    from evidence_bridge import _manifest, evidence_dir
    manifest = build_manifest("org/m", SHA, fake_reader, fake_api)
    folder = evidence_dir(tmp_path, "org/m", SHA)
    original = source_metadata_path(write_manifest(manifest, folder)).read_bytes()
    prepared = _manifest(tmp_path, "org/m", SHA)
    staged = write_manifest(prepared, tmp_path / "staged")
    assert source_metadata_path(staged).read_bytes() == original
    weaker = SourceMetadataCapture("org/m", SHA, source_kind="local")
    weaker.record_bytes("config.json", b'{"wrong":"downloaded subset"}')
    prepared.source_metadata = weaker.snapshot()
    write_manifest(prepared, folder)
    assert source_metadata_path(folder / (folder.name + ".manifest.json")).read_bytes() == original


def test_legacy_manifest_emits_explicit_not_captured_snapshot(tmp_path):
    from evidence.source_metadata import read_source_metadata, source_metadata_path
    target = write_manifest({"repo_id": "org/legacy", "revision": SHA}, tmp_path)
    value = read_source_metadata(source_metadata_path(target))
    assert value["status"] == "not_captured"
    assert value["documents"]["cardData.json"]["status"] == "not_captured"


def test_capture_reuses_info_and_config_and_never_fetches_vocabs_or_code(fake_reader, fake_api, manifest_repositories, tmp_path):
    from evidence.source_metadata import read_source_metadata, source_metadata_path
    excluded = ("tokenizer.json", "vocab.txt", "train.jsonl", "modeling_custom.py")
    for name in excluded:
        add_document(manifest_repositories, name, b"excluded fixture")
    add_document(manifest_repositories, "README.md", b"small source card")
    calls, reads = [], []
    original_info, original_read = fake_api.model_info, fake_reader.read

    def info(*args, **kwargs):
        calls.append(args[0])
        return original_info(*args, **kwargs)

    def read(remote, start, end):
        assert remote.filename not in excluded
        reads.append(remote.filename)
        return original_read(remote, start, end)

    fake_api.model_info, fake_reader.read = info, read
    manifest = build_manifest("org/m", SHA, fake_reader, fake_api)
    value = read_source_metadata(source_metadata_path(write_manifest(manifest, tmp_path)))
    assert calls == ["org/m"]
    assert reads.count("config.json") == 1
    assert reads.index("README.md") < next(i for i, name in enumerate(reads) if name.endswith(".safetensors"))
    assert not set(excluded) & set(value["documents"])
    assert value["documents"]["model_info.json"]["status"] == "captured"


def test_large_api_object_and_invalid_utf8_have_explicit_no_content_status(fake_reader, fake_api, manifest_repositories, tmp_path):
    from evidence.source_metadata import read_source_metadata, source_metadata_path
    _, infos = manifest_repositories
    infos["org/m"].large_metadata_field = "x" * (256 * 1024)
    add_document(manifest_repositories, "README.md", b"invalid utf8 \xff")
    value = read_source_metadata(source_metadata_path(write_manifest(build_manifest("org/m", SHA, fake_reader, fake_api), tmp_path)))
    assert value["documents"]["model_info.json"]["status"] == "oversize"
    assert value["documents"]["cardData.json"]["status"] == "captured"
    assert value["documents"]["README.md"]["status"] == "invalid_utf8"
    assert "content" not in value["documents"]["README.md"]


def test_plain_manifest_rewrite_rejects_existing_wrong_identity(fake_reader, fake_api, tmp_path):
    from evidence.source_metadata import source_metadata_path
    manifest = build_manifest("org/m", SHA, fake_reader, fake_api)
    path = source_metadata_path(write_manifest(manifest, tmp_path))
    value = json.loads(gzip.decompress(path.read_bytes()))
    value["identity"]["repo_id"] = "org/wrong"
    path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
    with pytest.raises(ValueError, match="identity"):
        write_manifest(dict(manifest), tmp_path)


def test_write_does_not_relabel_foreign_hub_snapshot(fake_reader, fake_api, tmp_path):
    manifest = build_manifest("org/m", SHA, fake_reader, fake_api)
    manifest["repo_id"] = "org/foreign"
    with pytest.raises(ValueError, match="identity"):
        write_manifest(manifest, tmp_path)


def test_observation_times_survive_filling_missing_documents(tmp_path):
    from datetime import datetime
    from evidence.source_metadata import SourceMetadataCapture, read_source_metadata, write_source_metadata
    capture = SourceMetadataCapture("org/m", SHA, source_kind="hub_api")
    capture.record_json("cardData.json", {"license": "mit"})
    path = write_source_metadata(capture.snapshot(), tmp_path / "model.manifest.json")
    original = read_source_metadata(path)["documents"]["cardData.json"]["observed_at"]
    later = SourceMetadataCapture("org/m", SHA, source_kind="hub_api")
    later.record_json("cardData.json", {"license": "changed"})
    later.record_bytes("README.md", b"later captured card")
    write_source_metadata(later.snapshot(), tmp_path / "model.manifest.json")
    snapshot = read_source_metadata(path)
    assert snapshot["documents"]["cardData.json"]["observed_at"] == original
    assert snapshot["documents"]["cardData.json"]["content"] == {"license": "mit"}
    assert datetime.fromisoformat(original).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(snapshot["documents"]["README.md"]["observed_at"]) >= datetime.fromisoformat(original)


def test_reader_rejects_unknown_captured_representation(tmp_path):
    from evidence.source_metadata import SourceMetadataCapture, read_source_metadata, write_source_metadata
    capture = SourceMetadataCapture("org/m", SHA)
    capture.record_bytes("README.md", b"card")
    path = write_source_metadata(capture.snapshot(), tmp_path / "model.manifest.json")
    value = json.loads(gzip.decompress(path.read_bytes()))
    value["documents"]["README.md"]["format"] = "unknown-format"
    value["documents"]["README.md"]["representation"] = "unknown-representation"
    path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
    with pytest.raises(ValueError, match="representation"):
        read_source_metadata(path)


def test_local_content_alias_records_original_identity(tiny_checkpoint, tmp_path):
    from evidence.hub import HubReader
    from evidence.source_metadata import read_source_metadata, source_metadata_path
    manifest = build_manifest(str(tiny_checkpoint), "local", HubReader(), None)
    original = {key: manifest[key] for key in ("repo_id", "revision")}
    manifest["repo_id"] = "lab/named-local"
    snapshot = read_source_metadata(source_metadata_path(write_manifest(manifest, tmp_path / "metadata")))
    assert snapshot["identity"]["repo_id"] == "lab/named-local"
    assert snapshot["source_identity"] == original
    assert snapshot["identity_binding"]["basis"] == "same_local_content_pin"


def test_initial_api_denial_is_distinct_from_later_weight_header_denial(fake_api, fake_reader, tmp_path):
    from evidence.hub import AccessDenied
    from evidence.source_metadata import read_source_metadata, source_metadata_path

    class DeniedAPI:
        def model_info(self, *args, **kwargs):
            raise AccessDenied("fixture denied")

    first = build_manifest("org/m", SHA, fake_reader, DeniedAPI())
    denied = read_source_metadata(source_metadata_path(write_manifest(first, tmp_path / "api")))
    assert denied["documents"]["model_info.json"]["status"] == "denied"
    assert denied["documents"]["cardData.json"]["status"] == "denied"

    def denied_header(remote):
        raise AccessDenied("fixture weight header denied")

    fake_reader.read_header = denied_header
    second = build_manifest("org/m", SHA, fake_reader, fake_api)
    later = read_source_metadata(source_metadata_path(write_manifest(second, tmp_path / "header")))
    assert first["access"] == second["access"] == "no_access"
    assert later["documents"]["model_info.json"]["status"] == "captured"
    assert later["documents"]["cardData.json"]["status"] == "captured"
