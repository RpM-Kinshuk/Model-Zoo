"""Hub lookups and bounded range reads, offline unless explicitly enabled."""

import json
import os
from unittest.mock import Mock

import pytest

from evidence.hub import AccessDenied, HubReader, OfflineError
from evidence.safetensors_header import MAX_HEADER_BYTES
from conftest import SHA


def test_resolve_once_and_range_read_cdn(fake_transport):
    reader = HubReader(token="t", transport=fake_transport)
    f1 = reader.resolve("org/m", "model.safetensors", SHA)
    assert reader.resolve("org/m", "model.safetensors", SHA) is f1 and reader.stats["resolves"] == 1
    assert f1.sha256 == "abc" and f1.size == 1000            # x-linked-etag / x-linked-size
    assert reader.read(f1, 10, 20) == fake_transport.blob[10:20]
    assert fake_transport.last_headers["Range"] == "bytes=10-19"
    assert "Authorization" not in fake_transport.last_headers


def test_expired_cdn_url_refreshes_once(fake_transport):
    fake_transport.expire_next_range_read()
    reader = HubReader(token="t", transport=fake_transport)
    f = reader.resolve("org/m", "model.safetensors", SHA)
    assert reader.read(f, 0, 8) == fake_transport.blob[:8]
    assert reader.stats["refreshes"] == 1 and reader.stats["resolves"] == 2


def test_gated_allowed_or_denied(fake_transport):
    fake_transport.gate("org/gated", allowed_token="owner")
    assert HubReader(token="owner", transport=fake_transport).resolve("org/gated", "model.safetensors", SHA)
    with pytest.raises(AccessDenied):
        HubReader(token="other", transport=fake_transport).resolve("org/gated", "model.safetensors", SHA)


def test_endpoint_and_offline(fake_transport, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    HubReader(token="t", transport=fake_transport).resolve("org/m", "model.safetensors", SHA)
    assert fake_transport.last_url.startswith("https://mirror.example/org/m/resolve/")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(OfflineError):
        HubReader(token="t", transport=fake_transport).resolve("org/m", "model.safetensors", SHA)


def test_local_mode_makes_no_network_calls(tiny_checkpoint, monkeypatch):
    monkeypatch.setattr("socket.socket.connect", Mock(side_effect=AssertionError("network")))
    reader = HubReader()
    f = reader.resolve(str(tiny_checkpoint), "model-00001-of-00002.safetensors", "local")
    assert "model.layers.0.self_attn.q_proj.weight" in reader.read_header(f)


@pytest.mark.parametrize("token", [None, ""])
def test_missing_credentials_never_calls_transport(token, tiny_checkpoint, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    transport = Mock()
    reader = HubReader(token=token, transport=transport)
    local = reader.resolve(str(tiny_checkpoint), "model-00001-of-00002.safetensors", "local")
    assert reader.read_header(local)
    with pytest.raises(AccessDenied, match="token"):
        reader.resolve("org/m", "model.safetensors", SHA)
    transport.assert_not_called()


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink"])
def test_local_filename_cannot_escape_checkpoint(kind, tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    if kind == "parent":
        filename = "../outside.safetensors"
    elif kind == "absolute":
        inside = root / "inside.safetensors"
        inside.write_bytes(b"inside")
        filename = str(inside)
    else:
        (root / "link.safetensors").symlink_to(outside)
        filename = "link.safetensors"
    transport = Mock()
    with pytest.raises(ValueError, match="checkpoint"):
        HubReader(transport=transport).resolve(str(root), filename, "local")
    transport.assert_not_called()


def test_lookup_metadata_env_token_and_small_file(fake_transport, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_marker")

    def transport(method, url, headers):
        if method == "HEAD":
            assert headers["Authorization"] == "Bearer hf_secret_marker"
            return 200, {"Content-Length": "1000", "ETag": '"small"', "X-Xet-Hash": "xet"}, b""
        assert "Authorization" not in headers
        return fake_transport(method, url, headers)

    reader = HubReader(transport=transport)
    remote = reader.resolve("org/m", "small.safetensors", SHA)
    assert remote.url.endswith(f"/resolve/{SHA}/small.safetensors")
    assert (remote.size, remote.sha256, remote.xet_hash) == (1000, "small", "xet")
    assert reader.read(remote, 0, 8) == fake_transport.blob[:8]


def test_retry_backoff_and_exhaustion(fake_transport, monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("evidence.hub.time.sleep", sleep)
    outcomes = [OSError("connection failed"), 503, 502]

    def transport(method, url, headers):
        if outcomes:
            outcome = outcomes.pop(0)
            if isinstance(outcome, OSError):
                raise outcome
            return outcome, {}, b""
        return fake_transport(method, url, headers)

    reader = HubReader(token="hf_secret_marker", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    assert [call.args[0] for call in sleep.call_args_list] == [1, 4, 16]
    assert reader.stats["retries"] == 3 and reader.stats["resolves"] == 4
    sleep.reset_mock()
    outcomes.extend([503] * 4)
    with pytest.raises(OSError, match="after retries"):
        reader.read(remote, 0, 8)
    assert [call.args[0] for call in sleep.call_args_list] == [1, 4, 16]
    assert reader.stats["range_reads"] == 4 and reader.stats["retries"] == 6


def test_lookup_rate_limit_waits_and_is_bounded(fake_transport, monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("evidence.hub.time.sleep", sleep)
    responses = [429]

    def transport(method, url, headers):
        if responses:
            return responses.pop(0), {"RateLimit": '"resolvers";r=0;t=7'}, b""
        return fake_transport(method, url, headers)

    reader = HubReader(token="hf_secret_marker", transport=transport)
    reader.resolve("org/m", "model.safetensors", SHA)
    sleep.assert_called_once_with(7)
    assert reader.stats["retries"] == 1
    responses.extend([429] * 4)
    sleep.reset_mock()
    with pytest.raises(OSError, match="rate limited"):
        reader.resolve("org/m", "other.safetensors", SHA)
    assert sleep.call_count == 3


def test_expired_refresh_is_reused_and_limited(fake_transport):
    reader = HubReader(token="hf_secret_marker", transport=fake_transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    fake_transport.expire_next_range_read()
    reader.read(remote, 0, 8)
    refreshed_url = fake_transport.last_url
    reader.read(remote, 8, 16)
    assert fake_transport.last_url == refreshed_url
    assert reader.stats["resolves"] == 2

    def forbidden(method, url, headers):
        return fake_transport(method, url, headers) if method == "HEAD" else (403, {}, b"")

    reader = HubReader(token="hf_secret_marker", transport=forbidden)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    with pytest.raises(OSError, match="403"):
        reader.read(remote, 0, 8)
    assert reader.stats["refreshes"] == 1 and reader.stats["resolves"] == 2
    assert reader.stats["range_reads"] == 2


def test_bounds_short_reads_and_content_range(fake_transport):
    reader = HubReader(token="hf_secret_marker", transport=fake_transport)
    with pytest.raises(ValueError, match="pinned"):
        reader.resolve("org/m", "model.safetensors", "main")
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    with pytest.raises(ValueError, match="bounds"):
        reader.read(remote, -1, 8)
    with pytest.raises(ValueError, match="bounds"):
        reader.read(remote, 0, 1001)
    assert reader.read(remote, 1000, 1000) == b"" and reader.stats["range_reads"] == 0
    fake_transport.blob = b"short"
    with pytest.raises(ValueError):
        reader.read(remote, 0, 8)

    reader = HubReader(transport=lambda *_: (206, {}, b"short"))
    with pytest.raises(ValueError, match="byte count"):
        reader.read(remote, 0, 8)


def test_header_reads_two_ranges_and_checks_file_bounds(fake_transport):
    raw = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    fake_transport.blob = (len(raw).to_bytes(8, "little") + raw).ljust(1000, b"\0")
    reader = HubReader(token="hf_secret_marker", transport=fake_transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    entry = reader.read_header(remote)["weight"]
    assert (entry.file, entry.start, entry.end) == ("model.safetensors", 8 + len(raw), 12 + len(raw))
    assert reader.stats["resolves"] == 1 and reader.stats["range_reads"] == 2
    assert reader.stats["bytes_read"] == 8 + len(raw)

    fake_transport.blob = (MAX_HEADER_BYTES + 1).to_bytes(8, "little") + b"\0" * 992
    with pytest.raises(ValueError, match="MAX_HEADER_BYTES"):
        reader.read_header(remote)
    assert reader.stats["range_reads"] == 3

    fake_transport.blob = (1000).to_bytes(8, "little") + b"\0" * 992
    with pytest.raises(ValueError, match="bounds"):
        reader.read_header(remote)
    assert reader.stats["range_reads"] == 4

    raw = json.dumps({"weight": {"dtype": "F32", "shape": [1000], "data_offsets": [0, 4000]}}).encode()
    fake_transport.blob = (len(raw).to_bytes(8, "little") + raw).ljust(1000, b"\0")
    with pytest.raises(ValueError, match="beyond the file size"):
        reader.read_header(remote)


def test_offline_blocks_cached_hub_reads_but_allows_local(fake_transport, tiny_checkpoint, monkeypatch):
    reader = HubReader(token="hf_secret_marker", transport=fake_transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(OfflineError):
        reader.read(remote, 0, 8)
    local = reader.resolve(str(tiny_checkpoint), "model-00001-of-00002.safetensors", "local")
    assert reader.read_header(local)
    assert reader.stats["resolves"] == 1 and reader.stats["range_reads"] == 0


@pytest.mark.skipif(os.environ.get("MODEL_ZOO_NETWORK_TESTS") != "1", reason="opt-in network smoke test")
def test_network_smoke():
    reader = HubReader()
    remote = reader.resolve("HuggingFaceTB/SmolLM2-135M", "model.safetensors",
                            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    assert "model.embed_tokens.weight" in reader.read_header(remote)
    assert reader.stats["resolves"] == 1 and reader.stats["range_reads"] == 2
