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
        assert headers["Authorization"] == "Bearer hf_secret_marker"
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


def test_cached_cdn_307_follows_relative_location_without_auth():
    urls = []

    def transport(method, url, headers):
        if method == "HEAD":
            return 302, {"X-Linked-Size": "8", "Location": "https://cdn.example/old/blob"}, b""
        urls.append(url)
        assert headers == {"Range": "bytes=0-7"}
        if len(urls) == 1:
            return 307, {"Location": "../new/blob"}, b""
        assert url == "https://cdn.example/new/blob"
        return 206, {"Content-Range": "bytes 0-7/8"}, b"abcdefgh"

    reader = HubReader(token="secret", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    assert reader.read(remote, 0, 8) == b"abcdefgh"


@pytest.mark.parametrize("redirect_status", [301, 302, 303, 307, 308])
def test_range_redirect_preserves_bounds_auth_and_caches_target(redirect_status):
    calls = []

    def transport(method, url, headers):
        calls.append((method, url, dict(headers)))
        if method == "HEAD":
            return 200, {"Content-Length": "16", "ETag": '"stable"'}, b""
        assert headers["Range"] in ("bytes=0-7", "bytes=8-15")
        if "/resolve/" in url:
            assert headers["Authorization"] == "Bearer secret"
            return redirect_status, {"Location": "/api/resolve-cache/org/m/blob?download=1"}, b""
        if url.startswith("https://huggingface.co/"):
            assert headers["Authorization"] == "Bearer secret"
            return redirect_status, {"Location": "https://cdn.example/blob?signed=secret"}, b""
        assert "Authorization" not in headers
        return 206, {"Content-Range": "bytes " + headers["Range"][6:] + "/16"}, b"abcdefgh"

    reader = HubReader(token="secret", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    assert reader.read(remote, 0, 8) == b"abcdefgh"
    cached = reader.resolve("org/m", "model.safetensors", SHA)
    assert (cached.repo_id, cached.filename, cached.revision, cached.size, cached.sha256) == (
        "org/m", "model.safetensors", SHA, 16, "stable")
    assert cached.url == "https://cdn.example/blob?signed=secret"
    assert reader.read(remote, 8, 16) == b"abcdefgh"
    assert [method for method, _, _ in calls] == ["HEAD", "GET", "GET", "GET", "GET"]
    assert calls[-1][1] == cached.url


@pytest.mark.parametrize("target", [
    "https://cdn.example/blob", "https://mirror.example.evil.example/blob",
    "http://mirror.example/blob", "https://mirror.example:444/blob", "https://huggingface.co/blob",
])
def test_range_redirect_only_authenticates_configured_origin(target, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example:443")

    def transport(method, url, headers):
        if method == "HEAD":
            return 302, {"X-Linked-Size": "8", "Location": "https://mirror.example/cache"}, b""
        if url == "https://mirror.example/cache":
            assert headers["Authorization"] == "Bearer secret"
            return 307, {"Location": target}, b""
        assert url == target and "Authorization" not in headers
        return 206, {"Content-Range": "bytes 0-7/8"}, b"abcdefgh"

    reader = HubReader(token="secret", transport=transport)
    assert reader.read(reader.resolve("org/m", "model.safetensors", SHA), 0, 8) == b"abcdefgh"


def test_lookup_follows_relative_redirect_without_file_metadata():
    calls = []

    def transport(method, url, headers):
        calls.append(url)
        assert method == "HEAD" and headers["Authorization"] == "Bearer secret"
        if "/resolve/" in url:
            return 307, {"Location": "/renamed/model?revision=pinned", "Content-Length": "0"}, b""
        assert url == "https://huggingface.co/renamed/model?revision=pinned"
        return 302, {"X-Linked-Size": "8", "Location": "https://cdn.example/blob"}, b""

    reader = HubReader(token="secret", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    assert remote.size == 8 and remote.url == "https://cdn.example/blob"
    assert reader.resolve("org/m", "model.safetensors", SHA) is remote and len(calls) == 2


@pytest.mark.parametrize("location", [None, "", "file:///tmp/blob", "https://user:pass@cdn.example/blob",
                                      "https://cdn.example:invalid/blob", "https://[invalid/blob",
                                      "\nhttps://cdn.example/blob", "/blob\x00"])
@pytest.mark.parametrize("method", ["HEAD", "GET"])
def test_malformed_redirects_fail_without_contacting_target(method, location):
    calls = []

    def transport(request_method, url, headers):
        calls.append((request_method, url))
        if request_method == "HEAD" and method == "GET":
            return 200, {"Content-Length": "8"}, b""
        return 307, {"Location": location, "X-Linked-Size": "8"}, b""

    reader = HubReader(token="secret", transport=transport)
    with pytest.raises(OSError, match="redirect") as error:
        remote = reader.resolve("org/m", "model.safetensors", SHA)
        reader.read(remote, 0, 8)
    assert "user:pass" not in str(error.value)
    assert len(calls) == (1 if method == "HEAD" else 2)


@pytest.mark.parametrize("cycle", [True, False])
@pytest.mark.parametrize("method", ["HEAD", "GET"])
def test_redirect_chain_is_bounded(cycle, method):
    urls = []

    def transport(method, url, headers):
        if method == "HEAD" and requested_method == "GET":
            return 200, {"Content-Length": "8"}, b""
        urls.append(url)
        return 307, {"Location": "/loop" if cycle else f"/hop/{len(urls)}"}, b""

    requested_method = method
    reader = HubReader(token="secret", transport=transport)
    with pytest.raises(OSError, match="redirect"):
        remote = reader.resolve("org/m", "model.safetensors", SHA)
        reader.read(remote, 0, 8)
    assert len(urls) <= 6
    if cycle:
        assert len(urls) == len(set(urls))


@pytest.mark.parametrize("response_headers, body, message", [
    ({"Content-Range": "bytes 1-8/16"}, b"abcdefgh", "Content-Range"),
    ({"Content-Range": "bytes 0-7/17"}, b"abcdefgh", "Content-Range"),
    ({"Content-Range": "bytes 0-7/16"}, b"abcdefg", "byte count"),
    ({"Content-Range": "bytes 0-7/16"}, b"abcdefghi", "byte count"),
])
def test_redirect_target_cached_only_after_range_validation(response_headers, body, message):
    def transport(method, url, headers):
        if method == "HEAD":
            return 200, {"Content-Length": "16"}, b""
        if "/resolve/" in url:
            return 307, {"Location": "https://cdn.example/blob"}, b""
        return 206, response_headers, body

    reader = HubReader(token="secret", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    with pytest.raises(ValueError, match=message):
        reader.read(remote, 0, 8)
    assert reader.resolve("org/m", "model.safetensors", SHA) is remote


def test_expired_cdn_401_refreshes_once_without_claiming_denial(fake_transport):
    reads = []

    def transport(method, url, headers):
        if method == "GET":
            reads.append(url)
            assert "Authorization" not in headers
            return 401, {}, b""
        return fake_transport(method, url, headers)

    reader = HubReader(token="secret", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    with pytest.raises(OSError, match="401") as error:
        reader.read(remote, 0, 8)
    assert not isinstance(error.value, AccessDenied)
    assert len(reads) == 2 and reads[0] != reads[1]
    assert reader.stats["refreshes"] == 1 and reader.stats["resolves"] == 2


def test_authenticated_hub_401_is_not_confirmed_denial_or_cdn_expiry():
    def transport(method, url, headers):
        assert headers["Authorization"] == "Bearer secret"
        if method == "HEAD":
            return 200, {"Content-Length": "8"}, b""
        return 401, {"X-Error-Message": "unauthorized secret"}, b"secret"

    reader = HubReader(token="secret", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    with pytest.raises(OSError, match="401") as error:
        reader.read(remote, 0, 8)
    assert not isinstance(error.value, AccessDenied) and "secret" not in str(error.value)
    assert reader.stats["refreshes"] == 0


@pytest.mark.parametrize("failure", ["size", "etag", "xet"])
def test_refresh_preserves_pinned_file_identity(failure):
    heads = []

    def transport(method, url, headers):
        if method == "GET":
            return 403, {}, b""
        heads.append(url)
        changed = len(heads) == 2
        return 302, {"Location": "https://cdn.example/blob", "X-Linked-Size":
                     "9" if changed and failure == "size" else "8", "X-Linked-ETag":
                     '"changed"' if changed and failure == "etag" else '"stable"',
                     "X-Xet-Hash": "changed" if changed and failure == "xet" else "stable"}, b""

    reader = HubReader(token="secret", transport=transport)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    with pytest.raises(ValueError, match="Pinned file"):
        reader.read(remote, 0, 8)
    assert reader.resolve("org/m", "model.safetensors", SHA) is remote


@pytest.mark.skipif(os.environ.get("MODEL_ZOO_NETWORK_TESTS") != "1", reason="opt-in network smoke test")
def test_network_smoke():
    reader = HubReader()
    remote = reader.resolve("HuggingFaceTB/SmolLM2-135M", "model.safetensors",
                            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    assert "model.embed_tokens.weight" in reader.read_header(remote)
    assert reader.stats["resolves"] == 1 and reader.stats["range_reads"] == 2


def test_local_cache_reads_make_no_network_calls(tmp_path, monkeypatch):
    path = tmp_path / "cached"
    path.write_bytes(b"checkpoint-bytes")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args, **kwargs: str(path))
    reader = HubReader(token="t", local_cache=True, transport=Mock(side_effect=AssertionError("network")))
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    assert remote.revision == SHA and remote.local_path == str(path)
    assert reader.read(remote, 2, 9) == b"eckpoin"
    assert reader.stats["resolves"] == 0


def test_local_cache_hit_needs_no_token(tmp_path, monkeypatch):
    path = tmp_path / "cached"
    path.write_bytes(b"local")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args, **kwargs: str(path))
    reader = HubReader(local_cache=True)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    assert reader.read(remote, 0, 5) == b"local"
    with pytest.raises(ValueError, match="pinned"):
        reader.resolve("org/m", "model.safetensors", "main")


def test_local_cache_falls_back_to_http_for_missing_files(fake_transport, monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args, **kwargs: None)
    reader = HubReader(token="t", transport=fake_transport, local_cache=True)
    remote = reader.resolve("org/m", "model.safetensors", SHA)
    assert remote.local_path is None
    assert reader.read(remote, 0, 8) == fake_transport.blob[:8]
    assert reader.stats["resolves"] == 1
