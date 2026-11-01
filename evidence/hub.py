"""Resolve pinned Hub files once, then read bounded byte ranges."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import time
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import quote, urljoin
from urllib.request import HTTPRedirectHandler, Request, build_opener

from measurement_config import is_commit_sha

from .safetensors_header import TensorEntry, header_length, parse_header


class AccessDenied(PermissionError):
    """The lookup token does not grant access to this file."""


class OfflineError(OSError):
    """Hub access is disabled by HF_HUB_OFFLINE."""


@dataclass(frozen=True)
class RemoteFile:
    repo_id: str
    filename: str
    revision: str
    url: str = field(repr=False)
    size: int
    sha256: str | None
    xet_hash: str | None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _transport(method, url, headers):
    # urllib's default ProxyHandler honors standard proxy environment variables.
    request = Request(url, headers=headers, method=method)
    try:
        response = build_opener(_NoRedirect()).open(request, timeout=60)
    except HTTPError as exc:
        response = exc
    with response:
        if method == "HEAD":
            body = b""
        else:
            start, end = map(int, headers["Range"][6:].split("-"))
            body = response.read(end - start + 2)
        return response.code, dict(response.headers), body


class HubReader:
    def __init__(self, token: str | None = None, transport: Callable | None = None):
        self._token = token if token is not None else os.environ.get("HF_TOKEN")
        self._transport = transport if transport is not None else _transport
        self._endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        self._files = {}
        self.stats = dict(resolves=0, range_reads=0, bytes_read=0, refreshes=0, retries=0)

    @staticmethod
    def _online():
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise OfflineError("Hub reads are disabled by HF_HUB_OFFLINE")

    def _request(self, method, url, headers):
        self._online()
        for attempt in range(4):
            self.stats["resolves" if method == "HEAD" else "range_reads"] += 1
            try:
                status, response_headers, body = self._transport(method, url, headers)
            except OSError:
                if attempt == 3:
                    raise OSError("Hub connection failed after retries") from None
                delay = (1, 4, 16)[attempt]
            else:
                response_headers = {key.lower(): value for key, value in response_headers.items()}
                if method == "GET":
                    self.stats["bytes_read"] += len(body)
                if method == "HEAD" and status in (401, 403):
                    raise AccessDenied("Hub lookup access denied")
                if method == "HEAD" and status == 429:
                    match = re.search(r"(?:^|[;,\s])t=(\d+)(?=[;,\s]|$)",
                                      response_headers.get("ratelimit", ""))
                    if match is None or attempt == 3:
                        raise OSError("Hub lookup rate limited")
                    delay = int(match.group(1))
                elif 500 <= status <= 599:
                    if attempt == 3:
                        raise OSError("Hub server failed after retries")
                    delay = (1, 4, 16)[attempt]
                else:
                    return status, response_headers, body
            self.stats["retries"] += 1
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")

    def _lookup(self, repo_id, filename, revision):
        if not self._token:
            raise AccessDenied("Hub lookup requires an HF_TOKEN or explicit token")
        url = f"{self._endpoint}/{quote(repo_id, safe='/')}/resolve/{revision}/{quote(filename, safe='/')}"
        headers = {"Authorization": f"Bearer {self._token}"}
        status, headers, _ = self._request("HEAD", url, headers)
        if status not in (200, 301, 302, 303, 307, 308):
            raise OSError(f"Unexpected Hub lookup status: {status}")
        location = headers.get("location")
        if status != 200 and not location:
            raise OSError("Hub redirect is missing its location")
        try:
            size = int(headers.get("x-linked-size", headers.get("content-length", "")))
        except (TypeError, ValueError):
            raise ValueError("Hub file size is missing or invalid") from None
        if size < 0:
            raise ValueError("Hub file size must be nonnegative")
        etag = headers.get("x-linked-etag", headers.get("etag"))
        remote = RemoteFile(repo_id, filename, revision, urljoin(url, location) if location else url,
                            size, etag.strip('"') if etag else None, headers.get("x-xet-hash"))
        self._files[(repo_id, filename, revision)] = remote
        return remote

    def resolve(self, repo_id, filename, revision) -> RemoteFile:
        repo_id = str(repo_id)
        key = (repo_id, filename, revision)
        local = Path(repo_id).is_dir()
        if not local:
            self._online()
            if not is_commit_sha(revision):
                raise ValueError("Hub reads require a pinned 40-character commit SHA")
        if key not in self._files:
            if local:
                root = Path(repo_id).resolve()
                if Path(filename).is_absolute():
                    raise ValueError("Local checkpoint filenames must be relative")
                path = (root / filename).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Local filename escapes the checkpoint directory")
                self._files[key] = RemoteFile(repo_id, filename, revision, str(path),
                                              path.stat().st_size, None, None)
            else:
                return self._lookup(repo_id, filename, revision)
        return self._files[key]

    def read(self, remote: RemoteFile, start: int, end: int) -> bytes:
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= remote.size:
            raise ValueError("Read range is outside file bounds")
        if Path(remote.repo_id).is_dir():
            with open(remote.url, "rb") as stream:
                stream.seek(start)
                body = stream.read(end - start)
            self.stats["bytes_read"] += len(body)
        else:
            self._online()
            if start == end:
                return b""
            key = (remote.repo_id, remote.filename, remote.revision)
            current = self._files.get(key, remote)
            headers = {"Range": f"bytes={start}-{end - 1}"}
            status, response_headers, body = self._request("GET", current.url, headers)
            if status == 403:
                self.stats["refreshes"] += 1
                current = self._lookup(*key)
                if current.size != remote.size:
                    raise ValueError("Pinned file size changed during refresh")
                status, response_headers, body = self._request("GET", current.url, headers)
            if status != 206:
                raise OSError(f"Unexpected Hub range status: {status}")
            content_range = response_headers.get("content-range")
            if content_range is not None and content_range != f"bytes {start}-{end - 1}/{remote.size}":
                raise ValueError("Hub Content-Range does not match the requested bytes")
        if len(body) != end - start:
            raise ValueError("File read returned an unexpected byte count")
        return body

    def read_header(self, remote: RemoteFile) -> dict[str, TensorEntry]:
        first8 = self.read(remote, 0, 8)
        size = header_length(first8)
        prefix = first8 + self.read(remote, 8, 8 + size)
        entries, _ = parse_header(prefix, remote.filename)
        if any(entry.end > remote.size for entry in entries.values()):
            raise ValueError("Tensor data extends beyond the file size")
        return entries
