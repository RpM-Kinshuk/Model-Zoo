"""Private, bounded stored-dtype parent blocks, shared safely by workers."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch


class BlockCache:
    def __init__(self, root: Path, max_bytes: int = 64 * 1024**3):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self.root, self.max_bytes = Path(root), max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        # Separate opens also serialize threads using distinct cache instances.
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _path(self, manifest, view_name, bounds, dtype, shape):
        identity = [manifest["repo_id"], manifest["revision"], view_name,
                    list(bounds), str(dtype), list(shape)]
        digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        return self.root / (digest + ".bin")

    def get(self, manifest, view_name, bounds, dtype, shape):
        path = self._path(manifest, view_name, bounds, dtype, shape)
        try:
            size = shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size()
            # Writers publish complete inodes by atomic rename. An open reader
            # can finish safely even if eviction unlinks that inode meanwhile.
            with path.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size != size or size == 0:
                    return None
                raw = stream.read()
            stamp = time.time_ns()
            try:
                os.utime(path, ns=(stamp, stamp))
            except OSError:
                pass  # Timestamp maintenance must not invalidate an already complete read.
        except OSError:
            return None
        if len(raw) != size:
            return None
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)

    def _entries(self):
        """Take one size/LRU snapshot; needed only for recovery or eviction."""
        entries = []
        with os.scandir(self.root) as directory:
            for entry in directory:
                if entry.name.endswith(".bin") and entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    entries.append((info.st_atime_ns, entry.name, info.st_size, entry.path))
        return entries

    def _accounted_bytes(self, usage_path):
        """Trust accounting only while directory membership is unchanged."""
        try:
            usage = json.loads(usage_path.read_text())
            if (type(usage["bytes"]) is int and usage["bytes"] >= 0
                    and usage["directory_mtime_ns"] == self.root.stat().st_mtime_ns):
                return usage["bytes"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def put(self, manifest, view_name, bounds, block):
        block = block.detach().cpu().contiguous()
        raw = block.view(torch.uint8).numpy().tobytes()
        if len(raw) > self.max_bytes:
            return
        path = self._path(manifest, view_name, bounds, block.dtype, block.shape)
        with self._locked():
            usage_path = self.root / ".usage.json"
            total = self._accounted_bytes(usage_path)
            if total is None:
                total = sum(entry[2] for entry in self._entries())
            # Invalidate before any mutation: an interrupted writer forces the
            # next writer to recount, including any newly published block.
            usage_path.unlink(missing_ok=True)
            previous_size = path.stat().st_size if path.exists() else 0
            fd, temporary = tempfile.mkstemp(suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                os.replace(temporary, path)
                stamp = time.time_ns()
                os.utime(path, ns=(stamp, stamp))
                total += len(raw) - previous_size
                if total > self.max_bytes:
                    entries = self._entries()
                    total = sum(entry[2] for entry in entries)
                    # Reclaim about 5% headroom so a full large cache does not
                    # scan again on every insert. Large blocks still use the cap.
                    target = self.max_bytes * 95 // 100
                    for _, _, size, entry in sorted(entries):
                        if total <= self.max_bytes and total - size < target:
                            break
                        os.unlink(entry)
                        total -= size
                # Creating the record changes directory mtime; writing its
                # contents does not. Legacy writers/external additions invalidate it.
                with usage_path.open("w") as stream:
                    json.dump(dict(bytes=total, directory_mtime_ns=self.root.stat().st_mtime_ns), stream)
            finally:
                Path(temporary).unlink(missing_ok=True)
