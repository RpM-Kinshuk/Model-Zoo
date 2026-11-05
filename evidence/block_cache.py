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

    def put(self, manifest, view_name, bounds, block):
        block = block.detach().cpu().contiguous()
        raw = block.view(torch.uint8).numpy().tobytes()
        if len(raw) > self.max_bytes:
            return
        path = self._path(manifest, view_name, bounds, block.dtype, block.shape)
        with self._locked():
            fd, temporary = tempfile.mkstemp(suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                os.replace(temporary, path)
                stamp = time.time_ns()
                os.utime(path, ns=(stamp, stamp))
                # Keep one stat result per entry and sort only when eviction
                # is actually necessary. Preserve the existing shared byte cap.
                entries = []
                with os.scandir(self.root) as directory:
                    for entry in directory:
                        if entry.name.endswith(".bin") and entry.is_file(follow_symlinks=False):
                            info = entry.stat(follow_symlinks=False)
                            entries.append((info.st_atime_ns, entry.name, info.st_size, entry.path))
                total = sum(entry[2] for entry in entries)
                if total > self.max_bytes:
                    for _, _, size, entry in sorted(entries):
                        if total <= self.max_bytes:
                            break
                        os.unlink(entry)
                        total -= size
            finally:
                Path(temporary).unlink(missing_ok=True)
