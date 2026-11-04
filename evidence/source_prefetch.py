"""One bounded loader thread; record/HDF5 writers stay on the caller."""
from concurrent.futures import ThreadPoolExecutor


class SourcePrefetch:
    def __init__(self, groups, load, estimate, *, window, max_bytes):
        self.groups, self.load, self.estimate = groups, load, estimate
        self.window, self.max_bytes, self.index = window, max_bytes, 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fingerprint-loader") if window == 2 else None
        self.pending = None

    def next(self):
        if self.index == len(self.groups):
            raise StopIteration
        group = self.groups[self.index]
        if self.pending is not None:
            pending, self.pending = self.pending, None
            payload = pending.result()
        else:
            payload = self.load(group) if self.window and self.estimate(group) <= self.max_bytes else None
        self.index += 1
        if self.pool is not None and self.index < len(self.groups):
            following = self.groups[self.index]
            if self.estimate(following) <= self.max_bytes:
                self.pending = self.pool.submit(self.load, following)
        return group, payload

    def close(self):
        if self.pending is not None:
            self.pending.cancel()
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending = None
