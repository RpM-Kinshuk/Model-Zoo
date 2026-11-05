"""Bounded parent blocks must survive races without returning torn data."""
from concurrent.futures import ThreadPoolExecutor
import os
import torch

from evidence.block_cache import BlockCache

IDENTITY = dict(repo_id="fixture/base", revision="a" * 40)


def test_block_cache_evicts_lru_within_budget(tmp_path):
    cache = BlockCache(tmp_path, max_bytes=32)
    block = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    cache.put(IDENTITY, "a", (0, 2), block)
    assert torch.equal(cache.get(IDENTITY, "a", (0, 2), block.dtype, block.shape), block)
    cache.put(IDENTITY, "b", (0, 2), block + 1)
    assert cache.get(IDENTITY, "a", (0, 2), block.dtype, block.shape) is None
    assert torch.equal(cache.get(IDENTITY, "b", (0, 2), block.dtype, block.shape), block + 1)
    assert sum(p.stat().st_size for p in tmp_path.glob("*.bin")) <= 32


def test_block_cache_holds_stored_dtype_and_bad_sizes_are_misses(tmp_path):
    cache = BlockCache(tmp_path, 1024)
    block = torch.ones(2, 4, dtype=torch.bfloat16)
    cache.put(IDENTITY, "a", (0, 2), block)
    assert sum(p.stat().st_size for p in tmp_path.glob("*.bin")) == 16
    assert torch.equal(cache.get(IDENTITY, "a", (0, 2), block.dtype, block.shape), block)
    assert cache.get(IDENTITY, "a", (0, 2), torch.float32, block.shape) is None
    next(tmp_path.glob("*.bin")).write_bytes(b"torn")
    assert cache.get(IDENTITY, "a", (0, 2), block.dtype, block.shape) is None


def test_block_cache_concurrent_put_get_evict(tmp_path):
    def exercise(index):
        cache = BlockCache(tmp_path, 64)
        name = str(index % 4)
        block = torch.full((2, 8), index % 4, dtype=torch.bfloat16)
        for _ in range(20):
            cache.put(IDENTITY, name, (0, 2), block)
            result = cache.get(IDENTITY, name, (0, 2), block.dtype, block.shape)
            assert result is None or torch.equal(result, block)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(exercise, range(16)))
    assert sum(p.stat().st_size for p in tmp_path.glob("*.bin")) <= 64
    assert len(list(tmp_path.glob("*.bin"))) == 2  # Four keys competed for two slots.


def test_block_cache_get_refreshes_lru(tmp_path):
    cache = BlockCache(tmp_path, 64)
    block = torch.ones(2, 4)
    for name in ("a", "b"):
        cache.put(IDENTITY, name, (0, 2), block)
    assert cache.get(IDENTITY, "a", (0, 2), block.dtype, block.shape) is not None
    cache.put(IDENTITY, "c", (0, 2), block)
    assert cache.get(IDENTITY, "b", (0, 2), block.dtype, block.shape) is None
    assert cache.get(IDENTITY, "a", (0, 2), block.dtype, block.shape) is not None


def test_cached_read_does_not_wait_for_writer_lock(tmp_path):
    from concurrent.futures import TimeoutError
    cache = BlockCache(tmp_path, 1024)
    block = torch.ones(2, 4)
    cache.put(IDENTITY, 'a', (0, 2), block)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with cache._locked():
            result = pool.submit(cache.get, IDENTITY, 'a', (0, 2), block.dtype, block.shape)
            assert torch.equal(result.result(timeout=.5), block)
    finally:
        pool.shutdown(wait=True)


def test_stale_network_cache_file_is_a_miss(tmp_path, monkeypatch):
    import errno
    from pathlib import Path
    cache = BlockCache(tmp_path, 1024)
    block = torch.ones(2, 4)
    cache.put(IDENTITY, 'a', (0, 2), block)
    original = Path.open
    def stale(path, *args, **kwargs):
        if path.suffix == '.bin':
            raise OSError(errno.ESTALE, 'stale file handle')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', stale)
    assert cache.get(IDENTITY, 'a', (0, 2), block.dtype, block.shape) is None
