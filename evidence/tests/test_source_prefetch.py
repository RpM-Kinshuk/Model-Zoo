from threading import Event

import pytest

from evidence.source_prefetch import SourcePrefetch


def test_next_group_is_read_while_current_payload_is_held():
    next_loaded = Event()
    reads = []
    def load(group):
        reads.append(group)
        if group == 1:
            next_loaded.set()
        return bytes([group])
    source = SourcePrefetch([0, 1, 2], load, lambda group: 1, window=2, max_bytes=1)
    try:
        group, payload = source.next()
        assert (group, payload) == (0, b'\0')
        assert next_loaded.wait(timeout=2), "next read was not overlapped"
        assert reads == [0, 1]  # A third group cannot start while current is held.
        assert source.next() == (1, b'\1')
        assert source.next() == (2, b'\2')
        with pytest.raises(StopIteration):
            source.next()
    finally:
        source.close()
    assert reads == [0, 1, 2]


def test_over_budget_group_is_not_preloaded():
    reads = []
    source = SourcePrefetch([0, 1], lambda group: reads.append(group) or group,
                            lambda group: 100 if group == 0 else 1, window=2, max_bytes=8)
    try:
        assert source.next() == (0, None)
        assert source.next() == (1, 1)
    finally:
        source.close()
    assert reads == [1]


def test_loader_error_propagates_and_close_is_safe():
    def load(group):
        if group == 1:
            raise OSError("bounded read failed")
        return group
    source = SourcePrefetch([0, 1], load, lambda group: 1, window=2, max_bytes=8)
    try:
        assert source.next() == (0, 0)
        with pytest.raises(OSError, match="bounded read failed"):
            source.next()
    finally:
        source.close()
        source.close()
