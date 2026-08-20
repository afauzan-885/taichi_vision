"""Regression tests for host BlockCache payload lifetime."""

from __future__ import annotations

from taichi_vision.taichi_aot.block import BlockCache, BlockRecord, BlockState


def _cache_with_payload():
    cache = BlockCache(max_entries=4, max_bytes=1024)
    record = BlockRecord(
        "tile",
        state=BlockState.READY,
        data=bytearray(b"stable-payload"),
        owner="test",
    )
    cache.put(record)
    return cache


def test_clear_defers_detach_until_final_lease_release():
    cache = _cache_with_payload()

    with cache.lease("tile") as leased:
        assert leased is not None
        cache.clear()

        # Clear is a logical barrier for new consumers, but the active
        # consumer keeps its payload attached until the context exits.
        assert leased.data == bytearray(b"stable-payload")
        assert cache.get("tile") is None
        with cache.lease("tile") as second:
            assert second is None
        assert cache.size_bytes == len(b"stable-payload")

    assert cache.size_bytes == 0
    assert len(cache) == 0
    assert leased.data is None


def test_invalidation_defers_detach_and_blocks_new_leases():
    cache = _cache_with_payload()

    with cache.lease("tile") as leased:
        assert leased is not None
        assert cache.invalidate("tile")
        assert leased.data == bytearray(b"stable-payload")
        assert cache.get("tile") is None
        with cache.lease("tile") as second:
            assert second is None

    assert cache.size_bytes == 0
    assert len(cache) == 0
