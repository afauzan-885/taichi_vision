"""Focused regression tests for device-residency lifetime hardening."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading


_RESIDENCY_PATH = Path(__file__).resolve().parents[1] / "residency.py"
_RESIDENCY_MODULE = "taichi_aot_residency_test_probe"
_SPEC = importlib.util.spec_from_file_location(_RESIDENCY_MODULE, _RESIDENCY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_RESIDENCY_MODULE] = _MODULE
_SPEC.loader.exec_module(_MODULE)
DeviceResidencyCache = _MODULE.DeviceResidencyCache


def test_dispose_callback_runs_outside_residency_lock():
    """A dispose callback may safely enter another thread that needs the cache."""

    cache = DeviceResidencyCache(max_bytes=16)
    callback_completed = threading.Event()
    worker_done = threading.Event()
    workers = []

    def dispose(_buffer):
        def inspect_cache():
            cache.stats()
            worker_done.set()

        worker = threading.Thread(target=inspect_cache)
        workers.append(worker)
        worker.start()
        assert worker_done.wait(1.0), "dispose callback ran while residency lock was held"
        callback_completed.set()

    assert cache.put("a", "owner", object(), 8, dispose=dispose) is not None
    cache.set_budget(0)

    assert callback_completed.is_set()
    for worker in workers:
        worker.join(timeout=1.0)
        assert not worker.is_alive()
    assert cache.size_bytes == 0


def test_same_key_replacement_uses_post_replacement_owner_quota():
    cache = DeviceResidencyCache(max_bytes=100)
    cache.configure_owner("owner", hard_bytes=100)
    disposed = []

    first = cache.put("tile", "owner", "old", 80, dispose=disposed.append)
    assert first is not None

    replacement = cache.put("tile", "owner", "new", 80, dispose=disposed.append)

    assert replacement is not None
    assert replacement.buffer == "new"
    assert cache.size_bytes == 80
    assert cache.stats()["owner_bytes"] == {"owner": 80}
    assert disposed == ["old"]


def test_rejected_replacement_preserves_previous_entry():
    """Admission planning must not evict the old key before rejection is known."""

    cache = DeviceResidencyCache(max_bytes=100)
    cache.configure_owner("owner", hard_bytes=100)
    disposed = []

    previous = cache.put("tile", "owner", "old", 80, dispose=disposed.append)
    assert previous is not None

    assert cache.put("tile", "owner", "too-large", 110, dispose=disposed.append) is None

    current = cache.get("tile")
    assert current is previous
    assert current.buffer == "old"
    assert cache.size_bytes == 80
    assert disposed == []


def test_clear_invalidates_active_lease_and_disposes_after_release():
    cache = DeviceResidencyCache(max_bytes=64)
    disposed = []
    assert cache.put("tile", "owner", "payload", 32, dispose=disposed.append) is not None

    with cache.lease("tile") as leased:
        assert leased is not None
        cache.clear()

        # The current lease remains valid, but clear is an immediate logical
        # barrier for every subsequent lookup/lease.
        assert leased.buffer == "payload"
        assert cache.get("tile") is None
        with cache.lease("tile") as second:
            assert second is None
        assert disposed == []

    assert disposed == ["payload"]
    assert cache.get("tile") is None
    assert cache.size_bytes == 0


def test_budget_zero_invalidates_active_lease_then_converges_on_release():
    cache = DeviceResidencyCache(max_bytes=64)
    disposed = []
    assert cache.put("tile", "owner", "payload", 32, dispose=disposed.append) is not None

    with cache.lease("tile") as leased:
        assert leased is not None
        cache.set_budget(0)
        assert cache.max_bytes == 0
        assert cache.get("tile") is None
        assert cache.size_bytes == 32
        assert disposed == []

    assert cache.size_bytes == 0
    assert disposed == ["payload"]


def test_invalidated_fenced_entry_is_reaped_after_fence_signals():
    ready = {"value": False}
    disposed = []
    cache = DeviceResidencyCache(max_bytes=64)

    assert cache.put(
        "tile",
        "owner",
        "payload",
        32,
        dispose=disposed.append,
        fence_ready=lambda: ready["value"],
    ) is not None

    cache.clear()
    assert cache.get("tile") is None
    assert cache.size_bytes == 32
    assert disposed == []

    ready["value"] = True
    # Any subsequent cache maintenance/lookup reaps the now-safe pending
    # eviction.  Engine memory-policy refreshes provide the same maintenance
    # opportunity in normal runtime operation.
    assert cache.get("tile") is None
    assert cache.size_bytes == 0
    assert disposed == ["payload"]
