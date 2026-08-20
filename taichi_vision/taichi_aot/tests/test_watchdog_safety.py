"""Contract tests for concurrent watchdog tracking and fatal exits."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import threading

import pytest


_engine_spec = importlib.util.spec_from_file_location(
    "taichi_vision.taichi_aot.engine_watchdog_test",
    Path(__file__).parents[1] / "engine.py",
)
assert _engine_spec and _engine_spec.loader
engine = importlib.util.module_from_spec(_engine_spec)
sys.modules[_engine_spec.name] = engine
_engine_spec.loader.exec_module(engine)


@pytest.fixture
def clean_tracking(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_AUTO_DESTROY_ENABLED", True)
    with engine._heartbeat_lock:
        engine._active_operations.clear()
        engine._active_lock_waits.clear()
    engine._tracking_local.operation_tokens = []
    engine._tracking_local.lock_wait_tokens = []
    yield
    with engine._heartbeat_lock:
        engine._active_operations.clear()
        engine._active_lock_waits.clear()


def test_watchdog_tracks_overlapping_operations_per_thread(clean_tracking) -> None:
    main_token = engine._op_begin("main-operation")
    worker_started = threading.Event()
    worker_release = threading.Event()

    def worker() -> None:
        token = engine._op_begin("worker-operation")
        worker_started.set()
        worker_release.wait(timeout=5)
        engine._op_end(token)

    thread = threading.Thread(target=worker, name="watchdog-test-worker")
    thread.start()
    assert worker_started.wait(timeout=5)

    snapshot = engine._watchdog_snapshot()
    assert snapshot["operation"]["name"] == "main-operation"
    assert len(engine._active_operations) == 2

    engine._op_end(main_token)
    worker_release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not engine._active_operations


def test_watchdog_tracks_overlapping_lock_waits_per_thread(clean_tracking) -> None:
    main_token = engine._lock_wait_begin("main-lock")
    worker_started = threading.Event()
    worker_release = threading.Event()

    def worker() -> None:
        token = engine._lock_wait_begin("worker-lock")
        worker_started.set()
        worker_release.wait(timeout=5)
        engine._lock_wait_end(token)

    thread = threading.Thread(target=worker, name="watchdog-test-lock-worker")
    thread.start()
    assert worker_started.wait(timeout=5)

    snapshot = engine._watchdog_snapshot()
    assert snapshot["lock_wait"]["name"] == "main-lock"
    assert len(engine._active_lock_waits) == 2

    engine._lock_wait_end(main_token)
    worker_release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not engine._active_lock_waits


def test_fatal_exit_skips_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup_calls: list[str] = []
    exit_codes: list[int] = []
    monkeypatch.setattr(
        engine, "_force_global_cleanup", lambda reason: cleanup_calls.append(reason)
    )
    monkeypatch.setattr(engine.os, "_exit", lambda code: exit_codes.append(code))

    engine._fatal_exit("test-watchdog", code=23)

    assert cleanup_calls == []
    assert exit_codes == [23]
