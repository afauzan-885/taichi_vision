"""Contract tests for concurrent watchdog tracking and fatal exits."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import threading
import textwrap

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


def test_dead_worker_tracking_is_pruned(clean_tracking) -> None:
    def worker() -> None:
        engine._op_begin("worker-exception")
        try:
            raise RuntimeError("injected worker failure")
        except RuntimeError:
            pass

    thread = threading.Thread(target=worker, name="watchdog-dead-worker")
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()

    snapshot = engine._watchdog_snapshot()
    assert snapshot["operation"] is None
    assert not engine._active_operations


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [("lock", 1), ("operation", 1), ("signal", 143)],
)
def test_fatal_watchdog_paths_reach_hard_exit_without_cleanup(
    mode: str, expected_code: int
) -> None:
    source = Path(__file__).parents[1] / "engine.py"
    script = textwrap.dedent(
        f"""
        import importlib.util, signal, sys, threading, time
        source = r"{source}"
        spec = importlib.util.spec_from_file_location(
            "taichi_vision.taichi_aot.engine_watchdog_child", source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module._WATCHDOG_INTERVAL_S = 0.01
        module._OP_TIMEOUT_S = 0.02
        module._LOCK_CONTENTION_S = 0.02
        gate = threading.Lock()
        class Held:
            arch = "cpu"
            def destroy(self):
                if {mode!r} == "operation":
                    while True:
                        time.sleep(0.1)
                gate.acquire()
        module.AOTEngine._instances.clear()
        module.AOTEngine._instances["held"] = Held()
        if {mode!r} == "lock":
            gate.acquire()
            module._lock_wait_begin("held-lock")
        elif {mode!r} == "operation":
            module._op_begin("held-operation")
        else:
            gate.acquire()
            module._signal_cleanup_handler(signal.SIGTERM, None)
        if {mode!r} != "signal":
            module._watchdog_run()
        """
    )
    env = os.environ.copy()
    env["AUTO_DESTROY"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        # Importing engine.py constructs the compatibility singleton and may
        # load the native bridge. Keep this watchdog assertion independent of
        # host CPU/loader startup latency while still bounding a real hang.
        timeout=30,
        env=env,
    )
    assert result.returncode == expected_code, result.stderr
    assert "fatal exit" in result.stderr
