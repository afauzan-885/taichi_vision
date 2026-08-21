"""Regression tests for the process-owned native runtime boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
import os

import pytest

from taichi_vision.taichi_aot.isolated_runtime import (
    IsolatedRuntime,
    IsolatedRuntimeError,
)


def _assert_bounded_failure(
    command, expected_text: str, timeout: float = 0.4, reap_before_request: bool = False
) -> None:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    client = IsolatedRuntime(process, timeout=timeout)
    started = time.monotonic()
    try:
        if reap_before_request:
            process.wait(timeout=5.0)
        with pytest.raises(IsolatedRuntimeError, match=expected_text):
            client._request("probe", {}, timeout=timeout)
        assert time.monotonic() - started < 3.0
        assert not client.alive
        assert process.poll() is not None
        assert not client._reader.is_alive()
    finally:
        client.close(force=True)


_FAKE_WORKER = textwrap.dedent(
    """
    import json
    import os
    import sys
    import time

    mode = os.environ.get("AOT_TEST_WORKER_MODE", "ok")
    for line in sys.stdin:
        request = json.loads(line)
        opcode = request.get("opcode")
        if opcode == "probe" and mode == "block":
            time.sleep(60)
        if opcode == "probe" and mode == "crash":
            os._exit(17)
        response = {
            "protocol": 1,
            "request_id": request.get("request_id", -1),
            "ok": True,
            "result": {"status": "ok"},
        }
        print(json.dumps(response, separators=(",", ":")), flush=True)
        if opcode == "destroy":
            break
    """
).strip()


def _fake_worker(mode: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["AOT_TEST_WORKER_MODE"] = mode
    return subprocess.Popen(
        [sys.executable, "-c", _FAKE_WORKER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )


def test_blocking_worker_is_terminated_at_ipc_boundary():
    _assert_bounded_failure(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        "timed out",
    )


def test_crashed_worker_returns_bounded_error():
    _assert_bounded_failure(
        [sys.executable, "-c", "import os; os._exit(17)"],
        r"worker (?:exited unexpectedly|is not alive)",
        timeout=2.0,
        reap_before_request=True,
    )


@pytest.mark.parametrize("mode", ["block", "crash"])
def test_failed_session_can_be_retried_with_a_fresh_worker(mode: str):
    process = _fake_worker(mode)
    failed = IsolatedRuntime(process, timeout=0.4)
    try:
        with pytest.raises(IsolatedRuntimeError):
            failed._request("probe", {}, timeout=0.2)
        assert process.poll() is not None
        assert not failed.alive
    finally:
        failed.close(force=True)

    replacement_process = _fake_worker("ok")
    replacement = IsolatedRuntime(replacement_process, timeout=0.4)
    try:
        assert replacement._request("probe", {}, timeout=1.0) == {"status": "ok"}
        assert replacement.alive
    finally:
        replacement.close()
    assert replacement_process.poll() is not None
    assert not replacement._reader.is_alive()


def test_destroy_is_idempotent_and_reaps_worker_and_reader():
    process = _fake_worker("ok")
    client = IsolatedRuntime(process, timeout=0.4)
    try:
        assert client._request("probe", {}, timeout=1.0) == {"status": "ok"}
        client.destroy()
        client.destroy(force=True)
        assert process.poll() is not None
        assert not client.alive
        assert not client._reader.is_alive()
    finally:
        client.close(force=True)
