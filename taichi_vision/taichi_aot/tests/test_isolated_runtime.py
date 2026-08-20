"""Regression tests for the process-owned native runtime boundary."""

from __future__ import annotations

import subprocess
import sys
import time

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
    finally:
        client.close(force=True)


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
