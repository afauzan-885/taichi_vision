"""Regression tests for recoverable lazy block-adapter registration."""

from __future__ import annotations

import taichi_vision.taichi_aot.block_adapters as adapters


def test_failed_family_is_retried_without_repeating_successful_families(monkeypatch):
    calls = {"good": 0, "flaky": 0}
    attempts = {"flaky": 0}

    def good(*, replace=False):
        calls["good"] += 1

    def flaky(*, replace=False):
        calls["flaky"] += 1
        attempts["flaky"] += 1
        if attempts["flaky"] == 1:
            raise RuntimeError("transient adapter import")

    monkeypatch.setattr(adapters, "register_test_good_adapters", good, raising=False)
    monkeypatch.setattr(adapters, "register_test_flaky_adapters", flaky, raising=False)
    monkeypatch.setattr(adapters, "_DEFAULT_ADAPTERS_INITIALIZED", False)
    monkeypatch.setattr(adapters, "_DEFAULT_ADAPTER_REGISTRATION_ERRORS", {})
    monkeypatch.setattr(adapters, "_DEFAULT_ADAPTER_REGISTRATION_STATUS", {})

    adapters.ensure_default_block_adapters()
    adapters.ensure_default_block_adapters()

    assert calls == {"good": 1, "flaky": 2}
    assert adapters._DEFAULT_ADAPTERS_INITIALIZED is True
    assert "register_test_flaky_adapters" not in adapters._DEFAULT_ADAPTER_REGISTRATION_ERRORS


def test_replace_retries_ready_families(monkeypatch):
    calls = []

    def helper(*, replace=False):
        calls.append(bool(replace))

    monkeypatch.setattr(adapters, "register_test_replace_adapters", helper, raising=False)
    monkeypatch.setattr(adapters, "_DEFAULT_ADAPTERS_INITIALIZED", True)
    monkeypatch.setattr(adapters, "_DEFAULT_ADAPTER_REGISTRATION_ERRORS", {})
    monkeypatch.setattr(adapters, "_DEFAULT_ADAPTER_REGISTRATION_STATUS", {})

    adapters.ensure_default_block_adapters(replace=True)
    assert calls == [True]
