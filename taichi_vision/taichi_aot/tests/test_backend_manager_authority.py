"""Focused tests for BackendManager capability/status authority."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


_BACKEND_MANAGER_PATH = Path(__file__).resolve().parents[1] / "backend_manager.py"


def _load_backend_manager(monkeypatch, *, candidates, capabilities):
    """Load backend_manager.py without importing the side-effecting AOT facade."""

    package_name = "aot_backend_manager_test_pkg"
    module_name = f"{package_name}.backend_manager"

    package = types.ModuleType(package_name)
    package.__path__ = [str(_BACKEND_MANAGER_PATH.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)

    caps_module = types.ModuleType(f"{package_name}.capabilities")
    caps_module.backend_candidates = lambda _device: list(candidates)
    caps_module.classify_device = lambda _device, backend: capabilities[backend]
    monkeypatch.setitem(sys.modules, caps_module.__name__, caps_module)

    root = types.ModuleType("taichi_vision")
    root.__path__ = []
    backend_config = types.ModuleType("taichi_vision.backend_config")

    def normalize_backend(value, *, allow_auto=True):
        token = str(value or "auto").strip().lower()
        if token == "auto" and allow_auto:
            return "auto"
        if token in {"cpu", "cuda", "vulkan", "opengl", "gles"}:
            return token
        return "auto" if allow_auto else "cpu"

    backend_config.normalize_backend = normalize_backend
    root.backend_config = backend_config
    monkeypatch.setitem(sys.modules, "taichi_vision", root)
    monkeypatch.setitem(sys.modules, "taichi_vision.backend_config", backend_config)

    spec = importlib.util.spec_from_file_location(module_name, _BACKEND_MANAGER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _caps(*, safe, reason="", vendor="unknown"):
    return types.SimpleNamespace(safe=safe, reason=reason, vendor=vendor)


def test_automatic_execution_never_re_admits_capability_rejection(monkeypatch):
    module = _load_backend_manager(
        monkeypatch,
        candidates=["vulkan", "cpu"],
        capabilities={
            "vulkan": _caps(safe=False, reason="missing compute capability"),
            "cpu": _caps(safe=True),
        },
    )
    manager = module.BackendManager(validated={})
    calls = []

    result, backend, errors = manager.run_with_fallback(
        lambda candidate: calls.append(candidate) or f"ran:{candidate}",
        requested="auto",
    )

    assert result == "ran:cpu"
    assert backend == "cpu"
    assert errors == {}
    assert calls == ["cpu"]


def test_explicit_rejected_backend_has_no_fabricated_cpu_selection(monkeypatch):
    module = _load_backend_manager(
        monkeypatch,
        candidates=["cpu"],
        capabilities={
            "vulkan": _caps(safe=False, reason="unsafe device"),
            "cpu": _caps(safe=True),
        },
    )
    manager = module.BackendManager(validated={})
    decision = manager.decide("vulkan")

    assert decision.selected is None
    assert decision.candidates == ["vulkan"]
    assert decision.eligible == []
    assert decision.rejected == {"vulkan": "unsafe device"}
    assert "explicit backend 'vulkan' rejected" in decision.reason

    calls = []
    with pytest.raises(RuntimeError, match="rejected before execution"):
        manager.run_with_fallback(
            lambda candidate: calls.append(candidate),
            requested="vulkan",
        )
    assert calls == []


def test_persisted_quarantine_remains_part_of_authoritative_filter(monkeypatch):
    module = _load_backend_manager(
        monkeypatch,
        candidates=["vulkan", "cpu"],
        capabilities={
            "vulkan": _caps(safe=True, reason="generic safe"),
            "cpu": _caps(safe=True),
        },
    )
    manager = module.BackendManager(validated={"vulkan": "quarantined"})
    decision = manager.decide("auto")

    assert decision.selected == "cpu"
    assert decision.eligible == ["cpu"]
    assert "vulkan" in decision.rejected


def test_exact_current_intel_manifest_can_supersede_legacy_generic_quarantine(monkeypatch):
    module = _load_backend_manager(
        monkeypatch,
        candidates=["vulkan", "cpu"],
        capabilities={
            "vulkan": _caps(
                safe=True,
                vendor="intel",
                reason="Intel Vulkan lifecycle manifest validated",
            ),
            "cpu": _caps(safe=True),
        },
    )
    manager = module.BackendManager(validated={"vulkan": "quarantined"})
    decision = manager.decide("auto")

    assert decision.selected == "vulkan"
    assert decision.eligible[0] == "vulkan"


def test_automatic_runtime_failure_advances_only_to_next_eligible_backend(monkeypatch):
    module = _load_backend_manager(
        monkeypatch,
        candidates=["vulkan", "opengl", "cpu"],
        capabilities={
            "vulkan": _caps(safe=True),
            "opengl": _caps(safe=False, reason="not qualified"),
            "cpu": _caps(safe=True),
        },
    )
    manager = module.BackendManager(validated={})
    calls = []

    def operation(backend):
        calls.append(backend)
        if backend == "vulkan":
            raise RuntimeError("sentinel Vulkan failure")
        return backend

    result, backend, errors = manager.run_with_fallback(operation, requested="auto")

    assert result == "cpu"
    assert backend == "cpu"
    assert calls == ["vulkan", "cpu"]
    assert "vulkan" in errors
    assert "opengl" not in errors


def test_explicit_runtime_failure_never_falls_back(monkeypatch):
    module = _load_backend_manager(
        monkeypatch,
        candidates=["cpu"],
        capabilities={
            "vulkan": _caps(safe=True),
            "cpu": _caps(safe=True),
        },
    )
    manager = module.BackendManager(validated={})
    calls = []

    with pytest.raises(RuntimeError, match="explicit backend failed"):
        manager.run_with_fallback(
            lambda backend: calls.append(backend) or (_ for _ in ()).throw(
                RuntimeError("sentinel")
            ),
            requested="vulkan",
        )

    assert calls == ["vulkan"]
