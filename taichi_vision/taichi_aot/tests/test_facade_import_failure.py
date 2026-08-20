"""Regression coverage for AOT facade import failure containment.

The package facade must not retry ``engine.py`` after a runtime/backend
initialization exception.  This test executes the facade under an isolated
package name and intercepts only the relative ``.engine`` import, so no real
GPU runtime or driver is initialized.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
import sys

import pytest


class _FailingEngineLoader(importlib.abc.Loader):
    def __init__(self, attempts):
        self._attempts = attempts

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._attempts.append(module.__name__)
        raise RuntimeError("sentinel engine initialization failure")


class _FailingEngineFinder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname, attempts):
        self._fullname = fullname
        self._attempts = attempts

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._fullname:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _FailingEngineLoader(self._attempts),
        )


def test_facade_does_not_retry_engine_after_runtime_error(monkeypatch):
    """One facade import must trigger at most one side-effecting engine import."""

    package_init = Path(__file__).resolve().parents[1] / "__init__.py"
    package_name = "taichi_aot_import_probe"
    engine_name = f"{package_name}.engine"
    attempts = []

    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init,
        submodule_search_locations=[str(package_init.parent)],
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    finder = _FailingEngineFinder(engine_name, attempts)
    monkeypatch.setenv("PIXEL_REFINE_USE_NATIVE_ENGINE", "1")

    sys.modules[package_name] = module
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(RuntimeError, match="sentinel engine initialization failure"):
            spec.loader.exec_module(module)
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        sys.modules.pop(engine_name, None)
        sys.modules.pop(package_name, None)

    assert attempts == [engine_name]
