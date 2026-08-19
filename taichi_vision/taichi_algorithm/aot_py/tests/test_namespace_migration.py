"""Static regression gates for the taichi_vision namespace migration."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def test_package_sources_do_not_reintroduce_the_legacy_namespace() -> None:
    legacy_namespace = "taichi_" + "library"
    checked_suffixes = {".py", ".md", ".toml", ".json", ".bat", ".ps1"}
    offenders = []
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in checked_suffixes:
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".cpu_aot_cache"} for part in path.parts):
            continue
        if legacy_namespace in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == [], f"legacy namespace found in package sources: {offenders}"


def test_distribution_metadata_uses_the_new_project_name() -> None:
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'name="taichi_vision"' in setup_text
