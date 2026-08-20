"""Regression tests for strict serialized safety booleans."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from taichi_vision.backend_config import parse_policy_bool


_GFX_PATH = Path(__file__).resolve().parents[1] / "gfx_capabilities.py"
_GFX_NAME = "taichi_aot_gfx_capabilities_test_probe"
_SPEC = importlib.util.spec_from_file_location(_GFX_NAME, _GFX_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GFX = importlib.util.module_from_spec(_SPEC)
sys.modules[_GFX_NAME] = _GFX
_SPEC.loader.exec_module(_GFX)


def test_parse_policy_bool_accepts_only_unambiguous_values():
    assert parse_policy_bool(True) is True
    assert parse_policy_bool(False) is False
    assert parse_policy_bool(1) is True
    assert parse_policy_bool(0) is False
    assert parse_policy_bool("true") is True
    assert parse_policy_bool("TRUE") is True
    assert parse_policy_bool("1") is True
    assert parse_policy_bool("yes") is True
    assert parse_policy_bool("on") is True
    assert parse_policy_bool("false") is False
    assert parse_policy_bool("FALSE") is False
    assert parse_policy_bool("0") is False
    assert parse_policy_bool("no") is False
    assert parse_policy_bool("off") is False

    assert parse_policy_bool("maybe") is None
    assert parse_policy_bool(2) is None
    assert parse_policy_bool(1.0) is None
    assert parse_policy_bool(object()) is None
    assert parse_policy_bool("maybe", default=False) is False


def test_vulkan_false_strings_do_not_enable_required_features():
    decision = _GFX.classify_vulkan(
        "1.1",
        features={"compute": "false", "ssbo": "0"},
        required_spirv="1.3",
    )

    assert decision.status == "unsupported"
    assert "compute" in decision.reason
    assert "ssbo" in decision.reason


def test_vulkan_true_strings_are_intentionally_supported():
    decision = _GFX.classify_vulkan(
        "1.1",
        features={"compute": "true", "storage_buffer": "1"},
        required_spirv="1.3",
    )

    assert decision.status == "native_candidate"


def test_ambiguous_graphics_feature_value_fails_closed():
    decision = _GFX.classify_vulkan(
        "1.1",
        features={"compute": "definitely", "ssbo": True},
        required_spirv="1.3",
    )

    assert decision.status == "unsupported"
    assert "compute" in decision.reason


def test_explicit_opengl_false_string_overrides_version_inference():
    decision = _GFX.classify_desktop_opengl(
        "4.3",
        compute_shader="false",
        ssbo="true",
    )

    assert decision.status == "unsupported"
    assert "missing compute shader or SSBO" in decision.reason


def test_explicit_gles_false_string_fails_closed():
    decision = _GFX.classify_gles(
        "3.1",
        compute_shader="true",
        ssbo="false",
    )

    assert decision.status == "unsupported"
