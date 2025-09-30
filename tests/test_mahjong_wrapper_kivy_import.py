"""Smoke test for the Kivy Mahjong wrapper import."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kivy") is None,
    reason="Kivy is not installed; skipping GUI smoke tests",
)


def test_kivy_wrapper_module_exposes_environment() -> None:
    """Ensure the Kivy wrapper module defines the MahjongEnv class."""

    module = importlib.import_module("mahjong_wrapper_kivy")
    assert hasattr(module, "MahjongEnv"), "MahjongEnv is missing from the Kivy wrapper"
