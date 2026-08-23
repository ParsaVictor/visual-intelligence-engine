"""Tests for the model cache configuration."""
from __future__ import annotations

import os
from pathlib import Path

from vie.cache import configure_cache


def test_no_cache_is_a_noop() -> None:
    assert configure_cache("") is None
    assert configure_cache(None) is None


def test_cache_creates_dirs_and_sets_env(tmp_path: Path, monkeypatch) -> None:
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        monkeypatch.delenv(var, raising=False)
    root = configure_cache(tmp_path / "drive" / "models")
    assert root is not None and root.is_dir()
    assert Path(os.environ["HF_HOME"]).is_dir()
    assert Path(os.environ["TORCH_HOME"]).is_dir()
    assert str(tmp_path) in os.environ["HUGGINGFACE_HUB_CACHE"]
