"""Persistent model cache.

Downloading multi-gigabyte weights on every Colab session is slow and burns
bandwidth. Pointing the HuggingFace and torch caches at a persistent directory
— a mounted Google Drive folder, say — downloads each model once and reuses it
across sessions.

Call :func:`configure_cache` before loading any model. It only sets environment
variables, so it is safe to call when no cache dir is configured (a no-op).
"""

from __future__ import annotations

import os
from pathlib import Path


def configure_cache(cache_dir: str | Path | None) -> Path | None:
    """Route HuggingFace, torch.hub and Ultralytics caches to ``cache_dir``.

    Returns the resolved path, or None when no cache is configured.
    """
    if not cache_dir:
        return None

    root = Path(cache_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    hf = root / "huggingface"
    hub = root / "torch_hub"
    for sub in (hf, hub):
        sub.mkdir(parents=True, exist_ok=True)

    # HuggingFace (transformers + hub downloads)
    os.environ["HF_HOME"] = str(hf)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf / "transformers")
    # torch.hub (MegaDetector / yolov5)
    os.environ["TORCH_HOME"] = str(hub)

    return root
