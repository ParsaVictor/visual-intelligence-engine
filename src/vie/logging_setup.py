"""Logging configuration.

The notebooks used ``print`` for everything and a bare ``except:`` that swallowed
the ImageNet class-list download failure — after which every query returned
empty with no warning and no way to tell a broken index from an empty gallery.

Anything that can silently degrade results must be able to say so.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the ``vie`` logger tree once."""
    root = logging.getLogger("vie")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vie.{name}")
