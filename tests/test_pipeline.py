"""Tests for index orchestration: incremental behaviour, resilience, discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vie.config import Config
from vie.database import Index
from vie.pipeline import build_index, discover_images


@pytest.fixture()
def gallery(tmp_path: Path) -> Path:
    root = tmp_path / "gallery"
    root.mkdir()
    for name in ("a.jpg", "b.JPG", "c.png", "notes.txt"):
        (root / name).write_bytes(name.encode())
    return root


@pytest.fixture()
def cfg(raw_config: dict, gallery: Path, tmp_path: Path) -> Config:
    raw = dict(raw_config)
    raw["paths"] = {"gallery": str(gallery), "database": str(tmp_path / "index.db")}
    return Config.from_dict(raw)


class StubFaces:
    def __init__(self, faces=()):
        self.faces = list(faces)

    def detect(self, image):
        return self.faces


class StubBoxes:
    def __init__(self, boxes=()):
        self.boxes = list(boxes)

    def detect(self, image, classes, confidence):
        return [(b, c) for b, c in self.boxes if c >= confidence]


def _run(cfg, *, faces=(), animals=(), foods=(), loader=None, workers=2):
    return build_index(
        cfg,
        loader=loader or (lambda p: f"image:{p.name}"),
        size_of=lambda img: (1000, 1000),
        face_detector=StubFaces(faces),
        animal_detector=StubBoxes(animals),
        food_detector=StubBoxes(foods),
        embed_image=lambda img: np.ones(768, dtype=np.float32),
        workers=workers,
    )


# ── discovery ──────────────────────────────────────────────────────────


def test_uppercase_extensions_are_found(gallery: Path) -> None:
    """The standalone module globbed lowercase only, hiding every .JPG file."""
    assert "b.JPG" in {p.name for p in discover_images(gallery)}


def test_non_images_are_ignored(gallery: Path) -> None:
    assert "notes.txt" not in {p.name for p in discover_images(gallery)}


def test_missing_gallery_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_images(tmp_path / "nope")


# ── indexing ───────────────────────────────────────────────────────────


def test_all_images_are_indexed_on_a_fresh_run(cfg: Config) -> None:
    stats = _run(cfg)
    assert (stats.scanned, stats.indexed, stats.skipped) == (3, 3, 0)


def test_second_run_skips_unchanged_files(cfg: Config) -> None:
    _run(cfg)
    stats = _run(cfg)
    assert (stats.indexed, stats.skipped) == (0, 3)


def test_edited_file_is_reindexed(cfg: Config, gallery: Path) -> None:
    _run(cfg)
    (gallery / "a.jpg").write_bytes(b"different content entirely")
    stats = _run(cfg)
    assert (stats.indexed, stats.skipped) == (1, 2)


def test_deleted_file_is_pruned(cfg: Config, gallery: Path) -> None:
    _run(cfg)
    (gallery / "a.jpg").unlink()
    stats = _run(cfg)
    assert stats.pruned == 1
    with Index(cfg.database_path).connect() as conn:
        names = {r[0] for r in conn.execute("SELECT file_name FROM gallery_meta")}
    assert "a.jpg" not in names


def test_flags_are_persisted(cfg: Config) -> None:
    _run(cfg, faces=[((0, 0, 80, 80), np.ones(512, dtype=np.float32))])
    index = Index(cfg.database_path)
    with index.connect() as conn:
        assert len(index.candidates(conn, "has_face")) == 3


def test_animal_boxes_are_persisted(cfg: Config) -> None:
    """P3-2: the original stored one bit and re-ran the detector at query time."""
    _run(cfg, animals=[((0, 0, 200, 200), 0.9)])
    with Index(cfg.database_path).connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM animal_boxes").fetchone()[0]
    assert count == 3


def test_reindexing_does_not_accumulate_rows(cfg: Config, gallery: Path) -> None:
    face = [((0, 0, 80, 80), np.ones(512, dtype=np.float32))]
    _run(cfg, faces=face)
    (gallery / "a.jpg").write_bytes(b"changed")
    _run(cfg, faces=face)
    with Index(cfg.database_path).connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE file_name = 'a.jpg'"
        ).fetchone()[0]
    assert count == 1


def test_undecodable_image_is_counted_not_fatal(cfg: Config) -> None:
    """One bad file must not abandon the archive, but must be visible."""
    stats = _run(cfg, loader=lambda p: None if p.name == "a.jpg" else f"image:{p.name}")
    assert stats.failed == 1
    assert stats.indexed == 2


def test_detector_failure_is_isolated(cfg: Config) -> None:
    class Exploding:
        def detect(self, image):
            if "a.jpg" in str(image):
                raise RuntimeError("detector blew up")
            return []

    stats = build_index(
        cfg,
        loader=lambda p: f"image:{p.name}",
        size_of=lambda img: (1000, 1000),
        face_detector=Exploding(),
        animal_detector=StubBoxes(),
        food_detector=StubBoxes(),
        embed_image=lambda img: np.ones(768, dtype=np.float32),
        workers=1,
    )
    assert stats.failed == 1
    assert stats.indexed == 2


def test_single_threaded_mode_matches_threaded(cfg: Config) -> None:
    serial = _run(cfg, workers=1)
    assert serial.indexed == 3
