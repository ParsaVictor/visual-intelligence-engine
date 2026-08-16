"""Regression tests for P2-6 (missing primary key) and P2-7 (incremental indexing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vie.database import Index, content_hash


@pytest.fixture()
def index(tmp_path: Path) -> Index:
    idx = Index(tmp_path / "index.db")
    idx.initialise()
    return idx


def _register(index: Index, conn, name: str, *, face=False, animal=False, food=False) -> None:
    index.upsert_image(
        conn, file_name=name, file_path=f"/g/{name}", hash_="h", size=1,
        has_face=face, has_animal=animal, has_food=food,
    )


def test_reindexing_the_same_face_does_not_duplicate_rows(index: Index) -> None:
    """P2-6: face_embeddings had no primary key, so re-runs accumulated duplicates."""
    emb = np.ones(512, dtype=np.float32)
    with index.connect() as conn:
        _register(index, conn, "a.jpg", face=True)
        for _ in range(3):
            index.add_face(conn, "a.jpg", (1, 2, 3, 4), emb)
        count = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
    assert count == 1


def test_two_distinct_faces_in_one_image_are_both_kept(index: Index) -> None:
    """Deduplication must key on the box, not just the file."""
    emb = np.ones(512, dtype=np.float32)
    with index.connect() as conn:
        _register(index, conn, "a.jpg", face=True)
        index.add_face(conn, "a.jpg", (0, 0, 10, 10), emb)
        index.add_face(conn, "a.jpg", (50, 50, 60, 60), emb)
        count = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
    assert count == 2


def test_clip_embedding_is_replaced_not_duplicated(index: Index) -> None:
    with index.connect() as conn:
        _register(index, conn, "a.jpg")
        index.add_clip(conn, "a.jpg", np.ones(768, dtype=np.float32))
        index.add_clip(conn, "a.jpg", np.zeros(768, dtype=np.float32))
        rows = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
    assert rows == 1


def test_candidate_prefilter_returns_only_flagged_images(index: Index) -> None:
    """The metadata prefilter is the core performance idea of the whole system."""
    with index.connect() as conn:
        _register(index, conn, "face.jpg", face=True)
        _register(index, conn, "animal.jpg", animal=True)
        _register(index, conn, "plain.jpg")
        faces = index.candidates(conn, "has_face")
        animals = index.candidates(conn, "has_animal")
    assert [r.file_name for r in faces] == ["face.jpg"]
    assert [r.file_name for r in animals] == ["animal.jpg"]


def test_unknown_flag_is_rejected(index: Index) -> None:
    with index.connect() as conn, pytest.raises(ValueError, match="unknown flag"):
        index.candidates(conn, "has_robots; DROP TABLE gallery_meta")


def test_unchanged_file_is_not_reindexed(index: Index) -> None:
    with index.connect() as conn:
        index.upsert_image(
            conn, file_name="a.jpg", file_path="/g/a.jpg", hash_="abc", size=1,
            has_face=False, has_animal=False, has_food=False,
        )
        assert index.needs_indexing(conn, "a.jpg", "abc") is False


def test_edited_file_is_reindexed(index: Index) -> None:
    """P2-7: keying on filename alone meant an edited photo was never re-indexed."""
    with index.connect() as conn:
        index.upsert_image(
            conn, file_name="a.jpg", file_path="/g/a.jpg", hash_="abc", size=1,
            has_face=False, has_animal=False, has_food=False,
        )
        assert index.needs_indexing(conn, "a.jpg", "def") is True


def test_new_file_needs_indexing(index: Index) -> None:
    with index.connect() as conn:
        assert index.needs_indexing(conn, "never-seen.jpg", "abc") is True


def test_deleted_files_are_pruned_with_their_embeddings(index: Index) -> None:
    with index.connect() as conn:
        _register(index, conn, "gone.jpg", face=True)
        _register(index, conn, "kept.jpg", face=True)
        index.add_face(conn, "gone.jpg", (0, 0, 1, 1), np.ones(512, dtype=np.float32))
        removed = index.prune_missing(conn, present={"kept.jpg"})
        remaining = conn.execute("SELECT COUNT(*) FROM gallery_meta").fetchone()[0]
        orphans = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
    assert (removed, remaining, orphans) == (1, 1, 0)


def test_content_hash_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "img.bin"
    path.write_bytes(b"one")
    first = content_hash(path)
    path.write_bytes(b"two")
    assert content_hash(path) != first
