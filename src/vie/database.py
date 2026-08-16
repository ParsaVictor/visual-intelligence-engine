"""SQLite index.

Schema changes against the original:

* ``face_embeddings`` had no primary key, so re-indexing accumulated duplicate
  rows for the same face. It now keys on ``(file_name, bbox)``.
* embeddings are raw float32 rather than pickle (see :mod:`vie.embeddings`).
* ``gallery_meta`` records a content hash and size, so an *edited* file is
  re-indexed instead of being skipped because its name was seen before, and
  deleted files can be pruned.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vie.embeddings import from_blob, to_blob

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gallery_meta (
    file_name    TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_size    INTEGER NOT NULL,
    has_face     INTEGER NOT NULL DEFAULT 0,
    has_animal   INTEGER NOT NULL DEFAULT 0,
    has_food     INTEGER NOT NULL DEFAULT 0
);

-- (file_name, bbox) is the natural key: one row per detected face. Without it
-- re-running the indexer duplicated every face in the gallery.
CREATE TABLE IF NOT EXISTS face_embeddings (
    file_name TEXT NOT NULL,
    bbox      TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (file_name, bbox),
    FOREIGN KEY (file_name) REFERENCES gallery_meta (file_name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS clip_embeddings (
    file_name TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (file_name) REFERENCES gallery_meta (file_name) ON DELETE CASCADE
);

-- Detector output persisted at index time. The original discarded it and
-- re-ran the detector on every candidate at query time.
CREATE TABLE IF NOT EXISTS animal_boxes (
    file_name  TEXT NOT NULL,
    bbox       TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (file_name, bbox),
    FOREIGN KEY (file_name) REFERENCES gallery_meta (file_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_meta_face   ON gallery_meta (has_face);
CREATE INDEX IF NOT EXISTS idx_meta_animal ON gallery_meta (has_animal);
CREATE INDEX IF NOT EXISTS idx_meta_food   ON gallery_meta (has_food);
"""


def content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """Stable content hash, so edited files are detected as changed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GalleryRow:
    file_name: str
    file_path: str
    has_face: bool
    has_animal: bool
    has_food: bool


class Index:
    """Thin, explicit wrapper over the SQLite index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialise(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # ---- write --------------------------------------------------------

    def upsert_image(
        self,
        conn: sqlite3.Connection,
        *,
        file_name: str,
        file_path: str,
        hash_: str,
        size: int,
        has_face: bool,
        has_animal: bool,
        has_food: bool,
    ) -> None:
        conn.execute(
            """INSERT INTO gallery_meta
                 (file_name, file_path, content_hash, file_size, has_face, has_animal, has_food)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (file_name) DO UPDATE SET
                 file_path = excluded.file_path, content_hash = excluded.content_hash,
                 file_size = excluded.file_size, has_face = excluded.has_face,
                 has_animal = excluded.has_animal, has_food = excluded.has_food""",
            (file_name, file_path, hash_, size, int(has_face), int(has_animal), int(has_food)),
        )

    def add_face(
        self, conn: sqlite3.Connection, file_name: str, box: tuple[int, int, int, int],
        embedding: np.ndarray,
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO face_embeddings (file_name, bbox, embedding) VALUES (?, ?, ?)",
            (file_name, ",".join(map(str, box)), to_blob(embedding)),
        )

    def add_clip(self, conn: sqlite3.Connection, file_name: str, embedding: np.ndarray) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO clip_embeddings (file_name, embedding) VALUES (?, ?)",
            (file_name, to_blob(embedding)),
        )

    def add_animal_box(
        self, conn: sqlite3.Connection, file_name: str, box: tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO animal_boxes (file_name, bbox, confidence) VALUES (?, ?, ?)",
            (file_name, ",".join(map(str, box)), float(confidence)),
        )

    def clear_derived(self, conn: sqlite3.Connection, file_name: str) -> None:
        """Drop previously derived rows for one image before re-indexing it."""
        for table in ("face_embeddings", "clip_embeddings", "animal_boxes"):
            conn.execute(f"DELETE FROM {table} WHERE file_name = ?", (file_name,))

    # ---- read ---------------------------------------------------------

    def needs_indexing(self, conn: sqlite3.Connection, file_name: str, hash_: str) -> bool:
        row = conn.execute(
            "SELECT content_hash FROM gallery_meta WHERE file_name = ?", (file_name,)
        ).fetchone()
        return row is None or row[0] != hash_

    def candidates(self, conn: sqlite3.Connection, flag: str) -> list[GalleryRow]:
        if flag not in {"has_face", "has_animal", "has_food"}:
            raise ValueError(f"unknown flag {flag!r}")
        rows = conn.execute(
            f"""SELECT file_name, file_path, has_face, has_animal, has_food
                FROM gallery_meta WHERE {flag} = 1"""
        ).fetchall()
        return [GalleryRow(r[0], r[1], bool(r[2]), bool(r[3]), bool(r[4])) for r in rows]

    def iter_faces(self, conn: sqlite3.Connection):
        query = """SELECT f.file_name, f.bbox, f.embedding, g.file_path
                   FROM face_embeddings f JOIN gallery_meta g USING (file_name)"""
        for file_name, bbox, blob, file_path in conn.execute(query):
            box = tuple(int(v) for v in bbox.split(","))
            yield file_name, box, from_blob(blob), file_path

    def iter_clip(self, conn: sqlite3.Connection):
        query = """SELECT c.file_name, c.embedding, g.file_path
                   FROM clip_embeddings c JOIN gallery_meta g USING (file_name)"""
        for file_name, blob, file_path in conn.execute(query):
            yield file_name, from_blob(blob), file_path

    def prune_missing(self, conn: sqlite3.Connection, present: set[str]) -> int:
        """Remove rows for files that no longer exist on disk."""
        known = {r[0] for r in conn.execute("SELECT file_name FROM gallery_meta")}
        gone = known - present
        for file_name in gone:
            self.clear_derived(conn, file_name)
            conn.execute("DELETE FROM gallery_meta WHERE file_name = ?", (file_name,))
        return len(gone)
