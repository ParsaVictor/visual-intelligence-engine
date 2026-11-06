"""SQLite index.

Schema changes against the original:

* ``face_embeddings`` had no primary key, so re-indexing accumulated duplicate
  rows for the same face. It now keys on ``(file_name, bbox)``.
* embeddings are raw float32 rather than pickle (see :mod:`vie.embeddings`).
* ``gallery_meta`` records a content hash and size, so an *edited* file is
  re-indexed instead of being skipped because its name was seen before, and
  deleted files can be pruned.

Migrations
----------
``CREATE TABLE IF NOT EXISTS`` does not change a table that already exists, so
before schema 2 an index built by an older version crashed on its first write
with ``no column named region``. :meth:`Index.initialise` now upgrades in place
and refuses an index written by a *newer* version rather than guessing.

``file_name`` is the gallery-relative POSIX path. For a flat gallery that is
exactly the bare file name older versions stored, so existing rows stay valid.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vie.embeddings import from_blob, to_blob

SCHEMA_VERSION = 4


class SchemaError(RuntimeError):
    """The index cannot be used by this version of the code."""


class EmbeddingModelMismatch(SchemaError):
    """Vectors from two different models must never be compared."""


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
    has_food     INTEGER NOT NULL DEFAULT 0,
    -- The two food gates are stored separately so their disagreement is
    -- visible: the detector covers COCO's ten Western food classes, the
    -- vision-language score covers everything else at no extra inference.
    food_by_detector INTEGER NOT NULL DEFAULT 0,
    food_by_vlm      INTEGER NOT NULL DEFAULT 0,
    food_vlm_score   REAL    NOT NULL DEFAULT 0.0,
    -- Boxes are stored in the coordinates of the capped image the models saw.
    -- Dividing by this factor maps them back onto the original file.
    index_scale  REAL    NOT NULL DEFAULT 1.0,
    mtime_ns     INTEGER NOT NULL DEFAULT 0,
    -- 1 once the image's object vectors are stored (see object_embeddings).
    objects_embedded INTEGER NOT NULL DEFAULT 0
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

-- One row per encoded region. 'whole' is the full frame; the rest are crops.
-- A single whole-image vector averages the scene, so a small subject is
-- diluted away -- encoding regions lets a query match the best one.
CREATE TABLE IF NOT EXISTS clip_embeddings (
    file_name TEXT NOT NULL,
    region    TEXT NOT NULL DEFAULT 'whole',
    embedding BLOB NOT NULL,
    PRIMARY KEY (file_name, region),
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

-- Top-k classifier output per animal crop. This table is what lets an
-- in-vocabulary species query run with no model loaded at all.
CREATE TABLE IF NOT EXISTS animal_predictions (
    file_name   TEXT NOT NULL,
    bbox        TEXT NOT NULL,
    class_id    INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY (file_name, bbox, class_id),
    FOREIGN KEY (file_name) REFERENCES gallery_meta (file_name) ON DELETE CASCADE
);

-- Same idea for food: a dish query never needs to re-run RT-DETR.
CREATE TABLE IF NOT EXISTS food_boxes (
    file_name  TEXT NOT NULL,
    bbox       TEXT NOT NULL,
    confidence REAL NOT NULL,
    class_id   INTEGER,
    PRIMARY KEY (file_name, bbox),
    FOREIGN KEY (file_name) REFERENCES gallery_meta (file_name) ON DELETE CASCADE
);

-- A vision-language vector of each detected animal and dish, cropped and padded
-- exactly as the search paths crop them. With these stored, a food query or an
-- out-of-vocabulary species query is a matrix product: no image is decoded and no
-- image model runs at query time.
CREATE TABLE IF NOT EXISTS object_embeddings (
    file_name  TEXT NOT NULL,
    kind       TEXT NOT NULL,          -- 'animal' or 'food'
    bbox       TEXT NOT NULL,          -- the detector's box
    crop       TEXT NOT NULL,          -- the padded box that was encoded
    confidence REAL NOT NULL,
    class_id   INTEGER,
    embedding  BLOB NOT NULL,
    PRIMARY KEY (file_name, kind, bbox),
    FOREIGN KEY (file_name) REFERENCES gallery_meta (file_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_meta_face   ON gallery_meta (has_face);
CREATE INDEX IF NOT EXISTS idx_meta_animal ON gallery_meta (has_animal);
CREATE INDEX IF NOT EXISTS idx_meta_food   ON gallery_meta (has_food);
CREATE INDEX IF NOT EXISTS idx_pred_class  ON animal_predictions (class_id);
"""

#: Every table derived from an image, and so cleared when that image changes.
DERIVED_TABLES = (
    "face_embeddings", "clip_embeddings", "animal_boxes", "animal_predictions", "food_boxes",
    "object_embeddings",
)

_ADDED_COLUMNS = {
    "food_by_detector": "INTEGER NOT NULL DEFAULT 0",
    "food_by_vlm": "INTEGER NOT NULL DEFAULT 0",
    "food_vlm_score": "REAL NOT NULL DEFAULT 0.0",
    "index_scale": "REAL NOT NULL DEFAULT 1.0",
    "mtime_ns": "INTEGER NOT NULL DEFAULT 0",
    # 1 once an image's object vectors were written; 0 for images indexed before,
    # which the search paths still serve by cropping at query time.
    "objects_embedded": "INTEGER NOT NULL DEFAULT 0",
}


def content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """Stable content hash, so edited files are detected as changed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def parse_box(text: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in text.split(","))
    return x1, y1, x2, y2


def format_box(box: tuple[int, int, int, int]) -> str:
    return ",".join(str(int(v)) for v in box)


@dataclass(frozen=True)
class GalleryRow:
    file_name: str
    file_path: str
    has_face: bool
    has_animal: bool
    has_food: bool
    index_scale: float = 1.0


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

    @contextmanager
    def image_transaction(self, conn: sqlite3.Connection) -> Iterator[None]:
        """Make everything written for one image land together or not at all.

        Without this, a failure after ``upsert_image`` left a row carrying the
        new content hash but no embeddings. The next run saw a matching hash and
        skipped the file, so the image stayed permanently unsearchable.
        """
        conn.execute("SAVEPOINT vie_image")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK TO vie_image")
            conn.execute("RELEASE vie_image")
            raise
        conn.execute("RELEASE vie_image")

    def initialise(self) -> None:
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            found = self.schema_version(conn)
            if found > SCHEMA_VERSION:
                raise SchemaError(
                    f"{self.path} has schema v{found} but this version of vie understands up "
                    f"to v{SCHEMA_VERSION}; upgrade vie rather than risk corrupting the index"
                )
            self._migrate(conn)
            conn.executescript(SCHEMA)
            self.set_meta(conn, "version", str(SCHEMA_VERSION))

    @staticmethod
    def schema_version(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Upgrade in place by looking at the columns that actually exist.

        Inspecting the tables rather than trusting the version number matters
        here: every index older than this one claimed ``version = 1`` whatever
        its real shape, because the value was overwritten on each start.
        """
        meta = self._columns(conn, "gallery_meta")
        if meta:
            for column, definition in _ADDED_COLUMNS.items():
                if column not in meta:
                    conn.execute(f"ALTER TABLE gallery_meta ADD COLUMN {column} {definition}")

        clip = self._columns(conn, "clip_embeddings")
        if clip and "region" not in clip:
            # The primary key has to change, and SQLite cannot alter a key.
            conn.execute("ALTER TABLE clip_embeddings RENAME TO clip_embeddings_old")
            conn.execute(
                """CREATE TABLE clip_embeddings (
                       file_name TEXT NOT NULL,
                       region    TEXT NOT NULL DEFAULT 'whole',
                       embedding BLOB NOT NULL,
                       PRIMARY KEY (file_name, region),
                       FOREIGN KEY (file_name) REFERENCES gallery_meta (file_name)
                           ON DELETE CASCADE
                   )"""
            )
            conn.execute(
                "INSERT INTO clip_embeddings (file_name, region, embedding) "
                "SELECT file_name, 'whole', embedding FROM clip_embeddings_old"
            )
            conn.execute("DROP TABLE clip_embeddings_old")

    # ---- metadata -----------------------------------------------------

    @staticmethod
    def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)", (key, value)
        )

    @staticmethod
    def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def bind_embedding_model(self, conn: sqlite3.Connection, model: str) -> None:
        """Record which model produced the vectors, refusing to mix two.

        Two models can share a dimension, in which case nothing would crash and
        every score would simply be meaningless.
        """
        found = self.get_meta(conn, "vision_language")
        has_vectors = conn.execute("SELECT 1 FROM clip_embeddings LIMIT 1").fetchone()
        if found and found != model and has_vectors:
            raise EmbeddingModelMismatch(
                f"the index was built with {found} but the config selects {model}; "
                f"run 'vie index --rebuild' to re-encode the gallery"
            )
        self.set_meta(conn, "vision_language", model)

    def check_embedding_model(self, conn: sqlite3.Connection, model: str) -> None:
        found = self.get_meta(conn, "vision_language")
        if found and found != model:
            raise EmbeddingModelMismatch(
                f"the index was built with {found} but the config selects {model}; "
                f"run 'vie index --rebuild', or point the config back at {found}"
            )

    def set_classifier_vocabulary(
        self, conn: sqlite3.Connection, name: str, class_names: list[str]
    ) -> None:
        self.set_meta(conn, "classifier", name)
        self.set_meta(conn, "classifier_vocabulary", json.dumps(class_names))

    def classifier_vocabulary(self, conn: sqlite3.Connection) -> list[str] | None:
        """Class names as they were at index time, so a query needs no torchvision."""
        raw = self.get_meta(conn, "classifier_vocabulary")
        return json.loads(raw) if raw else None

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
        food_by_detector: bool = False,
        food_by_vlm: bool = False,
        food_vlm_score: float = 0.0,
        index_scale: float = 1.0,
        mtime_ns: int = 0,
        objects_embedded: bool = False,
    ) -> None:
        conn.execute(
            """INSERT INTO gallery_meta
                 (file_name, file_path, content_hash, file_size, has_face, has_animal, has_food,
                  food_by_detector, food_by_vlm, food_vlm_score, index_scale, mtime_ns,
                  objects_embedded)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (file_name) DO UPDATE SET
                 file_path = excluded.file_path, content_hash = excluded.content_hash,
                 file_size = excluded.file_size, has_face = excluded.has_face,
                 has_animal = excluded.has_animal, has_food = excluded.has_food,
                 food_by_detector = excluded.food_by_detector,
                 food_by_vlm = excluded.food_by_vlm,
                 food_vlm_score = excluded.food_vlm_score,
                 index_scale = excluded.index_scale, mtime_ns = excluded.mtime_ns,
                 objects_embedded = excluded.objects_embedded""",
            (file_name, file_path, hash_, size, int(has_face), int(has_animal), int(has_food),
             int(food_by_detector), int(food_by_vlm), float(food_vlm_score),
             float(index_scale), int(mtime_ns), int(objects_embedded)),
        )

    def add_face(
        self, conn: sqlite3.Connection, file_name: str, box: tuple[int, int, int, int],
        embedding: np.ndarray,
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO face_embeddings (file_name, bbox, embedding) VALUES (?, ?, ?)",
            (file_name, format_box(box), to_blob(embedding)),
        )

    def add_clip(
        self, conn: sqlite3.Connection, file_name: str, embedding: np.ndarray,
        region: str = "whole",
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO clip_embeddings (file_name, region, embedding) "
            "VALUES (?, ?, ?)",
            (file_name, region, to_blob(embedding)),
        )

    def add_animal_box(
        self, conn: sqlite3.Connection, file_name: str, box: tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO animal_boxes (file_name, bbox, confidence) VALUES (?, ?, ?)",
            (file_name, format_box(box), float(confidence)),
        )

    def add_animal_prediction(
        self, conn: sqlite3.Connection, file_name: str, box: tuple[int, int, int, int],
        class_id: int, probability: float,
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO animal_predictions "
            "(file_name, bbox, class_id, probability) VALUES (?, ?, ?, ?)",
            (file_name, format_box(box), int(class_id), float(probability)),
        )

    def add_food_box(
        self, conn: sqlite3.Connection, file_name: str, box: tuple[int, int, int, int],
        confidence: float, class_id: int | None = None,
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO food_boxes (file_name, bbox, confidence, class_id) "
            "VALUES (?, ?, ?, ?)",
            (file_name, format_box(box), float(confidence),
             None if class_id is None else int(class_id)),
        )

    def clear_derived(self, conn: sqlite3.Connection, file_name: str) -> None:
        """Drop previously derived rows for one image before re-indexing it."""
        for table in DERIVED_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE file_name = ?", (file_name,))

    def clear_all(self, conn: sqlite3.Connection) -> None:
        """Forget every image, for a rebuild against a different model."""
        for table in DERIVED_TABLES:
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM gallery_meta")
        conn.execute("DELETE FROM schema_meta WHERE key != 'version'")

    # ---- read ---------------------------------------------------------

    def needs_indexing(self, conn: sqlite3.Connection, file_name: str, hash_: str) -> bool:
        row = conn.execute(
            "SELECT content_hash FROM gallery_meta WHERE file_name = ?", (file_name,)
        ).fetchone()
        return row is None or row[0] != hash_

    def known_files(self, conn: sqlite3.Connection) -> dict[str, tuple[str, int, int]]:
        """``file_name -> (hash, size, mtime_ns)`` for the incremental scan."""
        return {
            r[0]: (r[1], r[2], r[3])
            for r in conn.execute(
                "SELECT file_name, content_hash, file_size, mtime_ns FROM gallery_meta"
            )
        }

    def add_object(
        self, conn: sqlite3.Connection, file_name: str, kind: str, box: tuple,
        crop: tuple, confidence: float, class_id: int | None, embedding: np.ndarray,
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO object_embeddings "
            "(file_name, kind, bbox, crop, confidence, class_id, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_name, kind, format_box(box), format_box(crop), float(confidence),
             class_id, to_blob(embedding)),
        )

    def object_vectors(
        self, conn: sqlite3.Connection, kind: str
    ) -> dict[str, list[tuple[tuple, tuple, float, int | None, np.ndarray]]]:
        """``file_name -> [(box, crop, confidence, class_id, vector)]`` for one kind."""
        found: dict[str, list[tuple[tuple, tuple, float, int | None, np.ndarray]]] = {}
        for name, bbox, crop, confidence, class_id, blob in conn.execute(
            "SELECT file_name, bbox, crop, confidence, class_id, embedding "
            "FROM object_embeddings WHERE kind = ?", (kind,)
        ):
            found.setdefault(name, []).append(
                (parse_box(bbox), parse_box(crop), confidence, class_id, from_blob(blob))
            )
        return found

    def embedded_images(self, conn: sqlite3.Connection) -> set[str]:
        """Images whose object vectors are stored, so search needs no pixels."""
        return {r[0] for r in conn.execute(
            "SELECT file_name FROM gallery_meta WHERE objects_embedded = 1")}

    def whole_vectors(self, conn: sqlite3.Connection, names: set[str]) -> dict[str, np.ndarray]:
        """The stored whole-frame vector of each named image."""
        if not names:
            return {}
        return {
            name: from_blob(blob)
            for name, blob in conn.execute(
                "SELECT file_name, embedding FROM clip_embeddings WHERE region = 'whole'")
            if name in names
        }

    def refresh_files(
        self, conn: sqlite3.Connection, rows: list[tuple[str, str, int, int]]
    ) -> None:
        """Record where unchanged files now are, and their current size and mtime.

        ``rows`` is ``(file_name, file_path, size, mtime_ns)`` for files whose content
        hash matched. Without this, a file whose mtime moved (a copy, a sync, an
        extracted archive) is re-hashed on every run, and a gallery that moved keeps
        paths to where it used to be.
        """
        conn.executemany(
            "UPDATE gallery_meta SET file_path = ?, file_size = ?, mtime_ns = ? "
            "WHERE file_name = ?",
            [(path, size, mtime, name) for name, path, size, mtime in rows],
        )

    def candidates(self, conn: sqlite3.Connection, flag: str) -> list[GalleryRow]:
        if flag not in {"has_face", "has_animal", "has_food"}:
            raise ValueError(f"unknown flag {flag!r}")
        rows = conn.execute(
            f"""SELECT file_name, file_path, has_face, has_animal, has_food, index_scale
                FROM gallery_meta WHERE {flag} = 1"""
        ).fetchall()
        return [GalleryRow(r[0], r[1], bool(r[2]), bool(r[3]), bool(r[4]), r[5]) for r in rows]

    def scales(self, conn: sqlite3.Connection) -> dict[str, float]:
        return dict(conn.execute("SELECT file_name, index_scale FROM gallery_meta"))

    def iter_faces(self, conn: sqlite3.Connection):
        query = """SELECT f.file_name, f.bbox, f.embedding, g.file_path
                   FROM face_embeddings f JOIN gallery_meta g USING (file_name)"""
        for file_name, bbox, blob, file_path in conn.execute(query):
            yield file_name, parse_box(bbox), from_blob(blob), file_path

    def iter_clip(self, conn: sqlite3.Connection):
        """Yield every stored region vector.

        Callers collapse regions per image by taking the best-scoring one, so
        an image with a small subject can still win on its crop.
        """
        query = """SELECT c.file_name, c.embedding, g.file_path, c.region
                   FROM clip_embeddings c JOIN gallery_meta g USING (file_name)"""
        for file_name, blob, file_path, region in conn.execute(query):
            yield file_name, from_blob(blob), file_path, region

    def animal_boxes(self, conn: sqlite3.Connection) -> dict[str, list[tuple[tuple, float]]]:
        found: dict[str, list[tuple[tuple, float]]] = {}
        for name, bbox, confidence in conn.execute(
            "SELECT file_name, bbox, confidence FROM animal_boxes"
        ):
            found.setdefault(name, []).append((parse_box(bbox), confidence))
        return found

    def food_boxes(
        self, conn: sqlite3.Connection, classes: list[int] | None = None
    ) -> dict[str, list[tuple[tuple, float]]]:
        """Stored food boxes, restricted to the search classes.

        A ``NULL`` class means a detector that does not report one; it is kept
        rather than filtered, because dropping it would lose recall silently.
        """
        wanted = None if classes is None else set(classes)
        found: dict[str, list[tuple[tuple, float]]] = {}
        for name, bbox, confidence, class_id in conn.execute(
            "SELECT file_name, bbox, confidence, class_id FROM food_boxes"
        ):
            if wanted is not None and class_id is not None and class_id not in wanted:
                continue
            found.setdefault(name, []).append((parse_box(bbox), confidence))
        return found

    def animal_predictions(self, conn: sqlite3.Connection, class_ids: tuple[int, ...]):
        """Yield ``(file_name, file_path, box, class_id, probability)`` for these classes."""
        if not class_ids:
            return
        marks = ",".join("?" for _ in class_ids)
        query = f"""SELECT p.file_name, g.file_path, p.bbox, p.class_id, p.probability
                    FROM animal_predictions p JOIN gallery_meta g USING (file_name)
                    WHERE p.class_id IN ({marks})"""
        for file_name, file_path, bbox, class_id, probability in conn.execute(query, class_ids):
            yield file_name, file_path, parse_box(bbox), class_id, probability

    def prune_missing(self, conn: sqlite3.Connection, present: set[str]) -> int:
        """Remove rows for files that no longer exist on disk."""
        known = {r[0] for r in conn.execute("SELECT file_name FROM gallery_meta")}
        gone = known - present
        for file_name in gone:
            self.clear_derived(conn, file_name)
            conn.execute("DELETE FROM gallery_meta WHERE file_name = ?", (file_name,))
        return len(gone)
