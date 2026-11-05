"""Persisting video analyses, and searching across them.

Videos live in the same SQLite file as images, so a backup or a deletion covers
everything, and the same rule applies: the file holds biometric identifiers and
must be protected the way the footage itself is.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vie.config import Config
from vie.database import Index, format_box
from vie.embeddings import from_blob, to_blob
from vie.video.engine import VideoAnalysis
from vie.video.people import Interval, merge_times

VIDEO_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_key    TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_size    INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL DEFAULT 0,
    duration     REAL NOT NULL,
    analysis     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_keyframes (
    video_key TEXT NOT NULL,
    time      REAL NOT NULL,
    shot      INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (video_key, time),
    FOREIGN KEY (video_key) REFERENCES videos (video_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_faces (
    video_key TEXT NOT NULL,
    time      REAL NOT NULL,
    bbox      TEXT NOT NULL,
    person    INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (video_key, time, bbox),
    FOREIGN KEY (video_key) REFERENCES videos (video_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_transcript (
    video_key TEXT NOT NULL,
    start     REAL NOT NULL,
    end       REAL NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (video_key, start),
    FOREIGN KEY (video_key) REFERENCES videos (video_key) ON DELETE CASCADE
);
"""


@dataclass(frozen=True)
class Moment:
    """One place in one video where a query matched."""

    video_key: str
    file_path: str
    start: float
    end: float
    score: float
    kind: str = "visual"  # visual | speech | face
    text: str | None = None


def analysis_to_dict(analysis: VideoAnalysis) -> dict:
    """The JSON form, used by ``--json`` and stored in the index. No pixels, no vectors."""
    return {
        "info": asdict(analysis.info),
        "summary": analysis.summary,
        "sample_seconds": analysis.sample_seconds,
        "stages": analysis.stages,
        "problems": analysis.problems,
        "shots": [
            {
                "index": s.index,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "labels": {
                    group: {"label": label, "score": round(score, 4)}
                    for group, (label, score) in analysis.shot_labels[s.index].labels.items()
                },
                "caption": analysis.shot_labels[s.index].caption,
            }
            for s in analysis.shots
        ],
        "labels": [
            {
                "group": item.group,
                "label": item.label,
                "score": round(item.score, 4),
                "coverage": round(item.coverage, 4),
                "peak": round(item.peak, 4),
                "intervals": [[round(i.start, 2), round(i.end, 2)] for i in item.intervals],
            }
            for item in analysis.labels
        ],
        "people": [
            {
                "id": p.id,
                "sightings": len(p.sightings),
                "screen_time": round(p.screen_time, 2),
                "intervals": [[round(i.start, 2), round(i.end, 2)] for i in p.intervals],
            }
            for p in analysis.people
        ],
        "animals": {
            "species": analysis.animal_species,
            "times": [round(t, 2) for t in analysis.animal_times],
        },
        "food": {"times": [round(t, 2) for t in analysis.food_times]},
        "transcript": [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
            for s in analysis.transcript
        ],
        "language": analysis.language,
    }


class VideoStore:
    """Video tables inside the main index."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.index = Index(config.database_path)

    def initialise(self) -> None:
        self.index.initialise()
        with self.index.connect() as conn:
            conn.executescript(VIDEO_SCHEMA)

    def connect(self):
        return self.index.connect()

    def known(self, conn: sqlite3.Connection) -> dict[str, tuple[str, int, int]]:
        return {
            r[0]: (r[1], r[2], r[3])
            for r in conn.execute(
                "SELECT video_key, content_hash, file_size, mtime_ns FROM videos"
            )
        }

    def delete(self, conn: sqlite3.Connection, key: str) -> None:
        conn.execute("DELETE FROM videos WHERE video_key = ?", (key,))

    def prune(self, conn: sqlite3.Connection, present: set[str]) -> int:
        gone = {r[0] for r in conn.execute("SELECT video_key FROM videos")} - present
        for key in gone:
            self.delete(conn, key)
        return len(gone)

    def save(
        self, conn: sqlite3.Connection, key: str, path: Path, digest: str, analysis: VideoAnalysis
    ) -> None:
        stat = path.stat()
        with self.index.image_transaction(conn):
            self.delete(conn, key)
            conn.execute(
                "INSERT INTO videos (video_key, file_path, content_hash, file_size, mtime_ns, "
                "duration, analysis) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, str(path), digest, stat.st_size, stat.st_mtime_ns,
                 analysis.shots[-1].end if analysis.shots else analysis.info.duration,
                 json.dumps(analysis_to_dict(analysis), ensure_ascii=False)),
            )
            for frame in analysis.keyframes:
                conn.execute(
                    "INSERT INTO video_keyframes (video_key, time, shot, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (key, frame.time, frame.shot, to_blob(frame.vector)),
                )
            for person in analysis.people:
                for sighting in person.sightings:
                    conn.execute(
                        "INSERT OR REPLACE INTO video_faces "
                        "(video_key, time, bbox, person, embedding) VALUES (?, ?, ?, ?, ?)",
                        (key, sighting.time, format_box(sighting.box), person.id,
                         to_blob(sighting.embedding)),
                    )
            for segment in analysis.transcript:
                conn.execute(
                    "INSERT OR REPLACE INTO video_transcript (video_key, start, end, text) "
                    "VALUES (?, ?, ?, ?)",
                    (key, segment.start, segment.end, segment.text),
                )

    def analysis(self, conn: sqlite3.Connection, key: str) -> dict | None:
        row = conn.execute("SELECT analysis FROM videos WHERE video_key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def iter_videos(self, conn: sqlite3.Connection) -> Iterator[tuple[str, str, float, dict]]:
        for key, path, duration, analysis in conn.execute(
            "SELECT video_key, file_path, duration, analysis FROM videos ORDER BY video_key"
        ):
            yield key, path, duration, json.loads(analysis)

    # ---- search -------------------------------------------------------

    def _file_paths(self, conn: sqlite3.Connection) -> dict[str, str]:
        return dict(conn.execute("SELECT video_key, file_path FROM videos"))

    def search_visual(
        self,
        conn: sqlite3.Connection,
        query: np.ndarray,
        *,
        relative_floor: float = 0.85,
        top_k: int = 10,
    ) -> list[Moment]:
        """Find the moments that look like the query, across every video.

        Returns moments rather than frames: neighbouring keyframes that all match are
        one scene, and a person wants to seek to it once. A frame joins a moment when
        it scores within ``relative_floor`` of that video's best, since a raw cosine
        has no absolute meaning that holds from one query to the next.
        """
        rows = conn.execute(
            "SELECT video_key, time, embedding FROM video_keyframes ORDER BY video_key, time"
        ).fetchall()
        if not rows:
            return []
        paths = self._file_paths(conn)
        matrix = np.vstack([from_blob(r[2]) for r in rows])
        scores = matrix @ np.asarray(query, dtype=np.float32).ravel()

        per_video: dict[str, list[tuple[float, float]]] = {}
        for (key, time, _), score in zip(rows, scores.tolist(), strict=True):
            per_video.setdefault(key, []).append((time, score))

        gap = self.config.video.presence_gap
        moments: list[Moment] = []
        for key, entries in per_video.items():
            best = max(score for _, score in entries)
            floor = best * relative_floor if best > 0 else best
            selected = [(t, s) for t, s in entries if s >= floor]
            for span in merge_times([t for t, _ in selected], gap):
                inside = [s for t, s in selected if span.start <= t <= span.end]
                moments.append(Moment(key, paths.get(key, key), span.start, span.end, max(inside)))
        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:top_k]

    def search_speech(
        self, conn: sqlite3.Connection, query: str, top_k: int = 10
    ) -> list[Moment]:
        """Plain substring search over transcripts, in whatever language was spoken."""
        words = [w for w in query.split() if w]
        if not words:
            return []
        clause = " AND ".join("text LIKE ? ESCAPE '!'" for _ in words)
        values = [
            "%" + w.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
            for w in words
        ]
        rows = conn.execute(
            f"SELECT t.video_key, v.file_path, t.start, t.end, t.text FROM video_transcript t "
            f"JOIN videos v USING (video_key) WHERE {clause} ORDER BY t.video_key, t.start "
            f"LIMIT ?",
            (*values, top_k),
        ).fetchall()
        return [Moment(r[0], r[1], r[2], r[3], 1.0, "speech", r[4]) for r in rows]

    def search_face(
        self, conn: sqlite3.Connection, query: np.ndarray, top_k: int = 10
    ) -> list[Moment]:
        """Where a given face appears, as intervals, across every video."""
        rows = conn.execute(
            "SELECT f.video_key, v.file_path, f.time, f.embedding FROM video_faces f "
            "JOIN videos v USING (video_key)"
        ).fetchall()
        if not rows:
            return []
        matrix = np.vstack([from_blob(r[3]) for r in rows])
        scores = matrix @ np.asarray(query, dtype=np.float32).ravel()
        threshold = self.config.face.match_threshold

        hits: dict[str, list[tuple[float, float]]] = {}
        paths: dict[str, str] = {}
        for (key, path, time, _), score in zip(rows, scores.tolist(), strict=True):
            if score >= threshold:
                hits.setdefault(key, []).append((time, score))
                paths[key] = path

        moments: list[Moment] = []
        for key, entries in hits.items():
            for span in _spans(entries, self.config.video.presence_gap):
                inside = [s for t, s in entries if span.start <= t <= span.end]
                moments.append(Moment(key, paths[key], span.start, span.end, max(inside), "face"))
        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:top_k]


def _spans(entries: list[tuple[float, float]], gap: float) -> list[Interval]:
    return merge_times([t for t, _ in entries], gap)
