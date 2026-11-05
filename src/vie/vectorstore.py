"""Vector search over a contiguous matrix, with an optional HNSW index (P6-1).

The search paths used to walk SQLite rows and compute one ``np.dot`` per row in
Python, while the benchmark measured a single matrix multiply. The number the
FAISS decision rested on therefore described code that did not run. Everything
now goes through :class:`VectorCatalog`, so benchmark and product share a path.

Why exact remains the default
-----------------------------
One BLAS call returns the true answer, and measured at 100k vectors it takes 10 ms.
HNSW took 1 ms at recall@10 0.81 on the same data, so it is reserved for sizes where
exact search really hurts: ``search.backend: auto`` switches past ``ann_threshold``
when FAISS is installed, and logs when it is not. ``scripts/benchmark.py`` reports
latency and recall side by side so the threshold can be checked on real hardware.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vie.embeddings import from_blob
from vie.logging_setup import get_logger

log = get_logger("vectorstore")


@dataclass(frozen=True)
class SearchConfig:
    """How vectors are searched. Read from the ``search`` config section."""

    backend: str = "auto"              # auto | exact | hnsw
    ann_threshold: int = 500_000
    ann_candidates: int = 2_000
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 256
    crop_batch: int = 32


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, best first, without a full sort."""
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    if k >= scores.size:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


class ExactBackend:
    """Brute-force inner product. Always correct; linear in the vector count."""

    name = "exact"

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.matrix @ query
        order = top_k_indices(scores, k)
        return order, scores[order]


class HnswBackend:
    """FAISS HNSW over inner product, which is cosine on unit vectors."""

    name = "hnsw"

    def __init__(self, index: Any, ef_search: int) -> None:
        self.index = index
        self.index.hnsw.efSearch = ef_search

    @classmethod
    def build(cls, matrix: np.ndarray, settings: SearchConfig) -> HnswBackend:
        import faiss

        index = faiss.IndexHNSWFlat(matrix.shape[1], settings.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = settings.hnsw_ef_construction
        index.add(matrix)
        return cls(index, settings.hnsw_ef_search)

    @classmethod
    def load(cls, path: Path, settings: SearchConfig) -> HnswBackend:
        import faiss

        return cls(faiss.read_index(str(path)), settings.hnsw_ef_search)

    def save(self, path: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(path))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query.reshape(1, -1), k)
        keep = ids[0] >= 0
        return ids[0][keep].astype(np.int64), scores[0][keep]


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def choose_backend(count: int, settings: SearchConfig) -> str:
    """Resolve the configured backend for a table of ``count`` vectors."""
    if settings.backend not in {"auto", "exact", "hnsw"}:
        raise ValueError(f"unknown search backend {settings.backend!r}")
    if settings.backend == "exact":
        return "exact"
    if settings.backend == "hnsw":
        if not faiss_available():
            raise RuntimeError(
                "search.backend is 'hnsw' but faiss is not installed; "
                "run: pip install 'visual-intelligence-engine[ann]'"
            )
        return "hnsw"
    if count < settings.ann_threshold:
        return "exact"
    if faiss_available():
        return "hnsw"
    log.warning(
        "%d vectors is past the ANN threshold but faiss is not installed; "
        "staying on exact search (pip install 'visual-intelligence-engine[ann]')",
        count,
    )
    return "exact"


def _stack(blobs: Sequence[bytes], table: str) -> np.ndarray:
    vectors = [from_blob(b) for b in blobs]
    dims = {v.size for v in vectors}
    if len(dims) > 1:
        raise ValueError(
            f"embedding dimension mismatch in {table}: found {sorted(dims)}; "
            f"the index mixes two models, so run 'vie index --rebuild'"
        )
    return np.vstack(vectors).astype(np.float32)


#: The column that distinguishes rows of one image. Fixed strings, never user input.
_TABLES = {"clip_embeddings": "region", "face_embeddings": "bbox"}


class VectorCatalog:
    """One searchable table: row keys plus whichever backend suits its size."""

    def __init__(
        self, keys: list[tuple[str, str]], backend: ExactBackend | HnswBackend, dim: int
    ) -> None:
        self.keys = keys
        self.backend = backend
        self.dim = dim

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def open(
        cls,
        conn: sqlite3.Connection,
        table: str,
        settings: SearchConfig,
        cache_dir: Path | None = None,
    ) -> VectorCatalog:
        """Load a table, reusing a saved HNSW index when it is still current."""
        if table not in _TABLES:
            raise ValueError(f"unknown vector table {table!r}")

        detail = _TABLES[table]
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        kind = choose_backend(count, settings)
        fingerprint = _fingerprint(conn, table)

        keys: list[tuple[str, str]]
        if kind == "hnsw" and cache_dir is not None:
            saved = _load_saved(cache_dir, table, fingerprint, settings)
            if saved is not None:
                keys = [
                    (r[0], r[1])
                    for r in conn.execute(
                        f"SELECT file_name, {detail} FROM {table} ORDER BY rowid"
                    )
                ]
                return cls(keys, saved, saved.index.d)

        keys = []
        blobs: list[bytes] = []
        for file_name, value, blob in conn.execute(
            f"SELECT file_name, {detail}, embedding FROM {table} ORDER BY rowid"
        ):
            keys.append((file_name, value))
            blobs.append(blob)
        if not blobs:
            return cls([], ExactBackend(np.empty((0, 0), dtype=np.float32)), 0)

        matrix = _stack(blobs, table)
        if kind == "hnsw":
            log.info("building HNSW index over %d vectors", len(keys))
            backend: ExactBackend | HnswBackend = HnswBackend.build(matrix, settings)
            if cache_dir is not None:
                _save(cache_dir, table, fingerprint, backend)
        else:
            backend = ExactBackend(matrix)
        return cls(keys, backend, matrix.shape[1])

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, str, float]]:
        """Return ``(file_name, detail, score)``, best first."""
        vector = np.asarray(query, dtype=np.float32).ravel()
        if vector.size == 0:
            raise ValueError("empty query embedding")
        if not self.keys:
            return []
        if vector.size != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: index has {self.dim}, query has {vector.size}"
            )
        ids, scores = self.backend.search(vector, k)
        return [
            (*self.keys[int(i)], float(s)) for i, s in zip(ids, scores, strict=True)
        ]


def _fingerprint(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    count, top = conn.execute(f"SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM {table}").fetchone()
    return {"count": count, "max_rowid": top}


def _load_saved(
    cache_dir: Path, table: str, fingerprint: dict[str, int], settings: SearchConfig
) -> HnswBackend | None:
    index_path = cache_dir / f"{table}.faiss"
    stamp_path = cache_dir / f"{table}.faiss.json"
    if not (index_path.is_file() and stamp_path.is_file()):
        return None
    if json.loads(stamp_path.read_text(encoding="utf-8")) != fingerprint:
        return None
    return HnswBackend.load(index_path, settings)


def _save(
    cache_dir: Path, table: str, fingerprint: dict[str, int], backend: HnswBackend
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    backend.save(cache_dir / f"{table}.faiss")
    (cache_dir / f"{table}.faiss.json").write_text(json.dumps(fingerprint), encoding="utf-8")
