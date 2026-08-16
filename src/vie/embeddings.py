"""Embedding serialisation.

The original pipeline stored embeddings with ``pickle.dumps``. That is unsafe
and wasteful:

* ``pickle.loads`` on a database you did not create is arbitrary code
  execution. An index file is exactly the kind of artefact people copy between
  machines and download from colleagues.
* the face path pickled ``embedding.tolist()`` — a Python list of floats, not
  an array — which is several times larger than the raw bytes.

Raw little-endian float32 fixes both. It is also portable: any language with a
``frombuffer`` equivalent can read the index.
"""

from __future__ import annotations

import numpy as np

DTYPE = np.dtype(np.float32).newbyteorder("<")


def to_blob(vector: np.ndarray) -> bytes:
    """Serialise a 1-D embedding to little-endian float32 bytes."""
    array = np.asarray(vector, dtype=np.float32).ravel()
    if array.size == 0:
        raise ValueError("refusing to store an empty embedding")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding contains NaN or infinity")
    return array.astype(DTYPE, copy=False).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Deserialise bytes written by :func:`to_blob`."""
    if not blob:
        raise ValueError("empty embedding blob")
    if len(blob) % DTYPE.itemsize:
        raise ValueError(
            f"blob length {len(blob)} is not a multiple of {DTYPE.itemsize}; "
            f"the index may have been written by an older, pickle-based version"
        )
    return np.frombuffer(blob, dtype=DTYPE).astype(np.float32, copy=True)


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length so a dot product equals cosine similarity.

    InsightFace's ``normed_embedding`` is already unit length and CLIP features
    are normalised explicitly before storage, but going through one helper means
    the guarantee holds for anything added later.
    """
    array = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return array / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, safe against non-normalised input."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"dimension mismatch: {a.shape} vs {b.shape}")
    denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)
