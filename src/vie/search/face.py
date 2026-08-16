"""Biometric face search.

ArcFace embeddings from InsightFace ``buffalo_l`` are already L2-normalised
(``normed_embedding``), so a dot product *is* cosine similarity — which is why
the original could use ``np.dot`` directly and be correct. That part was right
and is kept.

What changed: the unified pipeline had lost the standalone module's per-image
early exit, so an image containing two matching faces was appended twice and
appeared twice in the ranked output. Results are now collapsed per image, and
the *strongest* face wins rather than whichever the scan reached first.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from vie.config import Config
from vie.geometry import Box
from vie.scoring import Match, best_per_image

#: One indexed face: (file_name, box, embedding, file_path).
IndexedFace = tuple[str, Box, np.ndarray, str]


def rank(
    query_embedding: np.ndarray,
    indexed: Iterable[IndexedFace],
    config: Config,
) -> list[Match]:
    """Rank gallery images by similarity to a query face.

    Complexity is O(N) in indexed faces. At archive sizes measured so far a full
    scan answers in milliseconds; see ``docs/design-decisions.md`` for why an
    approximate index is deliberately not used yet.
    """
    query = np.asarray(query_embedding, dtype=np.float32).ravel()
    if query.size == 0:
        raise ValueError("empty query embedding")

    matches: list[Match] = []
    for file_name, box, embedding, file_path in indexed:
        candidate = np.asarray(embedding, dtype=np.float32).ravel()
        if candidate.shape != query.shape:
            # A real mismatch means the index was built with a different model.
            # The original skipped these silently; surfacing it is the point.
            raise ValueError(
                f"embedding dimension mismatch for {file_name}: "
                f"index has {candidate.shape[0]}, query has {query.shape[0]}"
            )
        similarity = float(np.dot(query, candidate))
        if similarity >= config.face.match_threshold:
            matches.append(
                Match(
                    file_name=file_name,
                    file_path=file_path,
                    score=similarity,
                    box=box,
                )
            )
    return best_per_image(matches)


def select_query_face(faces: list[tuple[Box, np.ndarray]]) -> tuple[Box, np.ndarray] | None:
    """Pick the subject of a query photo: the largest detected face.

    A query image is a portrait of one person; bystanders are smaller. This
    heuristic came from the original and is kept because it is correct for the
    use case.
    """
    if not faces:
        return None
    return max(faces, key=lambda item: _area(item[0]))


def _area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def detection_size(width: int, height: int, config: Config) -> tuple[int, int]:
    """Choose the detector input size for a query image.

    Small crops are detected at 320 and larger photos at 640. The original also
    retried at 640 when the first pass found nothing, which is preserved by the
    caller in :mod:`vie.cli`.
    """
    size = (
        config.face.det_size_large
        if max(width, height) > config.face.small_image_px
        else config.face.det_size_small
    )
    return (size, size)
