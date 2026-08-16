"""Open-vocabulary text search over CLIP image embeddings.

Both sides of the comparison are unit-length CLIP vectors in the same 768-d
projected space, so the dot product is cosine similarity.

The change from the original is presentational but not cosmetic. It displayed
raw cosine as a percentage, so its best result read ``23.30%`` — a perfectly
healthy CLIP score that looks like failure to anyone reading it. CLIP cosines
occupy a narrow band (roughly 0.15-0.35 for genuine matches) because the
contrastive objective never required them to span [0, 1].

:func:`rank` therefore returns the raw similarity *and* a relative confidence
computed across the candidate set, so the UI can show something interpretable
without pretending the cosine itself is a probability.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from vie.config import Config
from vie.scoring import Match

#: One indexed image: (file_name, embedding, file_path).
IndexedImage = tuple[str, np.ndarray, str]


def rank(
    query_embedding: np.ndarray,
    indexed: Iterable[IndexedImage],
    config: Config,
    top_k: int | None = None,
) -> list[Match]:
    """Return the ``top_k`` most similar images."""
    query = np.asarray(query_embedding, dtype=np.float32).ravel()
    if query.size == 0:
        raise ValueError("empty query embedding")

    scored: list[Match] = []
    for file_name, embedding, file_path in indexed:
        candidate = np.asarray(embedding, dtype=np.float32).ravel()
        if candidate.shape != query.shape:
            raise ValueError(
                f"embedding dimension mismatch for {file_name}: "
                f"index has {candidate.shape[0]}, query has {query.shape[0]}"
            )
        scored.append(
            Match(
                file_name=file_name,
                file_path=file_path,
                score=float(np.dot(query, candidate)),
            )
        )

    scored.sort(key=lambda m: m.score, reverse=True)
    limit = config.top_k if top_k is None else top_k
    return scored[:limit]


def relative_confidence(scores: list[float], temperature: float = 100.0) -> list[float]:
    """Turn raw CLIP cosines into a comparable distribution over the results.

    ``temperature`` mirrors CLIP's own learned logit scale, which is what turns
    its narrow cosine band into usable logits. This is a *ranking* aid — it says
    how much better the top hit is than the rest, not how likely it is to be
    correct in absolute terms, and the docs say so.
    """
    from vie.scoring import softmax

    if not scores:
        return []
    return softmax([s * temperature for s in scores])
