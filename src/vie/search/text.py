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

#: One indexed region: (file_name, embedding, file_path, region).
#: Several rows share a file_name when multi-crop indexing is on.
IndexedRegion = tuple[str, np.ndarray, str, str]


def rank(
    query_embedding: np.ndarray,
    indexed: Iterable[IndexedRegion],
    config: Config,
    top_k: int | None = None,
) -> list[Match]:
    """Return the ``top_k`` most similar images.

    When several regions of one image are indexed, the image scores as its
    **best-matching region** rather than its average. That is the whole point
    of multi-crop: a backpack occupying 3% of a wide frame barely moves the
    whole-image vector, but dominates the quadrant it sits in.

    The winning region's name is carried through on the match, so the UI can
    say *where* it matched.
    """
    query = np.asarray(query_embedding, dtype=np.float32).ravel()
    if query.size == 0:
        raise ValueError("empty query embedding")

    best: dict[str, Match] = {}
    for file_name, embedding, file_path, region in indexed:
        candidate = np.asarray(embedding, dtype=np.float32).ravel()
        if candidate.shape != query.shape:
            raise ValueError(
                f"embedding dimension mismatch for {file_name}: "
                f"index has {candidate.shape[0]}, query has {query.shape[0]}"
            )
        score = float(np.dot(query, candidate))
        current = best.get(file_name)
        if current is None or score > current.score:
            best[file_name] = Match(
                file_name=file_name,
                file_path=file_path,
                score=score,
                label=region,
            )

    ranked = sorted(best.values(), key=lambda m: m.score, reverse=True)
    limit = config.top_k if top_k is None else top_k
    return ranked[:limit]


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
