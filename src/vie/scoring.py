"""Contrastive scoring and result aggregation.

The single best idea in the original project was in the food module: instead of
thresholding a raw cosine similarity, it scored the user's query against two
rival prompts and took a softmax. That gives the model an explicit "none of the
above" option, so the number that comes out is a calibrated posterior rather
than an uncalibrated distance.

The animal module did not do this — it used an ImageNet classifier with no
confidence floor at all, and shipped 21.5% top-1 results as confirmed matches.
This module makes the good pattern the shared default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def softmax(values: list[float]) -> list[float]:
    """Numerically stable softmax over a small list."""
    if not values:
        return []
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def contrastive_score(logits: list[float]) -> float:
    """Posterior probability of the hypothesis against its rivals.

    ``logits[0]`` is the query prompt; the rest are the negative prompts. The
    result is directly comparable to a threshold in [0, 1], unlike a raw
    cosine similarity, which for CLIP lives in a narrow band around 0.2-0.35
    even for a perfect match — the reason the original text search displayed
    "23.30%" for its best hit and looked like a failure to users.
    """
    if not logits:
        raise ValueError("logits must not be empty")
    if len(logits) == 1:
        raise ValueError(
            "contrastive scoring needs at least one rival prompt; with a single "
            "candidate the softmax is always 1.0"
        )
    return softmax(logits)[0]


@dataclass(frozen=True)
class Match:
    """One accepted result."""

    file_name: str
    file_path: str
    score: float
    box: tuple[int, int, int, int] | None = None
    label: str | None = None


def best_per_image(matches: list[Match]) -> list[Match]:
    """Collapse to one result per image, keeping the strongest.

    The standalone face notebook stopped at the first matching face in an
    image; the unified pipeline dropped that, so an image containing two
    matching faces was appended twice and appeared twice in the ranked output.

    Keeping the *best* rather than the *first* also fixes a subtler issue: the
    original early-exit recorded whichever instance the detector happened to
    return first, then used that score as the global ranking key — so an image
    whose second animal scored 95% could be ranked by its first at 26%.
    """
    best: dict[str, Match] = {}
    for match in matches:
        current = best.get(match.file_name)
        if current is None or match.score > current.score:
            best[match.file_name] = match
    return sorted(best.values(), key=lambda m: m.score, reverse=True)
