"""Animal search — a hybrid of a supervised classifier and CLIP.

Why hybrid
----------
Two models answer "which animal is this", and they are good at different things:

===================  ==========  ===========  ==========  ==================
model                params      GFLOPs/crop  ImageNet    vocabulary
===================  ==========  ===========  ==========  ==================
MobileNetV3-Large    5.5 M       0.217        74.0 %      398 animal classes
CLIP ViT-L/14        428 M       ~81          ~75.5 %     open
===================  ==========  ===========  ==========  ==================

CLIP costs roughly **373x the compute per crop** for about 1.5 points of
top-1 — a bad trade when the species is one the classifier already knows.
But it is the *only* option when it is not: ``quokka`` is not an ImageNet
class, so no supervised ImageNet model can ever return it.

So the classifier is the primary path and CLIP is the fallback:

.. code-block:: text

    query -> resolve against ImageNet vocabulary
               in vocabulary  -> precomputed predictions   (pure SQL, no inference)
               out of it      -> CLIP over candidate crops (expensive, rare)

The classifier also moves to **index time**. Each animal crop is classified
once when the image is ingested and its top-k predictions are stored, so an
in-vocabulary query runs no model at all — the original re-ran both MegaDetector
*and* the classifier on every candidate, for every query.

What the original got wrong, and what it got right
--------------------------------------------------
The defect was never MobileNetV3. It reported ``African elephant`` correctly.
The damage was done by a layer that collapsed 1000 class names into 8 buckets
with unanchored substring tests, so ``eleph``**ant** became *Insect*. That layer
is gone; queries now match the real class names with word boundaries
(:mod:`vie.species`).

MegaDetector stays as the localiser. A class-agnostic detector gives a
species-independent recall gate, and cropping before recognition is what lets an
animal occupying 4 % of a press photo survive the downscale to 224 px.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vie.config import Config
from vie.geometry import Box, is_large_enough, pad_box
from vie.scoring import Match, best_per_image, contrastive_score
from vie.species import Resolution, resolve


class CropScorer(Protocol):
    """Scores image crops against text prompts (CLIP)."""

    def score(self, crops: Sequence[object], prompts: Sequence[str]) -> list[list[float]]:
        ...


class CropClassifier(Protocol):
    """Classifies image crops into a fixed vocabulary (MobileNetV3)."""

    def classify(self, crops: Sequence[object], top_k: int) -> list[list[tuple[int, float]]]:
        """Return ``(class_id, probability)`` pairs per crop, best first."""


@dataclass(frozen=True)
class Candidate:
    """One detector output, ready for recognition."""

    file_name: str
    file_path: str
    box: Box
    detector_confidence: float


@dataclass(frozen=True)
class Prediction:
    """A stored classifier result for one crop."""

    file_name: str
    file_path: str
    box: Box
    class_id: int
    probability: float


def select_crops(
    candidates: Sequence[Candidate],
    config: Config,
    image_size: tuple[int, int],
) -> list[Candidate]:
    """Apply the size and padding rules before recognition."""
    width, height = image_size
    kept: list[Candidate] = []
    for candidate in candidates:
        if candidate.detector_confidence < config.animal.search_confidence:
            continue
        if not is_large_enough(candidate.box, config.animal.min_crop_px):
            continue
        kept.append(
            Candidate(
                candidate.file_name,
                candidate.file_path,
                pad_box(candidate.box, width, height, ratio=config.animal.padding_ratio),
                candidate.detector_confidence,
            )
        )
    return kept


# ── path 1: in-vocabulary, served from the index ───────────────────────


def rank_from_index(
    resolution: Resolution,
    predictions: Sequence[Prediction],
    config: Config,
    class_names: Sequence[str] | None = None,
) -> list[Match]:
    """Rank using classifier predictions computed at index time.

    No model runs here. This is the common case — a photo editor searching
    ``zebra`` or ``golden retriever`` — and it costs one indexed SQL scan.
    """
    if not resolution.in_vocabulary:
        raise ValueError("query is not in the classifier vocabulary; use CLIP instead")

    wanted = set(resolution.class_ids)
    matches: list[Match] = []
    for prediction in predictions:
        if prediction.class_id not in wanted:
            continue
        # The confidence floor the original never had: it accepted a 21.5 %
        # top-1 over 1000 classes as a confirmed match.
        if prediction.probability < config.animal.classifier_threshold:
            continue
        label = (
            class_names[prediction.class_id]
            if class_names is not None and prediction.class_id < len(class_names)
            else resolution.query
        )
        matches.append(
            Match(
                file_name=prediction.file_name,
                file_path=prediction.file_path,
                score=prediction.probability,
                box=prediction.box,
                label=label,
            )
        )
    return best_per_image(matches)


# ── path 2: out of vocabulary, CLIP ────────────────────────────────────


def rank_with_clip(
    query: str,
    candidates: Sequence[Candidate],
    crops: Sequence[object],
    scorer: CropScorer,
    config: Config,
) -> list[Match]:
    """Rank with CLIP scored against contrastive prompts.

    Reserved for species the classifier's vocabulary cannot express, because it
    costs ~373x the compute per crop.
    """
    if len(candidates) != len(crops):
        raise ValueError(f"{len(candidates)} candidates but {len(crops)} crops")
    if not candidates:
        return []

    query = query.strip()
    if not query:
        raise ValueError("empty query")

    logits = scorer.score(crops, config.animal.prompts_for(query))
    if len(logits) != len(crops):
        raise ValueError(f"scorer returned {len(logits)} rows for {len(crops)} crops")

    matches: list[Match] = []
    for candidate, row in zip(candidates, logits, strict=True):
        score = contrastive_score(list(row))
        if score < config.animal.match_threshold:
            continue
        matches.append(
            Match(
                file_name=candidate.file_name,
                file_path=candidate.file_path,
                score=score,
                box=candidate.box,
                label=query,
            )
        )
    return best_per_image(matches)


# ── the router ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Route:
    """Which engine will answer, and why."""

    engine: str                 # "classifier" | "clip"
    resolution: Resolution

    def explain(self) -> str:
        if self.engine == "classifier":
            return (
                f"'{self.resolution.query}' resolved to "
                f"{len(self.resolution.class_ids)} ImageNet class(es) via "
                f"{self.resolution.via} — answering from the index, no inference"
            )
        return (
            f"'{self.resolution.query}' is outside the classifier vocabulary — "
            f"falling back to CLIP over candidate crops"
        )


def choose_route(query: str, class_names: Sequence[str]) -> Route:
    """Decide which engine answers this query."""
    resolution = resolve(query, list(class_names))
    return Route("classifier" if resolution.in_vocabulary else "clip", resolution)
