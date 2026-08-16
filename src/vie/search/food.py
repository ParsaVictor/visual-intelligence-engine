"""Food search — detect tableware, then recognise the dish with CLIP.

This module's decision rule is the one good idea the original project already
had, kept intact: score the user's query against explicit rival prompts and take
a softmax, rather than thresholding a raw similarity. That gives the model a
"none of the above" option, which is why its demo scores read 96-99% while the
free-text path's raw cosines read 23%.

The animal path has since been rebuilt on the same pattern, so three of the four
search paths now share one recognition rule.

Fixed here: the search pass used a different COCO class list and a different
confidence than indexing, which re-introduced class 56 (``chair``) — furniture
was cropped and sent to CLIP. Both passes now read one config, and
:meth:`vie.config.Config.validate` refuses a search class the gate never indexed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vie.config import Config
from vie.geometry import Box, is_large_enough, pad_box
from vie.scoring import Match, best_per_image, contrastive_score


class CropScorer(Protocol):
    def score(self, crops: Sequence[object], prompts: Sequence[str]) -> list[list[float]]:
        """Return one logit vector per crop, aligned with ``prompts``."""


@dataclass(frozen=True)
class Candidate:
    file_name: str
    file_path: str
    box: Box
    detector_confidence: float


def select_crops(
    candidates: Sequence[Candidate],
    config: Config,
    image_size: tuple[int, int],
) -> list[Candidate]:
    """Filter and pad detections before recognition.

    Padding is by a fixed pixel count here rather than a ratio: a plate's
    identity is in its contents, and a constant margin recovers the rim without
    swallowing the neighbouring dish.
    """
    width, height = image_size
    kept: list[Candidate] = []
    for candidate in candidates:
        if candidate.detector_confidence < config.food.search_confidence:
            continue
        if not is_large_enough(candidate.box, config.food.min_crop_px):
            continue
        padded = pad_box(candidate.box, width, height, pixels=config.food.padding_px)
        kept.append(
            Candidate(
                candidate.file_name,
                candidate.file_path,
                padded,
                candidate.detector_confidence,
            )
        )
    return kept


def rank(
    query: str,
    candidates: Sequence[Candidate],
    crops: Sequence[object],
    scorer: CropScorer,
    config: Config,
) -> list[Match]:
    """Score crops against the dish name and return one result per image."""
    if len(candidates) != len(crops):
        raise ValueError(f"{len(candidates)} candidates but {len(crops)} crops")
    if not candidates:
        return []

    query = query.strip()
    if not query:
        raise ValueError("empty query")

    prompts = config.food.prompts_for(query)
    logits = scorer.score(crops, prompts)
    if len(logits) != len(crops):
        raise ValueError(f"scorer returned {len(logits)} rows for {len(crops)} crops")

    matches: list[Match] = []
    for candidate, row in zip(candidates, logits, strict=True):
        score = contrastive_score(list(row))
        if score < config.food.match_threshold:
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
