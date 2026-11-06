"""Food search: find plates and bowls, then recognise the dish with SigLIP.

The decision rule is the best idea the original project had, kept: score the query
against explicit rival prompts ("a completely different food", "an empty plate")
instead of thresholding a raw similarity, which gives the model a way to say "none
of these". :func:`vie.vlm.hypothesis_score` turns the logits into one number: a
softmax posterior for CLIP, and for SigLIP the sigmoid of the margin over the
strongest rival, which ``scripts/calibrate.py`` measured as the more stable cut.

Candidates come from RT-DETR boxes on COCO's tableware and food classes, plus whole
images the zero-cost SigLIP food gate admitted at index time, since a plate of
ghormeh sabzi fires none of COCO's ten Western food classes.

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
from vie.scoring import Match, best_per_image
from vie.vlm import hypothesis_score


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
        score = hypothesis_score(list(row), config.score_mode)
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
