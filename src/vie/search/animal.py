"""Animal search — CLIP zero-shot, replacing the ImageNet classifier.

What this replaces
------------------
The original module was MegaDetector (localise) → MobileNetV3-Small (label) →
a hand-written map collapsing ImageNet's 1000 class names into 8 buckets by
substring containment. That map was verified broken against the real class
list:

===========================  ========================  ==================
class                        assigned category         cause
===========================  ========================  ==================
``African elephant``         Insect / Arthropod        ``eleph``\\ **ant**
``giant panda``              Insect / Arthropod        ``gi``\\ **ant**
``wild boar``                Reptile / Amphibian       **boa**\\ ``r``
``sea lion``                 Cat                       **lion**
``mailbox``                  Large Mammal              **ox**
``bathtub``                  Small Mammal / Primate    **bat**
``beer bottle``              Insect / Arthropod        **bee**
===========================  ========================  ==================

Three separate defects came out of that design — the broken map, the absence of
any confidence floor, and unanchored query matching against a category string
that literally contains the word "non-animal". All three disappear here rather
than being patched, because the keyword map is gone entirely.

Why CLIP
--------
CLIP ViT-L/14 is *already* loaded for the food and free-text paths, already in
FP16, and already resident in VRAM. Using it here costs nothing extra and buys:

* an open vocabulary — ``red panda`` and ``fennec fox`` work despite not being
  ImageNet-1k classes;
* a calibrated decision, via the same contrastive-candidate trick the food
  module pioneered: score the query against explicit rival prompts and take a
  softmax, so the model can say "none of these";
* one recognition pattern shared by three of the four search paths.

MegaDetector is kept as the localiser. That part of the original design was
right: a class-agnostic detector gives a species-independent recall gate, and
cropping before recognition is what makes a small animal in a large press photo
survive the downscale to the model's input size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vie.config import Config
from vie.geometry import Box, is_large_enough, pad_box
from vie.scoring import Match, best_per_image, contrastive_score


class CropScorer(Protocol):
    """Anything that can score image crops against text prompts.

    Declaring this as a protocol keeps the search logic testable without a GPU,
    a model download, or a network call — the decision rules are what the
    regression tests need to pin, not CLIP's weights.
    """

    def score(self, crops: Sequence[object], prompts: Sequence[str]) -> list[list[float]]:
        """Return one logit vector per crop, aligned with ``prompts``."""


@dataclass(frozen=True)
class Candidate:
    """One detector output, ready for recognition."""

    file_name: str
    file_path: str
    box: Box
    detector_confidence: float


def select_crops(
    candidates: Sequence[Candidate],
    config: Config,
    image_size: tuple[int, int],
) -> list[Candidate]:
    """Apply the size and padding rules before recognition.

    The original applied padding but no minimum size, so a 12x9 px detection was
    upsampled roughly 20x and handed back a confident-looking species label.
    """
    width, height = image_size
    kept: list[Candidate] = []
    for candidate in candidates:
        if candidate.detector_confidence < config.animal.search_confidence:
            continue
        if not is_large_enough(candidate.box, config.animal.min_crop_px):
            continue
        padded = pad_box(candidate.box, width, height, ratio=config.animal.padding_ratio)
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
    """Score crops against the query and return one ranked result per image.

    ``candidates[i]`` must describe ``crops[i]``.
    """
    if len(candidates) != len(crops):
        raise ValueError(f"{len(candidates)} candidates but {len(crops)} crops")
    if not candidates:
        return []

    query = query.strip()
    if not query:
        raise ValueError("empty query")

    prompts = config.animal.prompts_for(query)
    logits = scorer.score(crops, prompts)
    if len(logits) != len(crops):
        raise ValueError(f"scorer returned {len(logits)} rows for {len(crops)} crops")

    matches: list[Match] = []
    for candidate, row in zip(candidates, logits, strict=True):
        score = contrastive_score(list(row))
        # The confidence floor the original module never had. A 21.5% top-1 over
        # 1000 classes was previously shipped as a confirmed match.
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
