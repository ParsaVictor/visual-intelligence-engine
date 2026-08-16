"""Index-time analysis of one image.

Stage order is load-bearing
---------------------------
The original ran **animal → food → face**, while the food stage tried to reject
boxes overlapping a detected face::

    if has_face == 1:          # line ~220
        for face in faces:     # `faces` not assigned until line ~248
            ...

``has_face`` is reset to 0 at the top of every iteration and the face detector
runs roughly thirty lines *later*, so the guard was always false. The advertised
cross-verification never executed once, and every food-shaped box sitting on a
human face was accepted.

Running **face → animal → food** makes the dependency satisfiable, and
:func:`analyse_image` enforces it structurally rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Protocol

from vie.config import Config
from vie.geometry import Box, is_large_enough, overlaps_any


class FaceDetector(Protocol):
    def detect(self, image: object) -> list[tuple[Box, object]]:
        """Return ``(box, embedding)`` per detected face."""


class BoxDetector(Protocol):
    def detect(
        self, image: object, classes: Sequence[int], confidence: float
    ) -> list[tuple[Box, float]]:
        """Return ``(box, confidence)`` for the requested classes."""


@dataclass
class ImageAnalysis:
    """Everything one image contributes to the index."""

    has_face: bool = False
    has_animal: bool = False
    has_food: bool = False
    faces: list[tuple[Box, object]] = field(default_factory=list)
    animal_boxes: list[tuple[Box, float]] = field(default_factory=list)
    food_boxes: list[Box] = field(default_factory=list)
    rejected_food_boxes: list[Box] = field(default_factory=list)


def analyse_image(
    image: object,
    size: tuple[int, int],
    config: Config,
    *,
    face_detector: FaceDetector,
    animal_detector: BoxDetector,
    food_detector: BoxDetector,
) -> ImageAnalysis:
    """Run all three index-time detectors in dependency order."""
    result = ImageAnalysis()

    # ── 1. faces ──────────────────────────────────────────────────────
    # Must run first: the food stage needs the boxes.
    result.faces = list(face_detector.detect(image))
    result.has_face = bool(result.faces)
    face_boxes = [box for box, _ in result.faces]

    # ── 2. animals ────────────────────────────────────────────────────
    # The gate is deliberately the permissive threshold. An image rejected
    # here can never be reached at search time, so the gate must never be
    # stricter than the search pass; Config.validate() enforces that.
    result.animal_boxes = [
        (box, conf)
        for box, conf in animal_detector.detect(
            image, [config.animal.animal_class_id], config.animal.gate_confidence
        )
        if is_large_enough(box, config.animal.min_crop_px)
    ]
    result.has_animal = bool(result.animal_boxes)

    # ── 3. food ───────────────────────────────────────────────────────
    for box, _ in food_detector.detect(
        image, config.food.gate_classes, config.food.gate_confidence
    ):
        if not is_large_enough(box, config.food.min_crop_px):
            continue
        # The cross-verification that never ran in the original.
        if overlaps_any(box, face_boxes, config.food.face_overlap_reject):
            result.rejected_food_boxes.append(box)
            continue
        result.food_boxes.append(box)
    result.has_food = bool(result.food_boxes)

    return result


#: Documented, asserted execution order. Tests pin this so a future refactor
#: cannot silently reintroduce the original bug.
STAGE_ORDER: tuple[str, ...] = ("face", "animal", "food")
