"""Index-time analysis of images.

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
:func:`analyse_batch` enforces it structurally rather than by convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from vie.config import Config
from vie.geometry import Box, is_large_enough, overlaps_any


class FaceDetector(Protocol):
    def detect(self, image: object) -> list[tuple[Box, object]]:
        """Return ``(box, embedding)`` per detected face."""


class BoxDetector(Protocol):
    def detect(
        self, image: object, classes: Sequence[int], confidence: float
    ) -> list[tuple[Any, ...]]:
        """Return ``(box, confidence)`` or ``(box, confidence, class_id)``.

        A detector may also offer ``detect_batch(images, classes, confidence)``,
        which :func:`analyse_batch` prefers (P6-2).
        """


@dataclass
class ImageAnalysis:
    """Everything one image contributes to the index."""

    has_face: bool = False
    has_animal: bool = False
    has_food: bool = False
    faces: list[tuple[Box, object]] = field(default_factory=list)
    animal_boxes: list[tuple[Box, float]] = field(default_factory=list)
    food_boxes: list[Box] = field(default_factory=list)
    food_detections: list[tuple[Box, float, int | None]] = field(default_factory=list)
    rejected_food_boxes: list[Box] = field(default_factory=list)


def _detect_many(
    detector: BoxDetector, images: Sequence[object], classes: Sequence[int], confidence: float
) -> list[list[tuple[Any, ...]]]:
    batched = getattr(detector, "detect_batch", None)
    if batched is not None and len(images) > 1:
        return batched(images, classes, confidence)
    return [detector.detect(image, classes, confidence) for image in images]


def analyse_batch(
    images: Sequence[object],
    config: Config,
    *,
    face_detector: FaceDetector,
    animal_detector: BoxDetector,
    food_detector: BoxDetector,
) -> list[ImageAnalysis]:
    """Run all three index-time detectors over a batch, in dependency order."""
    results = [ImageAnalysis() for _ in images]

    # ── 1. faces ──────────────────────────────────────────────────────
    # Must run first: the food stage needs the boxes.
    for result, image in zip(results, images, strict=True):
        result.faces = list(face_detector.detect(image))
        result.has_face = bool(result.faces)

    # ── 2. animals ────────────────────────────────────────────────────
    # The gate is deliberately the permissive threshold. An image rejected
    # here can never be reached at search time, so the gate must never be
    # stricter than the search pass; Config.validate() enforces that.
    animals = _detect_many(
        animal_detector, images, [config.animal.animal_class_id], config.animal.gate_confidence
    )
    for result, found in zip(results, animals, strict=True):
        result.animal_boxes = [
            (item[0], item[1])
            for item in found
            if is_large_enough(item[0], config.animal.min_crop_px)
        ]
        result.has_animal = bool(result.animal_boxes)

    # ── 3. food ───────────────────────────────────────────────────────
    foods = _detect_many(
        food_detector, images, config.food.gate_classes, config.food.gate_confidence
    )
    for result, found in zip(results, foods, strict=True):
        face_boxes = [box for box, _ in result.faces]
        for item in found:
            box, confidence = item[0], item[1]
            class_id = item[2] if len(item) > 2 else None
            if not is_large_enough(box, config.food.min_crop_px):
                continue
            # The cross-verification that never ran in the original.
            if overlaps_any(box, face_boxes, config.food.face_overlap_reject):
                result.rejected_food_boxes.append(box)
                continue
            result.food_boxes.append(box)
            result.food_detections.append((box, confidence, class_id))
        result.has_food = bool(result.food_boxes)

    return results


def analyse_image(
    image: object,
    size: tuple[int, int],
    config: Config,
    *,
    face_detector: FaceDetector,
    animal_detector: BoxDetector,
    food_detector: BoxDetector,
) -> ImageAnalysis:
    """Analyse one image. ``size`` is accepted for callers that already have it."""
    return analyse_batch(
        [image], config,
        face_detector=face_detector,
        animal_detector=animal_detector,
        food_detector=food_detector,
    )[0]


#: Documented, asserted execution order. Tests pin this so a future refactor
#: cannot silently reintroduce the original bug.
STAGE_ORDER: tuple[str, ...] = ("face", "animal", "food")
