"""Adapters wrapping each model behind the protocols the pipeline expects.

The pipeline and the search modules only know about small protocols
(:class:`vie.indexing.FaceDetector`, :class:`vie.indexing.BoxDetector`,
``CropScorer``). Everything model-specific — MegaDetector's raw tensor layout,
RT-DETR's result objects, InsightFace's ``Face`` records — is confined here, so
swapping a model touches one file and no tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.geometry import Box
from vie.logging_setup import get_logger
from vie.models import ModelBundle

log = get_logger("adapters")


class InsightFaceDetector:
    """InsightFace ``buffalo_l``: RetinaFace detection + ArcFace embeddings."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.app = bundle.face_app

    def detect(self, image: Any) -> list[tuple[Box, Any]]:
        faces = self.app.get(image)
        out: list[tuple[Box, Any]] = []
        for face in faces:
            box = tuple(int(v) for v in face.bbox[:4])
            # normed_embedding is unit length, which is what makes a dot
            # product equal cosine similarity downstream.
            out.append((box, face.normed_embedding))
        return out


class MegaDetectorAdapter:
    """MegaDetector v5a via the raw YOLOv5 head.

    The raw tensor is 0-indexed (0=animal, 1=person, 2=vehicle) — *not* the
    1/2/3 convention MegaDetector's published JSON API uses. Getting this wrong
    silently indexes vehicles as wildlife.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.model = bundle.animal_detector

    def detect(
        self, image: Any, classes: Sequence[int], confidence: float
    ) -> list[tuple[Box, float]]:
        import torch

        wanted = set(classes)
        with torch.no_grad():
            predictions = self.model(image).xyxy[0].cpu().numpy()

        out: list[tuple[Box, float]] = []
        for row in predictions:
            x1, y1, x2, y2, score, class_id = row[:6]
            if int(class_id) not in wanted or float(score) < confidence:
                continue
            out.append(((int(x1), int(y1), int(x2), int(y2)), float(score)))
        return out


class RTDetrAdapter:
    """RT-DETR-X restricted to the configured COCO classes."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.model = bundle.object_detector

    def detect(
        self, image: Any, classes: Sequence[int], confidence: float
    ) -> list[tuple[Box, float]]:
        results = self.model(image, classes=list(classes), conf=confidence, verbose=False)
        out: list[tuple[Box, float]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                out.append(((x1, y1, x2, y2), float(box.conf[0])))
        return out


def load_image(path: Path) -> Any:
    """Decode an image and cap its longest side.

    A 6000 px press photo costs decode time and PCIe bandwidth for detail no
    detector running at 640 px input can use.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    cap = 1600
    if max(height, width) > cap:
        scale = cap / max(height, width)
        image = cv2.resize(
            image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    return image


def image_size(image: Any) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height


def crop(image: Any, box: Box) -> Any:
    """Crop to a box and convert to the RGB PIL image CLIP expects."""
    import cv2
    from PIL import Image

    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))


@dataclass
class Detectors:
    face: InsightFaceDetector
    animal: MegaDetectorAdapter
    food: RTDetrAdapter
    loader: Any = load_image
    size_of: Any = image_size


def build_detectors(bundle: ModelBundle, config: Config) -> Detectors:
    return Detectors(
        face=InsightFaceDetector(bundle),
        animal=MegaDetectorAdapter(bundle),
        food=RTDetrAdapter(bundle),
    )
