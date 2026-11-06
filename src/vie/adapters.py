"""Adapters wrapping each model behind the protocols the pipeline expects.

The pipeline and the search modules only know about small protocols
(:class:`vie.indexing.FaceDetector`, :class:`vie.indexing.BoxDetector`,
``CropScorer``). Everything model-specific (MegaDetector's raw tensor layout,
RT-DETR's result objects, InsightFace's ``Face`` records) is confined here, so
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
from vie.models import ModelBundle, bgr_to_pil
from vie.quality import is_usable_face

log = get_logger("adapters")


def face_vectors(app: Any, image: Any, faces: Sequence[Any], flip: bool) -> list[Any]:
    """Unit identity vectors for detected faces, optionally mirror-averaged.

    InsightFace has already embedded each face as seen. With ``flip`` the aligned
    crop is mirrored and embedded once more, in one batch for the whole image, and
    the two raw vectors are summed before normalising. A face seen at an angle
    then carries evidence from both of its sides, which is where single-view
    embeddings lose the most (see scripts/face_benchmark.py).
    """
    import numpy as np

    if not faces:
        return []
    recogniser = getattr(app, "models", {}).get("recognition")
    if (not flip or recogniser is None
            or any(getattr(face, "kps", None) is None for face in faces)):
        return [face.normed_embedding for face in faces]

    from insightface.utils import face_align

    size = recogniser.input_size[0]
    mirrored = [
        np.ascontiguousarray(
            face_align.norm_crop(image, landmark=face.kps, image_size=size)[:, ::-1]
        )
        for face in faces
    ]
    extra = np.asarray(recogniser.get_feat(mirrored), dtype=np.float32).reshape(len(faces), -1)
    out = []
    for face, other in zip(faces, extra, strict=True):
        seen = getattr(face, "embedding", None)
        if seen is None:  # a FaceAnalysis built without its recognition module
            seen = recogniser.get_feat(
                [face_align.norm_crop(image, landmark=face.kps, image_size=size)]
            )[0]
        # Raw vectors are summed before normalising, as InsightFace's own flip test
        # does: a raw norm tracks how much identity a view carries, so the clearer
        # side of a turned face counts for more.
        vector = np.asarray(seen, dtype=np.float32).ravel() + other
        out.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
    return out


class InsightFaceDetector:
    """InsightFace ``buffalo_l``: RetinaFace detection + ArcFace embeddings.

    Applies a deliberately permissive quality gate. Anything RetinaFace is even
    moderately confident about gets indexed. The gate exists only to drop
    detections too degraded to carry identity, because such an embedding does
    not merely fail to match, it matches the *wrong person*.
    """

    def __init__(self, bundle: ModelBundle, config: Config) -> None:
        self.bundle = bundle
        self.config = config
        self.rejected = 0

    def detect(self, image: Any) -> list[tuple[Box, Any]]:
        app = self.bundle.face_app
        kept: list[tuple[Box, Any]] = []
        for face in app.get(image):
            box = tuple(int(v) for v in face.bbox[:4])
            score = float(getattr(face, "det_score", 1.0))
            if not is_usable_face(score, box, self.config.face.min_det_score,
                                  self.config.face.min_face_px):
                self.rejected += 1
                continue
            kept.append((box, face))
        # Unit length either way, which is what makes a dot product equal cosine
        # similarity downstream.
        vectors = face_vectors(app, image, [f for _, f in kept], self.config.face.flip_tta)
        return [(box, vector) for (box, _), vector in zip(kept, vectors, strict=True)]


class MegaDetectorAdapter:
    """MegaDetector v5a via the raw YOLOv5 head.

    The raw tensor is 0-indexed (0=animal, 1=person, 2=vehicle), *not* the 1/2/3
    convention MegaDetector's published JSON API uses. Getting this wrong
    silently indexes vehicles as wildlife.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def detect_batch(
        self, images: Sequence[Any], classes: Sequence[int], confidence: float
    ) -> list[list[tuple[Box, float]]]:
        import torch

        model = self.bundle.animal_detector
        # AutoShape applies its own confidence floor (0.25 by default) before we
        # see anything, which would silently undercut a more permissive gate.
        model.conf = min(confidence, 0.25)
        wanted = set(classes)
        # AutoShape reads NumPy input as RGB, while OpenCV hands us BGR.
        rgb = [image[:, :, ::-1].copy() for image in images]
        with torch.no_grad():
            results = model(rgb).xyxy

        out: list[list[tuple[Box, float]]] = []
        for predictions in results:
            found: list[tuple[Box, float]] = []
            for row in predictions.cpu().numpy():
                x1, y1, x2, y2, score, class_id = row[:6]
                if int(class_id) in wanted and float(score) >= confidence:
                    found.append(((int(x1), int(y1), int(x2), int(y2)), float(score)))
            out.append(found)
        return out

    def detect(
        self, image: Any, classes: Sequence[int], confidence: float
    ) -> list[tuple[Box, float]]:
        return self.detect_batch([image], classes, confidence)[0]


class RTDetrAdapter:
    """RT-DETR-X restricted to the configured COCO classes."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def detect_batch(
        self, images: Sequence[Any], classes: Sequence[int], confidence: float
    ) -> list[list[tuple[Box, float, int]]]:
        results = self.bundle.object_detector(
            list(images), classes=list(classes), conf=confidence, verbose=False
        )
        out: list[list[tuple[Box, float, int]]] = []
        for result in results:
            found: list[tuple[Box, float, int]] = []
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                found.append(((x1, y1, x2, y2), float(box.conf[0]), int(box.cls[0])))
            out.append(found)
        return out

    def detect(
        self, image: Any, classes: Sequence[int], confidence: float
    ) -> list[tuple[Box, float, int]]:
        return self.detect_batch([image], classes, confidence)[0]


def load_scaled(path: Path, cap: int) -> tuple[Any, float] | None:
    """Decode an image, cap its longest side, and report the factor applied.

    A 6000 px press photo costs decode time and PCIe bandwidth for detail no
    detector running at 640 px input can use.
    """
    import cv2
    import numpy as np

    # cv2.imread cannot open a non-ASCII path on Windows and just returns None,
    # which would hide every Persian-named file in the gallery.
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        return None
    height, width = image.shape[:2]
    if max(height, width) <= cap:
        return image, 1.0
    scale = cap / max(height, width)
    resized = cv2.resize(
        image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def load_image(path: Path, cap: int = 1600) -> Any:
    """Decode an image for query-time use, discarding the scale."""
    loaded = load_scaled(path, cap)
    return None if loaded is None else loaded[0]


def image_size(image: Any) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height


def crop(image: Any, box: Box) -> Any:
    """Crop to a box and convert to the RGB PIL image the encoders expect."""
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return bgr_to_pil(patch)


@dataclass
class Detectors:
    face: InsightFaceDetector
    animal: MegaDetectorAdapter
    food: RTDetrAdapter
    loader: Any
    size_of: Any = image_size
    crop: Any = crop


def build_detectors(bundle: ModelBundle, config: Config) -> Detectors:
    cap = config.face.max_index_dimension
    return Detectors(
        face=InsightFaceDetector(bundle, config),
        animal=MegaDetectorAdapter(bundle),
        food=RTDetrAdapter(bundle),
        loader=lambda path: load_scaled(path, cap),
    )
