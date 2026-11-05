"""A deterministic stand-in for the whole model stack.

The point of these fakes is to exercise the real wiring end to end. v0.2 had 198
passing unit tests and a species search that crashed on every call, because no
test ever went through ``cli -> query -> search``.

Images are small real PNGs whose dominant colour encodes what they "contain", and
the fake encoder maps colour into a vector, so the assertions read like behaviour:
a text query for "red" must find the red image, and only stays true if BGR/RGB
handling is right all the way through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Prompt words the fake understands, as (R, G, B) directions.
WORDS = {
    "red": (1.0, 0.0, 0.0),
    "zebra": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "kebab": (0.0, 1.0, 0.0),
    "food": (0.0, 1.0, 0.0),
    "meal": (0.0, 1.0, 0.0),
    "dish": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "quokka": (0.0, 0.0, 1.0),
}


def write_png(path: Path, rgb: tuple[int, int, int], size: tuple[int, int] = (120, 90)) -> Path:
    """Write a solid-colour image. ``size`` is (width, height)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path)
    return path


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class FakeEncoder:
    """Implements the surface of :class:`vie.models.VisionLanguageEncoder`."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.forward_passes = 0
        self.batch_sizes: list[int] = []

    def _embed_rgb(self, rgb: np.ndarray) -> np.ndarray:
        mean = rgb.reshape(-1, 3).mean(axis=0) / 255.0
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[:3] = mean
        return _unit(vector + 1e-3).astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        direction = np.zeros(3)
        lowered = text.lower()
        # A rival prompt names what it excludes: "a different food" is not food.
        negated = "different" in lowered or " no " in lowered
        for word, rgb in WORDS.items():
            if word in lowered and not negated:
                direction += np.array(rgb)
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[:3] = direction
        return _unit(vector + 1e-3).astype(np.float32)

    def encode_texts(self, texts) -> np.ndarray:
        return np.vstack([self.encode_text(t) for t in texts])

    def embed_batch(self, images, regions):
        self.forward_passes += 1
        self.batch_sizes.append(sum(len(r) for r in regions))
        grouped = []
        for image, boxes in zip(images, regions, strict=True):
            rows = []
            for name, (x1, y1, x2, y2) in boxes:
                # Images arrive as OpenCV BGR; reverse exactly as the real one must.
                rows.append((name, self._embed_rgb(image[y1:y2, x1:x2, ::-1])))
            grouped.append(rows)
        return grouped

    def score(self, crops, prompts) -> list[list[float]]:
        texts = self.encode_texts(prompts)
        return [
            ((self._embed_rgb(np.asarray(crop)) @ texts.T) * 12.0 - 6.0).tolist()
            for crop in crops
        ]

    def food_gate_scores(self, whole_vectors, config) -> list[float]:
        from vie.vlm import gate_score

        prompts = config.food.gate_prompts()
        logits = (whole_vectors @ self.encode_texts(prompts).T) * 12.0 - 6.0
        n_positive = len(config.food.clip_gate_positive)
        return [gate_score(row.tolist(), n_positive, config.score_mode) for row in logits]

    def logits_from_similarity(self, similarity):
        return similarity * 12.0 - 6.0

    def calibrated(self, cosine: float) -> float | None:
        return max(0.0, min(1.0, cosine))


class FakeFaces:
    """Finds a 'face' in images with a strong blue channel, and embeds it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect(self, image):
        self.calls.append("face")
        if float(image[:, :, 0].mean()) < 128:  # channel 0 is blue in BGR
            return []
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        return [((10, 10, 60, 70), vector)]


class FakeAnimals:
    """Finds an 'animal' covering the frame wherever red dominates."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image, classes, confidence):
        self.calls += 1
        if float(image[:, :, 2].mean()) < 128:  # channel 2 is red in BGR
            return []
        return [((5, 5, 100, 80), 0.9)]


class FakeFood:
    """A batching detector that returns the class id, like the real adapter."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def detect_batch(self, images, classes, confidence):
        self.batches.append(len(images))
        return [self.detect(image, classes, confidence) for image in images]

    def detect(self, image, classes, confidence):
        return []


class FakeClassifier:
    """Calls every crop class 340 ('zebra' in the vocabulary below)."""

    names = ["tench", "goldfish"] + [f"class {i}" for i in range(2, 340)] + ["zebra"]

    def __init__(self) -> None:
        self.crops_seen = 0

    def classify(self, crops, top_k):
        self.crops_seen += len(crops)
        return [[(340, 0.91), (0, 0.02)][:top_k] for _ in crops]


class FakeFaceApp:
    """The slice of InsightFace ``FaceAnalysis`` the query path uses."""

    class _Face:
        bbox = (0.0, 0.0, 50.0, 60.0)
        normed_embedding = np.eye(512, dtype=np.float32)[0]

    def prepare(self, ctx_id, det_size):
        pass

    def get(self, image):
        return [self._Face()]


class FakeBundle:
    """A bundle whose every attribute is observable, to prove what was loaded."""

    class _Runtime:
        is_cuda = False

    def __init__(self) -> None:
        self.runtime = self._Runtime()
        self.face_app = FakeFaceApp()
        self.class_names = FakeClassifier.names


class FakeStack:
    """Everything ``build_index`` and ``run_search`` need, wired like the CLI does."""

    def __init__(self, config) -> None:
        self.config = config
        self.encoder = FakeEncoder()
        self.faces = FakeFaces()
        self.animals = FakeAnimals()
        self.food = FakeFood()
        self.classifier = FakeClassifier()
        self.bundle = FakeBundle()
        self.runtime_loads = 0

    def load_runtime(self, config):
        self.runtime_loads += 1
        return self.bundle, self.encoder, self.encoder

    def index(self, **overrides):
        from vie.adapters import crop, image_size, load_scaled
        from vie.pipeline import build_index

        settings = dict(
            loader=lambda path: load_scaled(path, self.config.face.max_index_dimension),
            size_of=image_size,
            face_detector=self.faces,
            animal_detector=self.animals,
            food_detector=self.food,
            embed_batch=self.encoder.embed_batch,
            food_gate=lambda v: self.encoder.food_gate_scores(v, self.config),
            classify_crops=self.classifier.classify,
            crop=crop,
            class_names=lambda: FakeClassifier.names,
            workers=2,
        )
        settings.update(overrides)
        return build_index(self.config, **settings)
