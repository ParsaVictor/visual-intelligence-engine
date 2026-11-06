"""Model loading and the concrete encoders the search paths depend on.

Everything here is deliberately lazy, at two levels:

* importing :mod:`vie.models` does not pull in torch, so the core logic and its
  regression tests stay installable without a multi-gigabyte dependency tree;
* :class:`ModelBundle` loads each model on first use. It used to load all six for
  every command, so a text search spent minutes and many gigabytes on a face
  model, two detectors and a classifier it never touched.

Three problems from the original are fixed at this layer:

* **Unpinned supply chain.** ``torch.hub.load('ultralytics/yolov5', ...,
  trust_repo=True)`` cloned a third-party repository at runtime and executed it
  with the confirmation suppressed, so upstream changes altered behaviour
  silently. :data:`YOLOV5_PIN` pins the revision.
* **Device/dtype drift.** All casting goes through :class:`vie.device.Runtime`.
* **Preprocessing mismatch.** Each checkpoint is asked for its own reference
  transform via ``weights.transforms()``, so preprocessing cannot drift from the
  model, including when the classifier is swapped.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.device import Runtime, resolve
from vie.logging_setup import get_logger
from vie.vlm import is_siglip, similarity_to_probability

log = get_logger("models")

#: Pinned so a change to yolov5 master cannot alter detection behaviour.
YOLOV5_PIN = "v7.0"

MEGADETECTOR_URL = (
    "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
)

#: ``.pt`` files are pickles, so loading one executes code. Verifying the digest
#: of what was downloaded is what makes that acceptable. Digest of the official
#: release asset, computed from a fresh download on 2026-11-04.
MEGADETECTOR_SHA256 = "94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276"


def download_if_missing(path: str | Path, url: str, sha256: str = "") -> Path:
    """Fetch a weight file once and verify it when a digest is known."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s -> %s", url, path)
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)  # noqa: S310 - fixed, vetted URL
        partial.replace(path)
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise RuntimeError(
                f"{path} has SHA-256 {digest.hexdigest()}, expected {sha256}; refusing to "
                f"load it, because a .pt file executes code when unpickled"
            )
    return path


#: ImageNet-1k classifiers, all with the identical 1000-class vocabulary.
#: Swapping between them changes accuracy, never which species can be named.
CLASSIFIERS = {
    "mobilenet_v3_large": ("MobileNet_V3_Large_Weights", 74.0),
    "efficientnet_b0": ("EfficientNet_B0_Weights", 77.7),
    "convnext_tiny": ("ConvNeXt_Tiny_Weights", 82.5),
    "efficientnet_v2_s": ("EfficientNet_V2_S_Weights", 84.2),
}


def load_classifier(name: str, runtime: Runtime):
    """Load an ImageNet classifier plus its reference transform and labels."""
    from torchvision import models as tv

    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {name!r}; expected one of {sorted(CLASSIFIERS)}")

    weights_attr, published_acc = CLASSIFIERS[name]
    weights = getattr(tv, weights_attr).DEFAULT
    model = getattr(tv, name)(weights=weights).to(runtime.device).eval()
    log.info("loaded %s (%.1f%% published top-1)", name, published_acc)
    return model, weights.transforms(), list(weights.meta["categories"])


def load_weights_low_memory(name: str) -> Any:
    """Load a checkpoint holding only one copy of its weights in memory.

    ``from_pretrained`` first builds a randomly initialised model and then loads over
    it, so the peak is about twice the weights. When that exceeds what Windows can
    commit, the failure is not a Python error but a native access violation that kills
    the process with no message. Building on the ``meta`` device skips the random copy.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel

    # Memory-mapped: pages come straight from the file, so nothing is copied.
    with safe_open(hf_hub_download(name, "model.safetensors"), framework="pt") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
    with torch.device("meta"):
        model = AutoModel.from_config(AutoConfig.from_pretrained(name))
    model.load_state_dict(state, strict=False, assign=True)

    # Buffers that are rebuilt rather than saved are still placeholders at this point.
    for module in model.modules():
        for key, buffer in list(module.named_buffers(recurse=False)):
            if buffer.is_meta:
                if key == "position_ids":
                    size = buffer.shape[-1]
                    value = torch.arange(size).expand(buffer.shape).clone()
                else:
                    value = torch.zeros(buffer.shape, dtype=buffer.dtype)
                module.register_buffer(key, value, persistent=False)
    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    if leftover:
        raise RuntimeError(f"{name}: no weights found for {leftover[:5]}")
    return model


class ModelBundle:
    """Every model the system can use, each loaded once, on first access."""

    def __init__(self, config: Config) -> None:
        from vie.cache import configure_cache

        self.config = config
        self.score_mode = config.score_mode
        self.cache_dir = configure_cache(config.model_cache)
        if self.cache_dir:
            log.info("model cache: %s", self.cache_dir)
        self._lock = threading.RLock()
        self._loaded: dict[str, Any] = {}

    def _once(self, key: str, factory: Any) -> Any:
        with self._lock:
            if key not in self._loaded:
                self._loaded[key] = factory()
            return self._loaded[key]

    @property
    def runtime(self) -> Runtime:
        def build() -> Runtime:
            runtime = resolve(self.config.device, self.config.half_precision)
            log.info("device=%s half=%s", runtime.device, runtime.use_half)
            return runtime

        return self._once("runtime", build)

    def _vlm(self) -> tuple[Any, Any]:
        def build() -> tuple[Any, Any]:
            from transformers import AutoModel, AutoProcessor

            name = self.config.models["vision_language"]
            if self.config.low_memory:
                loaded = load_weights_low_memory(name)
            else:
                loaded = AutoModel.from_pretrained(name)
            model = self.runtime.prepare_model(loaded)
            processor = AutoProcessor.from_pretrained(name)
            log.info("loaded %s (score mode: %s)", name, self.score_mode)
            return model, processor

        return self._once("vlm", build)

    @property
    def clip_model(self) -> Any:
        return self._vlm()[0]

    @property
    def clip_processor(self) -> Any:
        return self._vlm()[1]

    @property
    def face_app(self) -> Any:
        def build() -> Any:
            from insightface.app import FaceAnalysis

            runtime = self.runtime
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if runtime.is_cuda
                else ["CPUExecutionProvider"]
            )
            kwargs: dict[str, Any] = {}
            if self.cache_dir:
                kwargs["root"] = str(self.cache_dir / "insightface")
            # The pack also ships two landmark models and an age/gender model, which
            # FaceAnalysis runs on every face by default. Nothing here reads their
            # output; skipping them halved the time per face (1144 -> 527 ms on a
            # four-face 1080p frame, CPU) with identical detections and embeddings.
            app = FaceAnalysis(name=self.config.models["face_pack"], providers=providers,
                               allowed_modules=["detection", "recognition"], **kwargs)
            size = self.config.face.det_size_large
            app.prepare(ctx_id=0 if runtime.is_cuda else -1, det_size=(size, size))
            log.info("loaded InsightFace %s", self.config.models["face_pack"])
            return app

        return self._once("face", build)

    def _weight_path(self, name: str) -> Path:
        return (self.cache_dir / "weights" / name) if self.cache_dir else Path(name)

    @property
    def animal_detector(self) -> Any:
        def build() -> Any:
            import os

            import torch

            weights = download_if_missing(
                self._weight_path(self.config.models["animal_detector"]),
                MEGADETECTOR_URL,
                MEGADETECTOR_SHA256,
            )
            # yolov5 v7.0 unpickles whole model objects, which torch >= 2.6 refuses
            # by default. Scoped to this one load of a pinned file.
            previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            try:
                model = torch.hub.load(
                    f"ultralytics/yolov5:{YOLOV5_PIN}", "custom",
                    path=str(weights), trust_repo=True, _verbose=False,
                ).to(self.runtime.device)
            finally:
                if previous is None:
                    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
                else:
                    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
            log.info("loaded MegaDetector (yolov5 pinned at %s)", YOLOV5_PIN)
            return model

        return self._once("animal", build)

    @property
    def object_detector(self) -> Any:
        def build() -> Any:
            from ultralytics import RTDETR

            model = RTDETR(str(self._weight_path(self.config.models["object_detector"])))
            log.info("loaded RT-DETR")
            return model

        return self._once("object", build)

    def _classifier(self) -> tuple[Any, Any, list[str]]:
        return self._once(
            "classifier", lambda: load_classifier(self.config.animal.classifier, self.runtime)
        )

    @property
    def classifier(self) -> Any:
        return self._classifier()[0]

    @property
    def classifier_transform(self) -> Any:
        return self._classifier()[1]

    @property
    def class_names(self) -> list[str]:
        return self._classifier()[2]


def load_all(config: Config) -> ModelBundle:
    """Create the bundle. Nothing is loaded until a model is first needed.

    Weights are frozen and nothing here is fine-tuned. The engineering value of
    this project is in how these are combined and indexed, not in training.
    """
    return ModelBundle(config)


class SpeciesClassifier:
    """Classifies animal crops into ImageNet-1k, batched.

    Runs at index time, once per crop, so its cost is paid on ingest rather than
    on every query. That is what makes a stronger, heavier backbone affordable.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def classify(self, crops: Sequence[Any], top_k: int) -> list[list[tuple[int, float]]]:
        import torch

        if not crops:
            return []
        transform = self.bundle.classifier_transform
        batch = torch.stack([transform(c) for c in crops]).to(self.bundle.runtime.device)
        with torch.no_grad():
            probs = torch.softmax(self.bundle.classifier(batch), dim=1)
        k = min(top_k, probs.shape[1])
        top = torch.topk(probs, k=k, dim=1)
        return [
            list(zip(idx.tolist(), val.tolist(), strict=True))
            for idx, val in zip(top.indices, top.values, strict=True)
        ]


def bgr_to_pil(patch: Any) -> Any:
    """OpenCV decodes BGR; every vision-language model expects RGB.

    Passing a BGR array straight to the processor raises no error. It just swaps
    red and blue in every embedding, so "a red car" retrieves blue ones.
    """
    from PIL import Image

    return Image.fromarray(patch[:, :, ::-1].copy())


def _as_features(output: Any) -> Any:
    """``get_*_features`` returns a tensor or, on newer versions, a model output."""
    return output if hasattr(output, "norm") else output.pooler_output


class VisionLanguageEncoder:
    """The shared image/text encoder: indexing, scoring and calibration.

    Vectors are unit length, which is what lets retrieval use a plain dot
    product as cosine similarity. This class satisfies the search paths'
    ``CropScorer`` protocol and is what the pipeline embeds with.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.batch_size = bundle.config.indexing.batch_size
        self._text_cache: dict[str, Any] = {}

    @property
    def is_siglip(self) -> bool:
        return is_siglip(self.bundle.config.models["vision_language"])

    # ---- encoding -----------------------------------------------------

    def _tokenize(self, texts: Sequence[str]) -> Any:
        # SigLIP was trained on text padded to its full 64 tokens, and its text
        # tower has no attention mask. padding=True produces a different, wrong
        # embedding with no error at all.
        padding = "max_length" if self.is_siglip else True
        return self.bundle.clip_processor(
            text=list(texts), return_tensors="pt", padding=padding, truncation=True
        )

    def encode_images(self, images: Sequence[Any], batch_size: int | None = None):
        """Encode RGB PIL images in chunks, returning an ``(n, dim)`` float32 array."""
        import numpy as np
        import torch

        if not images:
            return np.empty((0, 0), dtype=np.float32)
        step = batch_size or max(1, self.batch_size)
        chunks = []
        for start in range(0, len(images), step):
            inputs = self.bundle.clip_processor(
                images=list(images[start:start + step]), return_tensors="pt"
            )
            inputs = self.bundle.runtime.cast_inputs(dict(inputs))
            with torch.no_grad():
                features = _as_features(self.bundle.clip_model.get_image_features(**inputs))
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            chunks.append(features.float().cpu().numpy())
        return np.vstack(chunks)

    def encode_texts(self, texts: Sequence[str]):
        """Encode prompts, caching them: gates reuse the same few on every image."""
        import numpy as np
        import torch

        missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if missing:
            inputs = self._tokenize(missing)
            inputs = self.bundle.runtime.cast_inputs(dict(inputs))
            with torch.no_grad():
                features = _as_features(self.bundle.clip_model.get_text_features(**inputs))
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            for text, vector in zip(missing, features.float().cpu().numpy(), strict=True):
                self._text_cache[text] = vector
        return np.vstack([self._text_cache[t] for t in texts])

    def encode_text(self, text: str):
        return self.encode_texts([text])[0]

    def encode_image(self, image: Any):
        """Encode one BGR array or RGB PIL image."""
        return self.encode_images([bgr_to_pil(image) if hasattr(image, "shape") else image])[0]

    def embed_batch(
        self, images: Sequence[Any], regions: Sequence[Sequence[tuple[str, Any]]]
    ) -> list[list[tuple[str, Any]]]:
        """Encode every region of every BGR image in one batched pass.

        On a CPU this saves little: eight images took 26.9 s batched and 28.7 s one by
        one. The case for it is a GPU, which has not been measured here.
        """
        patches: list[Any] = []
        owners: list[tuple[int, str]] = []
        for owner, (image, boxes) in enumerate(zip(images, regions, strict=True)):
            for name, (x1, y1, x2, y2) in boxes:
                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                patches.append(bgr_to_pil(patch))
                owners.append((owner, name))

        grouped: list[list[tuple[str, Any]]] = [[] for _ in images]
        if not patches:
            return grouped
        features = self.encode_images(patches)
        for (owner, name), vector in zip(owners, features, strict=True):
            grouped[owner].append((name, vector))
        return grouped

    # ---- logits and calibration ---------------------------------------

    def _scale_and_bias(self) -> tuple[float, float]:
        model = self.bundle.clip_model
        scale = float(model.logit_scale.exp())
        bias = float(model.logit_bias) if hasattr(model, "logit_bias") else 0.0
        return scale, bias

    def logits_from_similarity(self, similarity: Any):
        """Apply the model's learned temperature and bias to cosine similarities."""
        scale, bias = self._scale_and_bias()
        return similarity * scale + bias

    def score(self, crops: Sequence[Any], prompts: Sequence[str]) -> list[list[float]]:
        """One logit vector per crop, aligned with ``prompts``.

        Text is encoded once and images in chunks, so a query over thousands of
        crops no longer pushes them all through the model in one batch.
        """
        if not crops:
            return []
        images = self.encode_images(crops)
        texts = self.encode_texts(prompts)
        return self.logits_from_similarity(images @ texts.T).tolist()

    def food_gate_scores(self, whole_vectors: Any, config: Config) -> list[float]:
        """The zero-inference food gate: stored image vectors against fixed prompts."""
        from vie.vlm import gate_score

        prompts = config.food.gate_prompts()
        logits = self.logits_from_similarity(whole_vectors @ self.encode_texts(prompts).T)
        n_positive = len(config.food.clip_gate_positive)
        return [gate_score(row.tolist(), n_positive, config.score_mode) for row in logits]

    def calibrated(self, cosine: float) -> float | None:
        """SigLIP's probability for a cosine; ``None`` for models without a bias."""
        if not self.is_siglip:
            return None
        scale, bias = self._scale_and_bias()
        return similarity_to_probability(cosine, scale, bias)


#: Names the rest of the code and earlier notebooks use for the two roles.
ClipCropScorer = VisionLanguageEncoder
ClipEmbedder = VisionLanguageEncoder


def classification_transform(size: int = 224):
    """The reference preprocessing chain for torchvision ImageNet weights.

    ``Resize(256) + CenterCrop(224)`` preserves aspect ratio. The tuple form
    ``Resize((224, 224))`` used in the original squashes it, which is worst on
    exactly the tall and wide crops (giraffe, dachshund) that fine-grained
    recognition is most sensitive to.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
