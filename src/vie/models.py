"""Model loading and the concrete encoders the search paths depend on.

Everything here is deliberately lazy. Importing :mod:`vie.models` must not pull
in torch, so the core logic and its regression tests stay installable without a
multi-gigabyte dependency tree.

Three problems from the original are fixed at this layer:

* **Unpinned supply chain.** ``torch.hub.load('ultralytics/yolov5', ...,
  trust_repo=True)`` cloned a third-party repository at runtime and executed it
  with the confirmation suppressed, so upstream changes altered behaviour
  silently. :data:`YOLOV5_PIN` pins the revision.
* **Device/dtype drift.** All casting goes through :class:`vie.device.Runtime`.
* **Preprocessing mismatch.** The original hard-coded a ``Resize((224, 224))``
  squash that does not match the weights it was feeding. Each checkpoint is now
  asked for its own reference transform via ``weights.transforms()``, so the
  preprocessing cannot drift from the model again — including when the
  classifier is swapped, since different checkpoints use different crops.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.device import Runtime, resolve
from vie.logging_setup import get_logger

log = get_logger("models")

#: Pinned so a change to yolov5 master cannot alter detection behaviour.
YOLOV5_PIN = "v7.0"

MEGADETECTOR_URL = (
    "https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
)


def download_if_missing(path: str | Path, url: str) -> Path:
    """Fetch a weight file once, reporting progress rather than failing silently."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, path)
    urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed, vetted URL
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
    """Load an ImageNet classifier plus its reference transform and labels.

    Using the weights' own ``transforms()`` is what keeps preprocessing correct:
    each checkpoint declares the resize and crop it was evaluated with, so there
    is no chance of repeating the original's mistake of hard-coding a
    ``Resize((224, 224))`` squash that does not match the weights.
    """
    from torchvision import models as tv

    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {name!r}; expected one of {sorted(CLASSIFIERS)}")

    weights_attr, published_acc = CLASSIFIERS[name]
    weights = getattr(tv, weights_attr).DEFAULT
    model = getattr(tv, name)(weights=weights).to(runtime.device).eval()
    log.info("loaded %s (%.1f%% published top-1)", name, published_acc)
    return model, weights.transforms(), list(weights.meta["categories"])


@dataclass
class ModelBundle:
    """Every model the pipeline uses, loaded once and kept resident."""

    runtime: Runtime
    clip_model: Any            # SigLIP or CLIP; see vie.vlm
    clip_processor: Any
    face_app: Any
    animal_detector: Any
    object_detector: Any
    classifier: Any = None
    classifier_transform: Any = None
    class_names: list[str] = None
    score_mode: str = "softmax"


def load_all(config: Config) -> ModelBundle:
    """Load every model onto the resolved device.

    Weights are frozen — nothing here is fine-tuned. The engineering value of
    this project is in how these are combined and indexed, not in training.
    """
    import torch  # noqa: F401 - imported for its side effect of initialising CUDA
    from insightface.app import FaceAnalysis
    from transformers import AutoModel, AutoProcessor
    from ultralytics import RTDETR

    runtime = resolve(config.device, config.half_precision)
    log.info("device=%s half=%s", runtime.device, runtime.use_half)

    vlm_name = config.models["vision_language"]
    clip_model = runtime.prepare_model(AutoModel.from_pretrained(vlm_name))
    clip_processor = AutoProcessor.from_pretrained(vlm_name)
    log.info("loaded %s (score mode: %s)", vlm_name, config.score_mode)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if runtime.is_cuda
        else ["CPUExecutionProvider"]
    )
    face_app = FaceAnalysis(name=config.models["face_pack"], providers=providers)
    face_app.prepare(ctx_id=0 if runtime.is_cuda else -1,
                     det_size=(config.face.det_size_large, config.face.det_size_large))
    log.info("loaded InsightFace %s", config.models["face_pack"])

    weights = download_if_missing(config.models["animal_detector"], MEGADETECTOR_URL)
    animal_detector = torch.hub.load(
        f"ultralytics/yolov5:{YOLOV5_PIN}", "custom", path=str(weights), trust_repo=True
    ).to(runtime.device)
    log.info("loaded MegaDetector (yolov5 pinned at %s)", YOLOV5_PIN)

    object_detector = RTDETR(config.models["object_detector"])
    log.info("loaded RT-DETR")

    classifier, transform, class_names = load_classifier(config.animal.classifier, runtime)

    return ModelBundle(
        runtime=runtime,
        clip_model=clip_model,
        clip_processor=clip_processor,
        face_app=face_app,
        animal_detector=animal_detector,
        object_detector=object_detector,
        classifier=classifier,
        classifier_transform=transform,
        class_names=class_names,
        score_mode=config.score_mode,
    )


class SpeciesClassifier:
    """Classifies animal crops into ImageNet-1k, batched.

    Runs at index time, once per crop, so its cost is paid on ingest rather than
    on every query. That is what makes a stronger, heavier backbone affordable:
    the original ran its classifier inside the query loop, which is why it had
    to settle for the smallest available model.
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


class ClipCropScorer:
    """Scores image crops against text prompts — the shared recognition path.

    Satisfies :class:`vie.search.animal.CropScorer` and the equivalent protocol
    in the food path.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def score(self, crops: Sequence[Any], prompts: Sequence[str]) -> list[list[float]]:
        import torch

        if not crops:
            return []
        processor, model = self.bundle.clip_processor, self.bundle.clip_model
        inputs = processor(
            text=list(prompts), images=list(crops), return_tensors="pt", padding=True
        )
        inputs = self.bundle.runtime.cast_inputs(dict(inputs))
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        return logits.float().cpu().tolist()


class ClipEmbedder:
    """Produces the unit-length image and text vectors stored in the index.

    ``get_image_features`` and ``get_text_features`` both apply their projection
    heads, so the two land in the same 768-d space for ViT-L/14. Normalising
    here is what lets retrieval use a plain dot product as cosine similarity.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def _normalise(self, tensor: Any) -> Any:
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True)

    def encode_regions(self, image: Any, regions: Sequence[tuple[str, Any]]):
        """Encode several crops of one image in a single batched forward pass.

        Returns ``(name, vector)`` pairs. Batching matters here: multi-crop
        multiplies the number of encodes per image, and one forward pass over
        six crops is far cheaper than six passes over one.
        """
        import torch
        from PIL import Image as PILImage

        if not regions:
            return []
        patches = []
        names = []
        for name, (x1, y1, x2, y2) in regions:
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            import cv2

            patches.append(PILImage.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
            names.append(name)
        if not patches:
            return []

        inputs = self.bundle.clip_processor(images=patches, return_tensors="pt")
        inputs = self.bundle.runtime.cast_inputs(dict(inputs))
        with torch.no_grad():
            features = self.bundle.clip_model.get_image_features(**inputs)
        features = self._normalise(features).float().cpu().numpy()
        return list(zip(names, features, strict=True))

    def encode_image(self, image: Any):
        import torch

        inputs = self.bundle.clip_processor(images=image, return_tensors="pt")
        inputs = self.bundle.runtime.cast_inputs(dict(inputs))
        with torch.no_grad():
            features = self.bundle.clip_model.get_image_features(**inputs)
        return self._normalise(features).float().cpu().numpy().ravel()

    def encode_text(self, text: str):
        import torch

        inputs = self.bundle.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = self.bundle.runtime.cast_inputs(dict(inputs))
        with torch.no_grad():
            features = self.bundle.clip_model.get_text_features(**inputs)
        return self._normalise(features).float().cpu().numpy().ravel()


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
