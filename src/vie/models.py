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
* **Preprocessing mismatch.** The reference transform for the torchvision
  weights is ``Resize(256) + CenterCrop(224)``. The original wrote exactly that,
  then shadowed it with a ``Resize((224, 224))`` squash that distorts the
  non-square crops fine-grained recognition depends on.
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


@dataclass
class ModelBundle:
    """Every model the pipeline uses, loaded once and kept resident."""

    runtime: Runtime
    clip_model: Any
    clip_processor: Any
    face_app: Any
    animal_detector: Any
    object_detector: Any


def load_all(config: Config) -> ModelBundle:
    """Load every model onto the resolved device.

    Weights are frozen — nothing here is fine-tuned. The engineering value of
    this project is in how these are combined and indexed, not in training.
    """
    import torch  # noqa: F401 - imported for its side effect of initialising CUDA
    from insightface.app import FaceAnalysis
    from transformers import CLIPModel, CLIPProcessor
    from ultralytics import RTDETR

    runtime = resolve(config.device, config.half_precision)
    log.info("device=%s half=%s", runtime.device, runtime.use_half)

    clip_model = CLIPModel.from_pretrained(config.models["clip"])
    clip_model = runtime.prepare_model(clip_model)
    clip_processor = CLIPProcessor.from_pretrained(config.models["clip"])
    log.info("loaded CLIP %s", config.models["clip"])

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

    return ModelBundle(
        runtime=runtime,
        clip_model=clip_model,
        clip_processor=clip_processor,
        face_app=face_app,
        animal_detector=animal_detector,
        object_detector=object_detector,
    )


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
