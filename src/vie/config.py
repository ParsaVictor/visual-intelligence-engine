"""Configuration loading and validation.

The original notebooks declared detection thresholds and COCO class lists twice
— once in the indexing pass, once in the search pass — and the two copies
drifted. Two real defects came out of that drift:

* the food search pass included COCO class 56 (``chair``), which the indexing
  pass had deliberately excluded, so furniture was cropped and sent to CLIP;
* the animal gate ran at confidence 0.40 while the search ran at 0.30, even
  though the gate is a hard filter — so the looser search threshold could not
  add any recall and only admitted low-confidence noise.

Both are instances of one invariant being violated:

    the index-time gate must be at least as permissive as the search pass

:meth:`Config.validate` enforces that at load time, which turns a whole class of
silent drift into an immediate, loud failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vie.vectorstore import SearchConfig
from vie.video.shots import ShotConfig

#: Ships inside the package. It used to live at the repository root, so anything
#: but an editable install failed with "config not found" on the first command.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


def merge(base: dict, extra: dict) -> dict:
    """Deep-merge ``extra`` over ``base`` without mutating either."""
    import copy

    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def with_model_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply ``model_overrides[<selected model>]`` on top of the rest of the config.

    Thresholds are on each model's own scale: SigLIP 2's logit margins run narrower
    than SigLIP's, so the same number means a stricter cut. Each model's measured
    thresholds live under its name and follow it automatically, so switching models
    never silently keeps another model's thresholds.
    """
    model = (raw.get("models") or {}).get("vision_language", "")
    override = (raw.get("model_overrides") or {}).get(model)
    return merge(raw, override) if override else raw


@dataclass(frozen=True)
class FaceConfig:
    match_threshold: float
    min_det_score: float
    min_face_px: int
    det_size_large: int
    det_size_small: int
    small_image_px: int
    max_index_dimension: int
    # Embed each face and its mirror image and average the two. Measured in
    # scripts/face_benchmark.py; costs one extra recognition pass per face.
    flip_tta: bool = True

    @property
    def recipe(self) -> str:
        """What produced the stored face vectors, so a change can be noticed."""
        return "arcface" + ("+flip" if self.flip_tta else "")


@dataclass(frozen=True)
class AnimalConfig:
    animal_class_id: int
    gate_confidence: float
    search_confidence: float
    padding_ratio: float
    min_crop_px: int
    classifier: str
    classifier_top_k: int
    classifier_threshold: float
    match_threshold: float
    negative_prompts: list[str]
    prompt_template: str

    def prompts_for(self, query: str) -> list[str]:
        """Build the contrastive prompt list for a query.

        The first entry is the hypothesis; the rest are rivals. Scoring is a
        softmax over all of them, so the model has to prefer the query over an
        explicit "something else" option rather than merely clearing a fixed
        cosine threshold.
        """
        return [self.prompt_template.format(query=query), *self.negative_prompts]


@dataclass(frozen=True)
class FoodConfig:
    gate_classes: list[int]
    search_classes: list[int]
    gate_confidence: float
    search_confidence: float
    min_crop_px: int
    padding_px: int
    match_threshold: float
    negative_prompts: list[str]
    face_overlap_reject: float
    clip_gate_enabled: bool
    clip_gate_threshold: float
    clip_gate_positive: list[str]
    clip_gate_negative: list[str]

    def prompts_for(self, query: str) -> list[str]:
        return [query, *self.negative_prompts]

    def gate_prompts(self) -> list[str]:
        """Positive prompts first, then negatives, for the zero-cost gate."""
        return [*self.clip_gate_positive, *self.clip_gate_negative]


@dataclass(frozen=True)
class IndexingConfig:
    batch_size: int = 8
    workers: int = 8
    commit_every: int = 50
    recursive: bool = True
    trust_mtime: bool = True


@dataclass(frozen=True)
class VideoConfig:
    sample_seconds: float = 1.0
    max_samples: int = 3600
    max_duration: float = 0
    max_side: int = 960
    keyframe_seconds: float = 6.0
    keyframes_per_shot: int = 3
    max_keyframes: int = 240
    shots: ShotConfig = field(default_factory=ShotConfig)
    faces: bool = True
    animals: bool = True
    food: bool = True
    captions: bool = False
    transcript: bool = False
    caption_model: str = "Salesforce/blip-image-captioning-base"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    face_cluster_threshold: float = 0.45
    face_exemplars: int = 4
    min_face_sightings: int = 2
    presence_gap: float = 12.0
    label_threshold: float = 0.40
    prompt_template: str = "a video frame showing {label}"
    taxonomy: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> VideoConfig:
        values = dict(raw)
        if "shots" in values:
            values["shots"] = ShotConfig(**values["shots"])
        return cls(**values)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)
    face: FaceConfig
    animal: AnimalConfig
    food: FoodConfig
    indexing: IndexingConfig
    search: SearchConfig
    video: VideoConfig
    seed: int
    device: str
    half_precision: bool
    low_memory: bool
    top_k: int
    multi_crop: bool
    crop_layout: str
    score_mode: str
    gallery_path: Path
    videos_path: Path
    database_path: Path
    model_cache: str
    models: dict[str, str]
    #: Phrasings a free-text query is averaged over; see ``encode_query``.
    query_templates: tuple[str, ...] = ("{query}",)
    #: Below this calibrated probability a free-text result is not shown at all.
    min_probability: float = 0.0

    # ---- construction -------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.is_file():
            raise FileNotFoundError(f"config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        config = cls.from_dict(raw)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = with_model_overrides(raw)
        return cls(
            raw=raw,
            face=FaceConfig(**raw["face"]),
            animal=AnimalConfig(**raw["animal"]),
            food=FoodConfig(**raw["food"]),
            indexing=IndexingConfig(**(raw.get("indexing") or {})),
            search=SearchConfig(**(raw.get("search") or {})),
            video=VideoConfig.from_dict(raw.get("video") or {}),
            seed=int(raw["project"]["seed"]),
            device=str(raw["runtime"]["device"]),
            half_precision=bool(raw["runtime"]["half_precision"]),
            low_memory=bool(raw["runtime"].get("low_memory", False)),
            top_k=int(raw["text"]["top_k"]),
            multi_crop=bool(raw["text"].get("multi_crop", False)),
            crop_layout=str(raw["text"].get("crop_layout", "1")),
            score_mode=str(raw["models"].get("score_mode", "softmax")),
            gallery_path=Path(raw["paths"]["gallery"]),
            videos_path=Path(raw["paths"].get("videos", "data/videos")),
            database_path=Path(raw["paths"]["database"]),
            model_cache=str(raw["paths"].get("model_cache", "") or ""),
            models=dict(raw["models"]),
            query_templates=tuple(raw["text"].get("query_templates") or ("{query}",)),
            min_probability=float(raw["text"].get("min_probability", 0.0)),
        )

    # ---- validation ---------------------------------------------------

    def validate(self) -> None:
        """Enforce the index/search invariants. Raises :class:`ConfigError`."""
        problems: list[str] = []
        if not self.query_templates or any("{query}" not in t for t in self.query_templates):
            problems.append("text.query_templates must each contain {query}")

        # A candidate the gate rejects is unreachable at search time, so the
        # gate must never be the stricter of the two.
        if self.animal.gate_confidence > self.animal.search_confidence:
            problems.append(
                f"animal.gate_confidence ({self.animal.gate_confidence}) is stricter than "
                f"animal.search_confidence ({self.animal.search_confidence}); an image the "
                f"gate rejects can never be reached by search"
            )
        if self.food.gate_confidence > self.food.search_confidence:
            problems.append(
                f"food.gate_confidence ({self.food.gate_confidence}) is stricter than "
                f"food.search_confidence ({self.food.search_confidence})"
            )

        # Searching for a class the gate never looked for is dead weight at
        # best and, as with COCO 56 (chair), actively harmful at worst.
        stray = sorted(set(self.food.search_classes) - set(self.food.gate_classes))
        if stray:
            problems.append(
                f"food.search_classes contains {stray} which food.gate_classes does not; "
                f"those classes were never indexed"
            )

        if not 0.0 <= self.food.face_overlap_reject <= 1.0:
            problems.append("food.face_overlap_reject must be a fraction in [0, 1]")

        for name, value in (
            ("face.match_threshold", self.face.match_threshold),
            ("animal.match_threshold", self.animal.match_threshold),
            ("animal.classifier_threshold", self.animal.classifier_threshold),
            ("food.match_threshold", self.food.match_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                problems.append(f"{name} must be in [0, 1], got {value}")

        if not self.animal.negative_prompts:
            problems.append("animal.negative_prompts must not be empty; without a rival "
                            "hypothesis the softmax degenerates to always-1.0")
        if not self.food.negative_prompts:
            problems.append("food.negative_prompts must not be empty")

        # SigLIP and CLIP are trained with different objectives: SigLIP uses an
        # independent sigmoid per pair, CLIP a softmax over the batch. Scoring
        # one with the other's rule silently miscalibrates every threshold.
        if self.score_mode not in {"softmax", "sigmoid"}:
            problems.append(
                f"models.score_mode must be 'softmax' (CLIP) or 'sigmoid' (SigLIP), "
                f"got {self.score_mode!r}"
            )
        model = self.models.get("vision_language", "")
        if "siglip" in model.lower() and self.score_mode != "sigmoid":
            problems.append(
                f"{model} is a SigLIP model but score_mode is {self.score_mode!r}; "
                f"SigLIP logits carry a learned bias and are meant for sigmoid"
            )
        if "clip-vit" in model.lower() and self.score_mode != "softmax":
            problems.append(
                f"{model} is a CLIP model but score_mode is {self.score_mode!r}"
            )

        if self.food.clip_gate_enabled and not self.food.clip_gate_positive:
            problems.append("food.clip_gate_positive must not be empty when the gate is on")
        if self.food.clip_gate_enabled and not self.food.clip_gate_negative:
            problems.append("food.clip_gate_negative must not be empty when the gate is on")

        for name, count in (
            ("indexing.batch_size", self.indexing.batch_size),
            ("indexing.workers", self.indexing.workers),
            ("indexing.commit_every", self.indexing.commit_every),
            ("search.crop_batch", self.search.crop_batch),
            ("search.ann_candidates", self.search.ann_candidates),
            ("face.max_index_dimension", self.face.max_index_dimension),
        ):
            if count < 1:
                problems.append(f"{name} must be at least 1, got {count}")
        if self.search.backend not in {"auto", "exact", "hnsw"}:
            problems.append(
                f"search.backend must be auto, exact or hnsw, got {self.search.backend!r}"
            )

        video = self.video
        if "{label}" not in video.prompt_template:
            problems.append("video.prompt_template must contain {label}")
        if video.sample_seconds <= 0 or video.keyframe_seconds <= 0:
            problems.append("video.sample_seconds and video.keyframe_seconds must be positive")
        if video.keyframes_per_shot < 1 or video.max_keyframes < 1 or video.max_samples < 1:
            problems.append("video keyframe and sample budgets must be at least 1")
        if not 0.0 <= video.label_threshold <= 1.0:
            problems.append("video.label_threshold must be in [0, 1]")
        for group, labels in video.taxonomy.items():
            if len(labels) < 2:
                problems.append(
                    f"video.taxonomy.{group} needs at least two labels: they are scored "
                    f"against each other, so a lone label would always win"
                )
        if video.whisper_device not in {"auto", "cpu", "cuda"}:
            problems.append("video.whisper_device must be auto, cpu or cuda")

        if problems:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))
