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

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


@dataclass(frozen=True)
class FaceConfig:
    match_threshold: float
    det_size_large: int
    det_size_small: int
    small_image_px: int
    max_index_dimension: int


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

    def prompts_for(self, query: str) -> list[str]:
        return [query, *self.negative_prompts]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)
    face: FaceConfig
    animal: AnimalConfig
    food: FoodConfig
    seed: int
    device: str
    half_precision: bool
    top_k: int
    gallery_path: Path
    database_path: Path
    models: dict[str, str]

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
        return cls(
            raw=raw,
            face=FaceConfig(**raw["face"]),
            animal=AnimalConfig(**raw["animal"]),
            food=FoodConfig(**raw["food"]),
            seed=int(raw["project"]["seed"]),
            device=str(raw["runtime"]["device"]),
            half_precision=bool(raw["runtime"]["half_precision"]),
            top_k=int(raw["text"]["top_k"]),
            gallery_path=Path(raw["paths"]["gallery"]),
            database_path=Path(raw["paths"]["database"]),
            models=dict(raw["models"]),
        )

    # ---- validation ---------------------------------------------------

    def validate(self) -> None:
        """Enforce the index/search invariants. Raises :class:`ConfigError`."""
        problems: list[str] = []

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

        if problems:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))
