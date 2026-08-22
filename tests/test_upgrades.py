"""Tests for the four module upgrades: face gate, classifier, food gate, multi-crop."""

from __future__ import annotations

import pytest

from vie.config import Config, ConfigError
from vie.quality import crop_regions, expected_vectors, is_usable_face
from vie.vlm import default_score_mode, gate_score, hypothesis_score, is_siglip

# ── face: permissive quality gate ──────────────────────────────────────


def test_good_face_passes(config: Config) -> None:
    assert is_usable_face(0.95, (0, 0, 120, 140), config.face.min_det_score,
                          config.face.min_face_px)


def test_marginal_face_still_passes(config: Config) -> None:
    """Deliberately permissive: if RetinaFace is at all confident, index it."""
    assert is_usable_face(0.35, (0, 0, 30, 34), config.face.min_det_score,
                          config.face.min_face_px)


def test_very_low_confidence_is_rejected(config: Config) -> None:
    assert not is_usable_face(0.10, (0, 0, 120, 140), config.face.min_det_score,
                              config.face.min_face_px)


def test_tiny_face_is_rejected(config: Config) -> None:
    """A 12px face cannot carry identity; its embedding matches the wrong person."""
    assert not is_usable_face(0.99, (0, 0, 12, 14), config.face.min_det_score,
                              config.face.min_face_px)


def test_gate_uses_the_shorter_side(config: Config) -> None:
    # Tall but narrow: a sliver of a face, not a face.
    assert not is_usable_face(0.9, (0, 0, 15, 200), config.face.min_det_score,
                              config.face.min_face_px)


def test_face_gate_is_permissive_by_default(config: Config) -> None:
    """Guards the intent: this gate must not become a selective filter."""
    assert config.face.min_det_score <= 0.35
    assert config.face.min_face_px <= 32


# ── animal: classifier choice ──────────────────────────────────────────


def test_classifier_is_configurable(config: Config) -> None:
    assert config.animal.classifier


def test_classifier_vocabulary_is_unchanged_by_the_upgrade(config: Config) -> None:
    """Every candidate is an ImageNet-1k model, so species coverage is identical.

    This is the point the upgrade turns on: swapping the classifier changes
    accuracy, never which species can be named.
    """
    imagenet_models = {
        "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny", "efficientnet_v2_s",
    }
    assert config.animal.classifier in imagenet_models


# ── food: the zero-cost CLIP gate ──────────────────────────────────────


def test_food_clip_gate_is_enabled(config: Config) -> None:
    assert config.food.clip_gate_enabled


def test_gate_prompts_put_positives_first(config: Config) -> None:
    prompts = config.food.gate_prompts()
    assert prompts[: len(config.food.clip_gate_positive)] == config.food.clip_gate_positive


def test_food_scores_positive_when_the_food_prompt_wins(config: Config) -> None:
    n_pos = len(config.food.clip_gate_positive)
    logits = [4.0] * n_pos + [-4.0] * len(config.food.clip_gate_negative)
    assert gate_score(logits, n_pos, config.score_mode) > config.food.clip_gate_threshold


def test_food_scores_negative_on_a_photo_without_food(config: Config) -> None:
    n_pos = len(config.food.clip_gate_positive)
    logits = [-4.0] * n_pos + [4.0] * len(config.food.clip_gate_negative)
    assert gate_score(logits, n_pos, config.score_mode) < 0


def test_gate_needs_both_sides() -> None:
    with pytest.raises(ValueError, match="positive"):
        gate_score([1.0, 2.0], 2, "sigmoid")
    with pytest.raises(ValueError, match="positive"):
        gate_score([1.0, 2.0], 0, "sigmoid")


# ── text: multi-crop layout ────────────────────────────────────────────


def test_default_layout_produces_six_vectors(config: Config) -> None:
    assert expected_vectors(config.crop_layout) == 6


def test_whole_frame_is_always_first() -> None:
    name, box = crop_regions(800, 600, "1+2x2+center")[0]
    assert name == "whole" and box == (0, 0, 800, 600)


def test_quadrants_tile_the_frame() -> None:
    regions = dict(crop_regions(800, 600, "1+2x2"))
    assert regions["tl"] == (0, 0, 400, 300)
    assert regions["br"] == (400, 300, 800, 600)


def test_centre_crop_covers_the_middle() -> None:
    """The 2x2 grid cuts through the centre, which is where the subject usually is."""
    regions = dict(crop_regions(800, 600, "1+2x2+center"))
    x1, y1, x2, y2 = regions["center"]
    assert x1 < 400 < x2 and y1 < 300 < y2


def test_single_layout_is_the_original_behaviour() -> None:
    assert expected_vectors("1") == 1


def test_invalid_image_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        crop_regions(0, 100, "1")


# ── vlm: CLIP and SigLIP are not interchangeable ───────────────────────


def test_siglip_is_detected() -> None:
    assert is_siglip("google/siglip-base-patch16-224")
    assert not is_siglip("openai/clip-vit-large-patch14")


def test_default_mode_follows_the_model() -> None:
    assert default_score_mode("google/siglip-base-patch16-224") == "sigmoid"
    assert default_score_mode("openai/clip-vit-large-patch14") == "softmax"


def test_confident_hypothesis_scores_high_in_both_modes() -> None:
    for mode in ("softmax", "sigmoid"):
        assert hypothesis_score([8.0, -4.0, -4.0], mode) > 0.9


def test_losing_hypothesis_scores_low_in_both_modes() -> None:
    for mode in ("softmax", "sigmoid"):
        assert hypothesis_score([-4.0, 8.0, -4.0], mode) < 0.2


def test_sigmoid_vetoes_a_stronger_rival() -> None:
    """SigLIP probabilities are absolute, so a rival must actively suppress."""
    alone = hypothesis_score([2.0, -8.0], "sigmoid")
    contested = hypothesis_score([2.0, 3.0], "sigmoid")
    assert contested < alone


def test_single_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="rival"):
        hypothesis_score([5.0], "sigmoid")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown score mode"):
        hypothesis_score([1.0, 2.0], "argmax")


# ── the pairing must stay consistent ───────────────────────────────────


def test_shipped_config_pairs_model_and_mode_correctly(config: Config) -> None:
    assert config.score_mode == default_score_mode(config.models["vision_language"])


def test_siglip_with_softmax_is_rejected(raw_config: dict) -> None:
    """Scoring SigLIP with softmax discards the learned bias that makes its
    output absolute — a silent miscalibration of every threshold."""
    import copy

    broken = copy.deepcopy(raw_config)
    broken["models"]["score_mode"] = "softmax"
    with pytest.raises(ConfigError, match="SigLIP"):
        Config.from_dict(broken).validate()


def test_clip_with_sigmoid_is_rejected(raw_config: dict) -> None:
    import copy

    broken = copy.deepcopy(raw_config)
    broken["models"]["vision_language"] = "openai/clip-vit-large-patch14"
    broken["models"]["score_mode"] = "sigmoid"
    with pytest.raises(ConfigError, match="CLIP"):
        Config.from_dict(broken).validate()
