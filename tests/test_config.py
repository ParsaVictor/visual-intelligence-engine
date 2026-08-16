"""Regression tests for P1-4: index and search must not use different constants."""

from __future__ import annotations

import copy

import pytest

from vie.config import Config, ConfigError


def test_shipped_config_is_valid(config: Config) -> None:
    config.validate()


def test_food_search_classes_are_a_subset_of_gate_classes(config: Config) -> None:
    """The original search pass looked for COCO 56 (chair), which indexing excluded."""
    assert set(config.food.search_classes) <= set(config.food.gate_classes)


def test_chair_is_not_searched(config: Config) -> None:
    """COCO 56 is 'chair'. It cropped furniture into CLIP in the original code."""
    assert 56 not in config.food.search_classes
    assert 56 not in config.food.gate_classes


def test_gate_is_never_stricter_than_search(config: Config) -> None:
    """A candidate the gate rejects is unreachable, so a stricter gate is a silent recall bug."""
    assert config.animal.gate_confidence <= config.animal.search_confidence
    assert config.food.gate_confidence <= config.food.search_confidence


def test_stricter_animal_gate_is_rejected(raw_config: dict) -> None:
    """This is exactly the original 0.40 gate / 0.30 search inversion."""
    broken = copy.deepcopy(raw_config)
    broken["animal"]["gate_confidence"] = 0.40
    broken["animal"]["search_confidence"] = 0.30
    with pytest.raises(ConfigError, match="stricter"):
        Config.from_dict(broken).validate()


def test_stray_food_search_class_is_rejected(raw_config: dict) -> None:
    broken = copy.deepcopy(raw_config)
    broken["food"]["search_classes"] = [*broken["food"]["search_classes"], 56]
    with pytest.raises(ConfigError, match=r"\[56\]"):
        Config.from_dict(broken).validate()


def test_empty_negative_prompts_are_rejected(raw_config: dict) -> None:
    """Without a rival hypothesis the contrastive softmax is always 1.0."""
    broken = copy.deepcopy(raw_config)
    broken["animal"]["negative_prompts"] = []
    with pytest.raises(ConfigError, match="negative_prompts"):
        Config.from_dict(broken).validate()


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_out_of_range_threshold_is_rejected(raw_config: dict, value: float) -> None:
    broken = copy.deepcopy(raw_config)
    broken["face"]["match_threshold"] = value
    with pytest.raises(ConfigError, match="face.match_threshold"):
        Config.from_dict(broken).validate()


def test_prompt_construction_puts_the_query_first(config: Config) -> None:
    prompts = config.animal.prompts_for("red panda")
    assert prompts[0] == "a photo of a red panda"
    assert len(prompts) == 1 + len(config.animal.negative_prompts)
