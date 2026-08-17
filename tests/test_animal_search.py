"""Tests for the hybrid animal path: classifier first, CLIP as fallback."""

from __future__ import annotations

import pytest

from vie.search.animal import (
    Candidate,
    Prediction,
    choose_route,
    rank_from_index,
    rank_with_clip,
    select_crops,
)

CLASSES = [
    "zebra", "lion", "sea lion", "African elephant", "Indian elephant",
    "golden retriever", "Bedlington terrier", "tabby, tabby cat",
    "Egyptian cat", "timber wolf", "ant", "bee", "mailbox", "bathtub",
]


class FakeScorer:
    def __init__(self, rows):
        self.rows = rows
        self.seen_prompts = None

    def score(self, crops, prompts):
        self.seen_prompts = list(prompts)
        return self.rows[: len(crops)]


def _candidate(name="a.jpg", box=(0, 0, 200, 200), conf=0.9) -> Candidate:
    return Candidate(name, f"/g/{name}", box, conf)


def _prediction(name="a.jpg", class_id=0, prob=0.9, box=(0, 0, 200, 200)) -> Prediction:
    return Prediction(name, f"/g/{name}", box, class_id, prob)


# ── routing: which engine answers ──────────────────────────────────────


@pytest.mark.parametrize("query", ["zebra", "lion", "golden retriever", "elephant"])
def test_in_vocabulary_queries_use_the_cheap_classifier(query: str) -> None:
    """~373x less compute per crop than CLIP, and served from the index."""
    assert choose_route(query, CLASSES).engine == "classifier"


@pytest.mark.parametrize("query", ["quokka", "fennec fox", "axolotl"])
def test_out_of_vocabulary_queries_fall_back_to_clip(query: str) -> None:
    """No supervised ImageNet model can ever return these — CLIP is the only option."""
    assert choose_route(query, CLASSES).engine == "clip"


def test_route_explains_itself() -> None:
    assert "no inference" in choose_route("zebra", CLASSES).explain()
    assert "CLIP" in choose_route("quokka", CLASSES).explain()


def test_elephant_routes_to_elephants_not_insects() -> None:
    """The original sent 'African elephant' to the Insect bucket via 'ant'."""
    route = choose_route("elephant", CLASSES)
    names = {CLASSES[i] for i in route.resolution.class_ids}
    assert "African elephant" in names and "ant" not in names


# ── path 1: served from the index, no model runs ───────────────────────


def test_indexed_query_returns_matching_class() -> None:
    zebra = CLASSES.index("zebra")
    results = rank_from_index(
        choose_route("zebra", CLASSES).resolution,
        [_prediction(class_id=zebra, prob=0.88)],
        _config(),
        CLASSES,
    )
    assert len(results) == 1
    assert results[0].label == "zebra"


def test_indexed_query_ignores_other_species() -> None:
    lion = CLASSES.index("lion")
    results = rank_from_index(
        choose_route("zebra", CLASSES).resolution,
        [_prediction(class_id=lion, prob=0.99)],
        _config(),
        CLASSES,
    )
    assert results == []


def test_low_probability_prediction_is_rejected(config) -> None:
    """The original accepted a 21.5% top-1 over 1000 classes as a match."""
    zebra = CLASSES.index("zebra")
    below = config.animal.classifier_threshold - 0.05
    results = rank_from_index(
        choose_route("zebra", CLASSES).resolution,
        [_prediction(class_id=zebra, prob=below)],
        config,
        CLASSES,
    )
    assert results == []


def test_generic_query_matches_any_breed(config) -> None:
    retriever = CLASSES.index("golden retriever")
    results = rank_from_index(
        choose_route("dog", CLASSES).resolution,
        [_prediction(class_id=retriever, prob=0.8)],
        config,
        CLASSES,
    )
    assert len(results) == 1


def test_one_image_with_two_animals_returns_once(config) -> None:
    zebra = CLASSES.index("zebra")
    results = rank_from_index(
        choose_route("zebra", CLASSES).resolution,
        [
            _prediction(class_id=zebra, prob=0.40, box=(0, 0, 100, 100)),
            _prediction(class_id=zebra, prob=0.95, box=(200, 200, 300, 300)),
        ],
        config,
        CLASSES,
    )
    assert len(results) == 1
    assert results[0].box == (200, 200, 300, 300), "kept the weaker instance"


def test_index_path_refuses_an_unresolved_query(config) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        rank_from_index(choose_route("quokka", CLASSES).resolution, [], config, CLASSES)


# ── path 2: CLIP fallback ──────────────────────────────────────────────


def test_clip_query_is_the_first_prompt(config) -> None:
    scorer = FakeScorer([[10.0, 0.0, 0.0]])
    rank_with_clip("quokka", [_candidate()], [object()], scorer, config)
    assert scorer.seen_prompts[0] == "a photo of a quokka"
    assert len(scorer.seen_prompts) > 1, "a rival prompt is required"


def test_clip_confident_match_is_returned(config) -> None:
    results = rank_with_clip(
        "quokka", [_candidate()], [object()], FakeScorer([[12.0, 0.0, 0.0]]), config
    )
    assert len(results) == 1 and results[0].label == "quokka"


def test_clip_query_losing_to_a_rival_is_rejected(config) -> None:
    assert rank_with_clip(
        "quokka", [_candidate()], [object()], FakeScorer([[0.0, 12.0, 0.0]]), config
    ) == []


def test_clip_ambiguous_match_is_rejected(config) -> None:
    assert rank_with_clip(
        "quokka", [_candidate()], [object()], FakeScorer([[5.0, 5.0, 5.0]]), config
    ) == []


def test_clip_empty_query_is_rejected(config) -> None:
    with pytest.raises(ValueError, match="empty query"):
        rank_with_clip("  ", [_candidate()], [object()], FakeScorer([[1.0, 0.0]]), config)


def test_clip_mismatched_lengths_are_rejected(config) -> None:
    with pytest.raises(ValueError, match="candidates"):
        rank_with_clip("quokka", [_candidate()], [object(), object()], FakeScorer([]), config)


# ── crop selection, shared by both paths ───────────────────────────────


def test_small_crops_are_dropped_before_recognition(config) -> None:
    assert select_crops([_candidate(box=(0, 0, 12, 9))], config, (1000, 1000)) == []


def test_low_detector_confidence_is_dropped(config) -> None:
    assert select_crops([_candidate(conf=0.1)], config, (1000, 1000)) == []


def test_padding_is_applied_and_clamped(config) -> None:
    kept = select_crops([_candidate(box=(5, 5, 105, 105))], config, (1000, 1000))
    assert kept[0].box[0] == 0      # clamped, not negative
    assert kept[0].box[2] > 105     # padded


@pytest.fixture()
def _cfg(config):
    return config


def _config():
    from vie.config import Config

    return Config.load()
