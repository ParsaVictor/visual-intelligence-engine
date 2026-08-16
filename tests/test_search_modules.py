"""Tests for the face, text and food search paths."""

from __future__ import annotations

import numpy as np
import pytest

from vie.search import face, food, text


# ── face ──────────────────────────────────────────────────────────────
def _unit(values: list[float]) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_identical_face_is_matched(config) -> None:
    emb = _unit([1.0, 0.0, 0.0])
    results = face.rank(emb, [("a.jpg", (0, 0, 1, 1), emb, "/g/a.jpg")], config)
    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_face_is_rejected(config) -> None:
    results = face.rank(
        _unit([1.0, 0.0, 0.0]),
        [("a.jpg", (0, 0, 1, 1), _unit([0.0, 1.0, 0.0]), "/g/a.jpg")],
        config,
    )
    assert results == []


def test_face_threshold_boundary(config) -> None:
    """A similarity just under the threshold must not be returned."""
    query = _unit([1.0, 0.0])
    below = config.face.match_threshold - 0.05
    stored = np.array([below, np.sqrt(1 - below**2)], dtype=np.float32)
    assert face.rank(query, [("a.jpg", (0, 0, 1, 1), stored, "/g/a.jpg")], config) == []


def test_two_matching_faces_in_one_image_return_once(config) -> None:
    """P1-6 again, this time through the real face path."""
    emb = _unit([1.0, 0.0, 0.0])
    weaker = _unit([0.9, 0.4, 0.0])
    indexed = [
        ("a.jpg", (0, 0, 10, 10), weaker, "/g/a.jpg"),
        ("a.jpg", (50, 50, 60, 60), emb, "/g/a.jpg"),
    ]
    results = face.rank(emb, indexed, config)
    assert len(results) == 1
    assert results[0].box == (50, 50, 60, 60), "should keep the stronger face"


def test_face_dimension_mismatch_is_reported(config) -> None:
    """The original skipped mismatched rows silently."""
    with pytest.raises(ValueError, match="dimension mismatch"):
        face.rank(
            _unit([1.0, 0.0, 0.0]),
            [("a.jpg", (0, 0, 1, 1), _unit([1.0, 0.0]), "/g/a.jpg")],
            config,
        )


def test_empty_query_embedding_is_rejected(config) -> None:
    with pytest.raises(ValueError, match="empty query"):
        face.rank(np.array([], dtype=np.float32), [], config)


def test_largest_face_is_selected_as_the_subject() -> None:
    faces = [((0, 0, 10, 10), "small"), ((0, 0, 100, 100), "large")]
    assert face.select_query_face(faces)[1] == "large"


def test_no_faces_gives_no_subject() -> None:
    assert face.select_query_face([]) is None


def test_detection_size_scales_with_the_image(config) -> None:
    assert face.detection_size(1200, 900, config)[0] == config.face.det_size_large
    assert face.detection_size(200, 150, config)[0] == config.face.det_size_small


# ── text ──────────────────────────────────────────────────────────────
def test_text_search_ranks_by_similarity(config) -> None:
    query = _unit([1.0, 0.0])
    indexed = [
        ("far.jpg", _unit([0.0, 1.0]), "/g/far.jpg"),
        ("near.jpg", _unit([1.0, 0.1]), "/g/near.jpg"),
    ]
    assert [m.file_name for m in text.rank(query, indexed, config)] == ["near.jpg", "far.jpg"]


def test_text_search_respects_top_k(config) -> None:
    query = _unit([1.0, 0.0])
    indexed = [(f"{i}.jpg", _unit([1.0, i / 10]), f"/g/{i}.jpg") for i in range(10)]
    assert len(text.rank(query, indexed, config, top_k=3)) == 3


def test_text_search_returns_everything_when_gallery_is_small(config) -> None:
    query = _unit([1.0, 0.0])
    indexed = [("a.jpg", _unit([1.0, 0.0]), "/g/a.jpg")]
    assert len(text.rank(query, indexed, config, top_k=50)) == 1


def test_relative_confidence_sums_to_one() -> None:
    assert sum(text.relative_confidence([0.30, 0.25, 0.20])) == pytest.approx(1.0)


def test_relative_confidence_separates_a_clear_winner() -> None:
    """Raw cosines of 0.31 vs 0.20 look nearly identical until scaled."""
    conf = text.relative_confidence([0.31, 0.20, 0.19])
    assert conf[0] > 0.9


def test_relative_confidence_of_empty_input() -> None:
    assert text.relative_confidence([]) == []


# ── food ──────────────────────────────────────────────────────────────
class FakeScorer:
    def __init__(self, rows):
        self.rows = rows
        self.seen_prompts = None

    def score(self, crops, prompts):
        self.seen_prompts = list(prompts)
        return self.rows[: len(crops)]


def _fc(name="a.jpg", box=(0, 0, 200, 200), conf=0.9) -> food.Candidate:
    return food.Candidate(name, f"/g/{name}", box, conf)


def test_food_query_is_the_first_prompt(config) -> None:
    scorer = FakeScorer([[10.0, 0.0, 0.0]])
    food.rank("French fries", [_fc()], [object()], scorer, config)
    assert scorer.seen_prompts[0] == "French fries"
    assert len(scorer.seen_prompts) == 3, "two rival prompts expected"


def test_confident_dish_is_returned(config) -> None:
    results = food.rank("pizza", [_fc()], [object()], FakeScorer([[12.0, 0.0, 0.0]]), config)
    assert len(results) == 1


def test_dish_losing_to_a_rival_is_rejected(config) -> None:
    assert food.rank("pizza", [_fc()], [object()], FakeScorer([[0.0, 12.0, 0.0]]), config) == []


def test_small_food_crop_is_dropped(config) -> None:
    assert food.select_crops([_fc(box=(0, 0, 20, 20))], config, (1000, 1000)) == []


def test_food_padding_is_fixed_pixels(config) -> None:
    kept = food.select_crops([_fc(box=(100, 100, 300, 300))], config, (1000, 1000))
    assert kept[0].box == (
        100 - config.food.padding_px,
        100 - config.food.padding_px,
        300 + config.food.padding_px,
        300 + config.food.padding_px,
    )


def test_food_padding_clamps_at_the_edge(config) -> None:
    kept = food.select_crops([_fc(box=(5, 5, 205, 205))], config, (1000, 1000))
    assert kept[0].box[0] == 0 and kept[0].box[1] == 0


def test_low_confidence_food_detection_is_dropped(config) -> None:
    assert food.select_crops([_fc(conf=0.05)], config, (1000, 1000)) == []


def test_one_image_with_two_dishes_returns_once(config) -> None:
    scorer = FakeScorer([[6.0, 0.0, 0.0], [14.0, 0.0, 0.0]])
    results = food.rank(
        "pizza", [_fc(), _fc(box=(400, 400, 600, 600))], [object(), object()], scorer, config
    )
    assert len(results) == 1


def test_food_empty_query_is_rejected(config) -> None:
    with pytest.raises(ValueError, match="empty query"):
        food.rank("  ", [_fc()], [object()], FakeScorer([[1.0, 0.0]]), config)
