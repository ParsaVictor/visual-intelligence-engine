"""Regression tests for P3-1 / P1-1 / P1-3 / P1-5: the rewritten animal path.

These pin the *decision rules*, which is where every original defect lived. The
CLIP weights are stubbed out — a fake scorer is enough to prove that a losing
query is rejected, that small crops never reach recognition, and that one image
cannot appear twice.
"""

from __future__ import annotations

import pytest

from vie.search.animal import Candidate, rank, select_crops


class FakeScorer:
    """Returns preset logits so the decision rules can be tested in isolation."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.seen_prompts: list[str] | None = None

    def score(self, crops, prompts):
        self.seen_prompts = list(prompts)
        return self.rows[: len(crops)]


def _candidate(name="a.jpg", box=(0, 0, 100, 100), conf=0.9) -> Candidate:
    return Candidate(name, f"/g/{name}", box, conf)


# ---- selection --------------------------------------------------------


def test_small_crops_are_dropped_before_recognition(config) -> None:
    """P3-6: a 12x9 detection upsampled to 224x224 used to get a species label."""
    kept = select_crops([_candidate(box=(0, 0, 12, 9))], config, (1000, 1000))
    assert kept == []


def test_large_crops_survive(config) -> None:
    kept = select_crops([_candidate(box=(0, 0, 200, 200))], config, (1000, 1000))
    assert len(kept) == 1


def test_low_detector_confidence_is_dropped(config) -> None:
    kept = select_crops([_candidate(conf=0.1)], config, (1000, 1000))
    assert kept == []


def test_padding_is_applied_and_clamped(config) -> None:
    kept = select_crops([_candidate(box=(5, 5, 105, 105))], config, (1000, 1000))
    assert kept[0].box[0] == 0  # clamped, not negative
    assert kept[0].box[2] > 105  # padded


# ---- ranking ----------------------------------------------------------


def test_query_is_the_first_prompt(config) -> None:
    scorer = FakeScorer([[10.0, 0.0, 0.0]])
    rank("zebra", [_candidate()], [object()], scorer, config)
    assert scorer.seen_prompts[0] == "a photo of a zebra"
    assert len(scorer.seen_prompts) > 1, "a rival prompt is required"


def test_confident_match_is_returned(config) -> None:
    scorer = FakeScorer([[10.0, 0.0, 0.0]])
    results = rank("zebra", [_candidate()], [object()], scorer, config)
    assert len(results) == 1
    assert results[0].score > config.animal.match_threshold


def test_query_losing_to_a_rival_is_rejected(config) -> None:
    """P1-3: the original had no confidence floor and shipped 21.5% as a match."""
    scorer = FakeScorer([[0.0, 10.0, 0.0]])
    assert rank("zebra", [_candidate()], [object()], scorer, config) == []


def test_ambiguous_match_is_rejected_at_the_default_threshold(config) -> None:
    scorer = FakeScorer([[5.0, 5.0, 5.0]])  # ~0.33 posterior
    assert rank("zebra", [_candidate()], [object()], scorer, config) == []


def test_open_vocabulary_query_is_not_restricted_to_a_class_list(config) -> None:
    """P1-1: 'red panda' is not an ImageNet-1k class and was unreachable before."""
    scorer = FakeScorer([[12.0, 0.0, 0.0]])
    results = rank("red panda", [_candidate()], [object()], scorer, config)
    assert results and results[0].label == "red panda"


def test_no_keyword_map_means_no_substring_collisions(config) -> None:
    """P1-5: searching 'animal' used to match the category 'object / non-animal'."""
    scorer = FakeScorer([[0.0, 8.0, 0.0]])
    assert rank("animal", [_candidate()], [object()], scorer, config) == []


def test_one_image_with_two_animals_returns_one_result(config) -> None:
    """P1-6: the strongest instance wins; the image is not listed twice."""
    scorer = FakeScorer([[6.0, 0.0, 0.0], [14.0, 0.0, 0.0]])
    candidates = [_candidate(box=(0, 0, 100, 100)), _candidate(box=(200, 200, 300, 300))]
    results = rank("cat", candidates, [object(), object()], scorer, config)
    assert len(results) == 1
    assert results[0].box == (200, 200, 300, 300), "kept the weaker instance"


def test_results_are_ranked_across_images(config) -> None:
    scorer = FakeScorer([[6.0, 0.0, 0.0], [14.0, 0.0, 0.0]])
    candidates = [_candidate("weak.jpg"), _candidate("strong.jpg")]
    results = rank("cat", candidates, [object(), object()], scorer, config)
    assert [m.file_name for m in results] == ["strong.jpg", "weak.jpg"]


def test_empty_candidate_list(config) -> None:
    assert rank("cat", [], [], FakeScorer([]), config) == []


def test_empty_query_is_rejected(config) -> None:
    with pytest.raises(ValueError, match="empty query"):
        rank("   ", [_candidate()], [object()], FakeScorer([[1.0, 0.0]]), config)


def test_mismatched_lengths_are_rejected(config) -> None:
    with pytest.raises(ValueError, match="candidates"):
        rank("cat", [_candidate()], [object(), object()], FakeScorer([]), config)
