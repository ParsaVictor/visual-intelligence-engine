"""Regression tests for P1-3 (no confidence floor) and P1-6 (duplicate results)."""

from __future__ import annotations

import pytest

from vie.scoring import Match, best_per_image, contrastive_score, softmax


def test_softmax_sums_to_one() -> None:
    assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_softmax_is_stable_for_large_logits() -> None:
    assert sum(softmax([1000.0, 1001.0])) == pytest.approx(1.0)


def test_query_winning_against_rivals_scores_high() -> None:
    assert contrastive_score([10.0, 1.0, 1.0]) > 0.95


def test_query_losing_to_a_rival_scores_low() -> None:
    """This is the case the original animal module could not express at all."""
    assert contrastive_score([1.0, 10.0, 1.0]) < 0.05


def test_ambiguous_case_lands_mid_range() -> None:
    score = contrastive_score([5.0, 5.0, 1.0])
    assert 0.4 < score < 0.6


def test_single_candidate_is_rejected() -> None:
    """Without a rival the softmax is always 1.0 — a meaningless 100% match."""
    with pytest.raises(ValueError, match="rival"):
        contrastive_score([5.0])


def test_empty_logits_are_rejected() -> None:
    with pytest.raises(ValueError):
        contrastive_score([])


# ---- P1-6: one result per image ---------------------------------------


def test_two_matching_faces_in_one_image_yield_one_result() -> None:
    """The unified pipeline lost the per-image break and returned this image twice."""
    matches = [
        Match("a.jpg", "/g/a.jpg", 0.61, box=(0, 0, 10, 10)),
        Match("a.jpg", "/g/a.jpg", 0.88, box=(50, 50, 60, 60)),
    ]
    result = best_per_image(matches)
    assert len(result) == 1


def test_the_strongest_instance_is_kept_not_the_first() -> None:
    """The original recorded whichever instance the detector returned first."""
    matches = [
        Match("a.jpg", "/g/a.jpg", 0.26, box=(0, 0, 10, 10)),
        Match("a.jpg", "/g/a.jpg", 0.95, box=(50, 50, 60, 60)),
    ]
    assert best_per_image(matches)[0].score == pytest.approx(0.95)


def test_results_are_ranked_by_score() -> None:
    matches = [
        Match("a.jpg", "/g/a.jpg", 0.50),
        Match("b.jpg", "/g/b.jpg", 0.90),
        Match("c.jpg", "/g/c.jpg", 0.70),
    ]
    assert [m.file_name for m in best_per_image(matches)] == ["b.jpg", "c.jpg", "a.jpg"]


def test_ranking_uses_the_best_instance_per_image() -> None:
    """An image whose best instance is strong must outrank one whose best is weak."""
    matches = [
        Match("weak.jpg", "/g/weak.jpg", 0.80),
        Match("strong.jpg", "/g/strong.jpg", 0.26),
        Match("strong.jpg", "/g/strong.jpg", 0.95),
    ]
    assert best_per_image(matches)[0].file_name == "strong.jpg"


def test_empty_input() -> None:
    assert best_per_image([]) == []
