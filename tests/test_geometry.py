"""Regression tests for P1-2 (face/food cross-verification) and crop safety."""

from __future__ import annotations

import pytest

from vie.geometry import (
    is_large_enough,
    overlap_fraction,
    overlaps_any,
    pad_box,
)


def test_disjoint_boxes_do_not_overlap() -> None:
    assert overlap_fraction((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_box_fully_inside_another_is_fully_covered() -> None:
    """A small food box sitting entirely on a face must report 1.0, not a low IoU."""
    food = (40, 40, 60, 60)
    face = (0, 0, 200, 200)
    assert overlap_fraction(food, face) == 1.0


def test_overlap_is_asymmetric() -> None:
    """The question is 'how much of the food box is face', not 'how similar are they'."""
    small, large = (40, 40, 60, 60), (0, 0, 200, 200)
    assert overlap_fraction(small, large) == 1.0
    assert overlap_fraction(large, small) < 0.05


def test_half_covered_box() -> None:
    assert overlap_fraction((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)


def test_food_box_on_a_face_is_rejected(config) -> None:
    """P1-2: the original filter never ran, so this case was always accepted."""
    food_box = (100, 100, 140, 140)
    faces = [(90, 90, 200, 200)]
    assert overlaps_any(food_box, faces, config.food.face_overlap_reject) is True


def test_food_box_beside_a_face_is_kept(config) -> None:
    food_box = (300, 300, 340, 340)
    faces = [(90, 90, 200, 200)]
    assert overlaps_any(food_box, faces, config.food.face_overlap_reject) is False


def test_no_faces_means_nothing_is_rejected(config) -> None:
    assert overlaps_any((0, 0, 10, 10), [], config.food.face_overlap_reject) is False


def test_padding_clamps_at_the_top_left_corner() -> None:
    """Unclamped padding produced a negative slice start, which NumPy silently wraps."""
    box = pad_box((5, 5, 50, 50), width=500, height=500, ratio=0.5)
    assert box[0] == 0 and box[1] == 0
    assert all(v >= 0 for v in box)


def test_padding_clamps_at_the_bottom_right_corner() -> None:
    box = pad_box((450, 450, 495, 495), width=500, height=500, ratio=0.5)
    assert box[2] <= 500 and box[3] <= 500


def test_ratio_padding_scales_with_the_box() -> None:
    small = pad_box((100, 100, 120, 120), 1000, 1000, ratio=0.15)
    large = pad_box((100, 100, 500, 500), 1000, 1000, ratio=0.15)
    assert (small[2] - small[0]) - 20 < (large[2] - large[0]) - 400


def test_pixel_padding_is_constant() -> None:
    box = pad_box((100, 100, 120, 120), 1000, 1000, pixels=20)
    assert box == (80, 80, 140, 140)


@pytest.mark.parametrize(
    ("box", "expected"),
    [((0, 0, 12, 9), False), ((0, 0, 40, 40), True), ((0, 0, 39, 100), False)],
)
def test_minimum_crop_size(box, expected) -> None:
    """A 12x9 detection upsampled to 224x224 still yields a confident-looking label."""
    assert is_large_enough(box, 40) is expected
