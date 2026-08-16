"""Bounding-box helpers.

Two of the original defects live here in spirit:

* the food/face cross-verification filter was written but never reached, so
  food-shaped boxes sitting on a human face were always accepted;
* animal crops were taken with a negative slice start when a detection touched
  the frame edge. NumPy does not raise on a negative slice — it wraps — so the
  classifier silently received the wrong pixels.

Both are fixed by using these helpers rather than inline arithmetic.
"""

from __future__ import annotations

Box = tuple[int, int, int, int]


def area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a: Box, b: Box) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def overlap_fraction(box: Box, other: Box) -> float:
    """Fraction of ``box`` covered by ``other``.

    Deliberately asymmetric: the question the food filter asks is "how much of
    this *food* box is actually face", not "how similar are these two boxes".
    IoU would answer the wrong question — a small food box entirely inside a
    large face box has a low IoU but an overlap fraction of 1.0.
    """
    box_area = area(box)
    if box_area == 0:
        return 0.0
    return intersection_area(box, other) / box_area


def overlaps_any(box: Box, others: list[Box], threshold: float) -> bool:
    """True if ``box`` is covered by any of ``others`` beyond ``threshold``."""
    return any(overlap_fraction(box, other) > threshold for other in others)


def pad_box(box: Box, width: int, height: int, *, ratio: float = 0.0, pixels: int = 0) -> Box:
    """Grow a box and clamp it to the image.

    ``ratio`` pads by a fraction of the box's own size, which keeps the added
    context proportional to the subject instead of over-padding small
    detections. ``pixels`` pads by a fixed amount. Use one or the other.

    Clamping is what makes this safe: an unclamped ``x1 - pad`` can go negative,
    and ``image[y1:y2, -7:x2]`` silently wraps to a wrong or empty crop instead
    of raising.
    """
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * ratio)) + pixels
    pad_y = int(round((y2 - y1) * ratio)) + pixels
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def is_large_enough(box: Box, min_px: int) -> bool:
    """Reject detections too small to carry recognisable detail.

    A 12x9 px crop upsampled to 224x224 still produces a confident-looking
    label, which is worse than producing nothing.
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_px and (y2 - y1) >= min_px
