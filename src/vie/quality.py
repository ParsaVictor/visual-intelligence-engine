"""Quality gates and multi-crop layout.

Face quality
------------
A detection too small or too uncertain produces an embedding that is not a
weak signal — it is a *misleading* one. At a match threshold of 0.46 a garbage
embedding does not simply fail to match; it matches the wrong person.

The gate here is deliberately permissive. It is not trying to be selective:
almost anything RetinaFace is confident about should be indexed, because recall
matters more than precision at index time and the search threshold does the
real filtering. It only removes detections that cannot carry identity at all.

Multi-crop layout
-----------------
A single whole-image embedding averages the whole scene. In a wide press photo
a backpack occupying 3% of the pixels contributes almost nothing to the vector,
which is why a query like "a man with a blue backpack" scored 23% on the
original demo — the model was not wrong, the representation had discarded the
subject before the query ever arrived.

Encoding a few regions as well lets the query match the best-matching region
instead of the averaged whole.
"""

from __future__ import annotations

from vie.geometry import Box


def is_usable_face(det_score: float, box: Box, min_det_score: float, min_px: int) -> bool:
    """Whether a detected face is worth embedding.

    Args:
        det_score: RetinaFace detection confidence.
        box: the detected face box.
        min_det_score: floor on confidence. Low by design.
        min_px: floor on the shorter side, in pixels.
    """
    if det_score < min_det_score:
        return False
    x1, y1, x2, y2 = box
    return min(x2 - x1, y2 - y1) >= min_px


def crop_regions(width: int, height: int, layout: str) -> list[tuple[str, Box]]:
    """Return the named regions to encode for one image.

    ``1``              whole frame only (one vector, original behaviour)
    ``1+2x2``          whole frame plus quadrants (5 vectors)
    ``1+2x2+center``   the above plus a centre crop (6 vectors)

    The centre crop matters because the 2x2 grid cuts straight through the
    middle of the frame, which is exactly where a photographer puts the
    subject — without it, a centred object is split across four quadrants and
    appears whole in none of them.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")

    regions: list[tuple[str, Box]] = [("whole", (0, 0, width, height))]
    if layout == "1":
        return regions

    if "2x2" in layout:
        mx, my = width // 2, height // 2
        regions += [
            ("tl", (0, 0, mx, my)),
            ("tr", (mx, 0, width, my)),
            ("bl", (0, my, mx, height)),
            ("br", (mx, my, width, height)),
        ]

    if "center" in layout:
        # Half-size box about the centre.
        qw, qh = width // 4, height // 4
        regions.append(("center", (qw, qh, width - qw, height - qh)))

    return regions


def expected_vectors(layout: str) -> int:
    """How many embeddings per image this layout produces."""
    return len(crop_regions(1000, 1000, layout))
