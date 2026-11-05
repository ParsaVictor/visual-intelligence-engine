"""Shot boundary detection and keyframe selection.

A colour histogram per sampled frame is enough here, for the same reason a
class-agnostic detector works as a gate: the job is cheap and only has to be good
enough to organise what the expensive models look at. Colour survives motion and
compression, so a person walking across a frame barely moves it while a cut does.

Two signals, because cuts come in two kinds:

* a **hard cut** is one large jump between neighbouring samples. A fixed threshold
  misfires on a busy action scene, so it is raised by the recent local variation;
* a **gradual change** (a dissolve or fade, or the camera panning somewhere new) has
  no single large jump, only drift from where the shot began.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShotConfig:
    hard_cut: float = 0.35
    adaptive_k: float = 4.0
    adaptive_window: int = 12
    drift: float = 0.55
    min_shot_seconds: float = 1.0


@dataclass(frozen=True)
class Sample:
    """What the first pass keeps about a sampled frame: no pixels."""

    time: float
    signature: np.ndarray
    sharpness: float


@dataclass(frozen=True)
class Shot:
    index: int
    start: float
    end: float
    first_sample: int
    last_sample: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_signature(image: np.ndarray) -> np.ndarray:
    """A normalised hue/saturation/value histogram, comparable across frame sizes."""
    import cv2

    small = cv2.resize(image, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 4], [0, 180, 0, 256, 0, 256])
    histogram = histogram.ravel().astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def frame_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian: low for motion blur and fades."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) > 480:
        factor = 480 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hellinger distance in [0, 1]: 0 for identical histograms, 1 for disjoint."""
    overlap = float(np.sqrt(a * b).sum())
    return float(np.sqrt(max(0.0, 1.0 - min(1.0, overlap))))


def detect_shots(samples: Sequence[Sample], duration: float, config: ShotConfig) -> list[Shot]:
    """Split a sampled video into shots covering ``[0, duration]`` without gaps."""
    if not samples:
        return []

    boundaries: list[int] = []
    recent: deque[float] = deque(maxlen=config.adaptive_window)
    anchor = 0
    start_time = samples[0].time

    for i in range(1, len(samples)):
        jump = signature_distance(samples[i - 1].signature, samples[i].signature)
        drift = signature_distance(samples[anchor].signature, samples[i].signature)
        threshold = config.hard_cut
        if len(recent) >= 3:
            # Only ever raises the bar: a static scene must not make noise look like a cut.
            threshold = max(threshold, float(np.mean(recent) + config.adaptive_k * np.std(recent)))
        long_enough = samples[i].time - start_time >= config.min_shot_seconds
        if long_enough and (jump >= threshold or drift >= config.drift):
            boundaries.append(i)
            anchor = i
            start_time = samples[i].time
            recent.clear()
        else:
            recent.append(jump)

    shots: list[Shot] = []
    edges = [0, *boundaries, len(samples)]
    for number, (first, stop) in enumerate(zip(edges, edges[1:], strict=False)):
        begin = 0.0 if number == 0 else samples[first].time
        finish = samples[stop].time if stop < len(samples) else max(duration, samples[-1].time)
        shots.append(Shot(number, begin, finish, first, stop - 1))
    return shots


def pick_keyframes(
    samples: Sequence[Sample],
    shots: Sequence[Shot],
    keyframe_seconds: float,
    per_shot_max: int,
    total_max: int,
) -> dict[int, list[int]]:
    """Choose the sample indices to run models on, per shot.

    Each shot is divided into equal stretches and the sharpest sample of each is
    taken, which avoids the blurred frame right after a cut. If the video would
    exceed ``total_max``, every shot first keeps one frame, so a brief shot is never
    invisible, and the remainder goes to the longest shots.
    """
    if not shots:
        return {}

    wanted = {
        shot.index: max(1, min(per_shot_max, int(shot.duration / keyframe_seconds) or 1))
        for shot in shots
    }
    if sum(wanted.values()) > total_max:
        kept = sorted(shots, key=lambda s: s.duration, reverse=True)
        wanted = {shot.index: 0 for shot in shots}
        budget = total_max
        # A film can have more shots than the budget: then the longest shots win.
        for shot in kept[:budget]:
            wanted[shot.index] = 1
        budget -= min(budget, len(shots))
        for shot in kept:
            if budget <= 0:
                break
            fits = min(per_shot_max, int(shot.duration / keyframe_seconds))
            extra = min(budget, max(0, fits - 1))
            wanted[shot.index] += extra
            budget -= extra

    chosen: dict[int, list[int]] = {}
    for shot in shots:
        count = wanted[shot.index]
        if count == 0:
            chosen[shot.index] = []
            continue
        indices = np.arange(shot.first_sample, shot.last_sample + 1)
        picks = []
        for stretch in np.array_split(indices, count):
            if stretch.size:
                picks.append(int(max(stretch, key=lambda i: samples[int(i)].sharpness)))
        chosen[shot.index] = picks
    return chosen
