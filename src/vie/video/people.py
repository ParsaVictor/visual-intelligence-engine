"""Grouping face sightings into people, and sightings into time intervals.

This says "the same person appears at 0:12 and 3:40". It does not say who that
person is. Naming someone is a separate, deliberate act: a face search against a
photo the operator supplies. Nothing here assigns an identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Sighting:
    time: float
    box: tuple[int, int, int, int]
    embedding: np.ndarray


@dataclass
class Person:
    id: int
    sightings: list[Sighting] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)

    @property
    def screen_time(self) -> float:
        return sum(i.duration for i in self.intervals)

    @property
    def best_sighting(self) -> Sighting:
        """The largest view of this face, for a thumbnail."""
        def size(s: Sighting) -> int:
            x1, y1, x2, y2 = s.box
            return max(0, x2 - x1) * max(0, y2 - y1)

        return max(self.sightings, key=size)


def merge_times(times: Sequence[float], gap: float, pad: float = 0.0) -> list[Interval]:
    """Join timestamps closer than ``gap`` into intervals."""
    if not times:
        return []
    ordered = sorted(times)
    intervals: list[Interval] = []
    start = last = ordered[0]
    for value in ordered[1:]:
        if value - last > gap:
            intervals.append(Interval(max(0.0, start - pad), last + pad))
            start = value
        last = value
    intervals.append(Interval(max(0.0, start - pad), last + pad))
    return intervals


def cluster_faces(
    sightings: Sequence[Sighting],
    threshold: float,
    merge_gap: float,
    min_sightings: int = 1,
) -> list[Person]:
    """Average-linkage agglomerative clustering of unit face embeddings.

    Average linkage rather than single, because single linkage chains: A resembles
    B, and B resembles C, and A and C end up as one person though they look nothing
    alike. For deciding that two sightings are the same human, merging too eagerly
    is the worse error. Sightings at the same timestamp are also never merged, since
    one person cannot appear twice in the same frame.

    Quadratic in the number of sightings, which is bounded by the keyframe budget.
    """
    if not sightings:
        return []

    vectors = np.vstack([s.embedding for s in sightings]).astype(np.float32)
    clusters: list[list[int]] = [[i] for i in range(len(sightings))]
    similarity = vectors @ vectors.T
    times = [s.time for s in sightings]

    def conflict(a: list[int], b: list[int]) -> bool:
        seen = {times[i] for i in a}
        return any(times[i] in seen for i in b)

    while len(clusters) > 1:
        best, pair = threshold, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                value = float(similarity[np.ix_(clusters[i], clusters[j])].mean())
                if value >= best and not conflict(clusters[i], clusters[j]):
                    best, pair = value, (i, j)
        if pair is None:
            break
        i, j = pair
        clusters[i].extend(clusters[j])
        del clusters[j]

    people = [
        Person(0, [sightings[i] for i in sorted(members, key=lambda m: times[m])])
        for members in clusters
        if len(members) >= min_sightings
    ]
    for person in people:
        person.intervals = merge_times([s.time for s in person.sightings], merge_gap, pad=0.5)
    people.sort(key=lambda p: (-p.screen_time, -len(p.sightings)))
    for number, person in enumerate(people, 1):
        person.id = number
    return people
