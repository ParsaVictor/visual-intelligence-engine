"""An annotated copy of a video: who is on screen, when, and what the shot is about.

The analysis pipeline looks at a few keyframes per shot, which is what makes it
fast. This module is the opposite trade: it visits every frame (detecting faces on
every ``detect_every``-th and tracking in between) to produce a watchable MP4 for a
demo or a review.

Identity, in order of preference:

1. **Named people.** Give ``known`` a few reference vectors per name, and a face
   is labelled when its track's mean vector clears ``match_threshold`` against
   that person and beats the runner-up by ``margin``. The margin rule comes from
   student-face-gate, where it removed wrong-person accepts at low thresholds.
2. **Anonymous people.** Otherwise tracks are grouped into "Person 1, 2, ..." by
   average similarity, on the same ArcFace scale as the rest of the system.

A track's identity is decided from its *mean* embedding over every frame seen so
far, not from one frame, so a single blurred or turned frame cannot relabel it.

Frames come from :func:`vie.video.decode.iter_decoded`, the decoder the analysis
uses, so a portrait phone video is upright and a truncated file still renders up to
the damage. The output has no audio track: it is for watching who appears, and
copying audio would tie the encoder to the source's codec.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from vie.logging_setup import get_logger

log = get_logger("video.render")


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    vector_sum: np.ndarray | None = None
    views: int = 0
    missed: int = 0
    person: int | None = None
    name: str | None = None
    score: float = 0.0

    @property
    def vector(self) -> np.ndarray | None:
        if self.vector_sum is None:
            return None
        return self.vector_sum / max(float(np.linalg.norm(self.vector_sum)), 1e-12)


@dataclass
class PersonRecord:
    id: int
    label: str
    colour: tuple[int, int, int]
    vector_sum: np.ndarray
    seconds: float = 0.0
    first: float = 0.0
    last: float = 0.0
    present: list[float] = field(default_factory=list)
    thumbnail: Any = None
    best_area: int = 0
    best_quality: float = 0.0

    @property
    def vector(self) -> np.ndarray:
        return self.vector_sum / max(float(np.linalg.norm(self.vector_sum)), 1e-12)


@dataclass
class RenderResult:
    path: Path
    frames: int
    seconds: float
    people: list[PersonRecord]
    fps_processed: float


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """IoU tracking with appearance as a tie-breaker, and identity by mean vector."""

    def __init__(
        self,
        *,
        known: dict[str, Sequence[np.ndarray]] | None = None,
        match_threshold: float = 0.46,
        margin: float = 0.05,
        cluster_threshold: float = 0.45,
        max_missed: int = 8,
    ) -> None:
        self.known = {
            name: np.vstack([v / np.linalg.norm(v) for v in vectors]).astype(np.float32)
            for name, vectors in (known or {}).items() if len(vectors)
        }
        self.match_threshold = match_threshold
        self.margin = margin
        self.cluster_threshold = cluster_threshold
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self.people: list[PersonRecord] = []
        self._next_track = 1

    def predict(self) -> None:
        """Move every track along its velocity, for frames with no detection."""
        for t in self.tracks:
            t.box = tuple(b + v for b, v in zip(t.box, t.velocity, strict=True))

    def update(self, detections: Sequence[tuple[Sequence[float], np.ndarray]]) -> None:
        unmatched = list(range(len(detections)))
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, (box, vector) in enumerate(detections):
                overlap = iou(track.box, box)
                appearance = float(track.vector @ vector) if track.vector is not None else 0.0
                if overlap > 0.15 or appearance > 0.55:
                    pairs.append((overlap + 0.5 * appearance, ti, di))
        used_t: set[int] = set()
        for _, ti, di in sorted(pairs, reverse=True):
            if ti in used_t or di not in unmatched:
                continue
            used_t.add(ti)
            unmatched.remove(di)
            self._absorb(self.tracks[ti], *detections[di])
        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for di in unmatched:
            track = Track(self._next_track, tuple(float(v) for v in detections[di][0]))
            self._next_track += 1
            self._absorb(track, *detections[di], fresh=True)
            self.tracks.append(track)
        for track in self.tracks:
            if track.missed == 0:
                self._identify(track)

    def _absorb(self, track: Track, box: Sequence[float], vector: np.ndarray,
                fresh: bool = False) -> None:
        box = tuple(float(v) for v in box)
        if not fresh:
            # Smooth the box a little: detections jitter by a few pixels per frame.
            track.velocity = tuple(0.5 * (n - o) for n, o in zip(box, track.box, strict=True))
            box = tuple(0.6 * n + 0.4 * o for n, o in zip(box, track.box, strict=True))
        track.box = box
        track.missed = 0
        track.vector_sum = vector.copy() if track.vector_sum is None else track.vector_sum + vector
        track.views += 1

    def _identify(self, track: Track) -> None:
        vector = track.vector
        if vector is None:
            return
        if self.known:
            scores = sorted(((float((m @ vector).max()), name) for name, m in self.known.items()),
                            reverse=True)
            best, name = scores[0]
            runner = scores[1][0] if len(scores) > 1 else -1.0
            if best >= self.match_threshold and best - runner >= self.margin:
                track.name, track.score = name, best
                track.person = self._person_for_name(name, vector)
            else:
                track.name, track.score, track.person = None, best, None
            return
        if track.person is not None:
            person = self.people[track.person - 1]
            track.score = float(person.vector @ vector)
            if track.views % 4 == 0:
                person.vector_sum = person.vector_sum + vector
            return
        if track.views < 3:
            return  # a glimpse or two is more often a false detection than a person
        best, choice = self.cluster_threshold, None
        for person in self.people:
            score = float(person.vector @ vector)
            busy = any(t.person == person.id and t is not track and t.missed == 0
                       for t in self.tracks)
            if score >= best and not busy:
                best, choice = score, person
        if choice is None:
            choice = self._new_person(f"Person {len(self.people) + 1}", vector)
            best = 1.0
        track.person, track.score = choice.id, best
        track.name = choice.label

    def _person_for_name(self, name: str, vector: np.ndarray) -> int:
        for person in self.people:
            if person.label == name:
                return person.id
        return self._new_person(name, vector).id

    def _new_person(self, label: str, vector: np.ndarray) -> PersonRecord:
        from vie.report.showcase import LANES

        person = PersonRecord(len(self.people) + 1, label,
                              LANES[len(self.people) % len(LANES)], vector.copy())
        self.people.append(person)
        return person


def _encoder(path: Path, width: int, height: int, fps: float):
    """An H.264 writer where the build has one, MPEG-4 Part 2 otherwise."""
    import av

    # faststart moves the index to the front, so a browser can play while loading.
    container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
    for codec in ("libx264", "h264", "libopenh264", "mpeg4"):
        try:
            # A fraction keeps 29.97 fps at 29.97; rounding to 30 drifts 3.6 s an hour.
            stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1001))
            break
        except Exception:  # noqa: BLE001 - try the next encoder this build has
            continue
    else:
        raise RuntimeError("no usable video encoder in this PyAV build")
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    if stream.codec_context.name == "libx264":
        stream.options = {"crf": "20", "preset": "veryfast"}
    return container, stream


def _sharpness(patch: np.ndarray) -> float:
    """Variance of the Laplacian, squashed to [0, 1): blur scores near zero."""
    import cv2

    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return value / (value + 60.0)


def _format_clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def render_annotated_video(
    source: str | Path,
    output: str | Path,
    *,
    detect: Callable[[np.ndarray], list[tuple[Sequence[float], np.ndarray]]],
    known: dict[str, Sequence[np.ndarray]] | None = None,
    title: str | None = None,
    shot_labels: Callable[[float], list[str]] | None = None,
    detect_every: int = 2,
    max_seconds: float | None = None,
    max_side: int = 1280,
    match_threshold: float = 0.46,
    cluster_threshold: float = 0.45,
    progress: Callable[[float], None] | None = None,
    min_seconds: float = 1.0,
) -> RenderResult:
    """Write an annotated MP4 and return who appeared in it.

    ``detect`` takes a BGR frame and returns ``(box, unit_vector)`` per face, which
    is what :class:`vie.adapters.InsightFaceDetector` already produces, so faces here
    use exactly the recipe the index uses.
    """
    import itertools
    import time

    import av
    from PIL import Image, ImageDraw

    from vie.report.showcase import (
        FAINT,
        HAIR,
        INK,
        MUTED,
        PAPER,
        SIGNAL,
        SIGNAL_INK,
        display_text,
        draw_brackets,
        font,
        pill,
    )
    from vie.video import decode

    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracker = FaceTracker(known=known, match_threshold=match_threshold,
                          cluster_threshold=cluster_threshold,
                          max_missed=max(4, detect_every * 4))

    info = decode.probe(source)
    fps = info.fps or 25.0
    duration = info.duration
    if max_seconds:
        duration = min(duration, max_seconds) if duration else max_seconds
    decoded = decode.iter_decoded(source, max_side=max_side)
    first = next(decoded, None)
    if first is None:
        raise decode.VideoDecodeError(f"{source} has no decodable frame")
    # From a decoded frame, not the header: rotation swaps width and height.
    out_h, out_w = (v // 2 * 2 for v in first.image.shape[:2])
    # Shown in the strip and counted in the header only once seen this long, so a
    # false detection that lasts a few frames never takes a lane.
    min_frames = max(1, int(fps * min_seconds / 2))
    unit = max(14, out_h // 34)
    lane_pitch = int(unit * 1.15)
    strip_h = (lane_pitch * 5 + unit) // 2 * 2   # room for five people, always even
    writer, out_stream = _encoder(output, out_w, out_h + strip_h, fps)

    f_title, f_label = font(int(unit * 1.25), True), font(unit, True)
    f_small = font(int(unit * .85))
    started = time.perf_counter()
    frames = 0
    t = 0.0
    try:
        for frame in itertools.chain([first], decoded):
            t = frame.time
            if max_seconds and t > max_seconds:
                break
            image = np.ascontiguousarray(frame.image[:out_h, :out_w])
            detected_now = frames % max(1, detect_every) == 0
            if detected_now:
                tracker.update(detect(image))
            else:
                tracker.predict()

            canvas = Image.new("RGB", (out_w, out_h + strip_h), INK)
            canvas.paste(Image.fromarray(image[:, :, ::-1]), (0, 0))
            draw = ImageDraw.Draw(canvas, "RGBA")

            on_screen = 0
            for track in tracker.tracks:
                if track.missed > detect_every:
                    continue
                x1, y1, x2, y2 = (int(v) for v in track.box)
                person = tracker.people[track.person - 1] if track.person else None
                colour = person.colour if person else (200, 205, 212)
                draw_brackets(draw, (x1, y1, x2, y2), colour, width=max(2, unit // 6))
                label = (f"{track.name}  {track.score * 100:.0f}%" if track.name
                         else "unknown" if tracker.known else "…")
                pill(draw, (x1, max(0, y1 - unit * 2)), label, f_label,
                     colour if person else (40, 44, 52),
                     SIGNAL_INK if person else PAPER, (unit // 2, unit // 4))
                if person is not None:
                    on_screen += 1
                    if not person.present or t - person.present[-1] > 1e-6:
                        person.present.append(t)
                    area = (x2 - x1) * (y2 - y1)
                    # Thumbnails only from a real detection, never a predicted box, and
                    # the sharpest large view: the largest alone is usually motion blur.
                    if (detected_now and track.missed == 0 and x1 >= 0 and y1 >= 0
                            and area * 2 > person.best_area):
                        face = image[y1:y2, x1:x2]
                        quality = area * _sharpness(face)
                        if face.size and quality > person.best_quality:
                            pad = int((x2 - x1) * 0.3)
                            crop = image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                            person.thumbnail = Image.fromarray(crop[:, :, ::-1].copy())
                            person.best_quality = quality
                            person.best_area = max(person.best_area, area)

            # ── heads-up display ─────────────────────────────────────
            hud_h = int(unit * 3.4)
            for k in range(hud_h):
                draw.line([(0, k), (out_w, k)], fill=(0, 0, 0, int(170 * (1 - k / hud_h))))
            draw.text((unit, unit * 0.6), display_text((title or source.stem)[:60]), font=f_title,
                      fill=PAPER)
            clock = _format_clock(t)
            cw = draw.textbbox((0, 0), clock, font=f_title)[2]
            draw.text((out_w - cw - unit, unit * 0.6), clock, font=f_title, fill=SIGNAL)
            # On the same dark backing as the chips, so it reads over a bright window.
            mark = "Visual Intelligence Engine"
            mw = draw.textbbox((0, 0), mark, font=f_small)[2] + unit
            pill(draw, (out_w - mw - unit, int(unit * 2.3)), mark, f_small, (0, 0, 0, 140),
                 MUTED, (unit // 2, unit // 5))
            lanes = [p for p in tracker.people if len(p.present) >= min_frames]
            chips = [f"{on_screen} on screen",
                     f"{len(lanes)} {'person' if len(lanes) == 1 else 'people'} so far"]
            if shot_labels is not None:
                chips = shot_labels(t)[:2] + chips
            x = unit
            for chip in chips:
                r = pill(draw, (x, int(unit * 2.3)), chip, f_small, (0, 0, 0, 140), PAPER,
                         (unit // 2, unit // 5))
                x = r[2] + unit // 2

            # ── presence timeline ────────────────────────────────────
            top = out_h
            draw.line([(0, top), (out_w, top)], fill=HAIR, width=1)
            lanes = lanes[:5]   # in order of first appearance, so lanes never reshuffle
            lane_h = max(4, lane_pitch - 6)
            span = duration or max(t, 1.0)
            label_w = int(unit * 7)
            for li, person in enumerate(lanes):
                ly = top + unit // 2 + li * lane_pitch
                draw.text((unit, ly - 3), display_text(person.label[:14]), font=f_small, fill=MUTED)
                for seen in person.present:
                    px = label_w + int((out_w - label_w - unit) * min(1.0, seen / span))
                    draw.line([(px, ly), (px, ly + lane_h)], fill=person.colour, width=2)
            if not lanes:
                draw.text((unit, top + strip_h // 3), "no faces yet", font=f_small, fill=FAINT)
            head = label_w + int((out_w - label_w - unit) * min(1.0, t / span))
            draw.line([(head, top + 4), (head, top + strip_h - 4)], fill=SIGNAL, width=2)

            out = av.VideoFrame.from_image(canvas)
            for packet in out_stream.encode(out):
                writer.mux(packet)
            frames += 1
            if progress is not None and frames % 30 == 0 and span:
                progress(min(1.0, t / span))
        for packet in out_stream.encode():
            writer.mux(packet)
    finally:
        writer.close()
        decoded.close()

    step = 1.0 / fps
    for person in tracker.people:
        if person.present:
            person.first, person.last = person.present[0], person.present[-1]
            person.seconds = len(person.present) * step
    # Under ``min_seconds`` on screen is more often a false detection than a person.
    # Kept in order of first appearance, which is also the order of their numbers.
    people = sorted((p for p in tracker.people if p.seconds >= min_seconds),
                    key=lambda p: p.first)
    elapsed = time.perf_counter() - started
    log.info("annotated %d frames in %.1fs -> %s", frames, elapsed, output)
    return RenderResult(output, frames, t, people, frames / elapsed if elapsed else 0.0)


def render_people_card(result: RenderResult, *, title: str, width: int = 1600) -> Any:
    """One image summarising who appeared and for how long."""
    from PIL import Image, ImageDraw

    from vie.report.showcase import (
        FAINT,
        HAIR,
        INK,
        INK_3,
        MUTED,
        PAPER,
        SIGNAL,
        display_text,
        focus_crop,
        font,
        rounded,
    )

    people = result.people[:12]
    margin, gap = 56, 36
    # Balanced rows (7 people: 4 + 3, not 5 + 2), tiles as large as the width allows.
    rows = max(1, -(-len(people) // 6))
    cols = max(1, -(-len(people) // rows))
    tile = min(300, (width - margin * 2 - gap * (cols - 1)) // cols)
    height = 230 + rows * (tile + 140) + 80
    canvas = Image.new("RGB", (width, height), INK)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((margin, 50), "VIDEO UNDERSTANDING · PEOPLE", font=font(18, True), fill=SIGNAL)
    draw.text((margin, 80), display_text(title[:60]), font=font(46, True), fill=PAPER)
    draw.text((margin, 142), f"{len(result.people)} people · {result.frames:,} frames · "
              f"{_format_clock(result.seconds)} analysed · {result.fps_processed:.1f} frames/s",
              font=font(18), fill=MUTED)
    draw.line([(margin, 190), (width - margin, 190)], fill=HAIR, width=1)
    span = max(result.seconds, 1e-6)
    for i, person in enumerate(people):
        r, c = divmod(i, cols)
        x, y = margin + c * (tile + gap), 220 + r * (tile + 140)
        if person.thumbnail is not None:
            face, _ = focus_crop(person.thumbnail, (tile, tile), None)
        else:
            face = Image.new("RGB", (tile, tile), INK_3)
        masked = rounded(face, 16)
        canvas.paste(masked, (x, y), masked)
        draw.rounded_rectangle((x, y, x + tile, y + tile), radius=16, outline=person.colour,
                               width=3)
        draw.text((x, y + tile + 12), display_text(person.label), font=font(22, True), fill=PAPER)
        draw.text((x, y + tile + 42), f"{person.seconds:.1f}s on screen", font=font(15),
                  fill=MUTED)
        draw.text((x, y + tile + 62), f"first seen {_format_clock(person.first)}", font=font(15),
                  fill=FAINT)
        by = y + tile + 94
        draw.rounded_rectangle((x, by, x + tile, by + 10), radius=5, fill=INK_3)
        for seen in person.present[::2]:
            px = x + int(tile * min(1.0, seen / span))
            draw.line([(px, by), (px, by + 10)], fill=person.colour, width=1)
    draw.text((margin, height - 50), "Faces are grouped by appearance; names come only from "
              "reference photos you supply.", font=font(15), fill=FAINT)
    return canvas
