"""Analyse one video end to end.

Two passes over the file, so memory stays flat however long the video is:

1. decode at low resolution and keep only a histogram and a sharpness score per
   sampled frame, then find shots and choose keyframes;
2. decode just the chosen frames at full resolution and run the models on them.

Every model is injected and every stage can be off. The minimum useful run is the
vision-language encoder alone, which works on a laptop CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vie.config import Config
from vie.geometry import is_large_enough, overlaps_any
from vie.logging_setup import get_logger
from vie.video import decode
from vie.video.people import Person, Sighting, cluster_faces, merge_times
from vie.video.shots import (
    Sample,
    Shot,
    detect_shots,
    frame_sharpness,
    frame_signature,
    pick_keyframes,
)
from vie.video.understand import (
    LabelScore,
    ShotLabels,
    SummaryFacts,
    TranscriptSegment,
    aggregate_labels,
    label_prompts,
    label_shots,
    shot_scores,
    summarise,
)

log = get_logger("video.engine")


class Encoder(Protocol):
    """The slice of :class:`vie.models.VisionLanguageEncoder` video needs."""

    def embed_batch(self, images: Sequence[Any], regions: Sequence[Any]) -> list[list[Any]]: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def logits_from_similarity(self, similarity: np.ndarray) -> np.ndarray: ...


class Captioner(Protocol):
    def caption(self, images: Sequence[Any]) -> list[str]: ...


class Transcriber(Protocol):
    def transcribe(
        self, path: str, duration: float
    ) -> tuple[list[TranscriptSegment], str | None]: ...


@dataclass
class Keyframe:
    time: float
    shot: int
    vector: np.ndarray
    thumbnail: bytes = b""  # small JPEG, for reports only
    brightness: float = 255.0  # mean of 0-255, so reports can skip black frames


@dataclass
class VideoAnalysis:
    """Everything known about one video."""

    info: decode.VideoInfo
    shots: list[Shot]
    keyframes: list[Keyframe]
    labels: list[LabelScore]
    shot_labels: list[ShotLabels]
    people: list[Person]
    animal_times: list[float] = field(default_factory=list)
    animal_species: list[str] = field(default_factory=list)
    food_times: list[float] = field(default_factory=list)
    transcript: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    summary: dict[str, str] = field(default_factory=dict)
    person_thumbnails: dict[int, bytes] = field(default_factory=dict)
    stages: dict[str, bool] = field(default_factory=dict)
    sample_seconds: float = 0.0
    problems: list[str] = field(default_factory=list)


def label_probabilities(
    vectors: np.ndarray,
    encoder: Encoder,
    prompts: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> np.ndarray:
    """For each keyframe, how strongly each label beats the others in its group.

    A softmax within the group, for SigLIP as well as CLIP. On real footage SigLIP's
    absolute sigmoid was about 0.001 for every label on every frame, right or wrong,
    so there is nothing to threshold, while the comparison correctly picked "a movie
    scene" for two films. The price: some label in a group always wins, so the group
    needs to be a sensible set of alternatives, and a low winning share means none fit.
    """
    logits = np.asarray(
        encoder.logits_from_similarity(vectors @ encoder.encode_texts(prompts).T),
        dtype=np.float64,
    )
    result = np.zeros_like(logits, dtype=np.float32)
    groups: dict[str, list[int]] = {}
    for column, (group, _) in enumerate(pairs):
        groups.setdefault(group, []).append(column)
    for columns in groups.values():
        part = logits[:, columns]
        exp = np.exp(part - part.max(axis=1, keepdims=True))
        result[:, columns] = exp / exp.sum(axis=1, keepdims=True)
    return result


def _encode_jpeg(image: np.ndarray, side: int = 360, quality: int = 80) -> bytes:
    import cv2

    height, width = image.shape[:2]
    factor = min(1.0, side / max(height, width))
    if factor < 1.0:
        image = cv2.resize(
            image, (max(1, int(width * factor)), max(1, int(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else b""


def _crop_jpeg(image: np.ndarray, box: tuple[int, int, int, int]) -> bytes:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = int((x2 - x1) * 0.35), int((y2 - y1) * 0.35)
    patch = image[max(0, y1 - pad_y):min(height, y2 + pad_y),
                  max(0, x1 - pad_x):min(width, x2 + pad_x)]
    return _encode_jpeg(patch, side=160, quality=85) if patch.size else b""


def choose_sampling(duration: float, seconds: float, max_samples: int) -> float:
    """Stretch the interval on long videos, so the first pass has a fixed ceiling."""
    return max(seconds, duration / max(1, max_samples))


class VideoEngine:
    """Runs the stages it was given. See :func:`build_engine` for the real wiring."""

    def __init__(
        self,
        config: Config,
        encoder: Encoder,
        *,
        face_detector: Any | None = None,
        animal_detector: Any | None = None,
        food_detector: Any | None = None,
        classify_crops: Callable[[Sequence[Any], int], list[list[tuple[int, float]]]] | None = None,
        class_names: Callable[[], list[str]] | None = None,
        crop: Callable[[Any, Any], Any] | None = None,
        captioner: Captioner | None = None,
        transcriber: Transcriber | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.settings = config.video
        self.encoder = encoder
        self.face_detector = face_detector
        self.animal_detector = animal_detector
        self.food_detector = food_detector
        self.classify_crops = classify_crops
        self.class_names = class_names
        self.crop = crop
        self.captioner = captioner
        self.transcriber = transcriber
        self._announce = on_stage or (lambda stage: log.info("%s", stage))

    # ---- pass 1 -------------------------------------------------------

    def _scan(self, path: Path, info: decode.VideoInfo) -> tuple[list[Sample], float]:
        settings = self.settings
        every = choose_sampling(info.duration, settings.sample_seconds, settings.max_samples)
        samples = [
            Sample(frame.time, frame_signature(frame.image), frame_sharpness(frame.image))
            for frame in decode.sample_frames(path, every, max_side=320,
                                              max_duration=settings.max_duration or None)
        ]
        return samples, every

    # ---- pass 2 -------------------------------------------------------

    def _detect_stages(
        self, images: list[np.ndarray], times: list[float]
    ) -> tuple[list[Sighting], dict[tuple, bytes], list[tuple[float, Any]], list[float]]:
        """Face, then animal, then food, the order the image pipeline depends on.

        Crops are taken here, so no full frame outlives its batch. Holding a 960p
        frame per sighting could otherwise reach gigabytes on a long video.
        """
        config = self.config
        sightings: list[Sighting] = []
        face_crops: dict[tuple, bytes] = {}
        animal_crops: list[tuple[float, Any]] = []
        food_times: list[float] = []

        for image, time in zip(images, times, strict=True):
            face_boxes = []
            if self.face_detector is not None:
                for box, embedding in self.face_detector.detect(image):
                    face_boxes.append(box)
                    sightings.append(Sighting(time, box, np.asarray(embedding, np.float32)))
                    face_crops[(time, box)] = _crop_jpeg(image, box)

            if self.animal_detector is not None:
                for item in self.animal_detector.detect(
                    image, [config.animal.animal_class_id], config.animal.search_confidence
                ):
                    if is_large_enough(item[0], config.animal.min_crop_px):
                        patch = self.crop(image, item[0]) if self.crop is not None else None
                        animal_crops.append((time, patch))

            if self.food_detector is not None:
                for item in self.food_detector.detect(
                    image, config.food.search_classes, config.food.search_confidence
                ):
                    box = item[0]
                    if is_large_enough(box, config.food.min_crop_px) and not overlaps_any(
                        box, face_boxes, config.food.face_overlap_reject
                    ):
                        food_times.append(time)
                        break
        return sightings, face_crops, animal_crops, food_times

    def _name_species(self, crops: list[Any]) -> list[str]:
        """Name animals only when the classifier is confident: absent beats wrong."""
        crops = [c for c in crops if c is not None]
        if not crops or self.classify_crops is None or self.class_names is None:
            return []
        names = self.class_names()
        votes: dict[str, float] = {}
        for top in self.classify_crops(crops, 1):
            for class_id, probability in top:
                if probability >= self.config.animal.classifier_threshold:
                    name = names[class_id].split(",")[0].strip()
                    votes[name] = votes.get(name, 0.0) + probability
        return [name for name, _ in sorted(votes.items(), key=lambda kv: -kv[1])]

    # ---- the whole thing ----------------------------------------------

    def analyse(self, path: str | Path) -> VideoAnalysis:
        path = Path(path)
        settings = self.settings
        info = decode.probe(path)

        self._announce(f"scanning {path.name} ({info.duration:.0f} s)")
        samples, every = self._scan(path, info)
        if not samples:
            raise decode.VideoDecodeError(f"no frames could be decoded from {path}")
        duration = max(info.duration, samples[-1].time)
        if settings.max_duration:
            duration = min(duration, settings.max_duration)

        shots = detect_shots(samples, duration, settings.shots)
        chosen = pick_keyframes(
            samples, shots, settings.keyframe_seconds, settings.keyframes_per_shot,
            settings.max_keyframes,
        )
        order = sorted(
            (samples[index].time, shot) for shot, picks in chosen.items() for index in picks
        )
        self._announce(f"{len(shots)} shots, {len(order)} keyframes")

        # Speech first, before the vision model is resident: on a small machine the two
        # do not fit in memory together.
        problems: list[str] = []
        transcript: list[TranscriptSegment] = []
        language = None
        if self.transcriber is not None and info.has_audio:
            self._announce("transcribing audio")
            try:
                transcript, language = self.transcriber.transcribe(str(path), duration)
            except Exception as exc:
                log.exception("transcription failed")
                problems.append(f"transcription failed: {exc}")

        vectors: list[np.ndarray] = []
        keyframes: list[Keyframe] = []
        sightings: list[Sighting] = []
        face_crops: dict[tuple, bytes] = {}
        animal_times: list[float] = []
        animal_crops: list[Any] = []
        food_times: list[float] = []
        caption_frames: dict[int, np.ndarray] = {}

        times = [time for time, _ in order]
        shot_by_time = dict(order)
        batch_images: list[np.ndarray] = []
        batch_times: list[float] = []

        def flush() -> None:
            if not batch_images:
                return
            regions = [[("whole", (0, 0, im.shape[1], im.shape[0]))] for im in batch_images]
            encoded = self.encoder.embed_batch(batch_images, regions)
            for time, image, rows in zip(batch_times, batch_images, encoded, strict=True):
                vector = dict(rows)["whole"]
                vectors.append(vector)
                keyframes.append(
                    Keyframe(time, shot_by_time[time], vector, _encode_jpeg(image),
                             float(image.mean()))
                )
                if self.captioner is not None:
                    caption_frames.setdefault(shot_by_time[time], image)

            found, crops, animals, foods = self._detect_stages(batch_images, batch_times)
            sightings.extend(found)
            face_crops.update(crops)
            animal_times.extend(t for t, _ in animals)
            animal_crops.extend(patch for _, patch in animals)
            food_times.extend(foods)
            batch_images.clear()
            batch_times.clear()

        self._announce("running models on keyframes")
        for wanted, frame in zip(
            times, decode.frames_at(path, times, settings.max_side), strict=False
        ):
            batch_images.append(frame.image)
            batch_times.append(wanted)
            if len(batch_images) >= self.config.indexing.batch_size:
                flush()
        flush()
        if not keyframes:
            raise decode.VideoDecodeError(f"no keyframes could be decoded from {path}")

        # What the video is about.
        matrix = np.vstack(vectors).astype(np.float32)
        pairs, prompts = label_prompts(settings.taxonomy, settings.prompt_template)
        probabilities = label_probabilities(matrix, self.encoder, prompts, pairs)
        per_shot = shot_scores(probabilities, [k.shot for k in keyframes], len(shots))
        labels = aggregate_labels(per_shot, shots, pairs, settings.label_threshold)
        shot_labels = label_shots(per_shot, pairs, settings.label_threshold, len(shots))

        # Who appears.
        people = cluster_faces(
            sightings, settings.face_cluster_threshold, settings.presence_gap,
            settings.min_face_sightings,
        )
        thumbnails = {
            person.id: face_crops[(person.best_sighting.time, person.best_sighting.box)]
            for person in people
        }

        species = self._name_species(animal_crops)

        # Optional layers.
        # A failure here must not cost the minutes of work already done, so it becomes
        # a visible warning on the report rather than an exception.
        if self.captioner is not None:
            self._announce("captioning shots")
            ordered = sorted(caption_frames)
            try:
                captions = self.captioner.caption([caption_frames[s] for s in ordered])
                for shot, text in zip(ordered, captions, strict=True):
                    shot_labels[shot].caption = text
            except Exception as exc:
                log.exception("captioning failed")
                problems.append(f"captions failed: {exc}")

        gap = settings.presence_gap
        animal_seconds = sum(i.duration for i in merge_times(animal_times, gap, pad=0.5))
        food_seconds = sum(i.duration for i in merge_times(food_times, gap, pad=0.5))
        summary = summarise(
            SummaryFacts(
                duration=duration,
                shot_count=len(shots),
                labels=labels,
                people=people,
                animals=species,
                animal_seconds=animal_seconds,
                food_seconds=food_seconds,
                faces_checked=self.face_detector is not None,
                transcript=transcript,
                language=language,
            )
        )

        return VideoAnalysis(
            info=info,
            shots=shots,
            keyframes=keyframes,
            labels=labels,
            shot_labels=shot_labels,
            people=people,
            animal_times=sorted(set(animal_times)),
            animal_species=species,
            food_times=sorted(set(food_times)),
            transcript=transcript,
            language=language,
            summary=summary,
            person_thumbnails=thumbnails,
            stages={
                "faces": self.face_detector is not None,
                "animals": self.animal_detector is not None,
                "food": self.food_detector is not None,
                "captions": self.captioner is not None,
                "transcript": self.transcriber is not None,
            },
            sample_seconds=every,
            problems=problems,
        )
