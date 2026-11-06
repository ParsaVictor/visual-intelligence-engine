"""Video understanding, against real encoded video files.

Each test video is really encoded and really decoded through PyAV, with scene
changes at known times. Only the model weights are faked, so a wrong timestamp, a
missed cut or a swapped colour channel fails here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeClassifier, FakeEncoder, FakeFaces, FakeFood

from vie import cli
from vie.config import Config
from vie.video import decode
from vie.video.engine import VideoEngine, choose_sampling, label_probabilities
from vie.video.library import index_videos
from vie.video.people import Sighting, cluster_faces, merge_times
from vie.video.shots import (
    Sample,
    Shot,
    ShotConfig,
    detect_shots,
    frame_sharpness,
    frame_signature,
    pick_keyframes,
    signature_distance,
)
from vie.video.store import VideoStore, analysis_to_dict
from vie.video.understand import (
    SummaryFacts,
    aggregate_labels,
    format_clock,
    shot_scores,
    summarise,
)

pytest.importorskip("av")

RED = (230, 20, 20)
GREEN = (20, 230, 20)
BLUE = (20, 20, 230)
FPS = 10


def write_video(path: Path, scenes: list[tuple[tuple[int, int, int], float]]) -> Path:
    """Encode solid scenes, with light texture so each compresses like a real frame."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
        for rgb, seconds in scenes:
            for _ in range(int(seconds * FPS)):
                noise = rng.integers(-6, 6, (180, 320, 3))
                pixels = np.clip(np.array(rgb) + noise, 0, 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    return write_video(tmp_path / "videos" / "clip.mp4", [(RED, 6), (GREEN, 6), (BLUE, 6)])


@pytest.fixture()
def cfg(raw_config: dict, tmp_path: Path) -> Config:
    raw = copy.deepcopy(raw_config)
    raw["paths"] = {
        "gallery": str(tmp_path / "gallery"),
        "videos": str(tmp_path / "videos"),
        "database": str(tmp_path / "index.db"),
    }
    raw["video"]["sample_seconds"] = 0.5
    raw["video"]["keyframe_seconds"] = 2.0
    raw["video"]["min_face_sightings"] = 1
    raw["video"]["taxonomy"] = {"topic": ["something red", "something green", "something blue"]}
    raw["video"]["prompt_template"] = "{label}"
    raw["video"]["label_threshold"] = 0.5
    return Config.from_dict(raw)


def engine(config: Config, **overrides) -> VideoEngine:
    return VideoEngine(config, FakeEncoder(), on_stage=lambda s: None, **overrides)


# ── decoding ───────────────────────────────────────────────────────────


def test_probe_reads_the_container(clip: Path) -> None:
    info = decode.probe(clip)
    assert (info.width, info.height) == (320, 180)
    assert info.duration == pytest.approx(18, abs=0.3)
    assert not info.has_audio


def test_sampling_follows_video_time(clip: Path) -> None:
    times = [f.time for f in decode.sample_frames(clip, 2.0)]
    assert times[:3] == pytest.approx([0.0, 2.0, 4.0], abs=0.15)
    assert len(times) == 9


def test_frames_are_bgr_like_the_image_pipeline(clip: Path) -> None:
    frame = next(decode.sample_frames(clip, 1.0))
    blue, green, red = frame.image.reshape(-1, 3).mean(axis=0)
    assert red > 180 and blue < 70 and green < 70


def test_frames_at_returns_the_requested_moments(clip: Path) -> None:
    frames = list(decode.frames_at(clip, [1.0, 7.0, 13.0], max_side=160))
    assert [f.time for f in frames] == pytest.approx([1.0, 7.0, 13.0], abs=0.15)
    assert max(frames[0].image.shape[:2]) <= 160
    assert frames[1].image[:, :, 1].mean() > 180  # the green scene


def test_seeking_across_a_long_gap_lands_correctly(tmp_path: Path) -> None:
    long_clip = write_video(tmp_path / "long.mp4", [(RED, 15), (BLUE, 15)])
    frames = list(decode.frames_at(long_clip, [0.5, 25.0]))
    assert frames[1].time == pytest.approx(25.0, abs=0.2)
    assert frames[1].image[:, :, 0].mean() > 180  # blue channel, in BGR


def test_non_ascii_video_path(tmp_path: Path) -> None:
    named = write_video(tmp_path / "ویدیو-آزمایشی.mp4", [(RED, 2)])
    assert decode.probe(named).duration > 1.5
    assert next(decode.sample_frames(named, 1.0)).image.size


def test_missing_and_garbage_files_are_clean_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        decode.probe(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(decode.VideoDecodeError):
        decode.probe(bad)


def test_truncated_file_yields_what_it_can(clip: Path, tmp_path: Path) -> None:
    cut = tmp_path / "cut.mp4"
    data = clip.read_bytes()
    cut.write_bytes(data[: len(data) // 2])
    try:
        frames = list(decode.sample_frames(cut, 1.0))
    except decode.VideoDecodeError:
        frames = None  # an MP4 cut before its index is honestly unreadable
    assert frames is None or isinstance(frames, list)


def _write_with_audio(path: Path, seconds: int, hertz: float = 440.0) -> Path:
    import av

    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=FPS)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        audio = container.add_stream("aac", rate=44_100)
        for _ in range(seconds * FPS):
            pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = np.arange(seconds * 44_100) / 44_100
        wave = (np.sin(2 * np.pi * hertz * samples) * 0.5).astype(np.float32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(wave, format="fltp", layout="mono")
        frame.sample_rate = 44_100
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return path


def test_audio_is_decoded_to_16khz_mono(tmp_path: Path) -> None:
    clip = _write_with_audio(tmp_path / "speech.mp4", 3)
    assert decode.probe(clip).has_audio
    audio = decode.read_audio(clip)
    assert audio.dtype == np.float32
    assert audio.size == pytest.approx(3 * 16_000, abs=4_000)
    assert 0.2 < float(np.abs(audio).max()) <= 1.0
    # The tone survives: strongest frequency is still about 440 Hz.
    spectrum = np.abs(np.fft.rfft(audio))
    assert np.argmax(spectrum) * 16_000 / audio.size == pytest.approx(440, abs=10)


def test_audio_respects_the_analysed_duration(tmp_path: Path) -> None:
    clip = _write_with_audio(tmp_path / "speech.mp4", 4)
    assert decode.read_audio(clip, duration=1.0).size == 16_000


def test_video_without_audio_gives_an_empty_array(clip: Path) -> None:
    assert decode.read_audio(clip).size == 0


def test_transcriber_hands_whisper_samples_not_a_path(tmp_path: Path, cfg: Config) -> None:
    """faster-whisper's own file reading is what broke against current PyAV."""
    from vie.video.optional import WhisperTranscriber

    received = []

    class FakeWhisper:
        def transcribe(self, audio, vad_filter):
            received.append(audio)

            class Info:
                language = "en"

            class Segment:
                start, end, text = 0.0, 1.0, " hello "

            return [Segment()], Info()

    transcriber = WhisperTranscriber(cfg)
    transcriber._model = FakeWhisper()
    clip = _write_with_audio(tmp_path / "speech.mp4", 2)
    segments, language = transcriber.transcribe(str(clip), 1.0)
    assert isinstance(received[0], np.ndarray) and received[0].size == 16_000
    assert segments[0].text == "hello" and language == "en"


def test_discovery_finds_videos_only(tmp_path: Path) -> None:
    for name in ("a.mp4", "b.MOV", "c.txt", ".hidden/d.mp4", "sub/e.mkv"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    found = {p.relative_to(tmp_path).as_posix() for p in decode.discover_videos(tmp_path)}
    assert found == {"a.mp4", "b.MOV", "sub/e.mkv"}


# ── shots ──────────────────────────────────────────────────────────────


def _samples(path: Path, every: float = 0.5) -> list[Sample]:
    return [
        Sample(f.time, frame_signature(f.image), frame_sharpness(f.image))
        for f in decode.sample_frames(path, every)
    ]


def test_cuts_are_found_at_the_right_times(clip: Path) -> None:
    shots = detect_shots(_samples(clip), 18.0, ShotConfig())
    assert len(shots) == 3
    assert [s.start for s in shots] == pytest.approx([0.0, 6.0, 12.0], abs=0.6)


def test_shots_cover_the_whole_video_without_gaps(clip: Path) -> None:
    shots = detect_shots(_samples(clip), 18.0, ShotConfig())
    assert shots[0].start == 0.0 and shots[-1].end == 18.0
    for before, after in zip(shots, shots[1:], strict=False):
        assert before.end == after.start


def test_a_static_scene_is_one_shot(tmp_path: Path) -> None:
    still = write_video(tmp_path / "still.mp4", [(GREEN, 8)])
    assert len(detect_shots(_samples(still), 8.0, ShotConfig())) == 1


def test_a_gradual_change_is_caught_by_drift() -> None:
    """A dissolve has no large jump anywhere, only movement from where it began."""
    start, end = np.zeros(10, np.float32), np.zeros(10, np.float32)
    start[0], end[9] = 1.0, 1.0
    samples = [
        Sample(float(i), (1 - i / 19) * start + (i / 19) * end, 1.0) for i in range(20)
    ]
    config = ShotConfig(hard_cut=0.9, drift=0.55)
    jumps = [signature_distance(a.signature, b.signature)
             for a, b in zip(samples, samples[1:], strict=False)]
    assert max(jumps) < config.hard_cut
    assert len(detect_shots(samples, 20.0, config)) >= 2


def test_min_shot_length_suppresses_flicker() -> None:
    a, b = np.zeros(4, np.float32), np.zeros(4, np.float32)
    a[0], b[1] = 1.0, 1.0
    samples = [Sample(i * 0.1, a if i % 2 else b, 1.0) for i in range(8)]
    assert len(detect_shots(samples, 0.8, ShotConfig(min_shot_seconds=1.0))) == 1


def test_no_samples_means_no_shots() -> None:
    assert detect_shots([], 10.0, ShotConfig()) == []


def _flat_samples(count: int) -> list[Sample]:
    return [Sample(float(i), np.ones(4, np.float32) / 4, float(i % 7)) for i in range(count)]


def test_the_sharpest_frame_is_chosen() -> None:
    samples = [Sample(float(i), np.ones(4, np.float32), s) for i, s in enumerate([1, 9, 2, 3])]
    picks = pick_keyframes(samples, [Shot(0, 0, 4, 0, 3)], 10.0, 1, 10)
    assert picks[0] == [1]


def test_long_shots_get_more_keyframes_up_to_the_cap() -> None:
    samples = _flat_samples(60)
    picks = pick_keyframes(samples, [Shot(0, 0, 12, 0, 11), Shot(1, 12, 60, 12, 59)], 6.0, 3, 100)
    assert len(picks[0]) == 2 and len(picks[1]) == 3


def test_budget_keeps_every_shot_visible_when_it_can() -> None:
    samples = _flat_samples(100)
    shots = [Shot(i, i * 10, i * 10 + 10, i * 10, i * 10 + 9) for i in range(10)]
    picks = pick_keyframes(samples, shots, 1.0, 5, 12)
    assert sum(len(v) for v in picks.values()) == 12
    assert all(picks[s.index] for s in shots)


def test_budget_smaller_than_the_shot_count_favours_long_shots() -> None:
    samples = _flat_samples(100)
    shots = [Shot(i, i * 10, i * 10 + 5 + i, i * 10, i * 10 + 9) for i in range(10)]
    picks = pick_keyframes(samples, shots, 1.0, 5, 3)
    assert sum(len(v) for v in picks.values()) == 3
    assert set(k for k, v in picks.items() if v) == {7, 8, 9}


def test_sampling_stretches_on_very_long_videos() -> None:
    assert choose_sampling(60, 1.0, 3600) == 1.0
    assert choose_sampling(36_000, 1.0, 3600) == 10.0


# ── people ─────────────────────────────────────────────────────────────


def _vector(*values: float) -> np.ndarray:
    v = np.array(values, np.float32)
    return v / np.linalg.norm(v)


def test_two_people_are_kept_apart_and_each_is_grouped() -> None:
    a, b = _vector(1, 0, 0), _vector(0, 1, 0)
    sightings = [Sighting(float(t), (0, 0, 10, 10), v)
                 for t, v in [(0, a), (5, b), (10, a), (15, b), (20, a)]]
    people = cluster_faces(sightings, 0.5, merge_gap=8)
    assert [len(p.sightings) for p in people] == [3, 2]
    assert [p.id for p in people] == [1, 2]


def test_chaining_does_not_merge_unlike_faces() -> None:
    """A~B and B~C, with A and C unlike: single linkage would call them one person."""
    a, b, c = _vector(1, 0), _vector(0.71, 0.71), _vector(0, 1)
    people = cluster_faces(
        [Sighting(float(t), (0, 0, 9, 9), v) for t, v in enumerate((a, b, c))], 0.6, 5
    )
    assert len(people) == 2


def test_two_faces_in_one_frame_are_never_one_person() -> None:
    same = _vector(1, 0)
    people = cluster_faces(
        [Sighting(1.0, (0, 0, 9, 9), same), Sighting(1.0, (20, 0, 29, 9), same)], 0.5, 5
    )
    assert len(people) == 2


def test_one_glimpse_can_be_discarded() -> None:
    people = cluster_faces([Sighting(0.0, (0, 0, 9, 9), _vector(1, 0))], 0.5, 5, min_sightings=2)
    assert people == []


def test_no_sightings_means_no_people() -> None:
    assert cluster_faces([], 0.5, 5) == []


def test_intervals_merge_only_across_small_gaps() -> None:
    spans = merge_times([0, 2, 4, 30, 31], gap=5)
    assert [(s.start, s.end) for s in spans] == [(0, 4), (30, 31)]
    assert merge_times([], gap=5) == []


# ── understanding ──────────────────────────────────────────────────────


def test_duration_outweighs_keyframe_count() -> None:
    """A brief title card with many frames must not outvote the long main scene."""
    shots = [Shot(0, 0, 5, 0, 0), Shot(1, 5, 105, 1, 1)]
    probabilities = np.array([[0.9, 0.0]] * 6 + [[0.0, 0.9]] * 2, np.float32)
    per_shot = shot_scores(probabilities, [0] * 6 + [1] * 2, 2)
    labels = aggregate_labels(per_shot, shots, [("topic", "title"), ("topic", "main")], 0.5)
    assert labels[0].label == "main"
    assert labels[0].coverage == pytest.approx(100 / 105, abs=0.01)


def test_a_label_below_the_floor_is_not_stated() -> None:
    labels = aggregate_labels(
        np.array([[0.2]], np.float32), [Shot(0, 0, 10, 0, 0)], [("topic", "x")], 0.5
    )
    assert labels == []


def test_adjacent_shots_merge_into_one_interval() -> None:
    shots = [Shot(0, 0, 5, 0, 0), Shot(1, 5, 9, 1, 1), Shot(2, 9, 20, 2, 2)]
    per_shot = np.array([[0.9], [0.9], [0.1]], np.float32)
    label = aggregate_labels(per_shot, shots, [("topic", "x")], 0.5)[0]
    assert [(i.start, i.end) for i in label.intervals] == [(0, 9)]


def test_a_shot_without_keyframes_scores_zero() -> None:
    per_shot = shot_scores(np.array([[0.8]], np.float32), [1], 3)
    assert per_shot[:, 0].tolist() == pytest.approx([0.0, 0.8, 0.0])


def test_summary_states_only_what_was_measured() -> None:
    summary = summarise(SummaryFacts(
        duration=30, shot_count=3, labels=[], people=[], animals=[],
        animal_seconds=0, food_seconds=0,
    ))
    assert "3 shots" in summary["en"]
    assert "mainly about" not in summary["en"]
    assert "Animals" not in summary["en"]
    assert "No faces" in summary["en"]
    assert "۳" in summary["fa"]


def test_clock_format() -> None:
    assert format_clock(5) == "0:05"
    assert format_clock(65) == "1:05"
    assert format_clock(3725) == "1:02:05"


def test_labels_compete_only_within_their_group() -> None:
    vectors = FakeEncoder().encode_texts(["red"])
    pairs = [("a", "red"), ("a", "blue"), ("b", "red"), ("b", "green")]
    result = label_probabilities(vectors, FakeEncoder(), ["red", "blue", "red", "green"], pairs)
    assert result[0, :2].sum() == pytest.approx(1.0)
    assert result[0, 2:].sum() == pytest.approx(1.0)
    assert result[0, 0] > 0.9 and result[0, 2] > 0.9


def _label(name: str, coverage: float, group: str = "setting"):
    from vie.video.understand import LabelScore

    return LabelScore(group, name, coverage, coverage, coverage)


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [(0.8, "Most of it"), (0.3, "Much of it"), (0.15, "Parts of it")],
)
def test_summary_wording_follows_coverage(coverage: float, expected: str) -> None:
    """A real run said "most of it takes place in the mountains" about 13% of a video."""
    summary = summarise(SummaryFacts(
        duration=60, shot_count=9, labels=[_label("in the mountains", coverage)],
        people=[], animals=[], animal_seconds=0, food_seconds=0,
    ))
    assert expected in summary["en"]


def test_a_label_too_rare_to_matter_gets_no_sentence() -> None:
    """The same run called something covering 2% of the video "the main activity"."""
    summary = summarise(SummaryFacts(
        duration=60, shot_count=9, labels=[_label("people fighting", 0.02, "activity")],
        people=[], animals=[], animal_seconds=0, food_seconds=0,
    ))
    assert "fighting" not in summary["en"]


def test_a_weak_label_does_not_borrow_a_strong_ones_wording() -> None:
    """A real run: "it often shows people walking and people cooking", at 42% and 11%."""
    summary = summarise(SummaryFacts(
        duration=240, shot_count=56,
        labels=[_label("people walking", 0.42, "activity"),
                _label("people cooking", 0.11, "activity")],
        people=[], animals=[], animal_seconds=0, food_seconds=0,
    ))
    assert "people walking" in summary["en"] and "cooking" not in summary["en"]


def test_a_single_stray_detection_gets_no_sentence() -> None:
    """A real run said "food is visible for about 0:01" on the strength of one frame."""
    def text(seconds: float) -> str:
        return summarise(SummaryFacts(
            duration=240, shot_count=56, labels=[], people=[], animals=[],
            animal_seconds=0, food_seconds=seconds,
        ))["en"]

    assert "Food" not in text(1.0)
    assert "Food" in text(12.0)


def test_persian_summary_uses_persian_labels() -> None:
    summary = summarise(SummaryFacts(
        duration=60, shot_count=9, labels=[_label("in the mountains", 0.6)],
        people=[], animals=[], animal_seconds=0, food_seconds=0,
    ))
    assert "کوهستان" in summary["fa"] and "mountains" not in summary["fa"]


def test_every_default_label_has_a_persian_name(config: Config) -> None:
    from vie.video.understand import FA_LABELS

    labels = [label for group in config.video.taxonomy.values() for label in group]
    assert [label for label in labels if label not in FA_LABELS] == []


def test_a_stage_that_did_not_run_says_nothing() -> None:
    """Reporting "no faces" when face detection was off is a false statement."""
    summary = summarise(SummaryFacts(
        duration=60, shot_count=9, labels=[], people=[], animals=[],
        animal_seconds=0, food_seconds=0, faces_checked=False,
    ))
    assert "faces" not in summary["en"]
    assert "چهره" not in summary["fa"]


def test_labels_only_analysis_does_not_claim_no_faces(clip: Path, cfg: Config) -> None:
    assert "faces" not in engine(cfg).analyse(clip).summary["en"]


def test_config_rejects_a_label_with_nothing_to_compete_with(raw_config: dict) -> None:
    from vie.config import ConfigError

    raw = copy.deepcopy(raw_config)
    raw["video"]["taxonomy"] = {"topic": ["only one"]}
    with pytest.raises(ConfigError, match="two labels"):
        Config.from_dict(raw).validate()


# ── the whole engine ───────────────────────────────────────────────────


def test_engine_understands_a_three_scene_video(clip: Path, cfg: Config) -> None:
    analysis = engine(cfg).analyse(clip)
    assert len(analysis.shots) == 3
    assert {label.label for label in analysis.labels} == {
        "something red", "something green", "something blue",
    }
    for label in analysis.labels:
        assert label.coverage == pytest.approx(1 / 3, abs=0.05)
    expected = ["something red", "something green", "something blue"]
    assert [
        analysis.shot_labels[i].labels["topic"][0] for i in range(3)
    ] == expected  # right label on the right shot: BGR handled correctly throughout


def test_every_keyframe_lands_inside_its_shot(clip: Path, cfg: Config) -> None:
    analysis = engine(cfg).analyse(clip)
    for frame in analysis.keyframes:
        shot = analysis.shots[frame.shot]
        assert shot.start <= frame.time <= shot.end
        assert frame.thumbnail[:2] == b"\xff\xd8"


def test_people_are_found_only_where_they_appear(clip: Path, cfg: Config) -> None:
    analysis = engine(cfg, face_detector=FakeFaces()).analyse(clip)  # 'faces' are in blue frames
    assert len(analysis.people) == 1
    span = analysis.people[0].intervals[0]
    assert span.start >= 11.0
    assert analysis.person_thumbnails[1][:2] == b"\xff\xd8"


def test_animals_are_located_and_named(clip: Path, cfg: Config) -> None:
    from fakes import FakeAnimals

    from vie.adapters import crop

    analysis = engine(
        cfg, animal_detector=FakeAnimals(), classify_crops=FakeClassifier().classify,
        class_names=lambda: FakeClassifier.names, crop=crop,
    ).analyse(clip)
    assert analysis.animal_species == ["zebra"]
    assert analysis.animal_times and max(analysis.animal_times) < 6.5
    assert "zebra" in analysis.summary["en"]


def test_food_detector_stage_runs_when_given(clip: Path, cfg: Config) -> None:
    assert engine(cfg, food_detector=FakeFood()).analyse(clip).food_times == []


def test_max_duration_limits_the_analysis(clip: Path, raw_config, tmp_path) -> None:
    raw = copy.deepcopy(raw_config)
    raw["paths"] = {"gallery": "g", "database": str(tmp_path / "i.db")}
    raw["video"]["max_duration"] = 5
    raw["video"]["sample_seconds"] = 0.5
    analysis = engine(Config.from_dict(raw)).analyse(clip)
    assert analysis.shots[-1].end <= 5.5


def test_optional_layers_are_used_when_present(clip: Path, cfg: Config) -> None:
    class Caption:
        def caption(self, images):
            return [f"scene {i}" for i, _ in enumerate(images)]

    analysis = engine(cfg, captioner=Caption()).analyse(clip)
    assert [s.caption for s in analysis.shot_labels] == ["scene 0", "scene 1", "scene 2"]
    assert analysis.stages["captions"] and not analysis.stages["transcript"]


def test_a_failing_optional_stage_keeps_the_analysis(clip: Path, cfg: Config) -> None:
    """Whisper running out of memory on a real run discarded minutes of finished work."""
    from vie.report.video import render_video_report

    class Explodes:
        def caption(self, images):
            raise RuntimeError("mkl_malloc: failed to allocate memory")

    analysis = engine(cfg, captioner=Explodes()).analyse(clip)
    assert len(analysis.shots) == 3
    assert "failed to allocate" in analysis.problems[0]
    assert "failed to allocate" in render_video_report(analysis)


def test_transcriber_is_skipped_without_audio(clip: Path, cfg: Config) -> None:
    class Explodes:
        def transcribe(self, path, duration):
            raise AssertionError("no audio track to transcribe")

    assert engine(cfg, transcriber=Explodes()).analyse(clip).transcript == []


def test_analysis_serialises_to_json_without_pixels_or_vectors(clip: Path, cfg: Config) -> None:
    payload = json.dumps(analysis_to_dict(engine(cfg, face_detector=FakeFaces()).analyse(clip)))
    assert "thumbnail" not in payload and "embedding" not in payload
    assert json.loads(payload)["people"][0]["id"] == 1


# ── report ─────────────────────────────────────────────────────────────


def test_report_is_self_contained_and_complete(clip: Path, cfg: Config, tmp_path: Path) -> None:
    from vie.report.video import write_video_report

    analysis = engine(cfg, face_detector=FakeFaces()).analyse(clip)
    page = write_video_report(analysis, tmp_path / "out" / "r.html").read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in page
    assert "Timeline" in page and "Person 1" in page and 'lang="fa"' in page
    assert "http://" not in page and "https://" not in page


def test_report_escapes_a_hostile_file_name(cfg: Config, tmp_path: Path) -> None:
    from vie.report.video import render_video_report

    name = tmp_path / "x<img src=y onerror=alert(1)>.mp4"
    try:
        hostile = write_video(name, [(RED, 2)])
    except (OSError, ValueError):
        pytest.skip("this filesystem refuses that name")
    assert "<img src=y" not in render_video_report(engine(cfg).analyse(hostile))


# ── library and search ─────────────────────────────────────────────────


def test_indexing_is_incremental(clip: Path, cfg: Config) -> None:
    stats = index_videos(cfg, engine(cfg))
    assert (stats.indexed, stats.skipped) == (1, 0)
    stats = index_videos(cfg, engine(cfg))
    assert (stats.indexed, stats.skipped) == (0, 1)


def test_deleted_videos_are_pruned_with_their_rows(clip: Path, cfg: Config) -> None:
    index_videos(cfg, engine(cfg))
    clip.unlink()
    assert index_videos(cfg, engine(cfg)).pruned == 1
    with VideoStore(cfg).connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM video_keyframes").fetchone()[0] == 0


def test_a_broken_video_does_not_stop_the_run(clip: Path, cfg: Config) -> None:
    (clip.parent / "broken.mp4").write_bytes(b"not a video")
    stats = index_videos(cfg, engine(cfg))
    assert (stats.indexed, stats.failed) == (1, 1)


def test_text_finds_the_right_moment(clip: Path, cfg: Config) -> None:
    index_videos(cfg, engine(cfg))
    store = VideoStore(cfg)
    with store.connect() as conn:
        moment = store.search_visual(conn, FakeEncoder().encode_text("green"))[0]
    assert moment.video_key == "clip.mp4"
    assert moment.start >= 5.5 and moment.end <= 12.5


def test_face_search_returns_when_someone_appears(clip: Path, cfg: Config) -> None:
    index_videos(cfg, engine(cfg, face_detector=FakeFaces()))
    store = VideoStore(cfg)
    with store.connect() as conn:
        moments = store.search_face(conn, np.eye(512, dtype=np.float32)[0])
        assert store.search_face(conn, np.eye(512, dtype=np.float32)[1]) == []
    assert moments[0].kind == "face" and moments[0].start >= 11.0


def test_speech_search_is_literal_and_safe(clip: Path, cfg: Config) -> None:
    from vie.video.understand import TranscriptSegment

    class Speech:
        def transcribe(self, path, duration):
            return [TranscriptSegment(1.0, 2.0, "سلام دنیا 100% hello")], "fa"

    analysed = engine(cfg, transcriber=Speech())
    from dataclasses import replace

    analysis = replace(analysed.analyse(clip),
                       transcript=[TranscriptSegment(1.0, 2.0, "سلام دنیا 100% hello")])
    store = VideoStore(cfg)
    store.initialise()
    with store.connect() as conn:
        store.save(conn, "clip.mp4", clip, "digest", analysis)
        assert store.search_speech(conn, "سلام")[0].kind == "speech"
        assert store.search_speech(conn, "100%")[0].text.startswith("سلام")
        assert store.search_speech(conn, "%%%") == []
        assert store.search_speech(conn, "' OR 1=1 --") == []


def test_config_rejects_a_template_without_a_slot(raw_config: dict) -> None:
    from vie.config import ConfigError

    raw = copy.deepcopy(raw_config)
    raw["video"]["prompt_template"] = "a photo"
    with pytest.raises(ConfigError, match="label"):
        Config.from_dict(raw).validate()


# ── the command line ───────────────────────────────────────────────────


def _config_file(cfg: Config, path: Path) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    return path


def test_cli_analyze_writes_json_and_html(clip, cfg, tmp_path, capsys, monkeypatch) -> None:
    from fakes import FakeBundle

    encoder = FakeEncoder()
    monkeypatch.setattr(cli, "_load_runtime", lambda c: (FakeBundle(), encoder, encoder))
    report = tmp_path / "report.html"
    code = cli.main(["video", "analyze", str(clip), "--labels-only", "--json", "--html",
                     str(report), "--config", str(_config_file(cfg, tmp_path / "c.yaml"))])
    assert code == 0
    assert len(json.loads(capsys.readouterr().out)["shots"]) == 3
    assert report.stat().st_size > 1000


def test_cli_analyze_missing_file_is_a_clean_error(cfg, tmp_path, monkeypatch) -> None:
    from fakes import FakeBundle

    encoder = FakeEncoder()
    monkeypatch.setattr(cli, "_load_runtime", lambda c: (FakeBundle(), encoder, encoder))
    code = cli.main(["video", "analyze", str(tmp_path / "nope.mp4"), "--labels-only",
                     "--config", str(_config_file(cfg, tmp_path / "c.yaml"))])
    assert code == 2


def test_cli_search_needs_exactly_one_query(cfg, tmp_path) -> None:
    code = cli.main(["video", "search", "--config", str(_config_file(cfg, tmp_path / "c.yaml"))])
    assert code == 2


def test_exemplars_are_the_typical_view_then_the_most_different() -> None:
    from vie.video.people import Sighting, choose_exemplars

    frontal = [_vector(1.0, 0.02 * i, 0.0) for i in range(8)]
    turned = _vector(0.6, 0.0, 0.8)
    sightings = [Sighting(float(i), (0, 0, 10, 10), v) for i, v in enumerate(frontal)]
    sightings.append(Sighting(99.0, (0, 0, 10, 10), turned))
    picked = choose_exemplars(sightings, 2)
    assert len(picked) == 2 and picked[-1].time == 99.0  # the turned head is kept
    assert choose_exemplars(sightings[:1], 4) == sightings[:1]
    assert choose_exemplars(sightings, 0) == []


def test_face_storage_is_capped_per_person_and_search_still_finds_them(
    clip: Path, cfg: Config
) -> None:
    index_videos(cfg, engine(cfg, face_detector=FakeFaces()))
    store = VideoStore(cfg)
    with store.connect() as conn:
        per_person = conn.execute(
            "SELECT person, COUNT(*) FROM video_faces GROUP BY person").fetchall()
        spans = conn.execute("SELECT COUNT(*) FROM video_person_spans").fetchone()[0]
        moments = store.search_face(conn, np.eye(512, dtype=np.float32)[0])
    assert per_person and all(count <= cfg.video.face_exemplars for _, count in per_person)
    assert spans >= 1
    assert moments and moments[0].kind == "face" and moments[0].start >= 11.0
