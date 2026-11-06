"""v0.5: face accuracy, the Engine API, result cards and the annotated video."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FakeStack, write_png

from vie.config import Config
from vie.database import Index
from vie.scoring import Match


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def cfg(raw_config: dict, tmp_path: Path) -> Config:
    root = tmp_path / "gallery"
    write_png(root / "red.png", (230, 20, 20))
    write_png(root / "blue.png", (20, 20, 230))
    raw = copy.deepcopy(raw_config)
    raw["paths"] = {"gallery": str(root), "database": str(tmp_path / "index.db")}
    return Config.from_dict(raw)


# ── mirror averaging ──────────────────────────────────────────────────


def test_mirror_vector_is_the_normalised_sum_of_both_views() -> None:
    pytest.importorskip("insightface")
    from vie.adapters import face_vectors

    seen, mirrored = np.array([3.0, 0, 0, 0]), np.array([0, 4.0, 0, 0], dtype=np.float32)

    class Rec:
        input_size = (112, 112)

        def get_feat(self, crops):
            assert crops[0].shape == (112, 112, 3)
            return np.vstack([mirrored for _ in crops])

    face = SimpleNamespace(embedding=seen, normed_embedding=unit(seen),
                           kps=np.array([[38, 52], [74, 52], [56, 72], [42, 92], [70, 92]],
                                        dtype=np.float32))
    app = SimpleNamespace(models={"recognition": Rec()})
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    (vector,) = face_vectors(app, image, [face], flip=True)
    np.testing.assert_allclose(vector, unit([3.0, 4.0, 0, 0]), rtol=1e-6)
    (plain,) = face_vectors(app, image, [face], flip=False)
    np.testing.assert_allclose(plain, unit(seen))


def test_mirror_falls_back_to_the_plain_vector_without_a_recogniser() -> None:
    from vie.adapters import face_vectors

    face = SimpleNamespace(normed_embedding=unit([1, 2, 3]))
    assert face_vectors(SimpleNamespace(), None, [face], flip=True)[0] is face.normed_embedding


# ── several query photos ──────────────────────────────────────────────


class Catalog:
    def __init__(self, rows):
        self.rows = rows
        self.backend = SimpleNamespace(name="exact")

    def __len__(self):
        return len(self.rows)

    def search(self, query, k):
        scored = [(name, bbox, float(vec @ query)) for name, bbox, vec in self.rows]
        return sorted(scored, key=lambda r: -r[2])[:k]


def test_a_second_query_photo_finds_the_pose_the_first_missed(cfg) -> None:
    from vie.search.face import rank_catalog_multi

    frontal, profile = unit([1, 0, 0]), unit([0, 1, 0])
    catalog = Catalog([("front.jpg", "0,0,10,10", frontal),
                       ("side.jpg", "0,0,10,10", unit([0.1, 1, 0])),
                       ("stranger.jpg", "0,0,10,10", unit([0, 0, 1]))])
    paths = {}
    one = rank_catalog_multi([frontal], catalog, paths, cfg)
    both = rank_catalog_multi([frontal, profile], catalog, paths, cfg)
    assert [m.file_name for m in one] == ["front.jpg"]
    assert sorted(m.file_name for m in both) == ["front.jpg", "side.jpg"]


def test_face_search_accepts_several_images_end_to_end(cfg, tmp_path) -> None:
    from vie.query import run_search

    stack = FakeStack(cfg)
    stack.index()
    query = cfg.gallery_path / "blue.png"
    args = SimpleNamespace(kind="face", query="", image=[query, query], top_k=None)
    assert [m.file_name for m in run_search(args, cfg, stack.load_runtime)] == ["blue.png"]


# ── changing the face recipe re-indexes only images with faces ───────


def test_a_new_face_recipe_reindexes_only_the_images_with_faces(cfg) -> None:
    stack = FakeStack(cfg)
    assert stack.index().indexed == 2
    assert stack.index().indexed == 0  # incremental: nothing changed, nothing read
    index = Index(cfg.database_path)
    with index.connect() as conn:
        assert index.get_meta(conn, "face_recipe") == "arcface+flip"
        index.set_meta(conn, "face_recipe", "arcface")  # an index from before v0.5
    stats = stack.index()
    assert stats.indexed == 1  # blue.png has the face; red.png is left alone
    assert stack.index().indexed == 0


# ── the Engine ────────────────────────────────────────────────────────


def test_load_config_merges_overrides_over_the_packaged_defaults(tmp_path) -> None:
    from vie.api import load_config

    config = load_config({"paths": {"gallery": str(tmp_path)}, "face": {"flip_tta": False}})
    assert config.gallery_path == tmp_path
    assert config.face.flip_tta is False
    assert config.face.match_threshold == 0.46  # untouched defaults survive


def test_pruning_most_of_the_index_at_once_is_refused(cfg) -> None:
    """An unmounted drive looks like every photo was deleted; rows must survive it."""
    stack = FakeStack(cfg)
    stack.index()
    for name in ("red.png", "blue.png"):
        (cfg.gallery_path / name).unlink()
    stats = stack.index()
    assert stats.prune_refused and stats.pruned == 0
    with Index(cfg.database_path).connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM gallery_meta").fetchone()[0] == 2
    forced = stack.index(prune_guard=0.0)
    assert forced.pruned == 2 and not forced.prune_refused


def test_engine_reports_a_refused_prune(cfg, monkeypatch) -> None:
    import vie.adapters
    import vie.models
    import vie.pipeline
    from vie.api import Engine

    monkeypatch.setattr(vie.pipeline, "build_index", lambda config, **kw: SimpleNamespace(
        scanned=0, indexed=0, skipped=0, pruned=0, failed=0, prune_refused=True))
    monkeypatch.setattr(vie.adapters, "build_detectors", lambda b, c: SimpleNamespace(
        loader=None, size_of=None, face=None, animal=None, food=None, crop=None))
    monkeypatch.setattr(vie.models, "SpeciesClassifier", lambda b: SimpleNamespace(classify=None))
    engine = Engine(cfg)
    engine._bundle, engine._encoder = SimpleNamespace(), SimpleNamespace(
        embed_batch=None, food_gate_scores=None)
    assert engine.index_images().pruning is False


def test_auto_face_query_is_a_crop_of_the_chosen_face(cfg, tmp_path) -> None:
    from PIL import Image

    from vie.api import Engine

    FakeStack(cfg).index()
    engine = Engine(cfg)
    assert engine.most_photographed_face() == cfg.gallery_path / "blue.png"
    crop = engine.most_photographed_face(crop_to=tmp_path / "query.jpg")
    with Image.open(crop) as image:
        # The fake face is 50x60 inside a 120x90 photo; padded, then clipped to it.
        assert image.size[0] <= 120 and image.size[1] <= 90 and image.size[0] > 50


def test_video_search_needs_exactly_one_of_query_or_image(cfg) -> None:
    from vie.api import Engine

    with pytest.raises(ValueError, match="exactly one"):
        Engine(cfg).search_videos()


# ── presentation ──────────────────────────────────────────────────────


def test_result_card_renders_with_boxes_and_query(tmp_path) -> None:
    pytest.importorskip("PIL")
    from vie.report.showcase import CardStats, save_search_card

    photos = [write_png(tmp_path / f"p{i}.png", (40 * i, 90, 160), (640, 480)) for i in range(5)]
    matches = [Match(p.name, str(p), 0.9 - i * 0.1, (100, 80, 300, 320), label="zebra")
               for i, p in enumerate(photos)]
    out = save_search_card(matches, tmp_path / "card.png", kind="animal", query="zebra",
                           query_images=[photos[0]], stats=CardStats(search_ms=3.2,
                                                                      indexed_images=5))
    from PIL import Image

    with Image.open(out) as image:
        assert image.size[0] == 1600 and image.size[1] > 600


def test_focus_crop_keeps_the_subject_in_the_tile() -> None:
    from PIL import Image

    from vie.report.showcase import focus_crop

    image = Image.new("RGB", (2000, 1000))
    _, box = focus_crop(image, (400, 300), (1700, 400, 1900, 600))
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 300


# ── the annotated video ───────────────────────────────────────────────


def test_tracker_keeps_identities_apart_and_stable() -> None:
    from vie.video.render import FaceTracker

    a, b = unit([1, 0, 0, 0]), unit([0, 1, 0, 0])
    tracker = FaceTracker()
    for step in range(6):
        tracker.update([((10 + step, 10, 60 + step, 60), a), ((200, 10, 250, 60), b)])
    labels = sorted(t.name for t in tracker.tracks)
    assert labels == ["Person 1", "Person 2"]
    assert len(tracker.people) == 2


def test_tracker_names_only_with_a_margin_over_the_runner_up() -> None:
    from vie.video.render import FaceTracker

    known = {"Ava": [unit([1, 0, 0])], "Ben": [unit([0.98, 0.2, 0])]}
    tracker = FaceTracker(known=known, margin=0.05)
    tracker.update([((0, 0, 50, 50), unit([1, 0.1, 0]))])
    assert tracker.tracks[0].name is None  # the two references are too alike to call it
    tracker = FaceTracker(known={"Ava": [unit([1, 0, 0])], "Ben": [unit([0, 1, 0])]})
    tracker.update([((0, 0, 50, 50), unit([1, 0.1, 0]))])
    assert tracker.tracks[0].name == "Ava"


def test_render_writes_a_playable_video(tmp_path) -> None:
    av = pytest.importorskip("av")
    from vie.video.render import render_annotated_video, render_people_card

    source = tmp_path / "in.mp4"
    container = av.open(str(source), mode="w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
    for i in range(30):
        frame = np.full((120, 160, 3), 40 + i * 3, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    face = unit([1, 0, 0, 0])
    result = render_annotated_video(source, tmp_path / "out.mp4",
                                    detect=lambda img: [((40, 30, 90, 90), face)],
                                    title="test", detect_every=2)
    assert result.frames == 30
    assert [p.label for p in result.people] == ["Person 1"]
    with av.open(str(result.path)) as check:
        assert sum(1 for _ in check.decode(video=0)) == 30
    assert render_people_card(result, title="test").size[0] == 1600


def test_shot_labeller_reads_labels_of_the_current_shot() -> None:
    from vie.api import shot_labeller

    analysis = SimpleNamespace(
        shots=[SimpleNamespace(index=1, start=5.0, end=9.0),
               SimpleNamespace(index=0, start=0.0, end=5.0)],
        shot_labels=[SimpleNamespace(shot=0, labels={"topic": ("an interview", 0.8),
                                                     "setting": ("in a studio", 0.6)}),
                     SimpleNamespace(shot=1, labels={"topic": ("a lecture", 0.3)})],
    )
    at = shot_labeller(analysis, threshold=0.4)
    assert at(2.0) == ["an interview", "in a studio"]
    assert at(6.0) == []          # below the presence threshold
    assert at(20.0) == []         # past the last shot
    assert shot_labeller(analysis, threshold=0.2)(6.0) == ["a lecture"]


def test_right_to_left_text_is_prepared_and_latin_is_untouched() -> None:
    from vie.report.showcase import display_text, font

    assert display_text("Person 1  93%") == "Person 1  93%"
    assert isinstance(display_text("دانشجو ۱۰۴۲"), str)
    assert font(20) is font(20)  # cached: looking up a face walks font folders


# ── incremental runs stay incremental ─────────────────────────────────


def test_a_touched_but_unchanged_file_is_hashed_once_then_never_again(cfg, monkeypatch) -> None:
    """A sync or a copy moves the mtime. Hashing then must happen once, not forever."""
    import os

    import vie.pipeline as pipeline

    stack = FakeStack(cfg)
    stack.index()
    target = cfg.gallery_path / "red.png"
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10**9))
    hashed = []
    real = pipeline.content_hash
    monkeypatch.setattr(pipeline, "content_hash", lambda p: hashed.append(p) or real(p))
    assert stack.index().indexed == 0
    assert hashed == [target]
    assert stack.index().indexed == 0
    assert hashed == [target]  # the new mtime was remembered


def test_a_moved_gallery_keeps_its_index_and_points_at_the_new_place(
    cfg, raw_config, tmp_path
) -> None:
    import shutil

    FakeStack(cfg).index()
    moved = tmp_path / "moved"
    shutil.copytree(cfg.gallery_path, moved)  # copytree keeps contents, not always mtimes
    shutil.rmtree(cfg.gallery_path)
    raw = copy.deepcopy(raw_config)
    raw["paths"] = {"gallery": str(moved), "database": str(cfg.database_path)}
    stats = FakeStack(Config.from_dict(raw)).index()
    assert stats.indexed == 0 and stats.pruned == 0
    with Index(cfg.database_path).connect() as conn:
        paths = {Path(p).parent for (p,) in conn.execute("SELECT file_path FROM gallery_meta")}
    assert paths == {moved}
