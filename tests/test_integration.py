"""End-to-end behaviour through the real wiring, with only the weights faked.

Each test here corresponds to something the v0.2 suite could not see: every piece
had a passing unit test, and the assembled system did not work. See docs/AUDIT.md.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FakeClassifier, FakeStack, write_png

from vie import cli
from vie.config import Config
from vie.database import SCHEMA_VERSION, EmbeddingModelMismatch, Index, SchemaError
from vie.embeddings import to_blob
from vie.query import run_search

RED = (230, 20, 20)
GREEN = (20, 230, 20)
BLUE = (20, 20, 230)


@pytest.fixture()
def gallery(tmp_path: Path) -> Path:
    root = tmp_path / "gallery"
    write_png(root / "red.png", RED)
    write_png(root / "green.png", GREEN)
    write_png(root / "blue.png", BLUE)
    return root


@pytest.fixture()
def cfg(raw_config: dict, gallery: Path, tmp_path: Path) -> Config:
    raw = copy.deepcopy(raw_config)
    raw["paths"] = {"gallery": str(gallery), "database": str(tmp_path / "index.db")}
    return Config.from_dict(raw)


@pytest.fixture()
def stack(cfg: Config) -> FakeStack:
    stack = FakeStack(cfg)
    assert stack.index().indexed == 3
    return stack


def request(kind: str, query: str = "", image=None, top_k=None) -> SimpleNamespace:
    return SimpleNamespace(kind=kind, query=query, image=image, top_k=top_k)


# ── all four advertised query paths actually run ───────────────────────


def test_text_search_finds_by_colour_through_the_full_path(stack, cfg) -> None:
    """Fails if BGR reaches the encoder unconverted: 'red' would return blue."""
    matches = run_search(request("text", "something red"), cfg, stack.load_runtime)
    assert matches[0].file_name == "red.png"


def test_species_search_in_vocabulary_runs_no_model(stack, cfg) -> None:
    """v0.2: AttributeError on every call, because search.animal has no rank()."""
    matches = run_search(request("animal", "zebra"), cfg, stack.load_runtime)
    assert [m.file_name for m in matches] == ["red.png"]
    assert matches[0].label == "zebra"
    assert stack.runtime_loads == 0  # the documented promise: pure SQL


def test_species_search_out_of_vocabulary_falls_back_to_the_encoder(stack, cfg) -> None:
    """'quokka' is not an ImageNet class; the encoder must answer, not crash."""
    matches = run_search(request("animal", "red quokka"), cfg, stack.load_runtime)
    assert stack.runtime_loads == 1
    assert isinstance(matches, list)


def test_food_search_reaches_a_dish_the_detector_never_fired_on(stack, cfg) -> None:
    """The stated reason for the food gate: kebab fires none of COCO's food classes."""
    matches = run_search(request("food", "kebab"), cfg, stack.load_runtime)
    assert [m.file_name for m in matches] == ["green.png"]


def test_face_search_returns_the_image_with_the_face(stack, cfg, gallery) -> None:
    query = gallery / "blue.png"
    matches = run_search(request("face", image=query), cfg, stack.load_runtime)
    assert [m.file_name for m in matches] == ["blue.png"]


def test_face_search_without_an_image_is_a_clean_error(stack, cfg) -> None:
    """The demo's Search button with nothing uploaded used to raise AttributeError."""
    with pytest.raises(ValueError, match="query image"):
        run_search(request("face"), cfg, stack.load_runtime)


@pytest.mark.parametrize("kind", ["animal", "food"])
def test_top_k_is_honoured_on_the_box_paths(stack, cfg, kind) -> None:
    """--top-k was accepted on the command line and silently ignored."""
    assert len(run_search(request(kind, "zebra", top_k=0), cfg, stack.load_runtime)) == 0


# ── upgrades that existed in code but were never called ────────────────


def test_multi_crop_is_actually_indexed(stack, cfg) -> None:
    index = Index(cfg.database_path)
    with index.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM clip_embeddings WHERE file_name = 'red.png'"
        ).fetchone()[0]
    assert count == 6


def test_food_gate_writes_its_columns(stack, cfg) -> None:
    with Index(cfg.database_path).connect() as conn:
        rows = dict(conn.execute("SELECT file_name, food_by_vlm FROM gallery_meta"))
    assert rows == {"green.png": 1, "red.png": 0, "blue.png": 0}


def test_species_classifier_runs_at_index_time(stack, cfg) -> None:
    assert stack.classifier.crops_seen == 1
    index = Index(cfg.database_path)
    with index.connect() as conn:
        assert index.classifier_vocabulary(conn) == FakeClassifier.names
        stored = list(index.animal_predictions(conn, (340,)))
    assert stored[0][0] == "red.png" and stored[0][3] == 340


@pytest.mark.parametrize(
    ("margin", "accepted"), [(0.5, False), (1.5, False), (6.0, True)]
)
def test_recognition_needs_a_clear_margin_over_the_best_rival(cfg, margin, accepted) -> None:
    """The measured rule, through both paths. SigLIP logits are all negative when
    healthy, so what decides is the gap, not the absolute value."""
    from vie.search import animal, food

    class Scorer:
        def score(self, crops, prompts):
            return [[-2.0, -2.0 - margin, -2.0 - margin - 1.0]]

    a = animal.Candidate("x", "x", (0, 0, 100, 100), 0.9)
    assert bool(animal.rank_with_clip("zebra", [a], [object()], Scorer(), cfg)) is accepted
    f = food.Candidate("x", "x", (0, 0, 100, 100), 0.9)
    assert bool(food.rank("kebab", [f], [object()], Scorer(), cfg)) is accepted


def test_recognition_reads_the_mode_from_config(cfg, monkeypatch) -> None:
    """v0.2 had a scoring function for each model and called neither from search."""
    from vie.search import animal

    modes: list[str] = []

    def spy(logits, mode):
        modes.append(mode)
        return 1.0

    monkeypatch.setattr(animal, "hypothesis_score", spy)

    class Scorer:
        def score(self, crops, prompts):
            return [[1.0, 0.0, 0.0]]

    a = animal.Candidate("x", "x", (0, 0, 100, 100), 0.9)
    animal.rank_with_clip("zebra", [a], [object()], Scorer(), cfg)
    assert modes == [cfg.score_mode]


def test_text_scores_are_calibrated_for_display(stack, cfg) -> None:
    match = run_search(request("text", "red"), cfg, stack.load_runtime)[0]
    assert match.confidence is not None
    assert match.display_score == match.confidence


def test_max_index_dimension_is_read_from_config(raw_config, tmp_path) -> None:
    """The key existed; the code hard-coded 1600 and ignored it."""
    raw = copy.deepcopy(raw_config)
    raw["face"]["max_index_dimension"] = 60
    write_png(tmp_path / "g" / "big.png", RED, size=(600, 300))
    raw["paths"] = {"gallery": str(tmp_path / "g"), "database": str(tmp_path / "i.db")}
    config = Config.from_dict(raw)
    stack = FakeStack(config)
    stack.index(animal_detector=stack.animals, classify_crops=None, class_names=None)
    with Index(config.database_path).connect() as conn:
        scale = conn.execute("SELECT index_scale FROM gallery_meta").fetchone()[0]
    assert scale == pytest.approx(0.1)


def test_boxes_are_returned_in_original_coordinates(raw_config, tmp_path) -> None:
    raw = copy.deepcopy(raw_config)
    raw["face"]["max_index_dimension"] = 300
    write_png(tmp_path / "g" / "big.png", RED, size=(1200, 900))
    raw["paths"] = {"gallery": str(tmp_path / "g"), "database": str(tmp_path / "i.db")}
    config = Config.from_dict(raw)
    stack = FakeStack(config)
    stack.index()
    match = run_search(request("animal", "zebra"), config, stack.load_runtime)[0]
    assert match.box == (20, 20, 400, 320)  # (5, 5, 100, 80) at scale 0.25


def test_non_ascii_file_names_are_indexed(raw_config, tmp_path) -> None:
    """cv2.imread returns None for non-ASCII paths on Windows."""
    raw = copy.deepcopy(raw_config)
    write_png(tmp_path / "g" / "عکس-قرمز.png", RED)
    raw["paths"] = {"gallery": str(tmp_path / "g"), "database": str(tmp_path / "i.db")}
    assert FakeStack(Config.from_dict(raw)).index().indexed == 1


def test_gallery_subfolders_with_the_same_file_name_stay_distinct(raw_config, tmp_path) -> None:
    raw = copy.deepcopy(raw_config)
    write_png(tmp_path / "g" / "2024" / "IMG_0001.png", RED)
    write_png(tmp_path / "g" / "2025" / "IMG_0001.png", GREEN)
    raw["paths"] = {"gallery": str(tmp_path / "g"), "database": str(tmp_path / "i.db")}
    config = Config.from_dict(raw)
    FakeStack(config).index()
    with Index(config.database_path).connect() as conn:
        names = {r[0] for r in conn.execute("SELECT file_name FROM gallery_meta")}
    assert names == {"2024/IMG_0001.png", "2025/IMG_0001.png"}


# ── pipeline guarantees ────────────────────────────────────────────────


def test_prefetch_is_bounded() -> None:
    """v0.2 decoded the entire gallery into RAM before processing the first image."""
    import time

    from vie.pipeline import _prefetch

    seen: list[int] = []

    def load(item: int) -> int:
        seen.append(item)
        return item

    stream = _prefetch(list(range(1000)), load, workers=4, window=8)
    next(stream)
    time.sleep(0.2)
    assert len(seen) <= 10
    assert [item for _, item in stream] == list(range(1, 1000))


def test_order_survives_prefetching() -> None:
    from vie.pipeline import _prefetch

    assert [v for _, v in _prefetch(list(range(50)), lambda x: x * 2, workers=4)] == [
        x * 2 for x in range(50)
    ]


def test_inference_is_batched(raw_config, tmp_path) -> None:
    raw = copy.deepcopy(raw_config)
    for i in range(6):
        write_png(tmp_path / "g" / f"{i}.png", RED)
    raw["paths"] = {"gallery": str(tmp_path / "g"), "database": str(tmp_path / "i.db")}
    config = Config.from_dict(raw)
    stack = FakeStack(config)
    stats = stack.index(batch_size=4)
    assert stats.indexed == 6
    assert stack.encoder.forward_passes == 2
    # Four and two images, six scene crops each, plus the fake animal every red image
    # holds: its object vector rides in the same forward pass, not a second one.
    assert stack.encoder.batch_sizes == [28, 14]
    assert stack.food.batches == [4, 2]


def test_a_failed_store_leaves_no_half_written_image(raw_config, tmp_path) -> None:
    """A row with the new hash and no vectors was never retried: unsearchable forever."""
    raw = copy.deepcopy(raw_config)
    write_png(tmp_path / "g" / "bad.png", RED)
    raw["paths"] = {"gallery": str(tmp_path / "g"), "database": str(tmp_path / "i.db")}
    config = Config.from_dict(raw)
    stack = FakeStack(config)

    def corrupt(images, regions):
        return [[("whole", np.full(8, np.nan, dtype=np.float32))] for _ in images]

    stats = stack.index(embed_batch=corrupt, food_gate=None)
    assert (stats.indexed, stats.failed) == (0, 1)
    with Index(config.database_path).connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM gallery_meta").fetchone()[0] == 0
    # ...so a healthy second run picks it up.
    assert stack.index().indexed == 1


def test_one_bad_image_does_not_cost_its_batch(raw_config, tmp_path) -> None:
    raw = copy.deepcopy(raw_config)
    for i in range(4):
        write_png(tmp_path / "g" / f"{i}.png", RED)
    raw["paths"] = {"gallery": str(tmp_path / "g"), "database": str(tmp_path / "i.db")}
    config = Config.from_dict(raw)
    stack = FakeStack(config)
    real = stack.encoder.embed_batch
    calls = {"n": 0}

    def flaky(images, regions):
        calls["n"] += 1
        if calls["n"] == 3:  # the first batch fails; this is a single-image retry
            raise RuntimeError("poison")
        if calls["n"] == 1:
            raise RuntimeError("batch failed")
        return real(images, regions)

    stats = stack.index(embed_batch=flaky, batch_size=4)
    assert (stats.indexed, stats.failed) == (3, 1)


def test_unchanged_files_are_not_rehashed(stack, cfg, monkeypatch) -> None:
    """An incremental run must not read every byte of the archive again."""
    import vie.pipeline as pipeline

    def explode(path):
        raise AssertionError(f"hashed {path}")

    monkeypatch.setattr(pipeline, "content_hash", explode)
    assert stack.index().skipped == 3


def test_rehash_detects_a_same_size_same_mtime_edit(stack, cfg, gallery) -> None:
    import os

    target = gallery / "red.png"
    stat = target.stat()
    write_png(target, RED, size=(121, 90))
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert stack.index(rehash=True).indexed == 1


# ── index integrity ────────────────────────────────────────────────────


def test_an_index_from_the_oldest_shape_is_upgraded_in_place(tmp_path) -> None:
    """v0.1 had no region column, and 'CREATE IF NOT EXISTS' never added one."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
           INSERT INTO schema_meta VALUES ('version', '1');
           CREATE TABLE gallery_meta (file_name TEXT PRIMARY KEY, file_path TEXT NOT NULL,
             content_hash TEXT NOT NULL, file_size INTEGER NOT NULL,
             has_face INTEGER NOT NULL DEFAULT 0, has_animal INTEGER NOT NULL DEFAULT 0,
             has_food INTEGER NOT NULL DEFAULT 0);
           CREATE TABLE clip_embeddings (file_name TEXT PRIMARY KEY, embedding BLOB NOT NULL);
           INSERT INTO gallery_meta (file_name, file_path, content_hash, file_size)
             VALUES ('a.jpg', '/a.jpg', 'h', 1);"""
    )
    conn.execute("INSERT INTO clip_embeddings VALUES ('a.jpg', ?)",
                 (to_blob(np.ones(4, dtype=np.float32)),))
    conn.commit()
    conn.close()

    index = Index(path)
    index.initialise()
    with index.connect() as conn:
        assert Index.schema_version(conn) == SCHEMA_VERSION
        rows = list(index.iter_clip(conn))
        assert rows[0][0] == "a.jpg" and rows[0][3] == "whole"
        index.upsert_image(conn, file_name="a.jpg", file_path="/a.jpg", hash_="h2", size=2,
                           has_face=False, has_animal=False, has_food=True, food_by_vlm=True)
        assert index.candidates(conn, "has_food")[0].file_name == "a.jpg"


def test_an_index_from_the_future_is_refused(tmp_path) -> None:
    path = tmp_path / "future.db"
    index = Index(path)
    index.initialise()
    with index.connect() as conn:
        index.set_meta(conn, "version", str(SCHEMA_VERSION + 1))
    with pytest.raises(SchemaError, match="upgrade vie"):
        index.initialise()


def test_mixing_embedding_models_is_refused(stack, cfg, raw_config) -> None:
    other = copy.deepcopy(cfg.raw)
    other["models"]["vision_language"] = "google/siglip-base-patch16-224"
    with pytest.raises(EmbeddingModelMismatch, match="--rebuild"):
        run_search(request("text", "red"), Config.from_dict(other), stack.load_runtime)


def test_rebuild_allows_a_new_embedding_model(stack, cfg) -> None:
    other = copy.deepcopy(cfg.raw)
    other["models"]["vision_language"] = "google/siglip-base-patch16-224"
    config = Config.from_dict(other)
    with pytest.raises(EmbeddingModelMismatch):
        FakeStack(config).index()
    assert FakeStack(config).index(rebuild=True).indexed == 3


# ── the command line itself ────────────────────────────────────────────


def _write_config(cfg: Config, path: Path) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(cfg.raw), encoding="utf-8")
    return path


def test_json_flag_is_accepted_after_the_subcommand(stack, cfg, tmp_path, capsys, monkeypatch):
    """The README's own example failed with 'unrecognized arguments: --json'."""
    monkeypatch.setattr(cli, "_load_runtime", stack.load_runtime)
    config_path = _write_config(cfg, tmp_path / "c.yaml")
    code = cli.main(["search", "text", "--query", "red", "--json", "--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload[0]["file_name"] == "red.png"
    assert payload[0]["raw_score"] != payload[0]["score"] or payload[0]["score"] <= 1


def test_json_flag_is_accepted_before_the_subcommand(stack, cfg, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_load_runtime", stack.load_runtime)
    config_path = _write_config(cfg, tmp_path / "c.yaml")
    code = cli.main(["--json", "--config", str(config_path), "search", "animal", "--query",
                     "zebra"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)[0]["label"] == "zebra"


def test_plain_output_and_html_report(stack, cfg, tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_load_runtime", stack.load_runtime)
    config_path = _write_config(cfg, tmp_path / "c.yaml")
    out = tmp_path / "reports" / "r.html"
    code = cli.main(["search", "animal", "--query", "zebra", "--html", str(out),
                     "--config", str(config_path)])
    assert code == 0
    assert "red.png" in capsys.readouterr().out
    page = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in page
    assert 'class="box"' in page
    assert "http://" not in page and "https://" not in page  # reports never call out


def test_report_escapes_hostile_file_names(tmp_path) -> None:
    from vie.report.search import render_search_report
    from vie.scoring import Match

    page = render_search_report(
        [Match("<script>alert(1)</script>.png", "/missing", 0.5)], kind="text", query="<b>q</b>"
    )
    assert "<script>alert" not in page and "<b>q</b>" not in page


def test_doctor_reports_without_loading_models(stack, cfg, tmp_path, capsys) -> None:
    config_path = _write_config(cfg, tmp_path / "c.yaml")
    code = cli.main(["doctor", "--json", "--config", str(config_path)])
    checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)}
    assert code == 0
    assert checks["images"]["detail"] == "3"
    assert checks["images without vectors"]["detail"] == "0"
    assert checks["schema"]["status"] == "ok"


def test_version_flag(capsys) -> None:
    from vie import __version__

    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_default_config_ships_inside_the_package() -> None:
    """It lived at the repo root, so only an editable install could find it."""
    import vie
    from vie.config import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH.is_file()
    assert Path(vie.__file__).parent in DEFAULT_CONFIG_PATH.parents


# ── every config key must be read by something ─────────────────────────


def test_no_dead_config_keys(raw_config: dict) -> None:
    """Most of what was broken here was a config key nothing read.

    Listed keys are the only ones allowed to be absent from the source; adding to
    this list should feel like what it is, an admission.
    """
    allowed_unread = {
        "project.name",   # descriptive
        "project.seed",   # nothing in the pipeline samples randomly
    }
    source = "\n".join(
        p.read_text(encoding="utf-8")
        for p in Path(__file__).resolve().parents[1].joinpath("src", "vie").rglob("*.py")
    )

    def walk(node: dict, prefix: str = "") -> list[str]:
        keys = []
        for key, value in node.items():
            path = f"{prefix}{key}"
            keys += walk(value, f"{path}.") if isinstance(value, dict) else [path]
        return keys

    dead = [
        path for path in walk(raw_config)
        if path not in allowed_unread and path.rsplit(".", 1)[-1] not in source
    ]
    assert dead == []
