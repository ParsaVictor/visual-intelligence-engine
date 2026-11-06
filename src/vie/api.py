"""One object for the whole system: index, search, understand video, present.

The CLI starts a process per command, so every command loads its models again.
A notebook, a demo or a service wants the opposite: load each model once and use
it for everything. :class:`Engine` is that, built only from the tested pieces the
CLI uses, so both paths give identical answers.

    from vie.api import Engine
    engine = Engine.from_overrides({"paths": {"gallery": "/data/photos"}})
    report = engine.index_images()          # first run: everything; later: only new files
    result = engine.search_face(["me.jpg"])
    engine.save_card(result, "face.png")

Speed comes from not doing work twice:

* an image whose size and modification time are unchanged is skipped without
  being opened, and an index that is up to date loads no model at all;
* every model loads on first use and stays loaded for the life of the engine;
* searches narrow candidates in SQL before any model runs.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from vie.config import DEFAULT_CONFIG_PATH, Config
from vie.logging_setup import get_logger
from vie.scoring import Match

log = get_logger("api")


def merge(base: dict, extra: dict) -> dict:
    """Deep-merge ``extra`` over ``base`` without mutating either."""
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(overrides: dict | None = None, base: str | Path | None = None) -> Config:
    """The packaged defaults (or ``base``), with ``overrides`` merged on top."""
    path = Path(base) if base else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = Config.from_dict(merge(raw, overrides or {}))
    config.validate()
    return config


@dataclass
class IndexReport:
    scanned: int
    indexed: int
    skipped: int
    pruned: int
    failed: int
    seconds: float
    pruning: bool = True

    @property
    def images_per_second(self) -> float:
        return self.indexed / self.seconds if self.seconds else 0.0

    def summary(self) -> str:
        return (f"{self.scanned} found · {self.indexed} newly indexed · {self.skipped} unchanged "
                f"(skipped) · {self.pruned} removed · {self.failed} failed · {self.seconds:.1f}s")


@dataclass
class SearchResult:
    kind: str
    query: str
    matches: list[Match]
    milliseconds: float
    query_images: list[Path] = field(default_factory=list)


class Engine:
    """Every capability of the system behind one long-lived object."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._bundle: Any = None
        self._encoder: Any = None
        self._video_engines: dict[Any, Any] = {}

    @classmethod
    def from_overrides(cls, overrides: dict | None = None,
                       base: str | Path | None = None) -> Engine:
        return cls(load_config(overrides, base))

    # ── models, each loaded once ──────────────────────────────────────

    @property
    def bundle(self) -> Any:
        if self._bundle is None:
            from vie.models import load_all

            self._bundle = load_all(self.config)
        return self._bundle

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            from vie.models import VisionLanguageEncoder

            self._encoder = VisionLanguageEncoder(self.bundle)
        return self._encoder

    def _runtime(self, _config: Config | None = None) -> tuple[Any, Any, Any]:
        return self.bundle, self.encoder, self.encoder

    # ── the archive ───────────────────────────────────────────────────

    def counts(self) -> dict[str, int]:
        """How much is indexed. Opens no model."""
        from vie.database import Index

        if not Path(self.config.database_path).exists():
            return {"images": 0, "faces": 0, "animals": 0, "food": 0, "videos": 0}
        index = Index(self.config.database_path)
        index.initialise()
        with index.connect() as conn:
            def one(sql: str) -> int:
                try:
                    return int(conn.execute(sql).fetchone()[0])
                except Exception:  # noqa: BLE001 - a table that does not exist yet
                    return 0
            return {
                "images": one("SELECT COUNT(*) FROM gallery_meta"),
                "faces": one("SELECT COUNT(*) FROM face_embeddings"),
                "animals": one("SELECT COUNT(*) FROM gallery_meta WHERE has_animal = 1"),
                "food": one("SELECT COUNT(*) FROM gallery_meta WHERE has_food = 1"),
                "videos": one("SELECT COUNT(*) FROM videos"),
            }

    def index_images(
        self,
        *,
        rebuild: bool = False,
        rehash: bool = False,
        prune: bool = True,
        progress: Callable[[Any], None] | None = None,
    ) -> IndexReport:
        """Bring the index up to date with the gallery folder.

        The first run reads every image. After that only new or changed files are
        read; everything else is skipped from its size and modification time, and if
        nothing changed no model is even loaded.

        Pruning removes rows for files no longer on disk, but it is refused when most
        of the archive seems to have vanished at once: that is a disconnected drive or
        a mistyped path far more often than a real deletion.
        """
        from vie.adapters import build_detectors
        from vie.models import SpeciesClassifier
        from vie.pipeline import build_index, discover_images

        config = self.config
        found = len(discover_images(config.gallery_path, config.indexing.recursive))
        known = self.counts()["images"]
        if prune and known and found < known * 0.5:
            log.warning("only %d of %d indexed images are on disk; not pruning. Check the "
                        "gallery path and run again.", found, known)
            prune = False

        bundle = self.bundle
        detectors = build_detectors(bundle, config)
        encoder = self.encoder
        classifier = SpeciesClassifier(bundle)
        started = time.perf_counter()
        stats = build_index(
            config,
            loader=detectors.loader,
            size_of=detectors.size_of,
            face_detector=detectors.face,
            animal_detector=detectors.animal,
            food_detector=detectors.food,
            embed_batch=encoder.embed_batch,
            food_gate=lambda vectors: encoder.food_gate_scores(vectors, config),
            classify_crops=classifier.classify,
            crop=detectors.crop,
            class_names=lambda: bundle.class_names,
            prune=prune,
            rebuild=rebuild,
            rehash=rehash,
            on_progress=progress,
        )
        return IndexReport(stats.scanned, stats.indexed, stats.skipped, stats.pruned,
                           stats.failed, time.perf_counter() - started, prune)

    # ── search ────────────────────────────────────────────────────────

    def _search(self, kind: str, *, query: str = "", images: Sequence[Path] = (),
                top_k: int | None = None) -> SearchResult:
        from vie.query import run_search

        args = SimpleNamespace(kind=kind, query=query, top_k=top_k,
                               image=[Path(p) for p in images] or None)
        started = time.perf_counter()
        matches = run_search(args, self.config, self._runtime)
        shown = query or " + ".join(Path(p).name for p in images)
        return SearchResult(kind, shown, matches, (time.perf_counter() - started) * 1000,
                            [Path(p) for p in images])

    def search_face(self, images: str | Path | Sequence[str | Path],
                    top_k: int | None = None) -> SearchResult:
        """Every photo of this person. Several photos of them find more."""
        if isinstance(images, str | Path):
            images = [images]
        return self._search("face", images=[Path(p) for p in images], top_k=top_k)

    def search_animal(self, query: str, top_k: int | None = None) -> SearchResult:
        return self._search("animal", query=query, top_k=top_k)

    def search_food(self, query: str, top_k: int | None = None) -> SearchResult:
        return self._search("food", query=query, top_k=top_k)

    def search_text(self, query: str, top_k: int | None = None) -> SearchResult:
        return self._search("text", query=query, top_k=top_k or self.config.top_k)

    def most_photographed_face(self, sample: int = 3000) -> Path | None:
        """The photo of whoever appears most often: a sensible demo query.

        Counts, for a sample of indexed faces, how many others clear the match
        threshold, and returns the image holding the best-connected face.
        """
        import numpy as np

        from vie.database import Index

        if not Path(self.config.database_path).exists():
            return None
        index = Index(self.config.database_path)
        with index.connect() as conn:
            rows = list(index.iter_faces(conn))
        if not rows:
            return None
        rng = np.random.default_rng(self.config.seed)
        if len(rows) > sample:
            rows = [rows[i] for i in rng.choice(len(rows), sample, replace=False)]
        vectors = np.vstack([np.asarray(r[2], dtype=np.float32) for r in rows])
        degree = (vectors @ vectors.T >= self.config.face.match_threshold).sum(axis=1)
        # Prefer a large face among the best connected, for a clear query picture.
        area = np.array([(b[2] - b[0]) * (b[3] - b[1]) for _, b, _, _ in rows], dtype=float)
        best = np.lexsort((-area, -degree))[0]
        return Path(rows[best][3])

    # ── presentation ──────────────────────────────────────────────────

    def save_card(self, result: SearchResult, path: str | Path, **kwargs: Any) -> Path:
        """A presentation-grade PNG or JPEG of one search."""
        from vie.report.showcase import CardStats, save_search_card

        return save_search_card(
            result.matches, path, kind=result.kind, query=result.query,
            query_images=result.query_images,
            stats=CardStats(search_ms=result.milliseconds,
                            indexed_images=self.counts()["images"]),
            **kwargs,
        )

    def save_html(self, result: SearchResult, path: str | Path) -> Path:
        from vie.report import write_search_report

        return write_search_report(result.matches, path, kind=result.kind, query=result.query)

    # ── faces as a function, for video ────────────────────────────────

    def face_detect(self) -> Callable[[Any], list[tuple[Any, Any]]]:
        """``frame -> [(box, unit vector)]`` with exactly the index's face recipe."""
        from vie.adapters import InsightFaceDetector

        detector = InsightFaceDetector(self.bundle, self.config)
        return detector.detect

    def known_people(self, folder: str | Path) -> dict[str, list[Any]]:
        """Reference vectors from ``folder/<name>/*.jpg``, one entry per person."""
        from vie.pipeline import IMAGE_SUFFIXES
        from vie.query import query_face_embedding

        root = Path(folder)
        people: dict[str, list[Any]] = {}
        if not root.is_dir():
            return people
        for person in sorted(p for p in root.iterdir() if p.is_dir()):
            vectors = []
            for photo in sorted(person.iterdir()):
                if photo.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    vectors.append(query_face_embedding(photo, self.config, self.bundle))
                except ValueError as exc:
                    log.warning("%s", exc)
            if vectors:
                people[person.name] = vectors
        return people

    # ── video ─────────────────────────────────────────────────────────

    def video_engine(self, stages: Any = None) -> Any:
        from vie.video.library import Stages, build_engine

        stages = stages or Stages.from_config(self.config)
        if stages not in self._video_engines:
            self._video_engines[stages] = build_engine(self.config, self.bundle, self.encoder,
                                                       stages)
        return self._video_engines[stages]

    def index_videos(self, *, stages: Any = None, prune: bool = True,
                     on_analysed: Callable[[str, Path, Any], None] | None = None) -> Any:
        """Analyse only new or changed videos; ``on_analysed`` sees each one."""
        from vie.video.library import index_videos

        return index_videos(self.config, self.video_engine(stages), prune=prune,
                            on_analysed=on_analysed)

    def analyse_video(self, path: str | Path, *, max_duration: float | None = None,
                      stages: Any = None) -> Any:
        if max_duration is None:
            return self.video_engine(stages).analyse(path)
        from vie.video.library import Stages, build_engine

        config = replace(self.config, video=replace(self.config.video, max_duration=max_duration))
        engine = build_engine(config, self.bundle, self.encoder,
                              stages or Stages.from_config(config))
        return engine.analyse(path)

    def render_video(self, source: str | Path, output: str | Path, *,
                     known: dict[str, list[Any]] | None = None, analysis: Any = None,
                     title: str | None = None, **kwargs: Any) -> Any:
        """An annotated MP4: tracked faces, names or people, labels and a timeline."""
        from vie.video.render import render_annotated_video

        return render_annotated_video(
            source, output, detect=self.face_detect(), known=known, title=title,
            shot_labels=shot_labeller(analysis) if analysis is not None else None,
            match_threshold=self.config.face.match_threshold,
            cluster_threshold=self.config.video.face_cluster_threshold,
            **kwargs,
        )

    def search_videos(self, query: str | None = None, image: str | Path | None = None,
                      top_k: int = 10) -> list[Any]:
        """Moments across every indexed video, by description, speech or face."""
        from vie.query import query_face_embedding
        from vie.video.store import VideoStore

        store = VideoStore(self.config)
        store.initialise()
        with store.connect() as conn:
            store.index.check_embedding_model(conn, self.config.models["vision_language"])
            if image is not None:
                vector = query_face_embedding(image, self.config, self.bundle)
                return store.search_face(conn, vector, top_k)
            moments = store.search_speech(conn, query, top_k)
            moments += store.search_visual(conn, self.encoder.encode_text(query), top_k=top_k)
            return sorted(moments, key=lambda m: -m.score)[:top_k]


def shot_labeller(analysis: Any) -> Callable[[float], list[str]]:
    """``time -> the shot's labels``, for the annotated video's header."""
    shots = list(getattr(analysis, "shots", []) or [])
    labels = {item.shot: item for item in getattr(analysis, "shot_labels", []) or []}

    def at(t: float) -> list[str]:
        for shot in shots:
            if shot.start <= t < shot.end + 1e-6:
                found = labels.get(shot.index)
                if found is None:
                    return []
                ranked = sorted(found.labels.items(), key=lambda kv: -kv[1][1])
                return [label for _, (label, score) in ranked if score >= 0.4][:2]
        return []

    return at
