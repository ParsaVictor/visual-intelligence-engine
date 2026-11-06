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

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from vie.config import DEFAULT_CONFIG_PATH, Config, merge
from vie.logging_setup import get_logger
from vie.scoring import Match

log = get_logger("api")


def load_config(overrides: dict | None = None, base: str | Path | None = None) -> Config:
    """The packaged defaults (or ``base``), with ``overrides`` merged on top."""
    path = Path(base) if base else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = merge(raw, overrides or {})
    # What the caller set explicitly wins over the selected model's thresholds:
    # re-applying it as that model's override puts it last.
    model = raw["models"]["vision_language"]
    explicit = {k: v for k, v in (overrides or {}).items()
                if k not in ("models", "model_overrides")}
    if explicit:
        raw.setdefault("model_overrides", {})
        raw["model_overrides"][model] = merge(raw["model_overrides"].get(model) or {}, explicit)
    config = Config.from_dict(raw)
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
    #: False when pruning was off or refused; see :func:`vie.pipeline.build_index`.
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
    #: Per result, a short tag drawn on its tile, such as "VIDEO 01:24"; same order.
    badges: list[str | None] = field(default_factory=list)


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

        Pruning removes rows for files no longer on disk, but :func:`build_index`
        refuses it when most of the archive seems to have vanished at once: that is a
        disconnected drive or a mistyped path far more often than a real deletion.
        """
        from vie.adapters import build_detectors
        from vie.models import SpeciesClassifier
        from vie.pipeline import build_index

        config = self.config
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
                           stats.failed, time.perf_counter() - started,
                           pruning=prune and not stats.prune_refused)

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

    def search_everything(
        self,
        query: str,
        *,
        top_k: int = 12,
        min_moments: int = 4,
        frames_dir: str | Path | None = None,
    ) -> SearchResult:
        """One description, every photo and every video moment that shows it.

        Photos are scored by their best region and video moments by their keyframes,
        both through SigLIP's calibrated probability that the description fits, so
        they rank on one scale. Video keyframes carry a whole-frame vector only, which
        reads a little lower than a photo's best region, so the strongest
        ``min_moments`` moments are kept even when photos outscore them.

        Each moment becomes a result whose picture is the frame at its start, grabbed
        from the video into ``frames_dir`` (a temporary folder by default).
        """
        import tempfile

        from vie.query import query_vector

        started = time.perf_counter()
        photos = self.search_text(query, top_k=top_k).matches if self.counts()["images"] else []
        moments: list[Any] = []
        if self.counts()["videos"]:
            from vie.video.store import VideoStore

            store = VideoStore(self.config)
            store.initialise()
            with store.connect() as conn:
                store.index.check_embedding_model(conn, self.config.models["vision_language"])
                vector = query_vector(self.encoder, query, self.config)
                moments = store.search_visual(conn, vector, top_k=top_k)
        calibrate = getattr(self.encoder, "calibrated", None)

        def probability(cosine: float) -> float:
            value = calibrate(cosine) if calibrate is not None else None
            return cosine if value is None else value

        frames = Path(frames_dir or tempfile.mkdtemp(prefix="vie-frames-"))
        clips: list[tuple[Match, str]] = []
        floor = self.config.min_probability if calibrate is not None else 0.0
        for moment in moments:
            if floor and probability(moment.score) < floor:
                continue  # the same "nothing found" floor as photos
            picture = grab_frame(moment.file_path, moment.start, frames)
            if picture is None:
                continue
            clock = f"{int(moment.start // 60)}:{int(moment.start % 60):02d}"
            clips.append((Match(file_name=f"{Path(moment.video_key).stem} @ {clock}",
                                file_path=str(picture), score=probability(moment.score)),
                          f"VIDEO {clock}"))

        ranked = [(m.display_score, m, None) for m in photos]
        ranked += [(m.score, m, badge) for m, badge in clips]
        ranked.sort(key=lambda item: -item[0])
        chosen = ranked[:top_k]
        missing = [item for item in ranked[top_k:] if item[2] is not None]
        have = sum(1 for item in chosen if item[2] is not None)
        for item in missing[: max(0, min_moments - have)]:
            # Make room by dropping the weakest photo, keeping the order by score.
            photo_positions = [i for i, entry in enumerate(chosen) if entry[2] is None]
            if not photo_positions:
                break
            chosen.pop(photo_positions[-1])
            chosen.append(item)
        chosen.sort(key=lambda item: -item[0])
        matches = [replace(m, confidence=score) for score, m, _ in chosen]
        return SearchResult("concept", query, matches, (time.perf_counter() - started) * 1000,
                            badges=[badge for _, _, badge in chosen])

    def most_photographed_face(self, crop_to: str | Path | None = None,
                               sample: int = 3000) -> Path | None:
        """A photo of whoever appears most often: a sensible demo query.

        Counts, for a sample of indexed faces, how many others clear the match
        threshold, and picks the best-connected face, preferring a large one.

        A query photo is searched by its *largest* face, which in a group photo
        need not be the face that was picked. Give ``crop_to`` and the picked face
        is saved there, padded, and that file is returned instead of the photo.
        """
        import numpy as np

        from vie.database import Index

        if not Path(self.config.database_path).exists():
            return None
        index = Index(self.config.database_path)
        with index.connect() as conn:
            rows = list(index.iter_faces(conn))
            scales = index.scales(conn)
        if not rows:
            return None
        rng = np.random.default_rng(self.config.seed)
        if len(rows) > sample:
            rows = [rows[i] for i in rng.choice(len(rows), sample, replace=False)]
        vectors = np.vstack([np.asarray(r[2], dtype=np.float32) for r in rows])
        degree = (vectors @ vectors.T >= self.config.face.match_threshold).sum(axis=1)
        area = np.array([(b[2] - b[0]) * (b[3] - b[1]) for _, b, _, _ in rows], dtype=float)
        # lexsort's primary key is the last one: connections first, then size.
        name, box, _, file_path = rows[int(np.lexsort((-area, -degree))[0])]
        if crop_to is None:
            return Path(file_path)

        from vie.report.showcase import open_rgb

        photo = open_rgb(file_path, max_side=100_000)
        # Boxes are stored at the scale the index saw; map back to the file.
        scale = scales.get(name, 1.0) or 1.0
        x1, y1, x2, y2 = (v / scale for v in box)
        pad = 0.6 * max(x2 - x1, y2 - y1)
        region = (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                  min(photo.width, int(x2 + pad)), min(photo.height, int(y2 + pad)))
        out = Path(crop_to)
        out.parent.mkdir(parents=True, exist_ok=True)
        photo.crop(region).save(out, quality=95)
        return out

    # ── presentation ──────────────────────────────────────────────────

    def save_card(self, result: SearchResult, path: str | Path, **kwargs: Any) -> Path:
        """A presentation-grade PNG or JPEG of one search."""
        from vie.report.showcase import CardStats, save_search_card

        return save_search_card(
            result.matches, path, kind=result.kind, query=result.query,
            query_images=result.query_images, badges=result.badges or None,
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
            shot_labels=(shot_labeller(analysis, self.config.video.label_threshold)
                         if analysis is not None else None),
            match_threshold=self.config.face.match_threshold,
            cluster_threshold=self.config.video.face_cluster_threshold,
            **kwargs,
        )

    def search_videos(self, query: str | None = None, image: str | Path | None = None,
                      top_k: int = 10) -> list[Any]:
        """Moments across every indexed video, by description and speech, or by face.

        Give exactly one of ``query`` or ``image``. For a query, transcript hits come
        first, as in ``vie video search``: a spoken word is an exact match, while a
        visual score is a cosine on another scale, so the two are not merged by score.
        """
        from vie.query import query_face_embedding, query_vector
        from vie.video.store import VideoStore

        if (query is None) == (image is None):
            raise ValueError("give exactly one of query or image")
        store = VideoStore(self.config)
        store.initialise()
        with store.connect() as conn:
            store.index.check_embedding_model(conn, self.config.models["vision_language"])
            if image is not None:
                vector = query_face_embedding(image, self.config, self.bundle)
                return store.search_face(conn, vector, top_k)
            spoken = store.search_speech(conn, query, top_k)
            seen = store.search_visual(conn, query_vector(self.encoder, query, self.config),
                                       top_k=top_k)
            return (spoken + seen)[:top_k]


def grab_frame(video: str | Path, at: float, folder: Path, max_side: int = 1280) -> Path | None:
    """Save the frame at ``at`` seconds as a JPEG in ``folder``; ``None`` if unreadable.

    Through the analysis decoder, so a rotated phone video comes out upright.
    """
    from PIL import Image

    from vie.video.decode import frames_at

    target = folder / f"{Path(video).stem}-{at:08.2f}.jpg"
    if target.exists():
        return target
    try:
        frame = next(iter(frames_at(video, [at], max_side=max_side)), None)
    except Exception:  # noqa: BLE001 - a missing or damaged video loses one tile, not the card
        log.warning("could not read %s at %.1f s", video, at)
        return None
    if frame is None:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.image[:, :, ::-1]).save(target, quality=90)
    return target


def shot_labeller(analysis: Any, threshold: float = 0.4) -> Callable[[float], list[str]]:
    """``time -> the current shot's strongest labels``, for the video's header.

    ``threshold`` is the same presence share as ``video.label_threshold``: a label
    the analysis would not call present in a shot is not shown on it either.
    """
    import bisect

    shots = sorted(getattr(analysis, "shots", []) or [], key=lambda shot: shot.start)
    starts = [shot.start for shot in shots]
    labels = {item.shot: item for item in getattr(analysis, "shot_labels", []) or []}

    def at(t: float) -> list[str]:
        position = bisect.bisect_right(starts, t) - 1
        if position < 0 or t > shots[position].end + 1e-6:
            return []
        found = labels.get(shots[position].index)
        if found is None:
            return []
        ranked = sorted(found.labels.values(), key=lambda pair: -pair[1])
        return [label for label, score in ranked if score >= threshold][:2]

    return at
