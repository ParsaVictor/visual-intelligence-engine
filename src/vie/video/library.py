"""Wiring the engine to real models, and keeping a folder of videos indexed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.database import content_hash
from vie.logging_setup import get_logger
from vie.video import decode
from vie.video.engine import VideoEngine
from vie.video.store import VideoStore

log = get_logger("video.library")


@dataclass(frozen=True)
class Stages:
    """Which optional stages a run uses. Starts from the config; flags override it."""

    faces: bool
    animals: bool
    food: bool
    captions: bool
    transcript: bool

    @classmethod
    def from_config(cls, config: Config) -> Stages:
        video = config.video
        return cls(video.faces, video.animals, video.food, video.captions, video.transcript)


def build_engine(config: Config, bundle: Any, encoder: Any, stages: Stages) -> VideoEngine:
    """Construct an engine using only the models this run needs."""
    from vie.adapters import InsightFaceDetector, MegaDetectorAdapter, RTDetrAdapter, crop
    from vie.models import SpeciesClassifier
    from vie.video.optional import BlipCaptioner, WhisperTranscriber

    return VideoEngine(
        config,
        encoder,
        face_detector=InsightFaceDetector(bundle, config) if stages.faces else None,
        animal_detector=MegaDetectorAdapter(bundle) if stages.animals else None,
        food_detector=RTDetrAdapter(bundle) if stages.food else None,
        classify_crops=SpeciesClassifier(bundle).classify if stages.animals else None,
        class_names=(lambda: bundle.class_names) if stages.animals else None,
        crop=crop,
        captioner=BlipCaptioner(config, bundle.runtime) if stages.captions else None,
        transcriber=WhisperTranscriber(config) if stages.transcript else None,
    )


@dataclass
class VideoIndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    pruned: int = 0
    failed: int = 0

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} indexed={self.indexed} skipped={self.skipped} "
            f"pruned={self.pruned} failed={self.failed}"
        )


def index_videos(config: Config, engine: VideoEngine, *, prune: bool = True) -> VideoIndexStats:
    """Analyse every new or changed video, committing after each one.

    A video can take minutes, so losing one to an interruption is the most this
    can cost. A file that fails is counted and logged, and the run carries on.
    """
    store = VideoStore(config)
    store.initialise()
    root = Path(config.videos_path)
    paths = decode.discover_videos(root, recursive=config.indexing.recursive)
    keys = {path: path.relative_to(root).as_posix() for path in paths}
    stats = VideoIndexStats(scanned=len(paths))

    with store.connect() as conn:
        store.index.bind_embedding_model(conn, config.models["vision_language"])
        if prune:
            stats.pruned = store.prune(conn, set(keys.values()))
        known = store.known(conn)
        conn.commit()

        for path in paths:
            key = keys[path]
            stat = path.stat()
            seen = known.get(key)
            if seen and seen[1] == stat.st_size and seen[2] == stat.st_mtime_ns:
                stats.skipped += 1
                continue
            digest = content_hash(path)
            if seen and seen[0] == digest:
                stats.skipped += 1
                continue
            try:
                store.save(conn, key, path, digest, engine.analyse(path))
                conn.commit()
                stats.indexed += 1
            except Exception:
                log.exception("failed to analyse %s", key)
                stats.failed += 1
    return stats
