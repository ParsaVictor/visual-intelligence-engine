"""Index-time orchestration.

Restores the engineering the standalone face notebook had and the unified
pipeline dropped:

* an 8-thread prefetch, so image decode overlaps GPU inference instead of
  serialising in front of it;
* a pre-resize cap, because a 6000 px press photo costs decode and transfer time
  for detail no detector at 640 px input will ever see;
* periodic commits, so an interrupted run over a large archive keeps its work.

It also replaces filename-only change detection with a content hash, so an
edited photo is re-indexed rather than skipped for having a familiar name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from vie.config import Config
from vie.database import Index, content_hash
from vie.indexing import BoxDetector, FaceDetector, analyse_image
from vie.logging_setup import get_logger

log = get_logger("pipeline")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def discover_images(root: str | Path) -> list[Path]:
    """List gallery images, case-insensitively.

    The standalone module globbed lowercase patterns only, so ``.JPG`` files —
    the default from most cameras and agency exports — were silently invisible.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"gallery not found: {root}")
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass
class IndexStats:
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


def _prefetch(
    paths: Sequence[Path],
    loader: Callable[[Path], object | None],
    workers: int,
) -> Iterator[tuple[Path, object | None]]:
    """Decode images on a thread pool, yielding in submission order."""
    if workers <= 1:
        for path in paths:
            yield path, loader(path)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from zip(paths, pool.map(loader, paths), strict=True)


def build_index(
    config: Config,
    *,
    loader: Callable[[Path], object | None],
    size_of: Callable[[object], tuple[int, int]],
    face_detector: FaceDetector,
    animal_detector: BoxDetector,
    food_detector: BoxDetector,
    embed_image: Callable[[object], object],
    workers: int = 8,
    commit_every: int = 50,
    prune: bool = True,
) -> IndexStats:
    """Index every changed image in the gallery.

    All model access is injected, which keeps this function testable end to end
    without a GPU — the orchestration logic is what carries the risk here, not
    the model calls.
    """
    stats = IndexStats()
    index = Index(config.database_path)
    index.initialise()

    paths = discover_images(config.gallery_path)
    stats.scanned = len(paths)
    log.info("found %d images in %s", len(paths), config.gallery_path)

    with index.connect() as conn:
        if prune:
            stats.pruned = index.prune_missing(conn, {p.name for p in paths})
            if stats.pruned:
                log.info("pruned %d rows for files no longer on disk", stats.pruned)

        pending: list[tuple[Path, str]] = []
        for path in paths:
            digest = content_hash(path)
            if index.needs_indexing(conn, path.name, digest):
                pending.append((path, digest))
            else:
                stats.skipped += 1

        if not pending:
            log.info("index is up to date (%s)", stats.summary())
            return stats

        log.info("indexing %d new or changed images", len(pending))
        digests = dict(pending)

        for path, image in _prefetch([p for p, _ in pending], loader, workers):
            if image is None:
                log.warning("could not decode %s", path.name)
                stats.failed += 1
                continue
            try:
                analysis = analyse_image(
                    image,
                    size_of(image),
                    config,
                    face_detector=face_detector,
                    animal_detector=animal_detector,
                    food_detector=food_detector,
                )

                # Re-indexing an edited file must not leave its old rows behind.
                index.clear_derived(conn, path.name)
                index.upsert_image(
                    conn,
                    file_name=path.name,
                    file_path=str(path),
                    hash_=digests[path],
                    size=path.stat().st_size,
                    has_face=analysis.has_face,
                    has_animal=analysis.has_animal,
                    has_food=analysis.has_food,
                )
                for box, embedding in analysis.faces:
                    index.add_face(conn, path.name, box, embedding)
                for box, confidence in analysis.animal_boxes:
                    index.add_animal_box(conn, path.name, box, confidence)
                index.add_clip(conn, path.name, embed_image(image))

                stats.indexed += 1
                if stats.indexed % commit_every == 0:
                    conn.commit()  # checkpoint: an interruption keeps this much
                    log.info("checkpoint at %d images", stats.indexed)
            except Exception:
                # One unreadable file must not abandon the whole archive, but it
                # must be visible — the original swallowed failures silently.
                log.exception("failed to index %s", path.name)
                stats.failed += 1

    log.info("indexing complete (%s)", stats.summary())
    return stats
