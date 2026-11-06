"""Index-time orchestration.

What this module guarantees, each pinned by a test in ``tests/test_pipeline.py``:

* **Bounded memory.** Decoding runs on a thread pool, ahead of inference by a
  fixed window. ``Executor.map`` submits everything at once, which on an archive
  of tens of thousands decoded the whole gallery into RAM before the first image
  was processed.
* **Batched inference (P6-2).** Detectors that can batch and the vision-language
  encoder receive whole batches.
* **Per-image atomicity.** All rows for one image land together or not at all, so
  a failure cannot leave a row whose hash says "done" with no embeddings behind it.
* **Failure isolation.** A batch that fails is retried one image at a time, so a
  single bad file costs itself rather than its seven neighbours.
* **Resumability.** Periodic commits keep work across an interruption.
* **Incremental runs.** Unchanged files are skipped without re-reading them.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.database import Index, content_hash, format_box
from vie.geometry import Box, pad_box
from vie.indexing import BoxDetector, FaceDetector, ImageAnalysis, analyse_batch
from vie.logging_setup import get_logger
from vie.quality import crop_regions

log = get_logger("pipeline")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def discover_images(root: str | Path, recursive: bool = True) -> list[Path]:
    """List gallery images, case-insensitively, skipping hidden directories.

    The standalone module globbed lowercase patterns only, so ``.JPG`` files, the
    default from most cameras and agency exports, were silently invisible.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"gallery not found: {root}")
    walker = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        p for p in walker
        if p.suffix.lower() in IMAGE_SUFFIXES
        and p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def gallery_key(path: Path, root: Path) -> str:
    """The index key for a file: its gallery-relative POSIX path.

    A bare file name collides as soon as two folders both hold ``IMG_0001.JPG``,
    and the second image would silently overwrite the first.
    """
    return path.relative_to(root).as_posix()


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    pruned: int = 0
    failed: int = 0
    #: Pruning was requested but refused, because too much of the index vanished.
    prune_refused: bool = False

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} indexed={self.indexed} skipped={self.skipped} "
            f"pruned={self.pruned} failed={self.failed}"
            + (" prune=refused" if self.prune_refused else "")
        )


def _prefetch(
    paths: Sequence[Path],
    loader: Callable[[Path], object | None],
    workers: int,
    window: int | None = None,
) -> Iterator[tuple[Path, object | None]]:
    """Decode on a thread pool, yielding in order, holding at most ``window`` results."""
    if workers <= 1:
        for path in paths:
            yield path, loader(path)
        return

    limit = window or workers * 2
    pending: deque = deque()
    source = iter(paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path in source:
            pending.append((path, pool.submit(loader, path)))
            if len(pending) >= limit:
                ready_path, future = pending.popleft()
                yield ready_path, future.result()
        while pending:
            ready_path, future = pending.popleft()
            yield ready_path, future.result()


def _chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _split_scale(decoded: Any) -> tuple[Any, float]:
    """A loader returns the image, or ``(image, scale)`` when it resized."""
    return decoded if isinstance(decoded, tuple) else (decoded, 1.0)


@dataclass
class _Item:
    path: Path
    key: str
    digest: str
    image: Any
    scale: float
    size: tuple[int, int]


@dataclass
class _Components:
    size_of: Callable[[Any], tuple[int, int]]
    face_detector: FaceDetector
    animal_detector: BoxDetector
    food_detector: BoxDetector
    embed_image: Callable[[Any], Any] | None
    embed_batch: Callable[[Sequence[Any], Sequence[Any]], list[list[tuple[str, Any]]]] | None
    food_gate: Callable[[Any], list[float]] | None
    classify_crops: Callable[[Sequence[Any], int], list[list[tuple[int, float]]]] | None
    crop: Callable[[Any, Any], Any] | None


#: Separates an object region's kind from its box in a region name: "food|10,20,90,80".
OBJECT_SEP = "|"


def object_regions(
    size: tuple[int, int], analysis: ImageAnalysis, config: Config
) -> list[tuple[str, Box, Box]]:
    """``(name, detector box, padded crop)`` for every animal and dish to embed.

    Padded exactly as :func:`vie.search.animal.select_crops` and
    :func:`vie.search.food.select_crops` pad at query time, so a stored vector is the
    vector those paths would have computed. Size floors were applied at detection;
    the confidence floor is applied at query time, against the stored confidence.
    Food boxes outside ``food.search_classes`` (a fork, a cup) are never searched,
    so they are not embedded either.
    """
    width, height = size
    out: list[tuple[str, Box, Box]] = []
    for box, _ in analysis.animal_boxes:
        crop = pad_box(box, width, height, ratio=config.animal.padding_ratio)
        out.append((f"animal{OBJECT_SEP}{format_box(box)}", box, crop))
    searchable = set(config.food.search_classes)
    for box, _, class_id in analysis.food_detections:
        if class_id is not None and class_id not in searchable:
            continue
        crop = pad_box(box, width, height, pixels=config.food.padding_px)
        out.append((f"food{OBJECT_SEP}{format_box(box)}", box, crop))
    return out


def _embed(
    items: list[_Item], analyses: list[ImageAnalysis], config: Config, parts: _Components
) -> list[list[tuple[str, Any]]]:
    """Every region vector of every image, objects included, in one batched pass."""
    if parts.embed_batch is not None:
        layout = config.crop_layout if config.multi_crop else "1"
        regions = [
            crop_regions(*item.size, layout)
            + [(name, crop) for name, _, crop in object_regions(item.size, analysis, config)]
            for item, analysis in zip(items, analyses, strict=True)
        ]
        return parts.embed_batch([item.image for item in items], regions)
    return [[("whole", parts.embed_image(item.image))] for item in items]


def _split_objects(
    vectors: list[tuple[str, Any]],
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """Separate scene regions (text search) from object regions (food and animals)."""
    scene = [(name, v) for name, v in vectors if OBJECT_SEP not in name]
    objects = {name: v for name, v in vectors if OBJECT_SEP in name}
    return scene, objects


def _food_gate_scores(
    vectors: list[list[tuple[str, Any]]], config: Config, parts: _Components
) -> list[float]:
    """Score the stored whole-image vectors, costing no extra image inference."""
    if parts.food_gate is None or not config.food.clip_gate_enabled:
        return [0.0] * len(vectors)
    import numpy as np

    whole = np.vstack([dict(regions)["whole"] for regions in vectors])
    return list(parts.food_gate(whole))


def _classify_animals(
    items: list[_Item], analyses: list[ImageAnalysis], config: Config, parts: _Components
) -> list[list[tuple[Any, int, float]]]:
    """Classify every animal crop once, so in-vocabulary queries need no model."""
    results: list[list[tuple[Any, int, float]]] = [[] for _ in items]
    if parts.classify_crops is None or parts.crop is None:
        return results

    crops: list[Any] = []
    owners: list[tuple[int, Any]] = []
    for position, (item, analysis) in enumerate(zip(items, analyses, strict=True)):
        width, height = item.size
        for box, _ in analysis.animal_boxes:
            padded = pad_box(box, width, height, ratio=config.animal.padding_ratio)
            patch = parts.crop(item.image, padded)
            if patch is not None:
                crops.append(patch)
                owners.append((position, box))
    if not crops:
        return results

    predictions = parts.classify_crops(crops, config.animal.classifier_top_k)
    for (position, box), top in zip(owners, predictions, strict=True):
        results[position].extend((box, class_id, prob) for class_id, prob in top)
    return results


def _write(
    index: Index,
    conn: Any,
    item: _Item,
    analysis: ImageAnalysis,
    regions: list[tuple[str, Any]],
    gate: float,
    predictions: list[tuple[Any, int, float]],
    config: Config,
    gate_enabled: bool,
    objects: dict[str, Any] | None = None,
) -> None:
    food_by_vlm = gate_enabled and gate >= config.food.clip_gate_threshold
    stat = item.path.stat()
    with index.image_transaction(conn):
        # Re-indexing an edited file must not leave its old rows behind.
        index.clear_derived(conn, item.key)
        index.upsert_image(
            conn,
            file_name=item.key,
            file_path=str(item.path),
            hash_=item.digest,
            size=stat.st_size,
            has_face=analysis.has_face,
            has_animal=analysis.has_animal,
            has_food=analysis.has_food or food_by_vlm,
            food_by_detector=analysis.has_food,
            food_by_vlm=food_by_vlm,
            food_vlm_score=gate,
            index_scale=item.scale,
            mtime_ns=stat.st_mtime_ns,
            objects_embedded=objects is not None,
        )
        for box, embedding in analysis.faces:
            index.add_face(conn, item.key, box, embedding)
        for box, confidence in analysis.animal_boxes:
            index.add_animal_box(conn, item.key, box, confidence)
        for box, class_id, probability in predictions:
            index.add_animal_prediction(conn, item.key, box, class_id, probability)
        for box, confidence, class_id in analysis.food_detections:
            index.add_food_box(conn, item.key, box, confidence, class_id)
        for name, vector in regions:
            index.add_clip(conn, item.key, vector, region=name)
        if objects is not None:
            confidence = {("animal", box): (c, None) for box, c in analysis.animal_boxes}
            confidence.update({("food", box): (c, k) for box, c, k in analysis.food_detections})
            for name, box, crop in object_regions(item.size, analysis, config):
                vector = objects.get(name)
                if vector is None:  # an empty crop the encoder skipped
                    continue
                kind = name.split(OBJECT_SEP, 1)[0]
                score, class_id = confidence[(kind, box)]
                index.add_object(conn, item.key, kind, box, crop, score, class_id, vector)


def _index_batch(
    items: list[_Item], index: Index, conn: Any, config: Config, parts: _Components,
    stats: IndexStats, on_progress: Callable[[IndexStats], None] | None,
) -> None:
    try:
        analyses = analyse_batch(
            [item.image for item in items], config,
            face_detector=parts.face_detector,
            animal_detector=parts.animal_detector,
            food_detector=parts.food_detector,
        )
        embedded = _embed(items, analyses, config, parts)
        split = [_split_objects(regions) for regions in embedded]
        vectors = [scene for scene, _ in split]
        objects = [found if parts.embed_batch is not None else None for _, found in split]
        gates = _food_gate_scores(vectors, config, parts)
        predictions = _classify_animals(items, analyses, config, parts)
    except Exception:
        if len(items) > 1:
            log.warning("batch of %d failed; retrying one at a time", len(items))
            for item in items:
                _index_batch([item], index, conn, config, parts, stats, on_progress)
            return
        # One unreadable file must not abandon the whole archive, but it must be
        # visible: the original swallowed failures silently.
        log.exception("failed to index %s", items[0].key)
        stats.failed += 1
        return

    gate_enabled = parts.food_gate is not None and config.food.clip_gate_enabled
    for item, analysis, regions, gate, found, stored in zip(
        items, analyses, vectors, gates, predictions, objects, strict=True
    ):
        try:
            _write(index, conn, item, analysis, regions, gate, found, config, gate_enabled,
                   stored)
            stats.indexed += 1
        except Exception:
            log.exception("failed to store %s", item.key)
            stats.failed += 1
    if on_progress is not None:
        on_progress(stats)


def _stale_face_images(index: Index, conn: Any, config: Config) -> set[str]:
    """Images whose face vectors came from a different recipe than the config's.

    Mixing recipes in one index would quietly shift every score, so those images
    are re-indexed. Only images that have faces pay for it, not the whole archive.
    An index written before the recipe was recorded used plain ArcFace.
    """
    found = index.get_meta(conn, "face_recipe") or "arcface"
    if found == config.face.recipe:
        return set()
    return {row[0] for row in conn.execute("SELECT file_name FROM gallery_meta WHERE has_face = 1")}


def build_index(
    config: Config,
    *,
    loader: Callable[[Path], object | None],
    size_of: Callable[[object], tuple[int, int]],
    face_detector: FaceDetector,
    animal_detector: BoxDetector,
    food_detector: BoxDetector,
    embed_image: Callable[[object], object] | None = None,
    embed_batch: Callable[[Sequence[Any], Sequence[Any]], list[list[tuple[str, Any]]]]
    | None = None,
    food_gate: Callable[[Any], list[float]] | None = None,
    classify_crops: Callable[[Sequence[Any], int], list[list[tuple[int, float]]]] | None = None,
    crop: Callable[[Any, Any], Any] | None = None,
    class_names: Callable[[], list[str]] | None = None,
    workers: int | None = None,
    batch_size: int | None = None,
    commit_every: int | None = None,
    prune: bool = True,
    prune_guard: float = 0.5,
    rebuild: bool = False,
    rehash: bool = False,
    on_progress: Callable[[IndexStats], None] | None = None,
) -> IndexStats:
    """Index every changed image in the gallery.

    All model access is injected, which keeps this function testable end to end
    without a GPU: the orchestration logic is what carries the risk here, not the
    model calls.

    ``prune`` drops rows for files no longer on disk, unless fewer than
    ``prune_guard`` of the indexed files are still there (``0`` disables the guard).
    """
    if embed_image is None and embed_batch is None:
        raise ValueError("build_index needs embed_batch or embed_image")

    settings = config.indexing
    workers = settings.workers if workers is None else workers
    batch_size = settings.batch_size if batch_size is None else batch_size
    commit_every = settings.commit_every if commit_every is None else commit_every

    stats = IndexStats()
    index = Index(config.database_path)
    index.initialise()

    root = Path(config.gallery_path)
    paths = discover_images(root, recursive=settings.recursive)
    stats.scanned = len(paths)
    log.info("found %d images in %s", len(paths), root)
    keys = {path: gallery_key(path, root) for path in paths}

    parts = _Components(
        size_of, face_detector, animal_detector, food_detector,
        embed_image, embed_batch, food_gate, classify_crops, crop,
    )

    with index.connect() as conn:
        if rebuild:
            index.clear_all(conn)
            log.info("rebuild requested: cleared the index")
        index.bind_embedding_model(conn, config.models["vision_language"])

        if prune:
            # An unmounted drive or a mistyped path looks exactly like "every photo
            # was deleted". Rows cost minutes of GPU each to rebuild, so pruning most
            # of the index at once is refused rather than trusted.
            indexed = conn.execute("SELECT COUNT(*) FROM gallery_meta").fetchone()[0]
            if indexed and len(keys) < indexed * prune_guard:
                stats.prune_refused = True
                log.warning(
                    "only %d of %d indexed images are on disk; not pruning. Check the "
                    "gallery path; to really drop them, run with --rebuild.",
                    len(keys), indexed,
                )
            else:
                stats.pruned = index.prune_missing(conn, set(keys.values()))
                if stats.pruned:
                    log.info("pruned %d rows for files no longer on disk", stats.pruned)

        known = index.known_files(conn)
        stored_paths = dict(conn.execute("SELECT file_name, file_path FROM gallery_meta"))
        stale_faces = _stale_face_images(index, conn, config)
        if stale_faces:
            log.info("face recipe changed to %s: re-indexing the %d images with faces",
                     config.face.recipe, len(stale_faces))
        trust_mtime = settings.trust_mtime and not rehash
        pending: list[tuple[Path, str]] = []
        refreshed: list[tuple[str, str, int, int]] = []
        for path in paths:
            key = keys[path]
            stat = path.stat()
            seen = known.get(key)
            if key in stale_faces:
                pending.append((path, content_hash(path)))
                continue
            if (
                trust_mtime and seen is not None
                and seen[1] == stat.st_size and seen[2] == stat.st_mtime_ns
            ):
                stats.skipped += 1
                if stored_paths.get(key) != str(path):
                    refreshed.append((key, str(path), stat.st_size, stat.st_mtime_ns))
                continue
            digest = content_hash(path)
            if seen is not None and seen[0] == digest:
                # Same content under a new mtime or path: remember both, so the
                # next run skips it from its stat alone instead of hashing again.
                stats.skipped += 1
                refreshed.append((key, str(path), stat.st_size, stat.st_mtime_ns))
                continue
            pending.append((path, digest))
        if refreshed:
            index.refresh_files(conn, refreshed)
            conn.commit()

        if not pending:
            index.set_meta(conn, "face_recipe", config.face.recipe)
            log.info("index is up to date (%s)", stats.summary())
            return stats

        log.info("indexing %d new or changed images", len(pending))
        digests = dict(pending)

        if classify_crops is not None and class_names is not None:
            index.set_classifier_vocabulary(conn, config.animal.classifier, class_names())

        def decoded_items() -> Iterator[_Item]:
            window = max(workers * 2, batch_size * 2)
            for path, decoded in _prefetch([p for p, _ in pending], loader, workers, window):
                if decoded is None:
                    log.warning("could not decode %s", keys[path])
                    stats.failed += 1
                    continue
                image, scale = _split_scale(decoded)
                yield _Item(path, keys[path], digests[path], image, scale, size_of(image))

        last_commit = 0
        for batch in _chunked(decoded_items(), max(1, batch_size)):
            _index_batch(batch, index, conn, config, parts, stats, on_progress)
            if stats.indexed - last_commit >= commit_every:
                conn.commit()  # checkpoint: an interruption keeps this much
                last_commit = stats.indexed
                log.info("checkpoint at %d images", stats.indexed)
        # Recorded only once every stale image was redone, so an interrupted run
        # picks the rest up next time.
        index.set_meta(conn, "face_recipe", config.face.recipe)

    log.info("indexing complete (%s)", stats.summary())
    return stats
