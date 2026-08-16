"""Query execution: turn a CLI request into ranked matches.

Every path follows the same shape — narrow the candidate set with SQL, then run
the expensive model only on what survives. That prefilter is the core idea of
the whole system, and it is why a query does not cost a full pass over the
archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vie.config import Config
from vie.database import Index
from vie.logging_setup import get_logger
from vie.scoring import Match
from vie.search import animal as animal_search
from vie.search import face as face_search
from vie.search import food as food_search
from vie.search import text as text_search

log = get_logger("query")


def run_search(args: Any, config: Config, load_runtime: Any) -> list[Match]:
    """Dispatch a search request."""
    index = Index(config.database_path)
    if not Path(config.database_path).exists():
        raise FileNotFoundError(
            f"no index at {config.database_path} — run 'vie index' first"
        )

    if args.kind == "face":
        return _search_face(args, config, index, load_runtime)
    if args.kind == "text":
        return _search_text(args, config, index, load_runtime)
    if args.kind == "animal":
        return _search_boxes(args, config, index, load_runtime, kind="animal")
    if args.kind == "food":
        return _search_boxes(args, config, index, load_runtime, kind="food")
    raise ValueError(f"unknown search kind {args.kind!r}")


def _search_face(args, config: Config, index: Index, load_runtime) -> list[Match]:
    from vie.adapters import load_image

    if not args.image.is_file():
        raise FileNotFoundError(f"query image not found: {args.image}")

    bundle, _, _ = load_runtime(config)
    image = load_image(args.image)
    if image is None:
        raise ValueError(f"could not decode {args.image}")

    height, width = image.shape[:2]
    app = bundle.face_app
    app.prepare(ctx_id=0 if bundle.runtime.is_cuda else -1,
                det_size=face_search.detection_size(width, height, config))
    faces = [(tuple(int(v) for v in f.bbox[:4]), f.normed_embedding) for f in app.get(image)]

    if not faces:
        # Retry once at full detection size — small or tightly cropped portraits
        # are the common failure and this recovers most of them.
        app.prepare(ctx_id=0 if bundle.runtime.is_cuda else -1,
                    det_size=(config.face.det_size_large, config.face.det_size_large))
        faces = [(tuple(int(v) for v in f.bbox[:4]), f.normed_embedding)
                 for f in app.get(image)]
    if not faces:
        raise ValueError("no face detected in the query image")

    _, embedding = face_search.select_query_face(faces)
    with index.connect() as conn:
        return face_search.rank(embedding, index.iter_faces(conn), config)


def _search_text(args, config: Config, index: Index, load_runtime) -> list[Match]:
    _, _, embedder = load_runtime(config)
    embedding = embedder.encode_text(args.query)
    with index.connect() as conn:
        return text_search.rank(embedding, index.iter_clip(conn), config, args.top_k)


def _search_boxes(args, config: Config, index: Index, load_runtime, *, kind: str) -> list[Match]:
    """Shared implementation for the two detect-then-recognise paths."""
    from vie.adapters import RTDetrAdapter, crop, image_size, load_image

    bundle, scorer, _ = load_runtime(config)
    module = animal_search if kind == "animal" else food_search
    flag = "has_animal" if kind == "animal" else "has_food"

    with index.connect() as conn:
        rows = index.candidates(conn, flag)
        stored_boxes: dict[str, list[tuple[tuple[int, int, int, int], float]]] = {}
        if kind == "animal":
            # Boxes were persisted at index time, so the detector does not have
            # to run again here — the original re-ran it on every candidate.
            for name, bbox, confidence in conn.execute(
                "SELECT file_name, bbox, confidence FROM animal_boxes"
            ):
                box = tuple(int(v) for v in bbox.split(","))
                stored_boxes.setdefault(name, []).append((box, confidence))

    log.info("%d candidate images after the %s prefilter", len(rows), flag)
    detector = RTDetrAdapter(bundle) if kind == "food" else None

    candidates: list[Any] = []
    crops: list[Any] = []
    for row in rows:
        image = load_image(Path(row.file_path))
        if image is None:
            log.warning("could not decode %s", row.file_name)
            continue
        size = image_size(image)

        if kind == "animal":
            raw = stored_boxes.get(row.file_name, [])
        else:
            raw = detector.detect(
                image, config.food.search_classes, config.food.search_confidence
            )

        found = [module.Candidate(row.file_name, row.file_path, box, conf) for box, conf in raw]
        for candidate in module.select_crops(found, config, size):
            patch = crop(image, candidate.box)
            if patch is None:
                continue
            candidates.append(candidate)
            crops.append(patch)

    if not candidates:
        return []
    return module.rank(args.query, candidates, crops, scorer, config)
