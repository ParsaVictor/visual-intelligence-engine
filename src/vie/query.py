"""Query execution: turn a request into ranked matches.

Every path follows the same shape: narrow the candidate set with SQL, then run
the expensive model only on what survives. That prefilter is the core idea of
the whole system, and it is why a query does not cost a full pass over the
archive.

Boxes are stored in the coordinates of the capped image the models saw. They are
mapped back onto the original file here, as the last step, so that every caller
receives coordinates it can actually use.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.database import Index
from vie.logging_setup import get_logger
from vie.scoring import Match, best_per_image
from vie.search import animal as animal_search
from vie.search import face as face_search
from vie.search import food as food_search
from vie.search import text as text_search
from vie.vectorstore import VectorCatalog

log = get_logger("query")


def run_search(args: Any, config: Config, load_runtime: Any) -> list[Match]:
    """Dispatch a search request."""
    if not Path(config.database_path).exists():
        raise FileNotFoundError(
            f"no index at {config.database_path} — run 'vie index' first"
        )
    index = Index(config.database_path)
    index.initialise()  # upgrades an index written by an older version

    if args.kind == "face":
        matches = _search_face(args, config, index, load_runtime)
    elif args.kind == "text":
        matches = _search_text(args, config, index, load_runtime)
    elif args.kind == "animal":
        matches = _search_animal(args, config, index, load_runtime)
    elif args.kind == "food":
        matches = _search_food(args, config, index, load_runtime)
    else:
        raise ValueError(f"unknown search kind {args.kind!r}")

    with index.connect() as conn:
        scales = index.scales(conn)
    limit = getattr(args, "top_k", None)
    return [_to_original(m, scales) for m in (matches if limit is None else matches[:limit])]


def _to_original(match: Match, scales: dict[str, float]) -> Match:
    scale = scales.get(match.file_name, 1.0)
    if match.box is None or scale == 1.0:
        return match
    return replace(match, box=tuple(int(round(v / scale)) for v in match.box))


def _ann_cache(config: Config) -> Path:
    return Path(str(config.database_path) + ".ann")


def _file_paths(conn: Any) -> dict[str, str]:
    return dict(conn.execute("SELECT file_name, file_path FROM gallery_meta"))


# ── face ───────────────────────────────────────────────────────────────


def query_face_embedding(image_path: Any, config: Config, bundle: Any) -> Any:
    """The embedding of the largest face in a query photo."""
    from vie.adapters import load_image

    if image_path is None:
        raise ValueError("a face search needs a query image")
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"query image not found: {image_path}")

    image = load_image(Path(image_path), config.face.max_index_dimension)
    if image is None:
        raise ValueError(f"could not decode {image_path}")

    height, width = image.shape[:2]
    app = bundle.face_app
    ctx = 0 if bundle.runtime.is_cuda else -1

    def detect() -> list[tuple[Any, Any]]:
        return [(tuple(int(v) for v in f.bbox[:4]), f.normed_embedding) for f in app.get(image)]

    app.prepare(ctx_id=ctx, det_size=face_search.detection_size(width, height, config))
    faces = detect()
    if not faces:
        # Retry once at full detection size: small or tightly cropped portraits
        # are the common failure and this recovers most of them.
        app.prepare(ctx_id=ctx, det_size=(config.face.det_size_large, config.face.det_size_large))
        faces = detect()
    if not faces:
        raise ValueError("no face detected in the query image")
    return face_search.select_query_face(faces)[1]


def _search_face(args, config: Config, index: Index, load_runtime) -> list[Match]:
    if args.image is None:
        raise ValueError("a face search needs a query image")
    if not Path(args.image).is_file():
        raise FileNotFoundError(f"query image not found: {args.image}")

    bundle, _, _ = load_runtime(config)
    embedding = query_face_embedding(args.image, config, bundle)
    with index.connect() as conn:
        catalog = VectorCatalog.open(conn, "face_embeddings", config.search, _ann_cache(config))
        return face_search.rank_catalog(embedding, catalog, _file_paths(conn), config)


# ── free text ──────────────────────────────────────────────────────────


def _search_text(args, config: Config, index: Index, load_runtime) -> list[Match]:
    query = args.query.strip()
    if not query:
        raise ValueError("empty query")
    with index.connect() as conn:
        index.check_embedding_model(conn, config.models["vision_language"])
        catalog = VectorCatalog.open(conn, "clip_embeddings", config.search, _ann_cache(config))
        paths = _file_paths(conn)
    _, _, encoder = load_runtime(config)
    vector = encoder.encode_text(query)
    matches = text_search.rank_catalog(vector, catalog, paths, config, args.top_k)
    return text_search.with_confidence(matches, getattr(encoder, "calibrated", None))


# ── species ────────────────────────────────────────────────────────────


def _search_animal(args, config: Config, index: Index, load_runtime) -> list[Match]:
    """Answer from stored predictions when possible, and only then load a model."""
    query = args.query.strip()
    if not query:
        raise ValueError("empty query")

    with index.connect() as conn:
        vocabulary = index.classifier_vocabulary(conn)
        if vocabulary:
            route = animal_search.choose_route(query, vocabulary)
            log.info("%s", route.explain())
            if route.engine == "classifier":
                predictions = [
                    animal_search.Prediction(name, path, box, class_id, prob)
                    for name, path, box, class_id, prob in index.animal_predictions(
                        conn, route.resolution.class_ids
                    )
                ]
                found = animal_search.rank_from_index(
                    route.resolution, predictions, config, vocabulary
                )
                if found:
                    return found
                log.info("no stored prediction cleared the floor; trying the open vocabulary")
        rows = index.candidates(conn, "has_animal")
        stored = index.animal_boxes(conn)

    log.info("%d candidate images after the has_animal prefilter", len(rows))
    return _score_crops(
        query, rows, stored, animal_search.select_crops, animal_search.rank_with_clip,
        animal_search.Candidate, config, load_runtime,
    )


# ── food ───────────────────────────────────────────────────────────────


def _search_food(args, config: Config, index: Index, load_runtime) -> list[Match]:
    query = args.query.strip()
    if not query:
        raise ValueError("empty query")
    with index.connect() as conn:
        rows = index.candidates(conn, "has_food")
        stored = index.food_boxes(conn, config.food.search_classes)

    log.info("%d candidate images after the has_food prefilter", len(rows))
    return _score_crops(
        query, rows, stored, food_search.select_crops, food_search.rank,
        food_search.Candidate, config, load_runtime, whole_frame_fallback=True,
    )


def _score_crops(
    query: str,
    rows: list[Any],
    stored: dict[str, list[tuple[Any, float]]],
    select_crops: Any,
    rank: Any,
    candidate_type: Any,
    config: Config,
    load_runtime: Any,
    *,
    whole_frame_fallback: bool = False,
) -> list[Match]:
    """Crop stored boxes and score them in bounded chunks.

    ``whole_frame_fallback`` covers an image admitted by the food gate with no
    box: a plate of ghormeh sabzi fires none of COCO's ten food classes, so the
    image as a whole is the only thing left to score.
    """
    from vie.adapters import crop, image_size, load_image

    if not rows:
        return []
    _, scorer, _ = load_runtime(config)
    step = max(1, config.search.crop_batch)

    matches: list[Match] = []
    candidates: list[Any] = []
    crops: list[Any] = []

    def flush() -> None:
        if candidates:
            matches.extend(rank(query, candidates, crops, scorer, config))
            candidates.clear()
            crops.clear()

    for row in rows:
        image = load_image(Path(row.file_path), config.face.max_index_dimension)
        if image is None:
            log.warning("could not decode %s", row.file_name)
            continue
        size = image_size(image)
        found = [
            candidate_type(row.file_name, row.file_path, box, confidence)
            for box, confidence in stored.get(row.file_name, [])
        ]
        selected = select_crops(found, config, size)
        if not selected and whole_frame_fallback:
            selected = [
                candidate_type(row.file_name, row.file_path, (0, 0, size[0], size[1]), 1.0)
            ]
        for candidate in selected:
            patch = crop(image, candidate.box)
            if patch is None:
                continue
            candidates.append(candidate)
            crops.append(patch)
            if len(crops) >= step:
                flush()
    flush()
    return best_per_image(matches)
