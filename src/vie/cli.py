"""Command-line interface.

    vie index
    vie search face  --image query.jpg
    vie search animal --query "red panda"
    vie search food   --query "French fries"
    vie search text   --query "a man with a blue backpack"

The notebooks were bound to Colab through ``google.colab.files``, ``cv2_imshow``
and hardcoded ``/content`` paths. Nothing here imports Colab, and every path
comes from the config, so the same code runs on Linux, Windows and Colab alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vie.config import Config
from vie.logging_setup import configure, get_logger
from vie.scoring import Match

log = get_logger("cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vie", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json", action="store_true", help="emit results as JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="build or update the index")
    index.add_argument("--workers", type=int, default=8)
    index.add_argument("--no-prune", action="store_true",
                       help="keep rows for files no longer on disk")

    search = sub.add_parser("search", help="query the index")
    kinds = search.add_subparsers(dest="kind", required=True)

    face = kinds.add_parser("face", help="find a person by photo")
    face.add_argument("--image", required=True, type=Path)

    for name, helptext in (
        ("animal", "find images containing a species"),
        ("food", "find images containing a dish"),
        ("text", "find images matching a description"),
    ):
        node = kinds.add_parser(name, help=helptext)
        node.add_argument("--query", required=True)
        node.add_argument("--top-k", type=int, default=None)

    return parser


def _render(matches: list[Match], as_json: bool) -> None:
    if as_json:
        print(json.dumps([
            {
                "file_name": m.file_name,
                "file_path": m.file_path,
                "score": round(m.score, 4),
                "box": list(m.box) if m.box else None,
                "label": m.label,
            }
            for m in matches
        ], indent=2, ensure_ascii=False))
        return

    if not matches:
        print("no matches")
        return
    width = max(len(m.file_name) for m in matches)
    for rank, m in enumerate(matches, 1):
        box = f"  box={m.box}" if m.box else ""
        print(f"[{rank:>2}] {m.file_name:<{width}}  {m.score * 100:6.2f}%{box}")


def _load_runtime(config: Config):
    """Import and construct the model stack. Deferred so ``--help`` stays instant."""
    from vie.models import ClipCropScorer, ClipEmbedder, load_all

    bundle = load_all(config)
    return bundle, ClipCropScorer(bundle), ClipEmbedder(bundle)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure(args.log_level)

    try:
        config = Config.load(args.config)
    except Exception as exc:
        log.error("%s", exc)
        return 2

    if args.command == "index":
        from vie.adapters import build_detectors
        from vie.pipeline import build_index

        bundle, _, embedder = _load_runtime(config)
        detectors = build_detectors(bundle, config)
        stats = build_index(
            config,
            loader=detectors.loader,
            size_of=detectors.size_of,
            face_detector=detectors.face,
            animal_detector=detectors.animal,
            food_detector=detectors.food,
            embed_image=embedder.encode_image,
            workers=args.workers,
            prune=not args.no_prune,
        )
        print(stats.summary())
        return 0

    if args.command == "search":
        from vie.query import run_search

        try:
            matches = run_search(args, config, _load_runtime)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 2
        except ValueError as exc:
            log.error("%s", exc)
            return 1
        _render(matches, args.json)
        return 0 if matches else 1

    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
