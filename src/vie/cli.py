"""Command-line interface.

    vie index
    vie search face   --image query.jpg
    vie search animal --query "red panda"
    vie search food   --query "French fries"
    vie search text   --query "a man with a blue backpack" --html results.html
    vie doctor

The notebooks were bound to Colab through ``google.colab.files``, ``cv2_imshow``
and hardcoded ``/content`` paths. Nothing here imports Colab, and every path
comes from the config, so the same code runs on Linux, Windows and Colab alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vie import __version__
from vie.config import Config
from vie.database import SchemaError
from vie.logging_setup import configure, get_logger
from vie.scoring import Match

log = get_logger("cli")


def _common() -> argparse.ArgumentParser:
    """Options accepted on either side of the subcommand.

    ``SUPPRESS`` stops a subparser's default overwriting a value parsed earlier,
    which is what makes ``vie --json search ...`` and ``vie search ... --json``
    equivalent. The README documented the second form while only the first worked.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help="path to a YAML config")
    common.add_argument("--log-level", default=argparse.SUPPRESS, help="DEBUG, INFO, WARNING...")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="emit results as JSON")
    return common


def _build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="vie", description=__doc__.splitlines()[0], parents=[common]
    )
    parser.add_argument("--version", action="version", version=f"vie {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="build or update the index", parents=[common])
    index.add_argument("--workers", type=int, default=None, help="decode threads")
    index.add_argument("--batch-size", type=int, default=None, help="images per forward pass")
    index.add_argument("--no-prune", action="store_true",
                       help="keep rows for files no longer on disk")
    index.add_argument("--rebuild", action="store_true",
                       help="forget everything and re-encode the gallery")
    index.add_argument("--rehash", action="store_true",
                       help="hash every file instead of trusting size and mtime")

    sub.add_parser("doctor", help="check the environment and the index", parents=[common])

    search = sub.add_parser("search", help="query the index", parents=[common])
    kinds = search.add_subparsers(dest="kind", required=True)

    face = kinds.add_parser("face", help="find a person by photo", parents=[common])
    face.add_argument("--image", required=True, type=Path)
    face.add_argument("--top-k", type=int, default=None)
    face.add_argument("--html", type=Path, default=None, help="write a visual report here")

    for name, helptext in (
        ("animal", "find images containing a species"),
        ("food", "find images containing a dish"),
        ("text", "find images matching a description"),
    ):
        node = kinds.add_parser(name, help=helptext, parents=[common])
        node.add_argument("--query", required=True)
        node.add_argument("--top-k", type=int, default=None)
        node.add_argument("--html", type=Path, default=None, help="write a visual report here")

    return parser


def match_to_dict(match: Match) -> dict:
    return {
        "file_name": match.file_name,
        "file_path": match.file_path,
        "score": round(match.display_score, 4),
        "raw_score": round(match.score, 4),
        "box": list(match.box) if match.box else None,
        "label": match.label,
    }


def _render(matches: list[Match], as_json: bool) -> None:
    if as_json:
        print(json.dumps([match_to_dict(m) for m in matches], indent=2, ensure_ascii=False))
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    if not matches:
        console.print("[yellow]no matches[/yellow]")
        return

    table = Table(box=None, header_style="dim", pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("file", overflow="fold")
    table.add_column("score", justify="right", style="bold")
    table.add_column("", style="cyan")
    table.add_column("where", style="dim")
    for rank, m in enumerate(matches, 1):
        value = max(0.0, min(1.0, m.display_score))
        bar = "█" * int(round(value * 12)) + "░" * (12 - int(round(value * 12)))
        where = " ".join(p for p in (m.label or "", str(m.box) if m.box else "") if p)
        table.add_row(str(rank), m.file_name, f"{m.display_score * 100:.1f}%", bar, where)
    console.print(table)


def _load_runtime(config: Config):
    """Create the lazy model bundle. Deferred so ``--help`` stays instant."""
    from vie.models import VisionLanguageEncoder, load_all

    bundle = load_all(config)
    encoder = VisionLanguageEncoder(bundle)
    return bundle, encoder, encoder


def _run_index(args: argparse.Namespace, config: Config) -> int:
    from vie.adapters import build_detectors
    from vie.models import SpeciesClassifier
    from vie.pipeline import build_index

    bundle, _, encoder = _load_runtime(config)
    detectors = build_detectors(bundle, config)
    classifier = SpeciesClassifier(bundle)
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
        workers=args.workers,
        batch_size=args.batch_size,
        prune=not args.no_prune,
        rebuild=args.rebuild,
        rehash=args.rehash,
    )
    print(stats.summary())
    return 0 if stats.failed == 0 else 3


def _run_search(args: argparse.Namespace, config: Config, as_json: bool) -> int:
    from vie.query import run_search

    try:
        matches = run_search(args, config, _load_runtime)
    except (FileNotFoundError, SchemaError) as exc:
        log.error("%s", exc)
        return 2
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    if args.html is not None:
        from vie.report import write_search_report

        query = str(args.image) if args.kind == "face" else args.query
        write_search_report(matches, args.html, kind=args.kind, query=query)
        log.info("report written to %s", args.html)
    _render(matches, as_json)
    return 0 if matches else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure(getattr(args, "log_level", "INFO"))
    as_json = getattr(args, "json", False)

    try:
        config = Config.load(getattr(args, "config", None))
    except Exception as exc:
        log.error("%s", exc)
        return 2

    if args.command == "doctor":
        from vie.doctor import run_doctor

        return run_doctor(config, as_json)
    if args.command == "index":
        try:
            return _run_index(args, config)
        except (FileNotFoundError, SchemaError) as exc:
            log.error("%s", exc)
            return 2
    if args.command == "search":
        return _run_search(args, config, as_json)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
