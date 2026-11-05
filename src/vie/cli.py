"""Command-line interface.

    vie index
    vie search face   --image query.jpg
    vie search animal --query "red panda"
    vie search food   --query "French fries"
    vie search text   --query "a man with a blue backpack" --html results.html
    vie video analyze clip.mp4 --html report.html
    vie video index
    vie video search --query "a goal being scored"
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

    _add_video_commands(sub, common)
    return parser


def _add_video_commands(sub: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    stages = argparse.ArgumentParser(add_help=False)
    stages.add_argument("--no-faces", action="store_true", help="skip face detection")
    stages.add_argument("--no-animals", action="store_true", help="skip animal detection")
    stages.add_argument("--no-food", action="store_true", help="skip food detection")
    stages.add_argument("--labels-only", action="store_true",
                        help="only the vision-language model: runs on a laptop CPU")
    stages.add_argument("--captions", action="store_true", help="add a caption per shot")
    stages.add_argument("--transcribe", action="store_true", help="transcribe the speech")

    video = sub.add_parser("video", help="understand and search videos", parents=[common])
    actions = video.add_subparsers(dest="action", required=True)

    analyze = actions.add_parser("analyze", help="describe one video",
                                 parents=[common, stages])
    analyze.add_argument("file", type=Path)
    analyze.add_argument("--html", type=Path, default=None, help="write a visual report here")
    analyze.add_argument("--max-duration", type=float, default=None,
                         help="analyse only the first N seconds")

    index = actions.add_parser("index", help="index the configured video folder",
                               parents=[common, stages])
    index.add_argument("--no-prune", action="store_true",
                       help="keep rows for files no longer on disk")

    search = actions.add_parser("search", help="find moments across indexed videos",
                                parents=[common])
    search.add_argument("--query", default=None, help="describe what you are looking for")
    search.add_argument("--image", type=Path, default=None, help="find this person")
    search.add_argument("--top-k", type=int, default=10)


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


def _stages(args: argparse.Namespace, config: Config):
    from vie.video.library import Stages

    base = Stages.from_config(config)
    only_labels = args.labels_only
    return Stages(
        faces=base.faces and not (args.no_faces or only_labels),
        animals=base.animals and not (args.no_animals or only_labels),
        food=base.food and not (args.no_food or only_labels),
        captions=base.captions or args.captions,
        transcript=base.transcript or args.transcribe,
    )


def _render_analysis(analysis, as_json: bool) -> None:
    from vie.video.store import analysis_to_dict

    if as_json:
        print(json.dumps(analysis_to_dict(analysis), indent=2, ensure_ascii=False))
        return

    from rich.console import Console
    from rich.table import Table

    from vie.video.understand import format_clock

    console = Console()
    console.print(f"\n{analysis.summary['en']}\n")
    table = Table(box=None, header_style="dim", pad_edge=False)
    for column in ("group", "label", "of running time", "first seen"):
        table.add_column(column)
    for item in analysis.labels[:12]:
        table.add_row(item.group, item.label, f"{item.coverage * 100:.0f}%",
                      format_clock(item.intervals[0].start))
    console.print(table)


def _run_video(args: argparse.Namespace, config: Config, as_json: bool) -> int:
    from dataclasses import replace

    from vie.video.decode import VideoDecodeError
    from vie.video.library import build_engine, index_videos

    try:
        if args.action == "search":
            return _run_video_search(args, config, as_json)

        if args.action == "analyze" and args.max_duration is not None:
            config = replace(config, video=replace(config.video, max_duration=args.max_duration))
        bundle, _, encoder = _load_runtime(config)
        engine = build_engine(config, bundle, encoder, _stages(args, config))

        if args.action == "index":
            stats = index_videos(config, engine, prune=not args.no_prune)
            print(stats.summary())
            return 0 if stats.failed == 0 else 3

        analysis = engine.analyse(args.file)
        if args.html is not None:
            from vie.report.video import write_video_report

            write_video_report(analysis, args.html)
            log.info("report written to %s", args.html)
        _render_analysis(analysis, as_json)
        return 0
    except (FileNotFoundError, SchemaError, VideoDecodeError) as exc:
        log.error("%s", exc)
        return 2


def _run_video_search(args: argparse.Namespace, config: Config, as_json: bool) -> int:
    from dataclasses import asdict

    from vie.video.store import VideoStore
    from vie.video.understand import format_clock

    if (args.query is None) == (args.image is None):
        log.error("give exactly one of --query or --image")
        return 2
    if not Path(config.database_path).exists():
        log.error("no index at %s: run 'vie video index' first", config.database_path)
        return 2

    store = VideoStore(config)
    store.initialise()
    bundle, _, encoder = _load_runtime(config)
    with store.connect() as conn:
        store.index.check_embedding_model(conn, config.models["vision_language"])
        if args.image is not None:
            from vie.query import query_face_embedding

            try:
                moments = store.search_face(
                    conn, query_face_embedding(args.image, config, bundle), args.top_k
                )
            except ValueError as exc:
                log.error("%s", exc)
                return 1
        else:
            moments = store.search_speech(conn, args.query, args.top_k)
            moments += store.search_visual(
                conn, encoder.encode_text(args.query), top_k=args.top_k
            )

    if as_json:
        print(json.dumps([asdict(m) for m in moments], indent=2, ensure_ascii=False))
    elif not moments:
        print("no matches")
    else:
        from rich.console import Console
        from rich.table import Table

        table = Table(box=None, header_style="dim", pad_edge=False)
        for column in ("video", "from", "to", "score", "matched on"):
            table.add_column(column, overflow="fold")
        for moment in moments:
            table.add_row(
                moment.video_key, format_clock(moment.start), format_clock(moment.end),
                f"{moment.score:.3f}", moment.text or moment.kind,
            )
        Console().print(table)
    return 0 if moments else 1


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
    if args.command == "video":
        return _run_video(args, config, as_json)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
