"""Run the whole project on one folder of images and videos.

    python scripts/run_dataset.py D:/datasets/my_set
    python scripts/run_dataset.py D:/datasets/my_set --text "a man with a backpack" \
        --animal "red panda" --food "pizza" --face D:/query/person.jpg

`vie` reads its folders from a config file, so testing a new dataset would otherwise
mean hand-editing a copy of it. This writes that copy, then indexes and searches.

Every dataset gets its own folder, ``data/runs/<name>/``, holding its config, index and
reports. Model weights are shared in ``data/models/``. All of it is under ``data/``,
which git ignores, because indexes and reports contain faces.

Defaults suit a CPU-only machine with about 8 GB of RAM. Use
``--model google/siglip-so400m-patch14-384`` where there is a GPU or plenty of memory.
Changing the model of an existing run needs ``--rebuild``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vie import cli
from vie.config import DEFAULT_CONFIG_PATH
from vie.pipeline import IMAGE_SUFFIXES, discover_images
from vie.video.decode import discover_videos

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google/siglip-base-patch16-224"


@dataclass
class Step:
    name: str
    code: int
    seconds: float
    report: str = ""


@dataclass
class Run:
    folder: Path
    directory: Path
    config: Path
    steps: list[Step] = field(default_factory=list)

    def execute(self, name: str, argv: list[str], report: Path | None = None) -> int:
        """Run one `vie` command in this process and record how it went."""
        print(f"\n=== {name}", flush=True)
        started = time.perf_counter()
        command = [*argv, "--config", str(self.config)]
        if report is not None:
            command += ["--html", str(report)]
        try:
            code = cli.main(command)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # one failed search must not lose the rest of the run
            print(f"failed: {exc!r}", file=sys.stderr)
            code = 70
        seconds = time.perf_counter() - started
        written = str(report) if report is not None and report.exists() else ""
        self.steps.append(Step(name, code, round(seconds, 1), written))
        print(f"--- {name}: exit {code}, {seconds:.1f} s", flush=True)
        return code


def describe(folder: Path) -> dict:
    """What is in the folder, so a wrong path or layout shows before hours of indexing."""
    images = discover_images(folder, recursive=True)
    videos = discover_videos(folder, recursive=True)
    parents = Counter(
        p.relative_to(folder).parts[0] if len(p.relative_to(folder).parts) > 1 else "."
        for p in images
    )
    known_videos = set(videos)
    ignored = Counter(
        p.suffix.lower() or "(none)"
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES and p not in known_videos
    )
    size = sum(p.stat().st_size for p in [*images, *videos]) / 1e6
    return {
        "images": len(images),
        "videos": len(videos),
        "megabytes": round(size, 1),
        "top_level_folders": dict(parents.most_common(20)),
        "ignored_file_types": dict(ignored.most_common(10)),
    }


def write_config(args: argparse.Namespace, directory: Path) -> Path:
    raw = copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")))
    raw["paths"] = {
        "gallery": str(args.folder),
        "videos": str(args.folder),
        "database": str(directory / "index.db"),
        "model_cache": str(args.models),
    }
    raw["models"]["vision_language"] = args.model
    raw["runtime"]["device"] = args.device
    raw["runtime"]["low_memory"] = True
    raw["indexing"]["batch_size"] = args.batch_size
    raw["indexing"]["workers"] = args.workers
    # A CPU indexes slowly, so an interruption should cost seconds of work, not minutes.
    raw["indexing"]["commit_every"] = 10
    raw["text"]["multi_crop"] = not args.no_multi_crop
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def report_path(directory: Path, kind: str, query: str) -> Path:
    slug = re.sub(r"[^\w.-]+", "_", query).strip("_")[:60] or "query"
    return directory / "reports" / f"{kind}-{slug}.html"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="the dataset folder")
    parser.add_argument("--name", help="run name; defaults to the folder's name")
    parser.add_argument("--text", action="append", default=[], help="describe an image")
    parser.add_argument("--animal", action="append", default=[], help="species to find")
    parser.add_argument("--food", action="append", default=[], help="dish to find")
    parser.add_argument("--face", action="append", default=[], type=Path, help="photo of a person")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-multi-crop", action="store_true",
                        help="one vector per image: about 6x faster, weaker on small objects")
    parser.add_argument("--full-video", action="store_true",
                        help="add faces, animals and food to videos, not only labels")
    parser.add_argument("--rebuild", action="store_true", help="re-encode everything")
    parser.add_argument("--inspect", action="store_true", help="only describe the folder")
    parser.add_argument("--models", type=Path, default=ROOT / "data" / "models",
                        help="where model weights are kept")
    args = parser.parse_args(argv)
    args.folder = args.folder.expanduser().resolve()
    args.models = args.models.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2

    contents = describe(args.folder)
    print(json.dumps(contents, indent=2, ensure_ascii=False))
    if args.inspect:
        return 0
    if not contents["images"] and not contents["videos"]:
        print("no images or videos found", file=sys.stderr)
        return 2

    name = re.sub(r"[^\w.-]+", "_", args.name or args.folder.name)
    directory = ROOT / "data" / "runs" / name
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    run = Run(args.folder, directory, write_config(args, directory))

    failed = False
    if contents["images"]:
        index_command = ["index", *(["--rebuild"] if args.rebuild else [])]
        failed |= run.execute("index images", index_command) not in {0}
    if contents["videos"]:
        stages = [] if args.full_video else ["--labels-only"]
        failed |= run.execute("index videos", ["video", "index", *stages]) not in {0}

    # Exit 1 from a search means "no matches", which is an answer, not a failure.
    limit = ["--top-k", str(args.top_k)]
    for kind, queries in (("text", args.text), ("animal", args.animal), ("food", args.food)):
        for query in queries:
            code = run.execute(
                f"search {kind}: {query}", ["search", kind, "--query", query, *limit],
                report_path(directory, kind, query),
            )
            failed |= code not in {0, 1}
    for image in args.face:
        code = run.execute(
            f"search face: {image.name}", ["search", "face", "--image", str(image), *limit],
            report_path(directory, "face", image.stem),
        )
        failed |= code not in {0, 1}

    failed |= run.execute("doctor", ["doctor"]) not in {0}

    summary = {
        "folder": str(args.folder),
        "model": args.model,
        "contents": contents,
        "steps": [vars(step) for step in run.steps],
    }
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nrun folder: {directory}\nsearch again with: vie search text --query \"...\" "
          f"--config \"{run.config}\"")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
