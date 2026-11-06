"""Regenerate assets/showcase/search/: food, species, free text, Persian, concept.

Runs the whole system with the real weights (SigLIP 2, ConvNeXt, MegaDetector,
RT-DETR, InsightFace) on public data, through :class:`vie.api.Engine`, exactly as
the notebook does, and saves one presentation card per search.

    python scripts/fetch_calibration_data.py DATA --per-label 2
    python scripts/make_search_showcase.py --data DATA --videos VIDEOS --out assets/showcase/search

DATA holds Food-101, cats-vs-dogs and mini-ImageNet images (see fetch_calibration_data.py);
VIDEOS holds intel-iot-devkit/sample-videos files (CC BY 4.0). The *Showcase* workflow
runs this on a CPU runner and commits the cards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from eval_search import PERSIAN

from vie.api import Engine
from vie.video.decode import frames_at


def build_gallery(data: Path, videos: Path, gallery: Path) -> list[str]:
    """A small mixed archive: dishes, pets, unrelated objects and classroom stills."""
    from PIL import Image

    if gallery.exists():
        shutil.rmtree(gallery)
    dishes = sorted(p for p in (data / "food").iterdir() if p.is_dir())[:6]
    picks = [(d, 2) for d in dishes]
    picks += [(data / "animal" / "cat", 3), (data / "animal" / "dog", 3)]
    picks += [(d, 1) for d in sorted((data / "other").iterdir())[:4]]
    for folder, count in picks:
        for i, source in enumerate(sorted(folder.glob("*.jpg"))[:count]):
            target = gallery / folder.parent.name / f"{folder.name}_{i}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
    for name, times in (("classroom.mp4", [4, 14, 24]),
                        ("face-demographics-walking-and-pause.mp4", [30, 60])):
        for frame in frames_at(videos / name, times, max_side=1280):
            target = gallery / "stills" / f"{Path(name).stem[:12]}_{frame.time:05.1f}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame.image[:, :, ::-1]).save(target, quality=92)
    return [d.name.replace("_", " ") for d in dishes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("assets/showcase/search"))
    parser.add_argument("--work", type=Path, default=Path("data/search-showcase"))
    args = parser.parse_args()

    gallery, clips = args.work / "gallery", args.work / "videos"
    dishes = build_gallery(args.data, args.videos, gallery)
    clips.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.videos / "classroom.mp4", clips / "classroom.mp4")

    engine = Engine.from_overrides({
        "paths": {"gallery": str(gallery), "videos": str(clips),
                  "database": str(args.work / "index.db")},
        "runtime": {"device": "cpu", "low_memory": True},
        "indexing": {"batch_size": 4, "workers": 4},
        "video": {"max_duration": 30, "captions": False, "transcript": False},
    })
    report: dict = {"model": engine.config.models["vision_language"], "searches": []}
    started = time.perf_counter()
    report["index"] = engine.index_images().summary()
    report["videos"] = engine.index_videos().summary()
    report["index_seconds"] = round(time.perf_counter() - started, 1)
    print(report["index"], "|", report["videos"], flush=True)

    frames = args.work / "frames"
    persian_dish = next((PERSIAN[d] for d in dishes if d in PERSIAN), None)
    searches = [
        ("food", dishes[0], lambda q: engine.search_food(q, top_k=8)),
        ("food", dishes[1], lambda q: engine.search_food(q, top_k=8)),
        ("animal", "cat", lambda q: engine.search_animal(q, top_k=8)),
        ("animal", "a puppy", lambda q: engine.search_animal(q, top_k=8)),
        ("text", "a bowl of soup", lambda q: engine.search_text(q, top_k=8)),
        ("concept", "people sitting in a classroom",
         lambda q: engine.search_everything(q, top_k=8, min_moments=2, frames_dir=frames)),
        ("concept", "گربه", lambda q: engine.search_everything(q, top_k=8, frames_dir=frames)),
        ("concept", "a theatre stage with actors",
         lambda q: engine.search_everything(q, top_k=8, frames_dir=frames)),
    ]
    if persian_dish:
        searches.append(("concept", persian_dish,
                         lambda q: engine.search_everything(q, top_k=8, frames_dir=frames)))

    args.out.mkdir(parents=True, exist_ok=True)
    for number, (kind, query, run) in enumerate(searches, 1):
        result = run(query)
        name = f"{number:02d}-{kind}"
        engine.save_card(result, args.out / f"{name}.jpg")
        entry = {"card": f"{name}.jpg", "kind": kind, "query": query,
                 "matches": len(result.matches), "milliseconds": round(result.milliseconds, 1),
                 "top": [(m.file_name, round(m.display_score, 4)) for m in result.matches[:5]]}
        report["searches"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
