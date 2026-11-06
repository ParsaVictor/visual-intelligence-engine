"""Regenerate the face showcase in assets/showcase/ from public CC-BY 4.0 videos.

Needs only InsightFace and PyAV (no torch, no GPU). Uses the same face recipe the
index uses (ArcFace buffalo_l with mirror averaging), the same search ranking
(:func:`vie.search.face.rank`) and the same renderers the notebook uses.

    python scripts/make_showcase.py --videos DIR --out assets/showcase

DIR holds files from https://github.com/intel-iot-devkit/sample-videos (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vie.adapters import face_vectors
from vie.config import Config
from vie.report.showcase import CardStats, save_search_card
from vie.search import face as face_search
from vie.video.render import render_annotated_video, render_people_card


def face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def detector(app, min_px: int = 24):
    def detect(image):
        found = [f for f in app.get(image)
                 if f.det_score >= 0.5
                 and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= min_px]
        return list(zip([tuple(float(v) for v in f.bbox[:4]) for f in found],
                        face_vectors(app, image, found, True), strict=True))
    return detect


def stills(video: Path, out: Path, times: list[float], prefix: str) -> list[Path]:
    """Save frames at the given seconds as JPEGs: a stand-in photo archive."""
    import av

    out.mkdir(parents=True, exist_ok=True)
    wanted = sorted(times)
    saved: list[Path] = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            if not wanted:
                break
            if frame.time is not None and frame.time >= wanted[0]:
                path = out / f"{prefix}_{wanted.pop(0):05.1f}s.jpg"
                frame.to_image().save(path, quality=92)
                saved.append(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("assets/showcase"))
    parser.add_argument("--work", type=Path, default=Path("data/showcase"))
    args = parser.parse_args()
    config = Config.load()
    app = face_app()
    detect = detector(app)
    v = args.videos

    # ── a photo archive, cut from the videos ──────────────────────────
    gallery = args.work / "gallery"
    photos = (
        stills(v / "head-pose-face-detection-female.mp4", gallery,
               [8, 22, 37, 51, 66, 80, 95, 110, 124], "lobby_cam1")
        + stills(v / "head-pose-face-detection-male.mp4", gallery,
                 [10, 30, 50, 70, 90, 110], "lobby_cam2")
        + stills(v / "head-pose-face-detection-female-and-male.mp4", gallery,
                 [15, 45, 75, 105, 135], "desk_cam")
        + stills(v / "face-demographics-walking-and-pause.mp4", gallery,
                 [6, 12, 25, 40, 55, 70, 85], "corridor")
        + stills(v / "classroom.mp4", gallery, [3, 12, 21, 30], "classroom")
    )
    queries = stills(v / "head-pose-face-detection-female.mp4", args.work / "queries",
                     [1.0, 129.0], "query")

    import cv2

    indexed = []
    for path in photos:
        image = cv2.imread(str(path))
        for box, vector in detect(image):
            indexed.append((path.name, tuple(int(b) for b in box), vector, str(path)))
    query_vectors = [detect(cv2.imread(str(q))) for q in queries]
    query_vectors = [max(found, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))[1]
                     for found in query_vectors if found]

    started = time.perf_counter()
    best: dict[str, object] = {}
    for vector in query_vectors:
        for match in face_search.rank(vector, indexed, config):
            if match.file_name not in best or match.score > best[match.file_name].score:
                best[match.file_name] = match
    matches = sorted(best.values(), key=lambda m: -m.score)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"{len(indexed)} faces in {len(photos)} photos; {len(matches)} matches")
    args.out.mkdir(parents=True, exist_ok=True)
    save_search_card(matches, args.out / "face-search.jpg", kind="face",
                     query="The same person, across cameras and poses",
                     query_images=queries,
                     stats=CardStats(search_ms=elapsed, indexed_images=len(photos)),
                     max_results=12)

    # ── videos ────────────────────────────────────────────────────────
    known = {}
    for name, video, at in (("Student 1042", "head-pose-face-detection-female.mp4", [2, 40, 128]),
                            ("Student 2177", "head-pose-face-detection-male.mp4", [3, 45, 120])):
        refs = stills(v / video, args.work / "enrolled" / name, at, "ref")
        known[name] = [max(found, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))[1]
                       for found in (detect(cv2.imread(str(r))) for r in refs) if found]

    jobs = [
        ("desk_named.mp4", "head-pose-face-detection-female-and-male.mp4",
         "Front desk: enrolled students", known, 45),
        ("corridor_people.mp4", "face-demographics-walking-and-pause.mp4",
         "Corridor: who passed, and when", None, 90),
        ("classroom_people.mp4", "classroom.mp4", "Classroom: every face, grouped", None, 20),
    ]
    for out_name, src, title, names, seconds in jobs:
        result = render_annotated_video(v / src, args.work / out_name, detect=detect, known=names,
                                        title=title, max_seconds=seconds, detect_every=2,
                                        max_side=1280)
        print(out_name, result.frames, "frames", f"{result.fps_processed:.1f} fps",
              [(p.label, round(p.seconds, 1)) for p in result.people])
        render_people_card(result, title=title).save(args.out / out_name.replace(".mp4", ".png"))


if __name__ == "__main__":
    main()
