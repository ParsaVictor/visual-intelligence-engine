"""Head-to-head face recognition benchmark: ArcFace (this repo) vs SFace (student-face-gate).

Both recognisers see exactly the same aligned 112x112 crops, cut by the same
detector with the same five landmarks, so the comparison isolates the recogniser.

Ground truth comes from public CC-BY 4.0 test videos (intel-iot-devkit/sample-videos)
without any hand labelling:

* **genuine pairs**: two frames of a single-person video. The person turns their
  head through large angles, so pairs with a big yaw difference are hard.
* **impostor pairs**: two different faces in the *same frame* are necessarily two
  different people, plus every pair across the two single-person videos (a woman
  and a man).

This is a stress test of pose, not a replacement for LFW or IJB-C. It is small,
and the numbers are a comparison between models, not a promise of field accuracy.

    python scripts/face_benchmark.py --videos DIR --sface face_recognition_sface_2021dec.onnx
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

SINGLE_PERSON = {
    "head-pose-face-detection-female.mp4": "female",
    "head-pose-face-detection-male.mp4": "male",
}
CROWD = [
    "classroom.mp4",
    "face-demographics-walking.mp4",
    "face-demographics-walking-and-pause.mp4",
    "people-detection.mp4",
    "head-pose-face-detection-female-and-male.mp4",
]


def extract(videos: Path, every: float, app) -> list[dict]:
    import av
    from insightface.utils import face_align

    faces: list[dict] = []
    for name in list(SINGLE_PERSON) + CROWD:
        path = videos / name
        if not path.exists():
            continue
        container = av.open(str(path))
        next_t, frame_no = 0.0, 0
        for frame in container.decode(video=0):
            t = float(frame.time or 0.0)
            if t + 1e-6 < next_t:
                continue
            next_t = t + every
            image = frame.to_ndarray(format="bgr24")
            found = [f for f in app.get(image)
                     if f.det_score >= 0.5
                     and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= 40]
            if name in SINGLE_PERSON and found:
                found = [max(found, key=lambda f: (f.bbox[2] - f.bbox[0])
                             * (f.bbox[3] - f.bbox[1]))]
            for f in found:
                yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0
                faces.append({
                    "video": name, "frame": frame_no, "time": t, "yaw": yaw,
                    "identity": SINGLE_PERSON.get(name),
                    "crop": face_align.norm_crop(image, landmark=f.kps, image_size=112),
                })
            frame_no += 1
        container.close()
    return faces


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def embed_arcface(rec, crops, flip: bool) -> np.ndarray:
    out = []
    for start in range(0, len(crops), 64):
        batch = crops[start:start + 64]
        feats = rec.get_feat(batch)
        if flip:
            feats = feats + rec.get_feat([np.ascontiguousarray(c[:, ::-1]) for c in batch])
        out.append(feats)
    return unit(np.vstack(out).astype(np.float32))


def embed_sface(model_path: str, crops, flip: bool) -> np.ndarray:
    import cv2

    sface = cv2.FaceRecognizerSF.create(model_path, "")
    out = []
    for crop in crops:
        v = sface.feature(crop).ravel()
        if flip:
            v = v + sface.feature(np.ascontiguousarray(crop[:, ::-1])).ravel()
        out.append(v)
    return unit(np.vstack(out).astype(np.float32))


def pairs(faces: list[dict]):
    genuine, hard, impostor = [], [], []
    by_id: dict[str, list[int]] = {}
    for i, f in enumerate(faces):
        if f["identity"]:
            by_id.setdefault(f["identity"], []).append(i)
    for members in by_id.values():
        for a, b in itertools.combinations(members, 2):
            genuine.append((a, b))
            if abs(faces[a]["yaw"] - faces[b]["yaw"]) >= 30:
                hard.append((a, b))
    ids = list(by_id.values())
    for x, y in itertools.combinations(ids, 2):
        impostor.extend(itertools.product(x, y))
    frames: dict[tuple[str, int], list[int]] = {}
    for i, f in enumerate(faces):
        if not f["identity"]:
            frames.setdefault((f["video"], f["frame"]), []).append(i)
    for members in frames.values():
        impostor.extend(itertools.combinations(members, 2))
    return genuine, hard, impostor


def scores(emb: np.ndarray, idx) -> np.ndarray:
    a, b = np.array(idx).T
    return np.einsum("ij,ij->i", emb[a], emb[b])


def metrics(gen: np.ndarray, hard: np.ndarray, imp: np.ndarray, threshold: float) -> dict:
    thresholds = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp >= t).mean() for t in thresholds])
    frr = np.array([(gen < t).mean() for t in thresholds])
    k = int(np.argmin(np.abs(far - frr)))
    zero_far = float(imp.max()) + 1e-6
    return {
        "eer": round(float((far[k] + frr[k]) / 2), 4),
        "separation_d_prime": round(float((gen.mean() - imp.mean())
                                          / np.sqrt((gen.var() + imp.var()) / 2)), 2),
        "genuine_mean": round(float(gen.mean()), 3),
        "impostor_max": round(float(imp.max()), 3),
        "tar_at_far0": round(float((gen >= zero_far).mean()), 4),
        "hard_tar_at_far0": round(float((hard >= zero_far).mean()), 4),
        "threshold": threshold,
        "tar_at_threshold": round(float((gen >= threshold).mean()), 4),
        "hard_tar_at_threshold": round(float((hard >= threshold).mean()), 4),
        "far_at_threshold": round(float((imp >= threshold).mean()), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--sface", required=True)
    parser.add_argument("--every", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("results/face_benchmark.json"))
    args = parser.parse_args()

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "landmark_3d_68"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = extract(args.videos, args.every, app)
    crops = [f["crop"] for f in faces]
    genuine, hard, impostor = pairs(faces)
    print(f"{len(faces)} faces, {len(genuine)} genuine pairs ({len(hard)} with yaw gap >= 30), "
          f"{len(impostor)} impostor pairs")

    full = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                        allowed_modules=["detection", "recognition"])
    full.prepare(ctx_id=-1, det_size=(640, 640))
    rec = full.models["recognition"]

    variants = {
        # Each model at the operating threshold its own project ships with.
        "ArcFace buffalo_l (this repo)": (lambda: embed_arcface(rec, crops, False), 0.46),
        "ArcFace buffalo_l + mirror": (lambda: embed_arcface(rec, crops, True), 0.46),
        "SFace (student-face-gate)": (lambda: embed_sface(args.sface, crops, False), 0.55),
        "SFace + mirror (student-face-gate)": (lambda: embed_sface(args.sface, crops, True), 0.55),
    }
    report = {"faces": len(faces), "genuine_pairs": len(genuine), "hard_pairs": len(hard),
              "impostor_pairs": len(impostor), "models": {}}
    for name, (embed, threshold) in variants.items():
        started = time.perf_counter()
        emb = embed()
        per_face = (time.perf_counter() - started) / len(crops) * 1000
        result = metrics(scores(emb, genuine), scores(emb, hard), scores(emb, impostor), threshold)
        result["ms_per_face_cpu"] = round(per_face, 1)
        report["models"][name] = result
        print(f"{name:38s} {json.dumps(result)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
