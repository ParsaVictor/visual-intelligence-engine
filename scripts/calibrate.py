"""Measure which scoring rule and threshold actually separate hits from misses.

Thresholds carried over from another model mean nothing: SigLIP gives a true match
0.21 where CLIP's softmax gave 0.97. So a threshold in the config should be a number
this script produced, for that model.

    python scripts/calibrate.py DATA_DIR --model google/siglip-base-patch16-224

``DATA_DIR`` holds ``<kind>/<label>/<image>``, where ``kind`` is ``food``, ``animal``
or ``other``. Each image is scored against its own label (a positive) and against
labels from other images (negatives). ``other`` images only ever serve as negatives.

Reports ROC AUC, which needs no threshold, and the threshold with the best F1, for
each candidate rule. The numbers describe this dataset. They are good for choosing
between rules and setting a sane default, not for promising accuracy on an archive.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vie.config import DEFAULT_CONFIG_PATH, Config
from vie.models import VisionLanguageEncoder, load_all
from vie.vlm import default_score_mode, gate_score, hypothesis_score


def auc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a random positive outranks a random negative."""
    pos, neg = np.asarray(positives), np.asarray(negatives)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def best_threshold(positives: list[float], negatives: list[float]) -> dict[str, float]:
    best = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for value in sorted(set(positives) | set(negatives)):
        tp = sum(p >= value for p in positives)
        fp = sum(n >= value for n in negatives)
        if not tp:
            continue
        precision, recall = tp / (tp + fp), tp / len(positives)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"threshold": value, "f1": f1, "precision": precision, "recall": recall}
    return {k: round(float(v), 4) for k, v in best.items()}


def softmax_rule(logits: list[float]) -> float:
    return hypothesis_score(logits, "softmax")


def absolute_rule(logits: list[float]) -> float:
    """SigLIP's own probability for the hypothesis, damped when a rival beats it."""
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    hypothesis, rival = float(probs[0]), float(probs[1:].max())
    return hypothesis if hypothesis > rival else hypothesis * hypothesis / (hypothesis + rival)


def at_threshold(positives: list[float], negatives: list[float], value: float) -> dict:
    tp, fp = sum(p >= value for p in positives), sum(n >= value for n in negatives)
    return {
        "threshold": round(value, 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / len(positives), 4),
    }


def margin_rule(logits: list[float]) -> float:
    return logits[0] - max(logits[1:])


def report(name: str, rules: dict, rows: list[tuple[list[float], bool]]) -> dict:
    results = {}
    for rule, score in rules.items():
        positives = [score(row) for row, hit in rows if hit]
        negatives = [score(row) for row, hit in rows if not hit]
        results[rule] = {
            "auc": round(auc(positives, negatives), 4),
            **best_threshold(positives, negatives),
            "median_positive": round(float(np.median(positives)), 4),
            "median_negative": round(float(np.median(negatives)), 4),
            "positives": len(positives),
            "negatives": len(negatives),
        }
        print(f"{name:<12} {rule:<16} {json.dumps(results[rule])}", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--negatives", type=int, default=4, help="wrong labels per image")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/calibration.json"))
    args = parser.parse_args()

    random.seed(0)
    raw = copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")))
    raw["models"]["vision_language"] = args.model
    raw["models"]["score_mode"] = default_score_mode(args.model)
    raw["runtime"]["low_memory"] = bool(os.environ.get("VIE_LOW_MEMORY"))
    raw["indexing"]["batch_size"] = 4
    config = Config.from_dict(raw)
    encoder = VisionLanguageEncoder(load_all(config))
    mode = config.score_mode

    items: dict[str, list[tuple[str, Path]]] = {}
    for path in sorted(p for p in args.data.rglob("*") if p.is_file()):
        kind, label = path.relative_to(args.data).parts[:2]
        items.setdefault(kind, []).append((label.replace("_", " "), path))

    rules = {"sigmoid_absolute": lambda row: absolute_rule(row),
             "softmax_rivals": softmax_rule, "logit_margin": margin_rule}
    output: dict = {"model": args.model, "images": {k: len(v) for k, v in items.items()}}
    vectors: dict[Path, np.ndarray] = {}

    def embed(entries: list[tuple[str, Path]]) -> None:
        todo = [p for _, p in entries if p not in vectors]
        for start in range(0, len(todo), 4):
            chunk = todo[start:start + 4]
            images = [Image.open(p).convert("RGB") for p in chunk]
            for path, vector in zip(chunk, encoder.encode_images(images), strict=True):
                vectors[path] = vector
        print(f"embedded {len(vectors)} images", flush=True)

    for entries in items.values():
        embed(entries)
    everything = [entry for entries in items.values() for entry in entries]

    def logits_for(path: Path, prompts: list[str]) -> list[float]:
        return encoder.logits_from_similarity(vectors[path] @ encoder.encode_texts(prompts).T)

    for kind, section, template in (
        ("food", config.food, "{query}"),
        ("animal", config.animal, config.animal.prompt_template),
    ):
        own = items.get(kind, [])
        if not own:
            continue
        labels = sorted({label for label, _ in own})
        rivals = list(section.negative_prompts)
        rows: list[tuple[list[float], bool]] = []
        for label, path in everything:
            wanted = [label] if label in labels else []
            wrong = random.sample([x for x in labels if x != label],
                                  min(args.negatives, len(labels) - len(wanted)))
            for query, hit in [*((w, True) for w in wanted), *((w, False) for w in wrong)]:
                prompts = [template.format(query=query), *rivals]
                rows.append((logits_for(path, prompts).tolist(), hit))
        output[kind] = report(kind, rules, rows)
        # What the code actually does now, and what v0.2 shipped, on the same pairs.
        positives = [hypothesis_score(row, mode) for row, hit in rows if hit]
        negatives = [hypothesis_score(row, mode) for row, hit in rows if not hit]
        output[kind]["configured"] = at_threshold(positives, negatives, section.match_threshold)
        shipped = {"food": 0.60, "animal": 0.55}[kind]
        positives = [softmax_rule(row) for row, hit in rows if hit]
        negatives = [softmax_rule(row) for row, hit in rows if not hit]
        output[kind]["v0.2_softmax"] = at_threshold(positives, negatives, shipped)
        print(kind, "configured", output[kind]["configured"],
              "v0.2", output[kind]["v0.2_softmax"], flush=True)

    # The zero-inference food gate: is this a picture of food at all?
    prompts = config.food.gate_prompts()
    count = len(config.food.clip_gate_positive)
    gate_rules = {
        "probability_gap": lambda row: gate_score(row, count, mode),
        "logit_margin": lambda row: max(row[:count]) - max(row[count:]),
    }
    rows = [
        (logits_for(path, prompts).tolist(), kind == "food")
        for kind, entries in items.items() for _, path in entries
    ]
    output["food_gate"] = report("food_gate", gate_rules, rows)
    positives = [gate_score(row, count, mode) for row, hit in rows if hit]
    negatives = [gate_score(row, count, mode) for row, hit in rows if not hit]
    output["food_gate"]["configured"] = at_threshold(
        positives, negatives, config.food.clip_gate_threshold
    )
    print("food_gate configured", output["food_gate"]["configured"], flush=True)

    # Free text: what does the model's own sigmoid give a right and a wrong caption?
    named = [entry for kind, entries in items.items() if kind != "other" for entry in entries]
    if named:
        cosines: dict[str, list[float]] = {"hit": [], "miss": []}
        labels = sorted({label for label, _ in named})
        for label, path in named:
            wrong = [x for x in labels if x != label][:args.negatives]
            cosines["hit"].append(float(vectors[path] @ encoder.encode_text(label)))
            cosines["miss"] += [float(vectors[path] @ encoder.encode_text(w)) for w in wrong]
        output["text"] = {
            "median_cosine_hit": round(float(np.median(cosines["hit"])), 4),
            "median_cosine_miss": round(float(np.median(cosines["miss"])), 4),
            "median_probability_hit": round(
                encoder.calibrated(float(np.median(cosines["hit"]))) or 0, 4
            ),
            "auc": round(auc(cosines["hit"], cosines["miss"]), 4),
        }
        print("text", json.dumps(output["text"]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
