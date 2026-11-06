"""Measure typed search: does a query find the photos it should, and only those?

    python scripts/fetch_calibration_data.py DATA --per-label 6
    python scripts/eval_search.py DATA --model google/siglip2-so400m-patch14-384

Every label in the labelled set (Food-101 dishes, cats, dogs) becomes a query, and
the images carrying that label are the right answers. Images are embedded exactly as
the index embeds them (whole frame plus multi-crop regions, best region wins), so the
numbers describe the product, not a simplified copy of it.

Reported per query variant:

* ``mAP``: mean average precision over the whole ranking, the standard retrieval
  measure. 1.0 means every right image ranks above every wrong one.
* ``top1``: how often the first result is right.
* ``r_precision``: precision within the first R results, R = number of right images.

Variants: the label alone, the label averaged over ``text.query_templates`` (prompt
ensembling), and the same in Persian, which only a multilingual model can answer.

For a "nothing found" answer it also reports SigLIP's calibrated probability of the
best right image for present concepts, against the best image at all for concepts
that are absent from the set ("a theatre stage", "a football match"). A floor between
the two lets search say "nothing here" instead of returning its five least-bad images.

The set is small: read differences of a few points as noise, differences of ten as real.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from vie.config import DEFAULT_CONFIG_PATH, Config
from vie.models import VisionLanguageEncoder, load_all
from vie.quality import crop_regions
from vie.vlm import default_score_mode

#: Food-101 classes and the two animals, in Persian, for the multilingual variant.
PERSIAN = {
    "apple pie": "پای سیب", "baby back ribs": "دنده کبابی", "baklava": "باقلوا",
    "beef carpaccio": "کارپاچیو گوشت", "beef tartare": "تارتار گوشت",
    "beet salad": "سالاد چغندر", "beignets": "دونات فرانسوی", "bibimbap": "بیبیمباپ",
    "bread pudding": "پودینگ نان", "breakfast burrito": "بوریتو صبحانه",
    "bruschetta": "بروسکتا", "caesar salad": "سالاد سزار", "cannoli": "کانولی",
    "caprese salad": "سالاد کاپرزه", "carrot cake": "کیک هویج", "ceviche": "سویچه",
    "cheesecake": "چیزکیک", "cheese plate": "بشقاب پنیر", "chicken curry": "کاری مرغ",
    "chicken quesadilla": "کسادیای مرغ", "chicken wings": "بال مرغ",
    "chocolate cake": "کیک شکلاتی", "chocolate mousse": "موس شکلات", "churros": "چوروس",
    "clam chowder": "سوپ صدف", "club sandwich": "کلاب ساندویچ", "crab cakes": "کیک خرچنگ",
    "creme brulee": "کرم بروله", "croque madame": "کروک مادام", "cup cakes": "کاپ کیک",
    "deviled eggs": "تخم‌مرغ شکم‌پر", "donuts": "دونات", "dumplings": "دامپلینگ",
    "edamame": "ادامامه", "eggs benedict": "تخم‌مرغ بندیکت", "escargots": "حلزون",
    "falafel": "فلافل", "filet mignon": "فیله مینیون", "fish and chips": "ماهی و سیب‌زمینی",
    "foie gras": "فوآگرا", "french fries": "سیب‌زمینی سرخ‌کرده",
    "french onion soup": "سوپ پیاز", "french toast": "نان تست فرانسوی",
    "fried calamari": "کالاماری سوخاری", "fried rice": "برنج سرخ‌کرده",
    "frozen yogurt": "ماست یخ‌زده", "garlic bread": "نان سیر", "gnocchi": "نیوکی",
    "greek salad": "سالاد یونانی", "grilled cheese sandwich": "ساندویچ پنیر کبابی",
    "grilled salmon": "ماهی سالمون کبابی", "guacamole": "گواکامولی", "gyoza": "گیوزا",
    "hamburger": "همبرگر", "hot and sour soup": "سوپ ترش و تند", "hot dog": "هات‌داگ",
    "huevos rancheros": "تخم‌مرغ مکزیکی", "hummus": "حمص", "ice cream": "بستنی",
    "lasagna": "لازانیا", "lobster bisque": "سوپ خرچنگ", "lobster roll sandwich": "ساندویچ خرچنگ",
    "macaroni and cheese": "ماکارونی و پنیر", "macarons": "ماکارون", "miso soup": "سوپ میسو",
    "mussels": "صدف", "nachos": "ناچو", "omelette": "املت", "onion rings": "حلقه پیاز",
    "oysters": "صدف خوراکی", "pad thai": "پد تای", "paella": "پائیا", "pancakes": "پنکیک",
    "panna cotta": "پاناکوتا", "peking duck": "اردک پکنی", "pho": "سوپ فو", "pizza": "پیتزا",
    "pork chop": "استیک خوک", "poutine": "پوتین", "prime rib": "دنده گوساله",
    "pulled pork sandwich": "ساندویچ گوشت ریش‌ریش", "ramen": "رامن", "ravioli": "راویولی",
    "red velvet cake": "کیک رد ولوت", "risotto": "ریزوتو", "samosa": "سمبوسه",
    "sashimi": "ساشیمی", "scallops": "گوش ماهی", "seaweed salad": "سالاد جلبک",
    "shrimp and grits": "میگو و بلغور", "spaghetti bolognese": "اسپاگتی بلونز",
    "spaghetti carbonara": "اسپاگتی کاربونارا", "spring rolls": "اسپرینگ رول",
    "steak": "استیک", "strawberry shortcake": "کیک توت‌فرنگی", "sushi": "سوشی",
    "tacos": "تاکو", "takoyaki": "تاکویاکی", "tiramisu": "تیرامیسو",
    "tuna tartare": "تارتار ماهی تن", "waffles": "وافل",
    "cat": "گربه", "dog": "سگ",
}
PERSIAN_TEMPLATES = ["{query}", "عکسی از {query}", "تصویری از {query}"]

#: Concepts with no image in the set, to see what search returns for "nothing".
ABSENT = ["a theatre stage with actors", "a football match in a stadium",
          "a snowy mountain", "a wedding ceremony", "a busy city street at night",
          "a classroom with students", "a car race", "people dancing at a concert"]


def average_precision(scores: np.ndarray, relevant: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = relevant[order]
    if not hits.any():
        return 0.0
    precision_at_hit = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float(precision_at_hit[hits].mean())


def evaluate(vectors: list[np.ndarray], labels: list[str], queries: dict[str, np.ndarray]
             ) -> dict:
    """Rank every image for every query, scoring an image by its best region."""
    aps, top1, rprec = [], [], []
    for label, query in queries.items():
        scores = np.array([float((regions @ query).max()) for regions in vectors])
        relevant = np.array([lbl == label for lbl in labels])
        r = int(relevant.sum())
        if not r:
            continue
        order = np.argsort(-scores, kind="stable")
        aps.append(average_precision(scores, relevant))
        top1.append(bool(relevant[order[0]]))
        rprec.append(float(relevant[order[:r]].mean()))
    return {"queries": len(aps), "mAP": round(float(np.mean(aps)), 4),
            "top1": round(float(np.mean(top1)), 4),
            "r_precision": round(float(np.mean(rprec)), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = copy.deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")))
    raw["models"]["vision_language"] = args.model
    raw["models"]["score_mode"] = default_score_mode(args.model)
    raw["runtime"]["low_memory"] = bool(os.environ.get("VIE_LOW_MEMORY"))
    raw["indexing"]["batch_size"] = 6
    config = Config.from_dict(raw)
    encoder = VisionLanguageEncoder(load_all(config))
    layout = config.crop_layout if config.multi_crop else "1"

    paths = sorted(p for p in args.data.rglob("*.jpg") if p.is_file())
    labels = [p.relative_to(args.data).parts[1].replace("_", " ") for p in paths]
    kinds = [p.relative_to(args.data).parts[0] for p in paths]
    started = time.perf_counter()
    vectors: list[np.ndarray] = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()  # BGR, as indexed
        h, w = image.shape[:2]
        regions = encoder.embed_batch([image], [crop_regions(w, h, layout)])[0]
        vectors.append(np.vstack([v for _, v in regions]))
    embed_s = time.perf_counter() - started
    whole = [v[:1] for v in vectors]  # region 0 is always the whole frame
    print(f"{args.model}: {len(paths)} images, {embed_s:.0f} s to embed", flush=True)

    targets = sorted({lbl for lbl, kind in zip(labels, kinds, strict=True) if kind != "other"})
    templates = list(config.query_templates)
    variants = {
        "label alone": {t: encoder.encode_query(t, ["{query}"]) for t in targets},
        "ensemble": {t: encoder.encode_query(t, templates) for t in targets},
    }
    translated = [t for t in targets if t in PERSIAN]
    if translated:
        variants["persian alone"] = {t: encoder.encode_query(PERSIAN[t], ["{query}"])
                                     for t in translated}
        variants["persian ensemble"] = {t: encoder.encode_query(PERSIAN[t], PERSIAN_TEMPLATES)
                                        for t in translated}

    report: dict = {"model": args.model, "images": len(paths), "queries": len(targets),
                    "templates": templates, "seconds_to_embed": round(embed_s, 1),
                    "multi_crop": {}, "whole_frame_only": {}}
    for name, queries in variants.items():
        report["multi_crop"][name] = evaluate(vectors, labels, queries)
        report["whole_frame_only"][name] = evaluate(whole, labels, queries)

    # What a "nothing found" floor would have to separate.
    calibrated = getattr(encoder, "calibrated", lambda cosine: None)
    present, absent = [], []
    for label, query in variants["ensemble"].items():
        best = max(float((v @ query).max()) for v, lbl in zip(vectors, labels, strict=True)
                   if lbl == label)
        present.append(calibrated(best))
    for concept in ABSENT:
        query = encoder.encode_query(concept, templates)
        absent.append(calibrated(max(float((v @ query).max()) for v in vectors)))
    if all(p is not None for p in present + absent):
        report["nothing_found"] = {
            "present_best_probability": {"min": min(present), "median": float(np.median(present))},
            "absent_best_probability": {"max": max(absent), "median": float(np.median(absent))},
            "absent_queries": dict(zip(ABSENT, absent, strict=True)),
        }

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\n{'variant':18s} {'multi-crop mAP':>15s} {'top1':>6s} {'whole mAP':>10s}")
    for name in variants:
        mc, wf = report["multi_crop"][name], report["whole_frame_only"][name]
        print(f"{name:18s} {mc['mAP']:>15.3f} {mc['top1']:>6.2f} {wf['mAP']:>10.3f}")
    out = args.out or Path(f"benchmarks/search-eval-{args.model.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
