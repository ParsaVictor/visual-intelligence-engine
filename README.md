<div align="center">

# Visual Intelligence Engine

**Multimodal search over an image archive — faces, species, food and free-text, on one index.**

[![Status](https://img.shields.io/badge/status-under%20reconstruction-orange)](ROADMAP.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

</div>

> ⚠️ **Current state: under reconstruction.** The system works end to end as four Colab
> notebooks, but it is being rebuilt into an installable package and several known
> correctness defects are still being fixed. See the [Roadmap](ROADMAP.md) for exactly
> what is done and what is not. Nothing here is oversold — if a number is not measured,
> it is not claimed.

---

## What it does

Four query paths over a single SQLite index of an image gallery:

| Query | You give it | It returns |
|:--|:--|:--|
| 👤 **Face** | a photo of a person | every image in the archive containing that person, ranked by similarity |
| 🦁 **Species** | a word (`cat`, `eagle`) | images containing that animal, with a box on the detection |
| 🍱 **Food** | a dish name (`French fries`) | images containing that dish, even in a crowded table scene |
| 🌐 **Free text** | any description | images matching the description semantically, no fixed vocabulary |

Built for a news agency's photo archive, where an editor needs to find *"that photo of
the minister at the ceremony"* across tens of thousands of unlabelled files.

## Models

Five pretrained models, none of them fine-tuned. The engineering is in how they are
combined, indexed and filtered — not in training.

| Model | Role | Output |
|:--|:--|:--|
| InsightFace `buffalo_l` (RetinaFace + ArcFace) | face detection & recognition | 512-d embedding |
| MegaDetector v5a | class-agnostic animal localiser | boxes |
| MobileNetV3-Small (ImageNet-1k) | species label *(being replaced — see P3-1)* | 1000-way |
| RT-DETR-X | food & tableware detection | boxes (COCO) |
| CLIP ViT-L/14 | image ↔ text semantic space | 768-d embedding |

## The idea that makes it fast

Heavy models run **once per image at index time**, never at query time. Each image gets
three cheap boolean flags — `has_face`, `has_animal`, `has_food` — plus its embeddings.
A query first narrows the candidate set with plain SQL, then runs vector search only on
what survives.

```
index:  image ──> detectors ──> flags + embeddings ──> SQLite
query:  SQL prefilter ──> vector search on candidates only ──> ranked results
```

## Status

| Area | State |
|:--|:--|
| Four query paths working | ✅ demonstrated |
| Installable package / CLI | ⬜ Phase 2 |
| Known correctness defects fixed | ⬜ Phase 1 |
| Tests & CI | ⬜ Phase 4 |
| Benchmarks | ⬜ Phase 4 — **no performance claims until measured** |

## Privacy

This system extracts and stores **biometric identifiers of identifiable people**. The
image gallery and the generated database are excluded from version control and must
never be committed. Anyone deploying it is responsible for having a lawful basis to
process those images. See [Roadmap P0-3](ROADMAP.md).

## License

To be determined pending an ownership decision on the original work — see P0-3.
