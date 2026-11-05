<div align="center">

# Visual Intelligence Engine

**Search images by what is in them: faces, species, food and free text, on one index.**

[![CI](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/ci.yml)
[![Real models](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/models.yml/badge.svg)](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/models.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

Find *"that photo of the minister at the ceremony"* in an archive of unlabelled press
photos, by face, by species, by dish, or by describing it in plain English.

<div dir="rtl">

جستجوی چندوجهی روی آرشیو تصویری: تشخیص چهره، گونه‌ی جانوری، غذا و جستجوی متنی آزاد.

</div>

</div>

---


| Face search | Species search |
|:--|:--|
| ![face](assets/demo/face-search.jpg) | ![animal](assets/demo/animal-search.jpg) |
| **Food search** | **Free-text search** |
| ![food](assets/demo/food-search.jpg) | ![text](assets/demo/text-search.jpg) |

*Screenshots from the original notebooks, which predate the current scoring.*

## What it does

| Query | Input | Returns |
|:--|:--|:--|
| **Face** | a photo of a person | every image containing that person, ranked by similarity |
| **Species** | a word: `zebra`, `quokka` | images containing that animal, with the detection boxed |
| **Food** | a dish: `French fries`, `ghormeh sabzi` | images of that dish, even ones no detector has a class for |
| **Free text** | any description | images matching semantically, with no fixed vocabulary |

## How it works

Heavy models run **once at index time**. A query narrows the candidates with plain SQL
first, then does expensive work only on what survives.

```
index:  image -> face, animal, food detectors + image vectors -> SQLite
query:  SQL prefilter -> vector search or stored predictions -> ranked results
```

A species the classifier knows is answered from stored predictions, with no model
loaded at all. One it does not know falls back to the vision-language model, so the
vocabulary stays open.

- [docs/architecture.md](docs/architecture.md): the data flow
- [docs/design-decisions.md](docs/design-decisions.md): why, including what measurement overturned
- [docs/AUDIT.md](docs/AUDIT.md): the audit behind v0.3

## Install

```bash
git clone https://github.com/ParsaVictor/visual-intelligence-engine.git
cd visual-intelligence-engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[models,gpu]"    # full stack, with CUDA
pip install -e ".[models,cpu]"    # full stack, without
pip install -e ".[dev]"           # logic and tests only, no torch
```

Optional extras: `ann` (FAISS), `demo` (Gradio).

**On a small machine** use `google/siglip-base-patch16-224` and `runtime.low_memory: true`.
If loading dies with no message or an access violation, memory ran out.

## Usage

Put images in `data/gallery/`, sub-folders included, then:

```bash
vie doctor                # is this machine ready, and is the index healthy?
vie index                 # incremental: only new or changed files
```

```bash
vie search face   --image query.jpg
vie search animal --query "red panda"
vie search food   --query "French fries"
vie search text   --query "a man with a blue backpack" --html results.html
vie search text   --query "sunset over the city" --json
```

Every threshold lives in [`src/vie/default.yaml`](src/vie/default.yaml). Pass your own with `--config`.

```bash
pip install -e ".[demo]"
python -m vie.demo
```

## Measured

Stated with their conditions, and with what they do not show.

**Hardware:** GitHub Actions runner, 4 vCPU, 16 GB RAM, no GPU, Python 3.12, torch 2.5.

### Vector search

1152-d clustered synthetic vectors, the size SigLIP so400m produces. This measures
search mechanics, not retrieval quality.

| Vectors | Load from SQLite | Exact search | HNSW search | HNSW recall@10 |
|:--|--:|--:|--:|--:|
| 1,000 | 11 ms | 0.04 ms | 0.13 ms | 1.00 |
| 10,000 | 83 ms | 0.6 ms | 0.6 ms | 0.98 |
| 100,000 | 785 ms | 10 ms | 1.1 ms | 0.81 |

Exact search stays the default until 500k vectors: 10 ms does not justify losing one true
result in five. Loading is the larger cost, and a resident process pays it only once.

### Scoring

`google/siglip-so400m-patch14-384`, the default, on 72 public labelled images: Food-101,
cats and dogs, and everyday objects. The set is small and easy, so read this as a
comparison between rules, not as expected accuracy.

| Path | v0.2 settings, precision | Now, precision | Now, recall |
|:--|--:|--:|--:|
| Species | 57% | 94% | 100% |
| Food | 46% | 92% | 100% |

`google/siglip-base-patch16-224` gave 84% to 100% and 48% to 90% on the same images, and is
the choice for a machine without a GPU. Reproduce with
`scripts/fetch_calibration_data.py` and `scripts/calibrate.py`.

### Encoding

SigLIP so400m on 4 vCPU: 3.5 s per image. Multi-crop encodes six regions per image, so
indexing without a GPU is slow. Batching saved about 6% on CPU.

**Not measured:** accuracy on a real archive, face recognition quality, and anything on
a GPU. Calibrate again before trusting a threshold on a different model.

## Testing

```bash
pytest              # no GPU, no downloads: real files and decoding, with the weights faked
pytest -m models    # the real weights
```

The fast suite goes through the real command line and database. Each test that names a
defect pins something that actually happened.

## Versions

| Tag | What it is |
|:--|:--|
| `v0.0.1` | The original artefacts. The notebooks' code was never committed |
| `v0.1.0` | First complete package, on CLIP |
| `v0.2.0` | SigLIP upgrade, as it stood before the audit |
| `v0.3.0` | The audit's fixes: everything connected, measured and tested |

Details in [CHANGELOG.md](CHANGELOG.md).

## Privacy

This system extracts and stores **biometric identifiers of identifiable people**. The
gallery, footage, index and generated reports are excluded from version control and must
never be committed. Reports embed thumbnails, faces included. Anyone deploying this is
responsible for having a lawful basis to process the images. See [SECURITY.md](SECURITY.md).

## Roadmap

[ROADMAP.md](ROADMAP.md) tracks the remaining work.

## Acknowledgements

[MegaDetector](https://github.com/agentmorris/MegaDetector) ·
[InsightFace](https://github.com/deepinsight/insightface) ·
[SigLIP](https://github.com/google-research/big_vision) ·
[Ultralytics](https://github.com/ultralytics/ultralytics)
## License

To be determined pending an ownership decision on the original work.
