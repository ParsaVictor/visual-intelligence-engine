<div align="center">

# Visual Intelligence Engine

**Search photos and videos by what is in them.**<br>
Faces, animals, food, plain-English descriptions, and moments inside a video, all from one index.

[![CI](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/ci.yml)
[![Real models](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/models.yml/badge.svg)](https://github.com/ParsaVictor/visual-intelligence-engine/actions/workflows/models.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/github/v/tag/ParsaVictor/visual-intelligence-engine?label=version&color=e8a33d)](CHANGELOG.md)

<div dir="rtl">

جستجو در عکس و ویدیو بر اساس محتوا: چهره، حیوان، غذا، توضیح متنی و لحظه‌های داخل ویدیو.

</div>

![The video report on a real film](assets/demo/video-report.gif)

<sub><code>vie video analyze</code> on four minutes of <i>Tears of Steel</i> (© Blender Foundation, CC BY 3.0)</sub>

</div>

## What you can ask

| Ask with | Example | You get |
|:--|:--|:--|
| A face | `vie search face --image person.jpg` | Every photo of that person |
| An animal | `vie search animal --query "red panda"` | Photos with the animal boxed |
| A dish | `vie search food --query "ghormeh sabzi"` | Photos of it, even with no detector class for it |
| A description | `vie search text --query "a man with a blue backpack"` | The closest matches |
| A video | `vie video analyze clip.mp4 --html report.html` | Shots, topics, people, speech and a summary |
| A moment | `vie video search --query "a goal being scored"` | Timestamps across all your videos |

## See it work

| Face search | Species search |
|:--|:--|
| ![Face search results](assets/demo/face-search.jpg) | ![Species search results](assets/demo/animal-search.jpg) |
| **Food search** | **Free-text search** |
| ![Food search results](assets/demo/food-search.jpg) | ![Free-text search results](assets/demo/text-search.jpg) |

<sub>Screenshots from the original notebooks, so the scores shown predate the current scoring.</sub>

## Architecture

![Files flow through analysis into a SQLite index, which search then filters and compares](assets/diagrams/architecture.png)

Three decisions carry most of the design:

- **Index once, query cheaply.** A species the classifier knows is answered from stored predictions, with no model loaded at all.
- **Ask against rivals.** Instead of thresholding a raw similarity, each query competes with prompts like "a different kind of animal", which gives the model a way to say no.
- **Measure, then decide.** Thresholds and the search backend come from measurements on real data, including where they overturned the original assumptions.

More in [the architecture](docs/architecture.md), [the design decisions](docs/design-decisions.md) and [the video pipeline](docs/video.md).

## How the project got here

![Five versions: notebooks, package, SigLIP, audit, video](assets/diagrams/journey.png)

| Step | What changed |
|:--|:--|
| **Fix what was wrong** | Species mapping, cross-checks that never ran, drifting thresholds |
| **Build a real package** | CLI, one config file, a database with migrations |
| **Upgrade the models** | CLIP to SigLIP, a stronger classifier, multi-crop search |
| **[Audit it honestly](docs/AUDIT.md)** | Found 25 defects behind 198 green tests, then fixed them all |
| **Test with real weights** | End-to-end tests, plus a CI job running every real model |
| **Understand video** | Shots, people, speech, timelines and search by moment |

Next up: on-screen text search, natural-language summaries and a REST API. See the [roadmap](ROADMAP.md) and [changelog](CHANGELOG.md).

## Quick start

```bash
git clone https://github.com/ParsaVictor/visual-intelligence-engine.git
cd visual-intelligence-engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[models,gpu,video,speech,demo]"    # no GPU? use "cpu" instead of "gpu"
```

Put photos in `data/gallery/` and videos in `data/videos/`, then:

```bash
vie doctor            # is this machine ready?
vie index             # index the photos
vie video index       # index the videos
python -m vie.demo    # open the web demo
```

Every search also takes `--json` for scripts and `--html report.html` for a shareable page.

**Smart Gallery.** Open `run_all.ipynb`, set the gallery folder and press Run All. It
indexes the folder and opens a page with the whole archive at a glance (people, animals,
dishes, subjects) and every search mode, including search by a person's name and by a
similar photo. See [the Smart Gallery](docs/smart-gallery.md).

**On a small machine,** set `google/siglip-base-patch16-224` and `runtime.low_memory: true` in [the config](src/vie/default.yaml), and use `--labels-only` for video. A fast GPU is recommended for large archives.

## Measured

Precision on 72 public labelled images with the default model, at 100% recall. The set is small and easy, so read this as a comparison between versions, not as expected accuracy on your archive.

| Search | v0.2 | Now |
|:--|--:|--:|
| Species | 57% | **94%** |
| Food | 46% | **92%** |

Vector search over 100,000 vectors on 4 CPU cores takes **10 ms**, and returns the exact answer.

Not yet measured: accuracy on a real archive, face recognition quality, and anything on a GPU. Reproduce the numbers above with [`scripts/calibrate.py`](scripts/calibrate.py) and [`scripts/benchmark.py`](scripts/benchmark.py).

## Privacy

This system stores **biometric identifiers of identifiable people**. Photos, footage, the index and generated reports are blocked from version control, and reports and the demo call no third party. If you deploy it, you need a lawful basis for processing those images. See [SECURITY.md](SECURITY.md).

## Built with

[SigLIP](https://github.com/google-research/big_vision) · [InsightFace](https://github.com/deepinsight/insightface) · [MegaDetector](https://github.com/agentmorris/MegaDetector) · [Ultralytics](https://github.com/ultralytics/ultralytics) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [PyAV](https://github.com/PyAV-Org/PyAV)

## License

Not yet chosen, so no reuse rights are granted for now. Each model above has its own licence.
