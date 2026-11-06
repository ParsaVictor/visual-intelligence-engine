# Changelog

Each version is a git tag. Check one out to get that state of the project.

## v0.5.0: one engine, more accurate faces, output worth showing

**Faces, measured against student-face-gate.** `scripts/face_benchmark.py` puts ArcFace
(this repo) and SFace (the gate) on identical aligned crops. ArcFace wins on every measure
(d′ 8.08 against 5.86; 96.9% against 83.6% of hard head poses at each model's own
threshold, with no wrong person accepted by either), so it stays. Two ideas came over from
the gate, each measured here:

* **Mirror averaging**, on by default (`face.flip_tta`): every face is embedded as seen and
  mirrored. Hard poses 96.9% -> 97.6%, d′ 8.08 -> 8.43. One extra 112x112 pass per face.
* **Several query photos**: `vie search face --image a.jpg --image b.jpg`. Each indexed face
  keeps its best score over the query photos.

Turning the recipe on re-indexes only the images that have faces, not the whole archive.

**`vie.api.Engine`.** Every capability behind one object that loads each model once:
incremental image and video indexing, the four searches, video search, result cards and
annotated video. It refuses to prune when most of the archive vanishes at once, which is a
disconnected drive far more often than a real deletion.

**Output.** `vie.report.showcase` renders a search as a presentation-grade image, cropped
around the matched subject. `vie.video.render` writes an annotated MP4: tracked faces,
names from reference photos (with a margin over the runner-up, as in the gate) or grouped
people, shot labels and a presence timeline, plus a card of who appeared for how long.

**`notebooks/0_run_everything.ipynb`.** The whole system with one *Run all* in Colab. The
index lives on Google Drive, so the first run reads the whole dataset and later runs read
only new or changed files: one photo added to 37 took 2.7 s instead of 49.7 s, unchanged
videos 0.1 s instead of 216 s. Every stage reports its own failure and the rest carry on.

## v0.4.0: video understanding

New `vie.video` package and `vie video` commands: shot detection, keyframes, zero-shot
labels per shot and per video, faces clustered into people with time intervals,
animals and food on a timeline, optional speech and captions, a factual summary in
English and Persian, a self-contained HTML report, and search across videos by text,
speech or face. See [docs/video.md](docs/video.md).

## v0.3.0: the system does what it says

The audit in [docs/AUDIT.md](docs/AUDIT.md) found that all tests passed while one of
the four search paths crashed and most of the v0.2 upgrades were never called.

**Fixed:** species search; swapped colour channels in every image vector; SigLIP
tokenizer padding; thresholds on the wrong scale, now measured; unbounded memory while
indexing; half-written images that were never retried; non-ASCII file names on
Windows; same-named files in different folders; box coordinates; `--json` and
`--top-k`; the default config missing from regular installs.

**Connected:** multi-crop, the food gate, the species classifier at index time, and
calibrated text scores.

**Added:** schema migrations; refusal to mix embedding models; batched inference (#37);
`VectorCatalog` with optional FAISS HNSW (#36); lazy model loading; `runtime.low_memory`;
`vie doctor`; HTML search reports; a measured benchmark (#28) and a calibration script;
end-to-end tests and a real-model CI job; a verified digest for downloaded weights.

**Changed:** the default config moved into the package, at `src/vie/default.yaml`. The
install extras are now `.[models,cpu]` or `.[models,gpu]`. Existing indexes upgrade on
first use, but must be rebuilt to gain what v0.2 never computed: multi-crop vectors,
correct colours, stored species predictions and the food gate.
`vie index --rebuild` does that.

## v0.2.0: SigLIP (tag on `61d88c5`)

The project as it stood on 2026-11-04. Moved to SigLIP so400m with a hybrid species
path, multi-crop and a food gate in code. Tests pass; see the audit for what ran.

## v0.1.0: first complete package (tag on `c86d763`)

Notebooks extracted into an installable package with a CLI, config, tests and CI, on
CLIP ViT-L/14. `pyproject.toml` in every commit up to v0.2.0 says `0.1.0`, because the
tags were added afterwards.

## v0.0.1: original artefacts (tag on `935cdc6`)

The earliest state in git: the roadmap, the original demo screenshots and the original
architecture diagram. **The original Colab notebooks' code was never committed**, so
this tag is a record of what the project looked like and not something that runs.
`notebooks/` holds later rewrites of them.
