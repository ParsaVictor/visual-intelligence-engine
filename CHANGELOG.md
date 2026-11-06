# Changelog

Each version is a git tag. Check one out to get that state of the project.

## v0.6.0: search the whole archive by what it shows

**One concept search over photos and videos.** `Engine.search_everything("a theatre
performance")` ranks photos and video moments together on SigLIP's calibrated
probability that the description fits, keeps the strongest moments in view, and
draws video tiles with their timestamp. The notebook has a section for it.

**Food and unknown species no longer touch pixels at query time.** Each detected
animal and dish is encoded at index time, padded exactly as search pads it, in the
same batched pass as the scene regions (schema v4). A food query, or a species
outside ImageNet, used to decode every candidate image and run the image tower on
its crops, so it took minutes on thousands of photos; it is now a matrix product
over stored vectors through the paths' own ranking rule. An older index is still
served by cropping.

**Persian queries and SigLIP 2.** `google/siglip2-so400m-patch14-384` is supported
(its tokenizer declares no length, so the 64 tokens both models trained on are now
explicit) and understands Persian. A query is averaged over several phrasings
(`text.query_templates`). `scripts/eval_search.py` and the *Search quality* workflow
measure both choices with the real weights.

**Measured defaults.** SigLIP 2 is the default vision-language model: equal on English
queries (mAP 1.0 both), 0.945 against 0.738 on Persian, and with thresholds calibrated
for it (`model_overrides`, food 0.92, species 0.84) food precision 0.94 against 0.89.
Each model's thresholds follow it automatically, and a threshold set explicitly still
wins. A Persian query is encoded as typed, since wrapping it in phrasings never helped.
`text.min_probability: 0.001` lets an absent concept return nothing. Changing the model
means `vie index --rebuild`; the index refuses to mix models.

**Video faces at a fixed cost per person.** Each person keeps up to four exemplar
views (the most typical, then the least alike) and their on-screen intervals,
instead of a vector for every sighting; face search returns all of a person's
intervals.

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

**Found in a strict review of the whole codebase, and fixed:**

* **Face indexing was doing 2.2x the work it needed.** InsightFace runs every model in
  its pack on every face by default, including two landmark models and age/gender that
  nothing here reads. Loading only detection and recognition: 1144 -> 527 ms on a
  four-face 1080p frame, with identical faces and vectors.
* **Unchanged files could be re-hashed forever.** A file whose content matched but
  whose mtime had moved (a copy, a Drive sync, an extracted zip) was skipped without the
  new mtime being stored, so every later run read it in full again. A gallery that
  moved kept pointing at its old paths. Both are now recorded on the first such run.
* **Pruning is guarded in the pipeline,** so `vie index` is protected too, not only the
  notebook: if fewer than half of the indexed files are on disk, nothing is deleted.
* **The annotated video ignored rotation flags** (portrait phone videos came out
  sideways) and gave up on a damaged file; it now uses the analysis decoder for both.
* **Persian names and file names** are shaped right to left on cards and videos.
* Comments that still described CLIP, MobileNetV3 and a 768-d space now describe what
  runs: SigLIP so400m, ConvNeXt-Tiny and the margin rule.

**`notebooks/0_run_everything.ipynb`.** The whole system with one *Run all* in Colab. The
index lives on Google Drive, so the first run reads the whole dataset and later runs read
only new or changed files: 37 photos took 18.4 s, and with one photo added the next run
took 1.1 s; two videos took 67.8 s the first time and 0.0 s the next. Every stage reports its own failure and the rest carry on.

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
