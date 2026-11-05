# Audit of v0.2.0

An audit of `main` at `61d88c5`, done on 2026-11-04. Findings were reproduced by
running the code, not by reading it alone. Each one is now pinned by a test in
`tests/test_integration.py` or `tests/test_video.py`.

## The headline

**198 of 198 tests passed, CI was green, and the system did not work.**

Every component had been upgraded and unit tested, and most upgrades were never
connected to anything. No test went through `cli -> query -> search`, so nothing could
notice. The README marked "Four query paths" as done, and one of the four crashed on
every call.

The lesson is the reason for `tests/test_integration.py` (the real wiring, with only
the weights faked) and `.github/workflows/models.yml` (the real weights).

## Things that did not work

| # | Severity | Finding | How it showed |
|:--|:--|:--|:--|
| 1 | Critical | `vie search animal` crashed on every call | `query.py` called `animal.rank()`, removed by `35a9959` when the path became a hybrid: `AttributeError` |
| 2 | Critical | Every stored image vector had red and blue swapped | `encode_image` passed OpenCV's BGR array straight to the processor, so "a red car" looks for blue ones. No error is ever raised |
| 3 | Critical | SigLIP text was tokenized wrongly | `padding=True` where SigLIP requires `padding="max_length"`. Different embeddings, no error |
| 4 | High | The species classifier never ran | `SpeciesClassifier`, `rank_from_index` and `choose_route` were unreachable, and the table they need did not exist. Every species query would have run a model, against the documented promise |
| 5 | High | The food gate never ran | `gate_score` and the `clip_gate_*` keys were read by nothing. The stated reason for it, finding non-Western dishes COCO cannot detect, did not happen |
| 6 | High | Multi-crop never ran | `encode_regions`, `crop_regions` and `multi_crop: true` were all unused, so only the whole-image vector existed. The config comment described a 6x storage cost nobody was paying |
| 7 | High | Indexing a large archive would exhaust memory | `Executor.map` decodes everything up front: 10,000 of 10,000 images were in RAM after consuming one |
| 8 | High | A failed image became permanently unsearchable | No per-image transaction, so a failure after `upsert_image` stored the new hash with no vectors, and the next run skipped it |
| 9 | Medium | Every command loaded all six models | A text search spent minutes and gigabytes on a face model, two detectors and a classifier |
| 10 | Medium | Upgrading an older index crashed | `CREATE TABLE IF NOT EXISTS` never adds a column, and the version was overwritten on every start. P2-6's "add a migration" was closed without one |
| 11 | Medium | Mixing embedding models was silent | Nothing recorded which model built an index. Two models of the same dimension would produce meaningless scores with no error |
| 12 | Medium | Only editable installs worked | The default config lived outside the package, so `pip install .` failed on the first command. CI only tested editable |
| 13 | Medium | `onnxruntime-gpu` in the only install path | Fails on macOS and is the wrong runtime without CUDA |
| 14 | Medium | Files with non-ASCII names were invisible on Windows | `cv2.imread` returns `None`, which hits every Persian file name |
| 15 | Medium | Same-named files in different folders | The gallery was flat, keyed by bare name |
| 16 | Low | The README's own example failed | `vie search text ... --json`: `--json` only worked before the subcommand |
| 17 | Low | `--top-k` was ignored on two paths | Accepted for animal and food, never applied |
| 18 | Low | Boxes were in the wrong coordinates | Returned for the downscaled image, not the file on disk |
| 19 | Low | `max_index_dimension` was ignored | The code hard-coded 1600 while the config said 1000 |
| 20 | Low | P3-3 was closed but not delivered | `relative_confidence` existed and was never called, so raw cosines were still displayed |
| 21 | Low | Food queries re-ran RT-DETR on every candidate | Boxes were persisted for animals but not food |
| 22 | Low | The demo crashed with no image uploaded | `AttributeError`, and Gradio refused to serve results from a gallery outside the working directory |
| 23 | Low | Weights downloaded with no integrity check | `.pt` files execute code when loaded, and the URL pointed at a repository that has since moved |
| 24 | Low | The benchmark measured different code | It timed one matrix multiply while the product ran a Python loop per row |
| 25 | Docs | Docs described an earlier system | CLIP ViT-L/14 and 768-d where the config says SigLIP so400m and 1152-d; "five models" followed by a list of four; "120 tests" where there were 198 |

### Correction on scoring

An early version of this audit called softmax over SigLIP logits a serious defect, as
the code's own comments did. Measurement did not support that. On real images softmax,
SigLIP's absolute sigmoid and the logit margin rank almost identically (AUC 0.999 to
1.0). The real defect was thresholds on the wrong scale: 0.55 and 0.60 carried over
from CLIP gave **46% precision on food and 57% on species** with the default model. The
README has the current numbers.

## What was already good

Worth saying plainly, because a list of defects makes it easy to miss:

- **The core idea is sound.** Heavy models once at index time, an SQL prefilter, and
  expensive work only on what survives.
- **The protocol boundary worked.** Search and pipeline code knowing nothing about
  torch is exactly what made this audit's end-to-end tests cheap to write.
- **Pure logic was correct.** Geometry, species resolution, the config invariants and
  the corrected stage order all held up.
- **Security instincts were right.** No pickle, biometric data blocked from git, no
  secrets anywhere in history.
- **The discipline was real.** Regression tests that name the defect they pin, and a
  README that refused to claim benchmark numbers it did not have.

## Still open

| Item | Why it is not done |
|:--|:--|
| **P0-3 ownership and licence** | A decision for the owner, not something code can settle |
| **Accuracy on a real archive** | The calibration set is 72 easy public images. It is good for choosing a rule and catching a wrong scale, not for promising accuracy |
| **Face recognition quality** | The path runs on CI with real weights, but nothing here measures how well it identifies people, and the CI images contain no faces to search for |
| **Docker image** | Rewritten, not built: this machine has no Docker |
| **GPU** | Nothing has been run on a GPU in this audit, including half precision |
