# Roadmap

How this project gets from four Colab notebooks to a repository that holds up to
international review. Every task below has an acceptance criterion, so "done" is
never a matter of opinion.

Progress is tracked on the GitHub Project board. Each task ID maps to one issue.

---

## Work categories

| | Category | What it covers |
|:--|:--|:--|
| 🔴 | **Correctness** | Defects that make the system return wrong answers |
| 🏗️ | **Architecture** | Notebook → package, config, CLI, data layer |
| 🧠 | **Models & Algorithms** | Model choice, decision rules, retrieval quality |
| ⚡ | **Performance** | Throughput, latency, memory, indexing cost |
| 🧪 | **Engineering Quality** | Tests, CI, reproducibility, error handling |
| 📖 | **Presentation** | README, diagram, demos, docs |
| 🔐 | **Security & Privacy** | Biometric data handling, supply chain, secrets |

---

## Phase 0 — Foundation & Safety

> Goal: a repository that is safe to work in. Nothing here changes behaviour.

| ID | Cat | Task | Acceptance criterion | Est. |
|:--|:--:|:--|:--|:--:|
| P0-1 | 🔐 | `.gitignore` blocking gallery, `*.db`, weights, embeddings | `git status` stays clean after a full pipeline run | 30 m |
| P0-2 | 📖 | Roadmap + project board + issues | Every task below exists as an issue | 1 h |
| P0-3 | 🔐 | Licensing and ownership decision | Ownership of the original work confirmed in writing; LICENSE chosen accordingly | — |
| P0-4 | 📖 | Honest placeholder README | States current maturity without overselling | 1 h |
| P0-5 | 🧪 | Pinned `requirements.txt` | `pip install -r requirements.txt` reproduces a working env from scratch | 1 h |

---

## Phase 1 — Critical correctness fixes

> Goal: stop the system returning wrong answers. **Highest value per hour in the
> whole roadmap — roughly 5 hours removes every known critical defect.**

| ID | Cat | Task | Acceptance criterion | Est. |
|:--|:--:|:--|:--|:--:|
| P1-1 | 🔴 | **Species mapping is broken.** Substring matching without word boundaries sends `African elephant` → *Insect* (via "eleph**ant**"), `wild boar` → *Reptile* (via "**boa**r"), `sea lion` → *Cat*. Reverse pollution too: `mailbox` → *Large Mammal*, `bathtub` → *Small Mammal*, `beer bottle` → *Insect*. | A test asserts the category of ≥30 named classes, including every case above, and passes | 3 h |
| P1-2 | 🔴 | **Face/food cross-verification never runs.** In `run_advanced_preprocessing` the food stage reads `has_face`/`faces` ~30 lines *before* the face stage assigns them, so the guard is always false. | Reorder to face → animal → food; a test proves a food box overlapping a face is rejected | 15 m |
| P1-3 | 🔴 | **No confidence floor on the classifier.** `torch.max` is used unconditionally, so a 21.5 % top-1 is presented as a confirmed match (visible in the shipped demo). | Minimum confidence is configurable and enforced; `Object / Non-Animal` crops are rejected | 30 m |
| P1-4 | 🔴 | **Index and search disagree.** Indexing uses COCO classes `[39…55]` @ conf 0.15; search uses `[45…56]` @ 0.25 — so search re-introduces class 56 (chair), which indexing deliberately excluded, and ignores 39–44. Animal path gates at 0.40 but searches at 0.30. | Both passes read the same constants from config; a test asserts they match | 30 m |
| P1-5 | 🔴 | Unanchored query matching: searching `animal` matches the category string `object / non-animal` and returns non-animals | Word-boundary matching; a test covers the `animal` case | 30 m |
| P1-6 | 🔴 | Duplicate results: the unified face search lost the per-image `break`, so one image with two matching faces appears twice | Results are unique per image; test covers a two-face image | 15 m |

---

## Phase 2 — Architecture

> Goal: turn four notebooks into one installable package with a real data layer.

| ID | Cat | Task | Acceptance criterion | Est. |
|:--|:--|:--|:--|:--|
| P2-1 | 🏗️ | Extract notebooks into `src/` modules; notebooks become thin callers | No business logic left in any notebook | 8 h |
| P2-2 | 🏗️ | Central `configs/default.yaml` — every threshold, class list and model name | No magic numbers in code; one config fully describes a run | 2 h |
| P2-3 | 🏗️ | CLI: `vie index` / `vie search face|animal|food|text` | Full pipeline runs from a terminal with no notebook | 4 h |
| P2-4 | 🏗️ | Remove Colab coupling (`google.colab.files`, `cv2_imshow`, `/content` paths) | Runs unchanged on Linux, Windows and Colab | 2 h |
| P2-5 | 🔐 | Replace `pickle` BLOBs with `np.float32.tobytes()` + `np.frombuffer` | Loading a hostile DB cannot execute code; embeddings ~4× smaller | 1 h |
| P2-6 | 🏗️ | Add primary key to `face_embeddings`; add a migration | Re-running indexing twice produces no duplicate rows | 1 h |
| P2-7 | 🏗️ | Content-hash based incremental indexing | Edited files are re-indexed; deleted files are pruned | 2 h |

---

## Phase 3 — Models & algorithms

> Goal: every module as strong as the best one. The food module's contrastive-candidate
> design is the internal benchmark — the others should match it.

| ID | Cat | Task | Acceptance criterion | Est. |
|:--|:--|:--|:--|:--|
| P3-1 | 🧠 | **Rewrite the animal module on CLIP zero-shot**, replacing the 250-keyword map. CLIP is already loaded, already FP16 and already indexed. Removes P1-1/P1-3/P1-5 by construction and makes the vocabulary open. | Species queries outside ImageNet-1k (e.g. "red panda") return correct results | 3 h |
| P3-2 | 🧠 | Persist detector boxes and top-k logits at index time | Animal query becomes a pure SQL scan — no detector re-run | 2 h |
| P3-3 | 🧠 | Calibrated scores for text search — raw cosine (0.23) reads as failure to users | Displayed score is interpretable and documented | 2 h |
| P3-4 | ⚡ | Restore the standalone face module's `ThreadPoolExecutor` prefetch, pre-resize and checkpointing, which the unified pipeline dropped | Interrupted indexing resumes without loss; GPU utilisation measurably up | 2 h |
| P3-5 | 🧠 | Correct preprocessing chain `Resize(256)+CenterCrop(224)` (already written in the standalone notebook, then shadowed and unused) | No aspect-ratio distortion; documented | 30 m |
| P3-6 | 🧠 | Minimum crop size before classification | Sub-40 px detections no longer produce confident labels | 30 m |

---

## Phase 4 — Engineering quality

| ID | Cat | Task | Acceptance criterion | Est. |
|:--|:--|:--|:--|:--|
| P4-1 | 🧪 | pytest suite over the framework-free logic | Mapping, config, DB and scoring covered; CI green | 4 h |
| P4-2 | 🧪 | GitHub Actions CI (lint + test, 3.10/3.12) | Every push runs it | 1 h |
| P4-3 | 🧪 | Structured logging replacing `print` | Log level configurable; no bare `except` | 2 h |
| P4-4 | 🧪 | **Benchmark harness + honest metrics table** | README reports measured numbers with hardware and dataset stated | 4 h |
| P4-5 | 🔐 | Pin `torch.hub` yolov5 to a commit or vendor it (currently clones unpinned master with `trust_repo=True`) | Build is reproducible and supply-chain safe | 1 h |
| P4-6 | 🧪 | Normalise the device/dtype guard (`device == "cuda"` vs `device.type == "cuda"` are mixed) | One idiom everywhere; no Half/Float mismatch possible | 15 m |

---

## Phase 5 — Presentation

| ID | Cat | Task | Acceptance criterion | Est. |
|:--|:--|:--|:--|:--|
| P5-1 | 📖 | **Correct the architecture diagram.** It currently claims FAISS (code uses a NumPy loop), a "120-species dictionary" (does not exist), MobileNetV3 at index time (runs only at query time) and stored MegaDetector crops (nothing is stored). | Diagram matches code claim for claim | 2 h |
| P5-2 | 📖 | Full README per the agreed section list | A stranger can run it without asking questions | 3 h |
| P5-3 | 📖 | `docs/design-decisions.md` — why contrastive candidates beat a cosine threshold, why not FAISS yet | Reads as engineering judgement, not omission | 2 h |
| P5-4 | 📖 | Demo assets: existing four screenshots + a short GIF | Visible above the fold in README | 2 h |
| P5-5 | 📖 | Gradio demo | Runs locally in one command | 4 h |

---

## Phase 6 — Scale & polish

> Nothing here is needed today. Listed so the growth path is explicit.

| ID | Cat | Task | Trigger |
|:--|:--|:--|:--|
| P6-1 | ⚡ | ANN index (FAISS / hnswlib) | **Only past ~100 k vectors.** At 65 images brute force answers in 0.0025 s; adding FAISS now would be over-engineering | 
| P6-2 | ⚡ | Batched inference for indexing | When indexing time becomes the bottleneck |
| P6-3 | 🏗️ | Docker image | When others need to reproduce the environment |
| P6-4 | 🏗️ | REST API (FastAPI) | When something other than a CLI must call it |

---

## Sequencing

```
Phase 0  ─────┐
              ├──> Phase 1 (critical fixes) ──> Phase 2 (architecture) ──┐
              │                                                          │
              └──> Phase 5-1 (fix diagram, can run in parallel)          │
                                                                         │
              Phase 3 (models) <────────────────────────────────────────┘
                    │
                    └──> Phase 4 (quality) ──> Phase 5 (presentation) ──> public release
                                                                              │
                                                                    Phase 6 when scale demands
```

**Do not make the repository public before Phase 1 and P0-3 are complete.** Publishing
a known-broken species mapping, or client-owned code without a licensing decision, is
harder to undo than to delay.
