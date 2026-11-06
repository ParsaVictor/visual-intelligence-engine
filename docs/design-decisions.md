# Design decisions

The reasoning behind choices that are not obvious from reading the code, and the
ones where the obvious option was deliberately rejected.

---

## Contrastive prompts instead of a similarity threshold

Species and dish search ask the model the same question, and it is not "how similar
is this crop to the query". It is:

```python
prompts = [
    "a photo of a red panda",                       # the hypothesis
    "a photo of a different kind of animal",        # rival
    "a photo of scenery with no animal in it",      # rival
]
```

A cosine similarity has no null hypothesis. Every crop produces *some* number, and there
is no principled place to cut it. Rivals give the model an explicit "none of the above".

### Which number to threshold was measured, and the first answer was wrong

The code used to document that SigLIP must be read through its absolute sigmoid, and
that softmax over its logits was a defect. `scripts/calibrate.py` compared the options
on real images (`siglip-base-patch16-224`, 72 public images):

| Rule | AUC, food / species | Best threshold, food / species | True match, typical |
|:--|--:|--:|--:|
| Absolute sigmoid | 1.000 / 1.000 | 0.05 / 0.003 | 63% / **2.4%** |
| Softmax over rivals | 0.999 / 1.000 | 0.98 / 0.93 | 100% / 97% |
| Logit margin | 0.999 / 1.000 | +4.1 / +2.5 | +8.3 / +3.3 |

All three rank about equally. What differs is whether there is a usable threshold. The
absolute sigmoid's best cut differs by a factor of eighteen between domains, and a
correct "cat" reads as 2%. SigLIP's learned bias makes every prompt improbable by
default, but the bias cancels in a difference. So recognition scores the **sigmoid of
the logit margin over the strongest rival**.

The thresholds it replaced were carried over from CLIP. On the same pairs, with the default
so400m model, they gave 46% precision on food and 57% on species. The numbers now are in the
README.

### The same lesson for video labels

SigLIP's absolute probability was about 0.001 for every video label on every frame, right
or wrong, so it cannot be thresholded. A softmax *within each label group* correctly chose
"a movie scene" for two films. The cost is that some label always wins, so a group has to
be a sensible set of alternatives, and a label needs real coverage to reach a summary.

---

## CLIP instead of a classifier plus a keyword map

The original mapped ImageNet's 1000 class names into 8 categories with
unanchored substring tests, in a fixed `if/elif` order. Verified against the real
class list, that produces:

| class | assigned | cause |
|:--|:--|:--|
| African elephant | Insect / Arthropod | `eleph`**`ant`** |
| giant panda | Insect / Arthropod | `gi`**`ant`** |
| wild boar | Reptile / Amphibian | **`boa`**`r` |
| sea lion | Cat | **`lion`** |
| mailbox | Large Mammal | **`ox`** |
| bathtub | Small Mammal | **`bat`** |
| beer bottle | Insect / Arthropod | **`bee`** |

Only 33 of 1000 classes reached "Large Mammal"; 658 fell through to
"Object / Non-Animal".

The fix was not to repair the map. Any hand-written map over a fixed vocabulary
has the same failure mode waiting in it, and no map can answer a query for a
species the classifier was never trained on. CLIP was **already loaded, already
FP16 and already indexed** for the other two paths, so switching cost nothing at
runtime and removed three defects at once — plus the 1000-class ceiling.

**What was kept:** MegaDetector as the localiser. That part of the original was
right. A class-agnostic detector gives a species-independent recall gate, and
cropping before recognition is what lets an animal occupying 4% of a press photo
survive the downscale to the model's input size — at full frame it is roughly
nine pixels across by the time it reaches the first convolution.

---

## Exact search until the numbers say otherwise

The architecture diagram once claimed FAISS while the code did a NumPy loop. The
assumption after that was a crossover at about 100k vectors. Measured through the real
code path, on 4 vCPU with 1152-d clustered synthetic vectors:

| Vectors | Load | Exact | HNSW | HNSW recall@10 |
|--:|--:|--:|--:|--:|
| 10,000 | 83 ms | 0.6 ms | 0.6 ms | 0.98 |
| 100,000 | 785 ms | 10 ms | 1.1 ms | 0.81 |

HNSW is faster at 100k, nobody can perceive the difference, and it silently loses one true
result in five at these settings. So `search.backend: auto` stays exact until 500k, and
FAISS is an optional extra. Load time is the larger cost at scale. A one-shot CLI pays it
on every query, while a resident process such as the demo pays it once.

On a 4 GB laptop under memory pressure the same load took 10 s rather than 0.8 s. Run
this where it will actually be used, because memory changes the answer by an order of
magnitude.

---

## Raw float32 instead of pickle

Embeddings are stored as little-endian float32 bytes, read back with
`np.frombuffer`.

`pickle.loads` executes arbitrary code. An index file is exactly the kind of
artefact that gets copied between machines and shared with colleagues, so
"loading the index" must not be a code-execution path. The original also pickled
`embedding.tolist()` — a Python list of floats rather than an array — which is
several times the size of the raw bytes for no benefit.

---

## Gate permissive, search strict

The index-time gate decides `has_animal` / `has_food`; the search pass decides
what is returned. The gate must be **at least as permissive**, because it is a
hard filter — an image it rejects is unreachable at any search threshold.

The original had this inverted for animals: it gated at 0.40 and searched at
0.30. The looser search threshold could not recover anything, and only admitted
low-confidence boxes on images that had already passed.

`Config.validate()` now rejects a configuration where the gate is stricter, or
where the search pass looks for a class the gate never indexed. That second rule
is how COCO class 56 (`chair`) got back into the food search after being
deliberately excluded from indexing.

---

## Device and dtype resolved together

`torch.device('cuda') == 'cuda'` evaluates to **False** (verified on torch 2.8).
The original used `device.type == "cuda"` in two places and `device == "cuda"` in
a third, so on a CUDA run the CLIP weights were cast to half precision while the
pixel values were left float32:

```
RuntimeError: expected scalar type Half but found Float
```

`vie.device.Runtime` resolves both together and casts through one method, so the
two cannot disagree. Half precision is only ever enabled on CUDA — fp16 on CPU is
slower than fp32 and unsupported for several ops.

---

## Model access behind protocols

The search and pipeline modules depend on small protocols (`FaceDetector`,
`BoxDetector`, `CropScorer`), never on torch directly. Everything model-specific
lives in `vie.adapters`.

This is what makes the decision rules testable. The regression tests that pin
each historical defect run in under two seconds, on any machine, with no GPU and
no model download — because what needs pinning is the logic, not the weights.

---

## Real weights in CI, because test doubles cannot see this

Every defect in this list passed a full green suite:

- SigLIP tokenized with the wrong padding (no error, wrong vectors)
- BGR sent to an RGB model (no error, wrong vectors)
- thresholds on another model's scale (no error, 46% precision)
- `faster-whisper` calling `av.open` with an option current PyAV removed

`tests/test_integration.py` runs the real wiring with fake weights in seconds on every
push. `.github/workflows/models.yml` runs real weights on real images and real footage.
Both exist because the fast suite alone was not enough.

---

## Fail where it is cheap, keep what is expensive

A video can take minutes to analyse, so an optional stage that fails produces a warning
on the report rather than an exception that discards the result. This came from a real
run, in which Whisper running out of memory threw away a finished analysis. The same idea
is why indexing commits periodically and retries a failed batch image by image.

## ArcFace over SFace, with two ideas taken from SFace's project

[student-face-gate](https://github.com/ParsaVictor/student-face-gate) recognises faces with
SFace (128-d, 39 MB, about 12 ms a face on a laptop CPU) because it must run on a gate PC
with no GPU. This system uses ArcFace from InsightFace `buffalo_l` (512-d, about 6x the CPU
cost). Accuracy decides here, so the two were measured head to head on identical aligned
crops (`scripts/face_benchmark.py`):

| | ArcFace | SFace |
|:--|--:|--:|
| Separation d′ between same and different people | **8.08** | 5.86 |
| Hard head poses accepted at the shipped threshold | **96.9%** | 83.6% |
| Equal error rate | **0.0%** | 0.2% |

ArcFace stays. What the gate does better is not the model but how it uses it, and two of
those habits carry over:

* **Mirror averaging.** Embed the aligned face and its mirror image, sum, normalise. A
  turned face then carries evidence from both sides. Measured here: hard poses 96.9% ->
  97.6%, d′ 8.08 -> 8.43, for one extra 112x112 pass per face, which is small next to
  detection. On by default as `face.flip_tta`.
* **Several photos of the person.** The gate enrols three photos and keeps the best match;
  with a fixed threshold that took accepted visits from 86.1% to 96.3% with no wrong person
  admitted. Here the same rule applies at query time: `--image` may be repeated.

The index records which recipe produced its face vectors. Changing it re-indexes only the
images that contain faces, so the upgrade does not cost a full rebuild.

The benchmark is a pose stress test on public CC BY 4.0 footage, not LFW. It has two
labelled identities for genuine pairs and uses "two faces in one frame are two people" for
impostors, so its absolute numbers are optimistic; the comparison between models is the point.
