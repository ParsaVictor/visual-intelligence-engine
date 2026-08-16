# Design decisions

The reasoning behind choices that are not obvious from reading the code, and the
ones where the obvious option was deliberately rejected.

---

## Contrastive prompts instead of a similarity threshold

Three of the four search paths ask CLIP the same question, and it is not "how
similar is this crop to the query". It is:

```python
prompts = [
    "a photo of a red panda",                       # the hypothesis
    "a photo of a different kind of animal",        # rival
    "a photo of scenery with no animal in it",      # rival
]
score = softmax(logits)[0]
```

**Why this beats thresholding a raw cosine.** A cosine similarity has no null
hypothesis. Every crop produces *some* number, and there is no principled place
to put the cut — the value depends on the query's wording, the crop's framing and
the domain. A softmax over rivals gives the model an explicit "none of the above"
option, so the output is a posterior over a closed set of explanations and a
fixed threshold means the same thing across queries.

The difference is visible in this project's own demo output. The food path, which
uses rivals, reports 96–99% for correct hits. The free-text path, which uses raw
cosine, reports **23.30%** for *its* best hit — a perfectly healthy CLIP score
that reads like failure. Same model, same image space; only the decision rule
differs.

This is also why the animal path was rebuilt. It previously used an ImageNet-1k
classifier with no confidence floor whatsoever, so a 1000-way softmax always
returned something, and a 21.5% top-1 shipped as a confirmed match with a green
box drawn on it.

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

## No approximate index (yet)

The architecture diagram claimed FAISS. The code did a plain NumPy loop, and
still does.

**This is the correct choice at current scale, not an omission.** Measured on the
demo archive, a full scan of every indexed face answers in **0.0025 s**. FAISS
would add a dependency, an index build step, a tuning surface and approximate
recall — to optimise something already three orders of magnitude below human
perception.

The crossover is around **10⁵ vectors**. Below it, a contiguous float32 matrix
multiply beats a graph traversal because it is one BLAS call with perfect cache
behaviour. Above it, memory bandwidth dominates and an approximate index wins.

Tracked as a roadmap item with that trigger recorded, so the decision is
revisited when the data justifies it rather than when it would look impressive.

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
