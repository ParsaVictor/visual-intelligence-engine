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

`tests/test_integration.py` runs the real wiring with fake weights in seconds on every
push. `.github/workflows/models.yml` runs real weights on real images and real footage.
Both exist because the fast suite alone was not enough.
