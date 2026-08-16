# Architecture

Two phases: heavy models run **once per image at index time**, queries run only
over what the index already knows.

> **On the original diagram.** `docs/architecture-original.png` is kept for
> history, but it claimed four things the code never did: a FAISS index (there
> was a plain NumPy loop, and no faiss import anywhere), a "120-species
> dictionary" (there were 8 keyword buckets — the number 120 came from a comment
> about ImageNet's 120 *dog breeds*), MobileNetV3 running at index time (it ran
> only at query time), and stored MegaDetector crops (nothing was stored). FP16
> was also listed as a global optimisation when only CLIP was ever half
> precision. The diagrams below describe what the code actually does.

## Indexing

```mermaid
flowchart TD
    A[gallery image] --> B{changed since last run?<br/>SHA-256 content hash}
    B -- no --> Z[skip]
    B -- yes --> C[decode + cap longest side<br/>threaded prefetch]

    C --> D[1 · InsightFace buffalo_l<br/>RetinaFace + ArcFace]
    D --> E[2 · MegaDetector v5a<br/>class-agnostic animal boxes]
    E --> F[3 · RT-DETR-X<br/>food and tableware boxes]
    F --> G{box overlaps a face?}
    G -- yes --> H[reject]
    G -- no --> I[keep]

    D --> J[(face_embeddings<br/>512-d, float32)]
    E --> K[(animal_boxes<br/>box + confidence)]
    I --> L[(gallery_meta<br/>has_face / has_animal / has_food)]
    C --> M[CLIP ViT-L/14 image encoder]
    M --> N[(clip_embeddings<br/>768-d, float32)]
```

**Stage order is load-bearing.** Faces must be detected before food, because the
food stage rejects boxes that sit on a face. The original ran animal → food →
face, so that guard read variables that had not been assigned yet and was always
false. `vie.indexing.STAGE_ORDER` is asserted by test.

## Querying

```mermaid
flowchart LR
    Q[query] --> R{kind}

    R -- face photo --> F1[ArcFace embedding of<br/>the largest face]
    F1 --> F2[scan face_embeddings<br/>dot product = cosine]
    F2 --> F3[threshold 0.46 · one result per image]

    R -- species --> A1[SQL: has_animal = 1]
    A1 --> A2[load stored boxes<br/>no detector re-run]
    A2 --> A3[CLIP vs contrastive prompts]

    R -- dish --> D1[SQL: has_food = 1]
    D1 --> D2[RT-DETR crops + padding]
    D2 --> D3[CLIP vs contrastive prompts]

    R -- free text --> T1[CLIP text encoder]
    T1 --> T2[scan clip_embeddings<br/>cosine · top-k]

    F3 --> OUT[ranked results]
    A3 --> OUT
    D3 --> OUT
    T2 --> OUT
```

## The prefilter

The single most important design decision. Each image carries three booleans, so
a species query never touches an image with no animal in it:

```sql
SELECT file_name, file_path FROM gallery_meta WHERE has_animal = 1;
```

The flags come from *class-agnostic* detectors, which is what makes them durable:
MegaDetector answers "is there an animal", not "which species". Swapping the
recogniser — as this project did, from MobileNetV3 to CLIP — requires no
re-indexing at all.

## Models

| Model | Stage | Precision | Output |
|:--|:--|:--|:--|
| InsightFace `buffalo_l` | index + face query | FP32 (ONNX) | 512-d unit vector |
| MegaDetector v5a | index | FP32 | boxes, class 0 = animal |
| RT-DETR-X | index + food query | FP32 | boxes, COCO classes |
| CLIP ViT-L/14 | index + 3 query paths | FP16 on CUDA | 768-d unit vector |

None are fine-tuned. The engineering is in the combination, the index and the
decision rules — not in training.

## Schema

```sql
gallery_meta(file_name PK, file_path, content_hash, file_size,
             has_face, has_animal, has_food)
face_embeddings(file_name, bbox, embedding, PRIMARY KEY (file_name, bbox))
clip_embeddings(file_name PK, embedding)
animal_boxes(file_name, bbox, confidence, PRIMARY KEY (file_name, bbox))
```

`face_embeddings` originally had no primary key, so every re-index duplicated
every face. `(file_name, bbox)` is the natural key: idempotent per face, while
still keeping two distinct faces in one photo.
