# Security and privacy

## This project processes biometric data

It detects faces and stores 512-dimensional ArcFace embeddings — biometric
identifiers of identifiable people. Treat the index the same way you would treat
the photographs themselves.

**Never commit:**

| Artefact | Why |
|:--|:--|
| `data/gallery/` | source photographs of identifiable people |
| `*.db` / `*.sqlite` | contains face embeddings — biometric identifiers |
| `*.pkl`, `*.npy` | cached embeddings |
| model weights | large, and redistributable only under their own licences |

`.gitignore` blocks all of these. Verify with `git status` after any pipeline
run — a clean tree is the check.

## Before deploying

- **Lawful basis.** Under GDPR, biometric data used to uniquely identify a person
  is a special category (Art. 9) and needs an explicit basis. Equivalent rules
  exist in many jurisdictions. Establish yours before indexing anyone.
- **Data minimisation.** Index only what you need, and delete embeddings when the
  underlying images are deleted — `vie index` prunes rows for files removed from
  the gallery, but only when it runs.
- **Access control.** The index is a plain SQLite file. Anyone who can read it can
  run face searches. Protect it accordingly.
- **Retention.** Decide how long embeddings live and enforce it.

## Supply chain

- The yolov5 revision loaded through `torch.hub` is **pinned** (`YOLOV5_PIN`).
  It was previously cloned from an unpinned master with `trust_repo=True`, so
  upstream changes could alter detection behaviour silently.
- Embeddings are stored as raw float32, **not pickle**. `pickle.loads` on an
  index file you did not create is arbitrary code execution, and index files get
  copied between machines.
- Model weights are downloaded from their official release URLs at first run.

## Reporting a vulnerability

Open a private security advisory through GitHub, or email
`1.parsa.karkooti@gmail.com`. Please do not open a public issue for anything
that exposes data.
