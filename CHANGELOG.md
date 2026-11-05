# Changelog

Each version is a git tag. Check one out to get that state of the project.

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
