# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` deliberately excludes torch. The core logic has no framework dependency,
so the whole suite installs and runs in seconds. Add `[models]` only when you
need to touch the model adapters themselves.

## Before opening a pull request

```bash
ruff check src tests
pytest
```

CI runs exactly those two commands on Python 3.10 and 3.12.

## The rule that matters most

**Every fixed defect gets a test that fails without the fix.**

This repository was reconstructed from notebooks that had shipped several
silent, wrong-answer bugs — a species map that classified elephants as insects,
a cross-verification filter that never executed, a search pass reading different
constants than the indexer. None of them announced themselves; they just
returned plausible nonsense.

So when you fix something, write the test first and watch it fail. Tests that
reference a `P1-*` or `P2-*` id exist to pin a specific historical defect, and
they carry a comment explaining what went wrong. Keep that habit.

## Design constraints

- **Model access goes behind a protocol.** Search and pipeline code depends on
  `FaceDetector`, `BoxDetector` or `CropScorer` — never on torch directly.
  Everything model-specific lives in `vie/adapters.py`. This is what keeps the
  decision rules testable without a GPU.
- **New behaviour goes in `configs/default.yaml`**, not into a new CLI flag, and
  if it has an invariant, encode it in `Config.validate()`.
- **No silent degradation.** A bare `except` that swallows a failure is how the
  original could return empty results forever without saying why. Log it or
  raise it.
- **Recognition uses contrastive prompts.** If you add a path that asks CLIP a
  question, give it rival prompts and threshold the softmax — do not threshold a
  raw cosine. See `docs/design-decisions.md`.

## Commit messages

Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`,
`chore:`. Explain *why* in the body, especially for a correctness fix — what
went wrong is more useful to the next reader than what changed.

## Reporting a defect

Include the config you ran, the query, the full traceback, and your Python and
torch versions. If it is a wrong-answer bug rather than a crash, include the
image characteristics — that class of bug is usually about framing, scale or
lighting rather than code.
