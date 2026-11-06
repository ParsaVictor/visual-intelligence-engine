# Notebooks

Standalone, self-contained versions of each module. They duplicate logic that
also lives in `src/vie/` on purpose: these are meant to be read top to bottom
and run in Colab without installing the package, so each one carries its own
config, model loading and helpers.

**Start with `0_run_everything.ipynb`.** It installs the maintained package and runs the
whole system on your Google Drive with one *Run all*: incremental indexing (only new files
after the first run), all searches, video analysis, annotated output videos and
presentation cards. The others below are standalone, read-top-to-bottom explanations.

| Notebook | Covers |
|:--|:--|
| `0_run_everything.ipynb` | **The complete system in Colab**, on the current package |
| `1_face_recognition.ipynb` | InsightFace `buffalo_l`, 512-d ArcFace, indexing and search |
| `2_animal_recognition.ipynb` | MegaDetector + classifier + CLIP fallback, with a cell that **proves** the original species-mapping defect against the real ImageNet class list |
| `3_food_recognition.ipynb` | RT-DETR-X + CLIP contrastive prompts |
| `4_cli.ipynb` | Writes a runnable `vie_cli.py` and tests that its config guards reject the original bugs |
| `5_full_pipeline.ipynb` | All four search paths on one index |

Every notebook opens with a table of what was wrong in the original and what
changed, so the reasoning travels with the code.

These predate v0.3 and still carry the scoring and thresholds of that time, so do not
treat their numbers as current. **The package in `src/` is the maintained version** — it has the test suite and
CI. Treat these as documentation you can execute.
