"""Download the small labelled set ``scripts/calibrate.py`` measures on.

    python scripts/fetch_calibration_data.py benchmarks/calibration-data

Public datasets, through the Hugging Face dataset viewer API, so nothing but a few
dozen JPEGs is fetched:

* ``ethz/food101``: real dishes, used as food positives;
* ``microsoft/cats_vs_dogs``: animal positives;
* ``timm/mini-imagenet``: artefact classes only, used as things that are neither.

The set is deliberately small and easy. It is enough to choose between scoring rules
and to catch a threshold on the wrong scale, not to quote an accuracy figure.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"


def rows(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    with urllib.request.urlopen(f"{API}?{query}", timeout=90) as response:  # noqa: S310
        return json.load(response)


def harvest(
    out: Path, kind: str, dataset: str, config: str, split: str, column: str,
    offsets: list[int], length: int, per_label: int, accept=lambda label: True,
) -> None:
    seen: dict[str, int] = {}
    for offset in offsets:
        try:
            data = rows(dataset, config, split, offset, length)
        except OSError as exc:
            print(f"skipping {dataset} at {offset}: {exc}")
            continue
        names = next(f["type"]["names"] for f in data["features"] if f["name"] == column)
        for row in data["rows"]:
            label = names[row["row"][column]]
            if not accept(label) or seen.get(label, 0) >= per_label:
                continue
            target = out / kind / label / f"{seen.get(label, 0)}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            source = row["row"]["image"]["src"]
            with urllib.request.urlopen(source, timeout=90) as response:  # noqa: S310
                target.write_bytes(response.read())
            seen[label] = seen.get(label, 0) + 1
    print(f"{kind}: {dataset}: {len(seen)} labels, {sum(seen.values())} images")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-label", type=int, default=3,
                        help="images per dish; animals get this times 8/3 (search eval: 6)")
    args = parser.parse_args()

    # Food-101 is ordered by class, 250 validation images each.
    harvest(args.out, "food", "ethz/food101", "default", "validation", "label",
            [i * 250 for i in range(0, 101, 9)], 100, args.per_label)
    harvest(args.out, "animal", "microsoft/cats_vs_dogs", "default", "train", "labels",
            [0, 12000], 100, max(8, args.per_label * 8 // 3))
    # WordNet ids: n03 and n04 are artefacts, n01 and n02 mostly animals.
    harvest(args.out, "other", "timm/mini-imagenet", "default", "validation", "label",
            list(range(0, 11000, 500)), 20, 2, lambda label: label.startswith(("n03", "n04")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
