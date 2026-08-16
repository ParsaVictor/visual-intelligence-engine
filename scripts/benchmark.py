"""Benchmark harness.

Produces the numbers the README deliberately does not claim yet. Reports
hardware, dataset size and configuration alongside every measurement, because a
latency figure without those three is not a result.

    python scripts/benchmark.py --queries benchmarks/queries.json

Latency is reported as median and p95 over repeated runs, not as a mean: the
first query pays for lazy CUDA initialisation and page cache misses, and a mean
lets that one outlier misrepresent every subsequent query.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.database import Index
from vie.logging_setup import configure, get_logger

log = get_logger("benchmark")


@dataclass
class Environment:
    python: str
    platform: str
    cpu_count: int | None
    torch_version: str | None = None
    cuda_device: str | None = None

    @classmethod
    def capture(cls) -> Environment:
        import os

        env = cls(
            python=platform.python_version(),
            platform=platform.platform(),
            cpu_count=os.cpu_count(),
        )
        try:
            import torch

            env.torch_version = torch.__version__
            if torch.cuda.is_available():
                env.cuda_device = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return env


@dataclass
class IndexProfile:
    images: int
    faces: int
    clip_vectors: int
    animal_boxes: int
    with_face: int
    with_animal: int
    with_food: int
    database_bytes: int


@dataclass
class Timing:
    label: str
    runs: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def measure(cls, label: str, fn: Any, runs: int = 20, warmup: int = 3) -> Timing:
        for _ in range(warmup):
            fn()
        samples: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        samples.sort()
        return cls(
            label=label,
            runs=runs,
            median_ms=round(statistics.median(samples), 3),
            p95_ms=round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 3),
            min_ms=round(samples[0], 3),
            max_ms=round(samples[-1], 3),
        )


@dataclass
class Report:
    environment: Environment
    index: IndexProfile
    timings: list[Timing] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        rows = "\n".join(
            f"| {t.label} | {t.median_ms:.2f} | {t.p95_ms:.2f} | {t.runs} |"
            for t in self.timings
        )
        gpu = self.environment.cuda_device or "CPU only"
        return (
            f"### Benchmark\n\n"
            f"**Hardware:** {gpu} · {self.environment.cpu_count} CPU threads · "
            f"Python {self.environment.python}\n\n"
            f"**Index:** {self.index.images} images · {self.index.faces} faces · "
            f"{self.index.clip_vectors} CLIP vectors · "
            f"{self.index.database_bytes / 1_048_576:.1f} MB\n\n"
            f"| Operation | Median (ms) | p95 (ms) | Runs |\n"
            f"|:--|--:|--:|--:|\n{rows}\n"
        )


def profile_index(config: Config) -> IndexProfile:
    index = Index(config.database_path)
    with index.connect() as conn:
        def count(sql: str) -> int:
            return int(conn.execute(sql).fetchone()[0])

        return IndexProfile(
            images=count("SELECT COUNT(*) FROM gallery_meta"),
            faces=count("SELECT COUNT(*) FROM face_embeddings"),
            clip_vectors=count("SELECT COUNT(*) FROM clip_embeddings"),
            animal_boxes=count("SELECT COUNT(*) FROM animal_boxes"),
            with_face=count("SELECT COUNT(*) FROM gallery_meta WHERE has_face = 1"),
            with_animal=count("SELECT COUNT(*) FROM gallery_meta WHERE has_animal = 1"),
            with_food=count("SELECT COUNT(*) FROM gallery_meta WHERE has_food = 1"),
            database_bytes=Path(config.database_path).stat().st_size,
        )


def benchmark_retrieval(config: Config) -> list[Timing]:
    """Measure the pure retrieval paths — no model inference involved.

    These are the parts whose complexity the FAISS question turns on, so they
    are worth measuring separately from encoding cost.
    """
    import numpy as np

    from vie.search import face as face_search
    from vie.search import text as text_search

    index = Index(config.database_path)
    timings: list[Timing] = []

    with index.connect() as conn:
        faces = list(index.iter_faces(conn))
        clips = list(index.iter_clip(conn))

    if faces:
        query = faces[0][2]
        timings.append(
            Timing.measure(
                f"face scan ({len(faces)} vectors)",
                lambda: face_search.rank(query, faces, config),
            )
        )
    if clips:
        query = clips[0][1]
        timings.append(
            Timing.measure(
                f"text scan ({len(clips)} vectors)",
                lambda: text_search.rank(query, clips, config),
            )
        )

    # Synthetic scaling curve: shows where a brute-force scan stops being free.
    rng = np.random.default_rng(0)
    for size in (1_000, 10_000, 100_000):
        vectors = rng.normal(size=(size, 512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        probe = vectors[0]
        timings.append(
            Timing.measure(
                f"synthetic 512-d scan ({size:,} vectors)",
                lambda v=vectors, p=probe: np.dot(v, p),
                runs=10,
            )
        )
    return timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the index and retrieval paths")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results.json"))
    args = parser.parse_args(argv)

    configure("INFO")
    config = Config.load(args.config)
    if not Path(config.database_path).exists():
        log.error("no index at %s — run 'vie index' first", config.database_path)
        return 2

    report = Report(
        environment=Environment.capture(),
        index=profile_index(config),
        timings=benchmark_retrieval(config),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.to_json(), encoding="utf-8")
    print(report.to_markdown())
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
