"""Benchmark harness.

Reports hardware, dataset size and configuration alongside every measurement,
because a latency figure without those three is not a result.

    python scripts/benchmark.py                    # scale curve only, no index needed
    python scripts/benchmark.py --index            # also profile the configured index
    python scripts/benchmark.py --encoder          # also time the real encoder on CPU/GPU

Latency is reported as median and p95 over repeated runs, not as a mean: the first
query pays for lazy initialisation and page-cache misses, and a mean lets that one
outlier misrepresent every subsequent query.

The scale curve runs through :class:`vie.vectorstore.VectorCatalog`, the same class
the search paths use, on a real SQLite table. Earlier versions timed a bare matrix
multiply while the product ran a Python loop, so the number did not describe the code.
Vectors are random unit vectors, so this measures search mechanics, not retrieval
quality; recall here means overlap with the exact answer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vie.config import Config
from vie.database import Index
from vie.embeddings import to_blob
from vie.logging_setup import configure, get_logger
from vie.vectorstore import ExactBackend, HnswBackend, SearchConfig, VectorCatalog, faiss_available

log = get_logger("benchmark")


@dataclass
class Environment:
    python: str
    platform: str
    cpu: str
    cpu_count: int | None
    torch_version: str | None = None
    cuda_device: str | None = None

    @classmethod
    def capture(cls) -> Environment:
        import os

        env = cls(
            python=platform.python_version(),
            platform=platform.platform(),
            cpu=platform.processor(),
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
    region_vectors: int
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
    recall_at_10: float | None = None

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
        )


@dataclass
class Report:
    environment: Environment
    index: IndexProfile | None = None
    timings: list[Timing] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        rows = "\n".join(
            f"| {t.label} | {t.median_ms:.2f} | {t.p95_ms:.2f} | "
            f"{'-' if t.recall_at_10 is None else f'{t.recall_at_10:.3f}'} |"
            for t in self.timings
        )
        env = self.environment
        gpu = env.cuda_device or "no GPU"
        profile = ""
        if self.index:
            profile = (
                f"**Index:** {self.index.images} images · {self.index.faces} faces · "
                f"{self.index.region_vectors} region vectors · "
                f"{self.index.database_bytes / 1_048_576:.1f} MB\n\n"
            )
        return (
            f"**Hardware:** {env.cpu or 'unknown CPU'} · {env.cpu_count} threads · {gpu} · "
            f"Python {env.python}\n\n{profile}"
            f"| Operation | Median (ms) | p95 (ms) | Recall@10 |\n"
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
            region_vectors=count("SELECT COUNT(*) FROM clip_embeddings"),
            animal_boxes=count("SELECT COUNT(*) FROM animal_boxes"),
            with_face=count("SELECT COUNT(*) FROM gallery_meta WHERE has_face = 1"),
            with_animal=count("SELECT COUNT(*) FROM gallery_meta WHERE has_animal = 1"),
            with_food=count("SELECT COUNT(*) FROM gallery_meta WHERE has_food = 1"),
            database_bytes=Path(config.database_path).stat().st_size,
        )


def benchmark_index(config: Config) -> list[Timing]:
    """The real retrieval path over the real index: no model inference involved."""
    index = Index(config.database_path)
    timings: list[Timing] = []
    with index.connect() as conn:
        for table, label in (("face_embeddings", "face scan"), ("clip_embeddings", "text scan")):
            catalog = VectorCatalog.open(conn, table, config.search)
            if not len(catalog):
                continue
            probe = np.random.default_rng(0).normal(size=catalog.dim).astype(np.float32)
            probe /= np.linalg.norm(probe)
            timings.append(
                Timing.measure(f"{label} ({len(catalog):,} vectors)",
                               lambda c=catalog, p=probe: c.search(p, 50))
            )
    return timings


def _unit_vectors(count: int, dim: int, seed: int = 0) -> np.ndarray:
    # Clustered rather than uniform: uniform random vectors in high dimensions are
    # nearly equidistant, which is the worst case for HNSW and unlike any embedding.
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(max(8, count // 200), dim)).astype(np.float32)
    vectors = centres[rng.integers(0, len(centres), count)]
    vectors = vectors + rng.normal(scale=0.5, size=(count, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def benchmark_scale(sizes: list[int], dim: int, queries: int = 50) -> list[Timing]:
    """Exact against HNSW at each size, with recall measured against the exact answer."""
    timings: list[Timing] = []
    settings = SearchConfig(backend="exact")
    for size in sizes:
        vectors = _unit_vectors(size, dim)
        probes = _unit_vectors(queries, dim, seed=1)

        # Through SQLite, so the load cost is the real one.
        with tempfile.TemporaryDirectory() as folder:
            index = Index(Path(folder) / "scale.db")
            index.initialise()
            with index.connect() as conn:
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.executemany(
                    "INSERT INTO clip_embeddings (file_name, region, embedding) "
                    "VALUES (?, 'whole', ?)",
                    ((str(i), to_blob(v)) for i, v in enumerate(vectors)),
                )
                conn.commit()
                start = time.perf_counter()
                catalog = VectorCatalog.open(conn, "clip_embeddings", settings)
                load_ms = (time.perf_counter() - start) * 1000
                db_mb = Path(folder, "scale.db").stat().st_size / 1_048_576
        log.info("%d vectors: load %.0f ms, %.0f MB on disk", size, load_ms, db_mb)
        timings.append(
            Timing(f"load {size:,} {dim}-d vectors from SQLite", 1, round(load_ms, 1),
                   round(load_ms, 1))
        )

        exact = ExactBackend(vectors)
        next_probe = itertools.cycle(probes).__next__

        timings.append(
            Timing.measure(f"exact search, {size:,} vectors",
                           lambda b=catalog.backend, p=next_probe: b.search(p(), 10), runs=30)
        )

        if faiss_available():
            ann_settings = SearchConfig(backend="hnsw")
            start = time.perf_counter()
            hnsw = HnswBackend.build(vectors, ann_settings)
            build_s = time.perf_counter() - start
            log.info("%d vectors: HNSW build %.1f s", size, build_s)
            truth = [set(exact.search(p, 10)[0].tolist()) for p in probes]
            found = [set(hnsw.search(p, 10)[0].tolist()) for p in probes]
            recall = float(np.mean(
                [len(t & f) / 10 for t, f in zip(truth, found, strict=True)]
            ))
            timing = Timing.measure(
                f"HNSW search, {size:,} vectors (build {build_s:.1f} s)",
                lambda h=hnsw, p=next_probe: h.search(p(), 10), runs=30,
            )
            timing.recall_at_10 = round(recall, 4)
            timings.append(timing)
    return timings


def benchmark_encoder(config: Config) -> list[Timing]:
    """Real encoder latency on whatever device is present."""
    from PIL import Image

    from vie.models import VisionLanguageEncoder, load_all

    encoder = VisionLanguageEncoder(load_all(config))
    picture = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    name = config.models["vision_language"]
    return [
        Timing.measure(f"{name}: encode 1 image", lambda: encoder.encode_images([picture]),
                       runs=5, warmup=1),
        Timing.measure(f"{name}: encode 8 images, one batch",
                       lambda: encoder.encode_images([picture] * 8, batch_size=8),
                       runs=3, warmup=1),
        Timing.measure(f"{name}: encode 8 images, one by one",
                       lambda: encoder.encode_images([picture] * 8, batch_size=1),
                       runs=3, warmup=0),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the index and retrieval paths")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--dim", type=int, default=1152, help="vector size; so400m produces 1152")
    parser.add_argument("--index", action="store_true", help="profile the configured index")
    parser.add_argument("--encoder", action="store_true", help="time the real encoder")
    args = parser.parse_args(argv)

    configure("INFO")
    config = Config.load(args.config)
    report = Report(environment=Environment.capture())

    if args.index:
        if not Path(config.database_path).exists():
            log.error("no index at %s — run 'vie index' first", config.database_path)
            return 2
        report.index = profile_index(config)
        report.timings += benchmark_index(config)

    report.timings += benchmark_scale([int(s) for s in args.sizes.split(",")], args.dim)
    if args.encoder:
        report.timings += benchmark_encoder(config)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.to_json(), encoding="utf-8")
    print(report.to_markdown())
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
