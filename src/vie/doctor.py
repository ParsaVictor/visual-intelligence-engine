"""``vie doctor``: can this machine run this, and is the index healthy?

Every failure this project has shipped was silent: a score on the wrong scale, an
upgrade nothing called, an index that no longer matched its models. This command
makes that state visible without loading a single model.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from vie import __version__
from vie.config import Config
from vie.database import SCHEMA_VERSION, Index

#: (import name, what is missing without it)
OPTIONAL_MODULES = (
    ("torch", "every model"),
    ("transformers", "free-text, species fallback and food search"),
    ("insightface", "face search"),
    ("onnxruntime", "face search"),
    ("ultralytics", "food detection"),
    ("cv2", "image decoding"),
    ("faiss", "the ANN backend (exact search still works)"),
    ("gradio", "the interactive demo"),
)


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)


def _check_environment(report: Report, config: Config) -> None:
    report.add("vie", "ok", __version__)
    report.add("python", "ok", platform.python_version())
    for module, purpose in OPTIONAL_MODULES:
        found = importlib.util.find_spec(module) is not None
        report.add(module, "ok" if found else "warn", "" if found else f"needed for {purpose}")

    if importlib.util.find_spec("torch") is not None:
        import torch

        gpu = torch.cuda.is_available()
        report.add(
            "device",
            "ok" if gpu else "warn",
            torch.cuda.get_device_name(0) if gpu else "CPU only: indexing will be slow",
        )

    free = shutil.disk_usage(Path(".").resolve()).free / 1e9
    report.add("disk", "ok" if free > 15 else "warn", f"{free:.0f} GB free")


def _check_index(report: Report, config: Config) -> None:
    gallery = Path(config.gallery_path)
    report.add(
        "gallery",
        "ok" if gallery.is_dir() else "warn",
        str(gallery) if gallery.is_dir() else f"{gallery} does not exist yet",
    )

    path = Path(config.database_path)
    if not path.exists():
        report.add("index", "warn", f"no index at {path}: run 'vie index'")
        return

    index = Index(path)
    with index.connect() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "schema_meta" not in tables:
            report.add("index", "fail", f"{path} is not a vie index")
            return
        version = index.schema_version(conn)
        if version > SCHEMA_VERSION:
            report.add("schema", "fail", f"v{version} is newer than this vie (v{SCHEMA_VERSION})")
            return
        report.add("schema", "ok" if version == SCHEMA_VERSION else "warn",
                   f"v{version}" + ("" if version == SCHEMA_VERSION else
                                    f": upgraded to v{SCHEMA_VERSION} on next use"))
        if version != SCHEMA_VERSION:
            return

        def count(sql: str) -> int:
            return int(conn.execute(sql).fetchone()[0])

        images = count("SELECT COUNT(*) FROM gallery_meta")
        report.add("images", "ok" if images else "warn", str(images))

        built_with = index.get_meta(conn, "vision_language")
        selected = config.models["vision_language"]
        if built_with is None:
            report.add("embedding model", "warn", "not recorded: built by an older version")
        else:
            report.add("embedding model", "ok" if built_with == selected else "fail",
                       built_with if built_with == selected else
                       f"index has {built_with}, config selects {selected}: "
                       f"run 'vie index --rebuild'")

        # Rows that older versions could leave behind.
        unembedded = count(
            "SELECT COUNT(*) FROM gallery_meta g WHERE NOT EXISTS "
            "(SELECT 1 FROM clip_embeddings c WHERE c.file_name = g.file_name)"
        )
        report.add("images without vectors", "ok" if not unembedded else "fail", str(unembedded))
        report.add("faces", "ok", str(count("SELECT COUNT(*) FROM face_embeddings")))
        report.add("region vectors", "ok", str(count("SELECT COUNT(*) FROM clip_embeddings")))

        with_animals = count("SELECT COUNT(*) FROM gallery_meta WHERE has_animal = 1")
        predictions = count("SELECT COUNT(*) FROM animal_predictions")
        report.add(
            "species predictions",
            "ok" if predictions or not with_animals else "warn",
            f"{predictions} for {with_animals} images with animals"
            + ("" if predictions or not with_animals else
               ": every species query will run a model; re-index to fix"),
        )
        report.add("food gates", "ok",
                   f"detector {count('SELECT COUNT(*) FROM gallery_meta WHERE food_by_detector=1')}"
                   f", vision-language "
                   f"{count('SELECT COUNT(*) FROM gallery_meta WHERE food_by_vlm=1')}")


def collect(config: Config) -> Report:
    report = Report()
    _check_environment(report, config)
    _check_index(report, config)
    return report


def run_doctor(config: Config, as_json: bool = False) -> int:
    report = collect(config)
    if as_json:
        print(json.dumps([vars(c) for c in report.checks], indent=2, ensure_ascii=False))
        return 1 if report.failed else 0

    from rich.console import Console
    from rich.table import Table

    marks = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]", "fail": "[red]fail[/red]"}
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_column(overflow="fold")
    for check in report.checks:
        table.add_row(check.name, marks[check.status], check.detail)
    Console().print(table)
    return 1 if report.failed else 0
