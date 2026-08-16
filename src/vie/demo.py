"""Gradio demo exposing all four search paths.

    python -m vie.demo

Loads the model stack once and holds it, so only the first query pays for it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vie.config import Config
from vie.logging_setup import configure, get_logger
from vie.scoring import Match

log = get_logger("demo")

DESCRIPTION = """
# Visual Intelligence Engine

Search an image archive by face, species, dish, or plain description.
Run `vie index` first to build the index.
"""


class _Request:
    """Mimics the argparse namespace `run_search` expects."""

    def __init__(self, kind: str, query: str = "", image: Path | None = None,
                 top_k: int | None = None) -> None:
        self.kind = kind
        self.query = query
        self.image = image
        self.top_k = top_k


def _gallery_items(matches: list[Match]) -> list[tuple[str, str]]:
    return [
        (m.file_path, f"{m.file_name} — {m.score * 100:.1f}%")
        for m in matches
        if Path(m.file_path).is_file()
    ]


def build_app(config: Config) -> Any:
    import gradio as gr

    state: dict[str, Any] = {}

    def runtime(cfg: Config):
        if "bundle" not in state:
            from vie.models import ClipCropScorer, ClipEmbedder, load_all

            bundle = load_all(cfg)
            state["bundle"] = (bundle, ClipCropScorer(bundle), ClipEmbedder(bundle))
        return state["bundle"]

    def search(kind: str, query: str, image_path: str | None):
        from vie.query import run_search

        try:
            request = _Request(
                kind,
                query=query or "",
                image=Path(image_path) if image_path else None,
                top_k=None,
            )
            matches = run_search(request, config, lambda c: runtime(c))
        except (FileNotFoundError, ValueError) as exc:
            return [], f"⚠️ {exc}"
        if not matches:
            return [], "No matches."
        return _gallery_items(matches), f"{len(matches)} match(es)."

    with gr.Blocks(title="Visual Intelligence Engine") as app:
        gr.Markdown(DESCRIPTION)

        with gr.Tab("👤 Face"):
            face_in = gr.Image(type="filepath", label="Photo of the person")
            face_btn = gr.Button("Search", variant="primary")
            face_out = gr.Gallery(label="Results", columns=4, height=420)
            face_msg = gr.Markdown()
            face_btn.click(
                lambda p: search("face", "", p), [face_in], [face_out, face_msg]
            )

        for tab, kind, placeholder in (
            ("🦁 Species", "animal", "red panda"),
            ("🍱 Food", "food", "French fries"),
            ("🌐 Free text", "text", "a man with a blue backpack"),
        ):
            with gr.Tab(tab):
                box = gr.Textbox(label="Query", placeholder=placeholder)
                btn = gr.Button("Search", variant="primary")
                out = gr.Gallery(label="Results", columns=4, height=420)
                msg = gr.Markdown()
                btn.click(
                    lambda q, k=kind: search(k, q, None), [box], [out, msg]
                )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the Gradio demo")
    parser.add_argument("--config", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    configure("INFO")
    config = Config.load(args.config)
    if not Path(config.database_path).exists():
        log.error("no index at %s — run 'vie index' first", config.database_path)
        return 2

    build_app(config).launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
