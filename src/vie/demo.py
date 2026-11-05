"""Gradio demo: every search path.

    python -m vie.demo

Models load lazily and stay resident, so a text search does not pay for the face
model and only the first query of each kind is slow.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from vie import __version__
from vie.config import Config
from vie.logging_setup import configure, get_logger
from vie.scoring import Match

log = get_logger("demo")

CSS = """
:root, .dark {
  --vie-ink: #0b0d10; --vie-line: #262b33; --vie-signal: #ffb454;
  --vie-muted: #98a0ab;
}
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#vie-head { padding: 26px 2px 12px; }
#vie-head .eyebrow {
  font: 500 12px/1 ui-monospace, Consolas, monospace; letter-spacing: .14em;
  text-transform: uppercase; color: var(--vie-signal); margin: 0 0 12px;
}
#vie-head h1 { font-size: 34px; line-height: 1.15; letter-spacing: -.02em; margin: 0 0 8px; }
#vie-head p { color: var(--vie-muted); margin: 0; }
#vie-head [dir="rtl"] { margin-top: 4px; }
.vie-status { font: 13px/1.5 ui-monospace, Consolas, monospace; color: var(--vie-muted); }
button.primary { font-weight: 600 !important; }
"""

HEADER = f"""
<div id="vie-head">
  <p class="eyebrow">Visual Intelligence Engine {__version__}</p>
  <h1>Search images by what is in them.</h1>
  <p>Faces, species, dishes and free text.</p>
  <p dir="rtl">جستجو در عکس‌ها بر اساس محتوای آن‌ها.</p>
</div>
"""


class _Request:
    """Mimics the argparse namespace `run_search` expects."""

    def __init__(self, kind: str, query: str = "", image: Path | None = None,
                 top_k: int | None = None) -> None:
        self.kind = kind
        self.query = query
        self.image = image
        self.top_k = top_k


def gallery_items(matches: list[Match]) -> list[tuple[str, str]]:
    def caption(m: Match) -> str:
        return f"{m.display_score * 100:.1f}%  ·  {Path(m.file_name).name}"

    return [(m.file_path, caption(m)) for m in matches if Path(m.file_path).is_file()]


def _build_theme(gr: Any) -> Any:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.slate,
        # System fonts only: a web font would tell Google each time this page opened.
        font=["Segoe UI", "Tahoma", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "Consolas", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    )


def _styling_kwargs(target: Any, gr: Any) -> dict[str, Any]:
    """Gradio moved ``theme`` and ``css`` from ``Blocks`` to ``launch`` in version 6."""
    accepted = inspect.signature(target).parameters
    wanted = {"theme": _build_theme(gr), "css": CSS, "title": "Visual Intelligence Engine"}
    return {key: value for key, value in wanted.items() if key in accepted}


def build_app(config: Config) -> Any:
    import gradio as gr

    state: dict[str, Any] = {}

    def runtime(cfg: Config):
        if "bundle" not in state:
            from vie.models import VisionLanguageEncoder, load_all

            bundle = load_all(cfg)
            encoder = VisionLanguageEncoder(bundle)
            state["bundle"] = (bundle, encoder, encoder)
        return state["bundle"]

    def search(kind: str, query: str, image_path: str | None):
        from vie.database import SchemaError
        from vie.query import run_search

        try:
            request = _Request(
                kind, query=query or "", image=Path(image_path) if image_path else None
            )
            matches = run_search(request, config, runtime)
        except (FileNotFoundError, ValueError, SchemaError) as exc:
            return [], f"{exc}"
        if not matches:
            return [], "Nothing in the index cleared the threshold."
        noun = "image" if len(matches) == 1 else "images"
        return gallery_items(matches), f"{len(matches)} {noun}, best first."

    # Analytics off: this interface shows photographs of identifiable people.
    with gr.Blocks(analytics_enabled=False, **_styling_kwargs(gr.Blocks, gr)) as app:
        gr.HTML(HEADER)

        with gr.Tab("Free text"):
            _text_tab(gr, search, "text", "a man with a blue backpack")
        with gr.Tab("Face"):
            with gr.Row():
                with gr.Column(scale=1):
                    face_in = gr.Image(type="filepath", label="Photo of the person")
                    face_btn = gr.Button("Find this person", variant="primary")
                    face_msg = gr.Markdown(elem_classes="vie-status")
                with gr.Column(scale=3):
                    face_out = gr.Gallery(label="Results", columns=4, height=520)
            face_btn.click(lambda p: search("face", "", p), [face_in], [face_out, face_msg])
        with gr.Tab("Species"):
            _text_tab(gr, search, "animal", "red panda")
        with gr.Tab("Food"):
            _text_tab(gr, search, "food", "ghormeh sabzi")
    return app


def _text_tab(gr: Any, search: Any, kind: str, placeholder: str) -> None:
    with gr.Row():
        box = gr.Textbox(label="Query", placeholder=placeholder, scale=5, autofocus=True)
        button = gr.Button("Search", variant="primary", scale=1)
    message = gr.Markdown(elem_classes="vie-status")
    output = gr.Gallery(label="Results", columns=4, height=560)

    def run(query: str):
        return search(kind, query, None)

    button.click(run, [box], [output, message])
    box.submit(run, [box], [output, message])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the Gradio demo")
    parser.add_argument("--config", default=None)
    parser.add_argument("--share", action="store_true",
                        help="expose publicly via a Gradio link: the index holds biometric data")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    configure("INFO")
    config = Config.load(args.config)
    if not Path(config.database_path).exists():
        log.warning("no index at %s: image search is empty until 'vie index' runs",
                    config.database_path)

    import gradio as gr

    app = build_app(config)
    # Gradio refuses to serve files outside the working directory unless told to,
    # which broke every result as soon as the gallery lived anywhere else.
    allowed = [str(Path(config.gallery_path).resolve())]
    launch_args: dict[str, Any] = {
        "server_port": args.port,
        "share": args.share,
        "allowed_paths": allowed,
    }
    launch_args.update(_styling_kwargs(app.launch, gr))
    launch_args.pop("title", None)
    app.launch(**launch_args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
