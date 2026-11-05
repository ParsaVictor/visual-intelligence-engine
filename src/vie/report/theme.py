"""The visual language shared by every report and the demo.

Near-black ink with one warm signal colour, and hairlines rather than boxes, so
the photographs stay the brightest thing on the page. Numbers use tabular figures
so a column of scores lines up. Persian text falls back to whichever good Arabic
script face the system has, because a report must not fetch a web font.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

CSS = """
:root {
  color-scheme: dark;
  --ink: #0b0d10; --ink-2: #12151a; --ink-3: #1a1e25; --hair: #262b33;
  --paper: #eef0f2; --muted: #98a0ab; --faint: #626a75;
  --signal: #ffb454; --signal-ink: #1c1405;
  --lane-1: #ffb454; --lane-2: #6cc4ff; --lane-3: #8fe388; --lane-4: #e58cff; --lane-5: #ff8a7a;
  --radius: 10px;
  --sans: "Inter", "Segoe UI Variable", "Segoe UI", "Vazirmatn", "IRANSans", "Tahoma",
          system-ui, -apple-system, sans-serif;
  --mono: "JetBrains Mono", "Cascadia Code", "SF Mono", Consolas, ui-monospace, monospace;
}
@media (prefers-color-scheme: light) {
  :root {
    color-scheme: light;
    --ink: #f6f6f3; --ink-2: #ffffff; --ink-3: #eeeeea; --hair: #dcdcd6;
    --paper: #15171a; --muted: #5d636b; --faint: #8a9099;
    --signal: #c2650a; --signal-ink: #fff8f0;
    --lane-1: #c2650a; --lane-2: #0a78b8; --lane-3: #2f8a3a; --lane-4: #9a3fb8; --lane-5: #c4442f;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--ink); color: var(--paper);
  font: 15px/1.6 var(--sans); font-feature-settings: "tnum" 1;
}
.page { max-width: 1180px; margin: 0 auto; padding: 56px 28px 96px; }
.eyebrow {
  font: 500 12px/1 var(--mono); letter-spacing: .14em; text-transform: uppercase;
  color: var(--signal); margin: 0 0 18px;
}
h1 {
  font-size: clamp(28px, 4.4vw, 46px); line-height: 1.12; letter-spacing: -.022em;
  font-weight: 650; margin: 0 0 14px; overflow-wrap: anywhere;
}
h2 {
  font-size: 13px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); margin: 56px 0 18px; padding-top: 18px; border-top: 1px solid var(--hair);
}
.lede { color: var(--muted); font-size: 17px; max-width: 72ch; margin: 0; }
.meta { font: 12.5px/1.6 var(--mono); color: var(--faint); margin-top: 16px; overflow-wrap: anywhere; }
.footer {
  margin-top: 72px; padding-top: 18px; border-top: 1px solid var(--hair);
  font: 12px/1.6 var(--mono); color: var(--faint);
}
.stats { display: flex; flex-wrap: wrap; gap: 14px 44px; margin-top: 30px; }
.stat b { display: block; font-size: 30px; line-height: 1.1; font-weight: 620; letter-spacing: -.02em; }
.stat span { font-size: 12.5px; color: var(--muted); }
.empty {
  padding: 56px 24px; text-align: center; color: var(--muted);
  border: 1px dashed var(--hair); border-radius: var(--radius); margin-top: 36px;
}

/* search results */
.grid {
  display: grid; gap: 22px; margin-top: 40px;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr));
}
.card { min-width: 0; }
.frame {
  position: relative; border-radius: var(--radius); overflow: hidden; background: var(--ink-3);
  line-height: 0;
}
.frame img { width: 100%; height: auto; display: block; }
.box {
  position: absolute; border: 2px solid var(--signal); border-radius: 3px;
  box-shadow: 0 0 0 1px rgba(0,0,0,.55);
}
.rank {
  position: absolute; top: 10px; left: 10px; padding: 3px 8px; border-radius: 5px;
  background: rgba(8,10,12,.78); color: #fff; font: 600 12px/1.4 var(--mono);
}
.card .name { font-size: 13px; margin-top: 11px; overflow-wrap: anywhere; }
.card .label { font-size: 12.5px; color: var(--muted); overflow-wrap: anywhere; }
.scoreline { display: flex; align-items: center; gap: 10px; margin-top: 7px; }
.scoreline b { font: 600 13px/1 var(--mono); min-width: 52px; }
.track { flex: 1; height: 3px; background: var(--hair); border-radius: 2px; overflow: hidden; }
.track i { display: block; height: 100%; background: var(--signal); }
"""


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def thumbnail_uri(path: str | Path, max_side: int = 640, quality: int = 82) -> str | None:
    """Embed a file as a JPEG data URI, or ``None`` if it cannot be read."""
    import io

    from PIL import Image, ImageOps

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
    except OSError:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def original_size(path: str | Path) -> tuple[int, int] | None:
    """Displayed width and height, honouring EXIF rotation."""
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).size
    except OSError:
        return None


def document(title: str, body: str, extra_css: str = "", script: str = "") -> str:
    """Wrap a body in the shared page shell."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}{extra_css}</style></head>"
        f'<body><main class="page">{body}</main>'
        + (f"<script>{script}</script>" if script else "")
        + "</body></html>"
    )
