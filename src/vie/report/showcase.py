"""Presentation-grade result images: one PNG per search, ready to put in a deck.

The HTML reports are for working with results. These are for showing them: a
fixed-size image that looks the same in a slide, a chat message or a README.
Same visual language as :mod:`vie.report.theme`: near-black ink, one warm signal
colour, hairlines, and the photographs as the brightest thing on the page.

Each tile is cropped around the detection rather than the centre of the photo,
so the face, animal or dish that matched is what the viewer sees first.

Only Latin text is drawn. Pillow does not shape Arabic script without libraqm,
and unshaped Persian is worse than none; the notebook prints Persian around it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vie import __version__
from vie.scoring import Match

INK = (11, 13, 16)
INK_2 = (18, 21, 26)
INK_3 = (26, 30, 37)
HAIR = (38, 43, 51)
PAPER = (238, 240, 242)
MUTED = (152, 160, 171)
FAINT = (98, 106, 117)
SIGNAL = (255, 180, 84)
SIGNAL_INK = (28, 20, 5)
LANES = [(255, 180, 84), (108, 196, 255), (143, 227, 136), (229, 140, 255), (255, 138, 122),
         (255, 222, 110), (120, 230, 214), (255, 160, 200)]

KIND_TITLES = {
    "face": "Face search",
    "animal": "Species search",
    "food": "Dish search",
    "text": "Free-text search",
    "video": "Video understanding",
}


def font(size: int, bold: bool = False) -> Any:
    """A real TrueType face wherever one exists; Pillow's default otherwise."""
    from PIL import ImageFont

    names = (["Inter-SemiBold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
              "Arial Bold.ttf", "arialbd.ttf", "segoeuib.ttf"] if bold else
             ["Inter-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
              "Arial.ttf", "arial.ttf", "segoeui.ttf"])
    candidates: list[Path] = []
    try:
        import matplotlib

        candidates += [Path(matplotlib.get_data_path()) / "fonts" / "ttf" / n for n in names]
    except Exception:  # noqa: BLE001 - matplotlib is optional here
        pass
    for root in ("/usr/share/fonts", "/usr/local/share/fonts", "C:/Windows/Fonts",
                 "/Library/Fonts", "/System/Library/Fonts"):
        base = Path(root)
        if base.is_dir():
            for name in names:
                candidates += list(base.rglob(name))[:1]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text_width(draw: Any, text: str, face: Any) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=face)
    return right - left


def _ellipsize(draw: Any, text: str, face: Any, width: int) -> str:
    if _text_width(draw, text, face) <= width:
        return text
    while text and _text_width(draw, text + "…", face) > width:
        text = text[:-1]
    return text + "…"


def open_rgb(path: str | Path, max_side: int = 1600) -> Any:
    """Open any image as RGB, honouring EXIF rotation, capped in size."""
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def focus_crop(image: Any, size: tuple[int, int], box: Sequence[int] | None,
               box_scale: float = 1.0) -> tuple[Any, tuple[int, int, int, int] | None]:
    """Cover-fit ``image`` into ``size``, centred on ``box`` when there is one.

    Returns the tile and the box in tile coordinates. The box is kept whole when it
    fits; a subject bigger than the tile is zoomed out to fit instead.
    """
    from PIL import Image

    tw, th = size
    w, h = image.size
    scale = max(tw / w, th / h)
    if box is not None:
        bx1, by1, bx2, by2 = (v * box_scale for v in box)
        bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
        # Leave room around the subject: it should fill about half the tile.
        scale = max(scale, min(tw / (bw * 2.2), th / (bh * 2.2)))
        scale = min(scale, max(tw / w, th / h) * 3.0)
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    else:
        cx, cy = w / 2, h / 2
    resized = image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                           Image.Resampling.LANCZOS)
    rw, rh = resized.size
    left = int(min(max(cx * scale - tw / 2, 0), max(rw - tw, 0)))
    top = int(min(max(cy * scale - th / 2, 0), max(rh - th, 0)))
    tile = Image.new("RGB", size, INK_3)
    tile.paste(resized.crop((left, top, left + tw, top + th)),
               ((tw - min(tw, rw)) // 2, (th - min(th, rh)) // 2))
    if box is None:
        return tile, None
    x1, y1, x2, y2 = (round(v * box_scale * scale) for v in box)
    return tile, (x1 - left, y1 - top, x2 - left, y2 - top)


def rounded(image: Any, radius: int) -> Any:
    from PIL import Image, ImageDraw

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius=radius, fill=255)
    out = Image.new("RGBA", image.size)
    out.paste(image, (0, 0), mask)
    return out


def draw_brackets(draw: Any, box: Sequence[int], colour: tuple[int, int, int],
                  width: int = 3, frac: float = 0.24) -> None:
    """Corner brackets plus a hairline: reads as a detection without hiding it."""
    x1, y1, x2, y2 = box
    draw.rectangle(box, outline=colour + (110,), width=1)
    arm = max(8, int(min(x2 - x1, y2 - y1) * frac))
    for (x, y, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        draw.line([(x, y), (x + dx * arm, y)], fill=colour, width=width)
        draw.line([(x, y), (x, y + dy * arm)], fill=colour, width=width)


def pill(draw: Any, xy: tuple[int, int], text: str, face: Any,
         fill: tuple[int, int, int], ink: tuple[int, int, int], pad: tuple[int, int] = (10, 5)
         ) -> tuple[int, int, int, int]:
    x, y = xy
    width = _text_width(draw, text, face)
    _, top, _, bottom = draw.textbbox((0, 0), text, font=face)
    rect = (x, y, x + width + pad[0] * 2, y + (bottom - top) + pad[1] * 2)
    draw.rounded_rectangle(rect, radius=(rect[3] - rect[1]) // 2, fill=fill)
    draw.text((x + pad[0], y + pad[1] - top), text, font=face, fill=ink)
    return rect


@dataclass
class CardStats:
    """Optional numbers for the header, all shown only when given."""

    search_ms: float | None = None
    indexed_images: int | None = None
    candidates: int | None = None


def render_search_card(
    matches: Sequence[Match],
    *,
    kind: str,
    query: str,
    query_images: Sequence[str | Path] = (),
    stats: CardStats | None = None,
    columns: int = 4,
    max_results: int = 8,
    width: int = 1600,
) -> Any:
    """Render a search as one image. Returns a Pillow image."""
    from PIL import Image, ImageDraw

    shown = list(matches[:max_results])
    margin, gap = 56, 22
    side = 300 if query_images else 0
    grid_w = width - margin * 2 - (side + gap if side else 0)
    columns = max(1, min(columns, len(shown) or 1))
    tile_w = (grid_w - gap * (columns - 1)) // columns
    tile_h = int(tile_w * 0.75)
    caption_h = 92
    rows = max(1, -(-len(shown) // columns))
    header_h = 240
    body_h = rows * (tile_h + caption_h) + (rows - 1) * gap if shown else 220
    if side:
        body_h = max(body_h, 60 + len(query_images[:3]) * (side + 16))
    height = header_h + body_h + 120

    canvas = Image.new("RGB", (width, height), INK)
    draw = ImageDraw.Draw(canvas, "RGBA")
    f_eyebrow, f_title = font(18, True), font(46, True)
    f_name, f_small, f_score = font(19, True), font(16), font(26, True)

    # ── header ────────────────────────────────────────────────────────
    draw.text((margin, 50), KIND_TITLES.get(kind, "Search").upper(), font=f_eyebrow, fill=SIGNAL)
    title = _ellipsize(draw, query, f_title, width - margin * 2)
    draw.text((margin, 80), title, font=f_title, fill=PAPER)
    chips: list[str] = [f"{len(matches)} match" + ("" if len(matches) == 1 else "es")]
    if matches:
        chips.append(f"best {max(m.display_score for m in matches) * 100:.1f}%")
    if stats and stats.search_ms is not None:
        chips.append(f"search {stats.search_ms:.0f} ms" if stats.search_ms >= 1
                     else f"search {stats.search_ms:.2f} ms")
    if stats and stats.indexed_images is not None:
        chips.append(f"{stats.indexed_images:,} images indexed")
    x = margin
    for n, chip in enumerate(chips):
        w = _text_width(draw, chip, f_small) + 28
        draw.rounded_rectangle((x, 150, x + w, 184), radius=17, fill=INK_2,
                               outline=SIGNAL if n == 0 else HAIR, width=1)
        draw.text((x + 14, 157), chip, font=f_small, fill=PAPER)
        x += w + 10
    draw.line([(margin, header_h - 26), (width - margin, header_h - 26)], fill=HAIR, width=1)

    top = header_h
    left = margin
    # ── query column ─────────────────────────────────────────────────
    if side:
        draw.text((margin, top), "QUERY", font=f_eyebrow, fill=MUTED)
        y = top + 34
        for path in query_images[:3]:
            try:
                tile, _ = focus_crop(open_rgb(path), (side, side), None)
            except Exception:  # noqa: BLE001 - one unreadable query must not stop the card
                continue
            canvas.paste(rounded(tile, 14), (margin, y), rounded(tile, 14))
            draw.rounded_rectangle((margin, y, margin + side, y + side), radius=14,
                                   outline=SIGNAL, width=2)
            y += side + 16
        left = margin + side + gap
        draw.line([(left - gap // 2, top), (left - gap // 2, top + body_h)], fill=HAIR, width=1)

    # ── results ───────────────────────────────────────────────────────
    if not shown:
        draw.text((left, top + 60), "Nothing in the index cleared the threshold.",
                  font=font(24), fill=MUTED)
    for i, match in enumerate(shown):
        r, c = divmod(i, columns)
        x0 = left + c * (tile_w + gap)
        y0 = top + r * (tile_h + caption_h + gap)
        try:
            photo = open_rgb(match.file_path, 2400)
            from vie.report.theme import original_size

            original = original_size(match.file_path)
            box_scale = photo.size[0] / original[0] if original and original[0] else 1.0
            tile, box = focus_crop(photo, (tile_w, tile_h), match.box, box_scale)
        except Exception:  # noqa: BLE001
            tile, box = Image.new("RGB", (tile_w, tile_h), INK_3), None
        layer = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(layer)
        if box is not None:
            draw_brackets(tdraw, box, SIGNAL if i == 0 else PAPER, width=4 if i == 0 else 3)
        # Soft shade at the top so the rank badge reads on any photo.
        for k in range(46):
            tdraw.line([(0, k), (tile_w, k)], fill=(0, 0, 0, int(120 * (1 - k / 46))))
        tile = Image.alpha_composite(tile.convert("RGBA"), layer)
        masked = rounded(tile.convert("RGB"), 14)
        canvas.paste(masked, (x0, y0), masked)
        draw.rounded_rectangle((x0, y0, x0 + tile_w, y0 + tile_h), radius=14, outline=HAIR, width=1)
        pill(draw, (x0 + 12, y0 + 12), f"{i + 1:02d}", f_small,
             SIGNAL if i == 0 else (0, 0, 0, 150), SIGNAL_INK if i == 0 else PAPER)

        cy = y0 + tile_h + 14
        value = max(0.0, min(1.0, match.display_score))
        score = f"{value * 100:.1f}%"
        sw = _text_width(draw, score, f_score)
        draw.text((x0 + tile_w - sw, cy - 2), score, font=f_score,
                  fill=SIGNAL if i == 0 else PAPER)
        name = _ellipsize(draw, Path(match.file_name).name, f_name, tile_w - sw - 16)
        draw.text((x0, cy), name, font=f_name, fill=PAPER)
        if match.label:
            draw.text((x0, cy + 28), _ellipsize(draw, match.label, f_small, tile_w - sw - 16),
                      font=f_small, fill=MUTED)
        by = cy + 60
        draw.rounded_rectangle((x0, by, x0 + tile_w, by + 6), radius=3, fill=INK_3)
        draw.rounded_rectangle((x0, by, x0 + max(6, int(tile_w * value)), by + 6), radius=3,
                               fill=SIGNAL if i == 0 else MUTED)

    # ── footer ────────────────────────────────────────────────────────
    fy = height - 70
    draw.line([(margin, fy), (width - margin, fy)], fill=HAIR, width=1)
    draw.text((margin, fy + 20), f"Visual Intelligence Engine {__version__}", font=f_small,
              fill=FAINT)
    note = "scores are calibrated per search path · boxes come from the index"
    draw.text((width - margin - _text_width(draw, note, f_small), fy + 20), note, font=f_small,
              fill=FAINT)
    return canvas


def save_search_card(matches: Sequence[Match], path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = render_search_card(matches, **kwargs)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=90, optimize=True, progressive=True)
    else:
        image.save(path, optimize=True)
    return path
