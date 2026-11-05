"""The search results page."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from vie import __version__
from vie.report.theme import document, esc, original_size, thumbnail_uri
from vie.scoring import Match

KIND_TITLES = {
    "face": "Face search",
    "animal": "Species search",
    "food": "Dish search",
    "text": "Free-text search",
}


def _box_overlay(match: Match) -> str:
    """Draw the detection as percentages, so it survives thumbnailing."""
    if match.box is None:
        return ""
    size = original_size(match.file_path)
    if size is None or not all(size):
        return ""
    width, height = size
    x1, y1, x2, y2 = match.box
    return (
        f'<div class="box" style="left:{x1 / width * 100:.2f}%;top:{y1 / height * 100:.2f}%;'
        f'width:{(x2 - x1) / width * 100:.2f}%;height:{(y2 - y1) / height * 100:.2f}%"></div>'
    )


def _card(rank: int, match: Match) -> str:
    uri = thumbnail_uri(match.file_path)
    picture = (
        f'<img src="{uri}" alt="{esc(match.file_name)}">'
        if uri else '<div class="empty">file not readable</div>'
    )
    value = max(0.0, min(1.0, match.display_score))
    label = f'<div class="label">{esc(match.label)}</div>' if match.label else ""
    return (
        f'<article class="card"><div class="frame">{picture}{_box_overlay(match)}'
        f'<span class="rank">{rank:02d}</span></div>'
        f'<div class="name">{esc(match.file_name)}</div>{label}'
        f'<div class="scoreline"><b>{match.display_score * 100:.1f}%</b>'
        f'<span class="track"><i style="width:{value * 100:.1f}%"></i></span></div></article>'
    )


def render_search_report(matches: list[Match], *, kind: str, query: str) -> str:
    title = KIND_TITLES.get(kind, "Search")
    shown = Path(query).name if kind == "face" else query
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if matches:
        best = max(m.display_score for m in matches)
        stats = (
            f'<div class="stats"><div class="stat"><b>{len(matches)}</b><span>images</span></div>'
            f'<div class="stat"><b>{best * 100:.1f}%</b><span>best score</span></div></div>'
        )
        results = (
            '<section class="grid">'
            + "".join(_card(i, m) for i, m in enumerate(matches, 1))
            + "</section>"
        )
    else:
        stats = ""
        results = '<div class="empty">Nothing in the index cleared the threshold.</div>'

    body = (
        f'<p class="eyebrow">{esc(title)}</p><h1>{esc(shown)}</h1>{stats}{results}'
        f'<div class="footer">Visual Intelligence Engine {__version__} · {stamp}'
        f" · boxes are in original image coordinates</div>"
    )
    return document(f"{title}: {shown}", body)


def write_search_report(
    matches: list[Match], path: str | Path, *, kind: str, query: str
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_search_report(matches, kind=kind, query=query), encoding="utf-8")
    return path
