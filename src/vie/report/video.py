"""The video report: one page that answers "what is in this video?".

The timeline is the centre of the page. Every lane shares one time axis, so you can
read straight down from a moment and see which shot it is, what it was labelled, who
is on screen, and what was being said. Hovering a keyframe moves a shared cursor;
everything is inline and no script is needed to read the page.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

from vie import __version__
from vie.report.theme import document, esc
from vie.video.engine import Keyframe, VideoAnalysis
from vie.video.people import Interval, merge_times
from vie.video.understand import format_clock, top_by_group

_CSS = """
.page { max-width: 1240px; }
.summary-switch { display: inline-flex; gap: 2px; margin: 28px 0 12px; }
.summary-switch button {
  all: unset; cursor: pointer; font: 500 12px/1 var(--mono); padding: 7px 11px;
  border: 1px solid var(--hair); color: var(--muted);
}
.summary-switch button:first-child { border-radius: 6px 0 0 6px; }
.summary-switch button:last-child { border-radius: 0 6px 6px 0; }
.summary-switch button[aria-pressed="true"] { background: var(--signal); color: var(--signal-ink); border-color: var(--signal); }
.summary-switch button:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }
.summary { font-size: clamp(18px, 2.1vw, 23px); line-height: 1.55; max-width: 62ch; letter-spacing: -.008em; margin: 0; }
.summary[lang="fa"] { direction: rtl; text-align: right; line-height: 1.9; }
.summary[hidden] { display: none; }

.reel {
  display: grid; gap: 3px; margin-top: 40px; border-radius: var(--radius); overflow: hidden;
  grid-template-columns: repeat(auto-fill, minmax(min(33%, 150px), 1fr));
}
.reel figure { margin: 0; position: relative; background: var(--ink-3); aspect-ratio: 16 / 9; }
.reel img { width: 100%; height: 100%; object-fit: cover; display: block; }
.reel figcaption {
  position: absolute; left: 5px; bottom: 5px; padding: 1px 5px; border-radius: 3px;
  background: rgba(8,10,12,.75); color: #fff; font: 500 10.5px/1.4 var(--mono);
}

.timeline { position: relative; margin-top: 8px; }
.lane { display: grid; grid-template-columns: 150px 1fr; align-items: center; gap: 14px; min-height: 26px; padding: 3px 0; }
.lane > .name { font-size: 12.5px; color: var(--muted); text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.track-area { position: relative; height: 18px; background: var(--ink-2); border-radius: 3px; overflow: hidden; }
.seg { position: absolute; top: 0; bottom: 0; min-width: 2px; }
.lane.shots .seg { background: var(--ink-3); border-right: 1px solid var(--ink); }
.lane.shots .seg:nth-child(even) { background: var(--hair); }
.axis { display: grid; grid-template-columns: 150px 1fr; gap: 14px; margin-top: 8px; }
.axis .ticks { position: relative; height: 16px; font: 11px/1 var(--mono); color: var(--faint); }
.axis .ticks span { position: absolute; transform: translateX(-50%); }
.axis .ticks span:first-child { transform: none; }
.axis .ticks span:last-child { transform: translateX(-100%); }
.lane-group { font: 500 11px/1 var(--mono); letter-spacing: .1em; text-transform: uppercase; color: var(--faint); margin: 16px 0 4px; padding-left: 164px; }

.bars { display: grid; gap: 34px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); }
.bars h3 { font-size: 12.5px; color: var(--muted); font-weight: 500; margin: 0 0 12px; text-transform: capitalize; }
.bar { margin-bottom: 13px; }
.bar .row { display: flex; justify-content: space-between; gap: 12px; font-size: 14px; }
.bar .row b { font: 500 12.5px/1.6 var(--mono); color: var(--muted); }
.bar .track { display: block; height: 3px; background: var(--hair); border-radius: 2px; overflow: hidden; margin-top: 5px; }
.bar .track i { display: block; height: 100%; background: var(--signal); }

.people { display: flex; flex-wrap: wrap; gap: 22px; }
.person { width: 124px; }
.person img { width: 124px; height: 124px; object-fit: cover; border-radius: 10px; display: block; background: var(--ink-3); }
.person b { display: block; font-size: 13.5px; margin-top: 8px; font-weight: 600; }
.person span { font: 12px/1.5 var(--mono); color: var(--muted); }

.shot-grid {
  display: grid; gap: 22px 14px;
  grid-template-columns: repeat(auto-fill, minmax(min(46%, 188px), 1fr));
}
.shot-card img, .shot-card .blank {
  width: 100%; aspect-ratio: 16 / 9; object-fit: cover; display: block;
  border-radius: 6px; background: var(--ink-3);
}
.shot-card .when { font: 12px/1.5 var(--mono); color: var(--muted); margin-top: 7px; }
.tag { display: block; font-size: 12.5px; line-height: 1.45; color: var(--muted); }
.caption { font-size: 13px; line-height: 1.45; color: var(--paper); margin-top: 3px; }

.transcript { display: grid; gap: 10px; max-height: 460px; overflow-y: auto; padding-right: 10px; }
.line { display: grid; grid-template-columns: 62px 1fr; gap: 14px; }
.line time { font: 12.5px/1.7 var(--mono); color: var(--faint); }
.line p { margin: 0; unicode-bidi: plaintext; text-align: start; }
.problem { margin: 40px 0 0; padding: 12px 14px; border: 1px solid var(--lane-5); border-radius: 8px; color: var(--paper); font-size: 14px; overflow-wrap: anywhere; }
.note { font-size: 13px; color: var(--faint); margin-top: 10px; max-width: 72ch; }

@media (max-width: 640px) {
  .page { padding: 34px 16px 64px; }
  .lane, .axis { grid-template-columns: 84px 1fr; gap: 8px; }
  .lane-group { padding-left: 92px; }
  .axis .ticks span:nth-child(even) { display: none; }
}
"""

_SCRIPT = """
document.querySelectorAll('.summary-switch button').forEach(function (button) {
  button.addEventListener('click', function () {
    document.querySelectorAll('.summary-switch button').forEach(function (other) {
      other.setAttribute('aria-pressed', String(other === button));
    });
    document.querySelectorAll('.summary').forEach(function (text) {
      text.hidden = text.lang !== button.dataset.lang;
    });
  });
});
"""


def _jpeg_uri(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _percent(value: float, duration: float) -> float:
    return max(0.0, min(100.0, value / duration * 100)) if duration > 0 else 0.0


def _segments(intervals: list[Interval], duration: float, colour: str) -> str:
    parts = []
    for span in intervals:
        left = _percent(span.start, duration)
        width = max(0.25, _percent(span.end, duration) - left)
        title = f"{format_clock(span.start)} to {format_clock(span.end)}"
        parts.append(
            f'<div class="seg" style="left:{left:.3f}%;width:{width:.3f}%;background:{colour}" '
            f'title="{esc(title)}"></div>'
        )
    return "".join(parts)


def _lane(name: str, intervals: list[Interval], duration: float, colour: str) -> str:
    return (
        f'<div class="lane"><div class="name" title="{esc(name)}">{esc(name)}</div>'
        f'<div class="track-area">{_segments(intervals, duration, colour)}</div></div>'
    )


def _axis(duration: float) -> str:
    count = 6
    ticks = "".join(
        f'<span style="left:{i / count * 100:.2f}%">{format_clock(duration * i / count)}</span>'
        for i in range(count + 1)
    )
    return f'<div class="axis"><div></div><div class="ticks">{ticks}</div></div>'


def _reel(keyframes: list[Keyframe], limit: int = 24) -> str:
    """A contact sheet of the whole video, leaving out black frames and fades."""
    usable = [k for k in keyframes if k.thumbnail and k.brightness >= 25] or keyframes
    if not usable:
        return ""
    step = len(usable) / min(limit, len(usable))
    chosen = [usable[int(i * step)] for i in range(min(limit, len(usable)))]
    figures = "".join(
        f'<figure><img src="{_jpeg_uri(k.thumbnail)}" alt="Frame at {format_clock(k.time)}">'
        f"<figcaption>{format_clock(k.time)}</figcaption></figure>"
        for k in chosen
    )
    return f'<section class="reel" aria-label="Frames from across the video">{figures}</section>'


def _timeline(analysis: VideoAnalysis, duration: float) -> str:
    shot_segments = "".join(
        f'<div class="seg" style="left:{_percent(s.start, duration):.3f}%;'
        f'width:{max(0.1, _percent(s.end, duration) - _percent(s.start, duration)):.3f}%" '
        f'title="Shot {s.index + 1}: {format_clock(s.start)} to {format_clock(s.end)}"></div>'
        for s in analysis.shots
    )
    rows = [
        f'<div class="lane shots"><div class="name">{len(analysis.shots)} shots</div>'
        f'<div class="track-area">{shot_segments}</div></div>'
    ]

    colours = {"topic": "var(--lane-1)", "setting": "var(--lane-2)", "activity": "var(--lane-3)"}
    for group, items in top_by_group(analysis.labels, per_group=4).items():
        rows.append(f'<div class="lane-group">{esc(group)}</div>')
        colour = colours.get(group, "var(--lane-1)")
        rows += [_lane(item.label, list(item.intervals), duration, colour) for item in items]

    if analysis.people:
        rows.append('<div class="lane-group">People</div>')
        rows += [
            _lane(f"Person {p.id}", p.intervals, duration, "var(--lane-4)")
            for p in analysis.people[:8]
        ]

    gap = max(analysis.sample_seconds * 3, 8.0)
    sightings: list[tuple[str, list[Interval]]] = []
    if analysis.animal_times:
        sightings.append(("Animals", merge_times(analysis.animal_times, gap, pad=0.5)))
    if analysis.food_times:
        sightings.append(("Food", merge_times(analysis.food_times, gap, pad=0.5)))
    if analysis.transcript:
        spans = [Interval(s.start, s.end) for s in analysis.transcript]
        sightings.append(("Speech", spans))
    if sightings:
        rows.append('<div class="lane-group">Also detected</div>')
        rows += [_lane(name, spans, duration, "var(--lane-5)") for name, spans in sightings]

    return f'<section class="timeline">{"".join(rows)}{_axis(duration)}</section>'


def _bars(analysis: VideoAnalysis) -> str:
    grouped = top_by_group(analysis.labels, per_group=5)
    if not grouped:
        return ""
    columns = []
    for group, items in grouped.items():
        bars = "".join(
            f'<div class="bar"><div class="row"><span>{esc(i.label)}</span>'
            f"<b>{i.coverage * 100:.0f}%</b></div>"
            f'<span class="track"><i style="width:{i.coverage * 100:.1f}%"></i></span></div>'
            for i in items
        )
        columns.append(f"<div><h3>{esc(group)}</h3>{bars}</div>")
    return (
        '<h2>What it contains</h2><section class="bars">' + "".join(columns) + "</section>"
        '<p class="note">Percentages are the share of the running time where the label clears '
        "its threshold. Labels are zero-shot guesses from a fixed list, not ground truth.</p>"
    )


def _people(analysis: VideoAnalysis) -> str:
    if not analysis.people:
        return ""
    cards = []
    for person in analysis.people[:16]:
        data = analysis.person_thumbnails.get(person.id)
        picture = f'<img src="{_jpeg_uri(data)}" alt="Person {person.id}">' if data else ""
        cards.append(
            f'<div class="person">{picture}<b>Person {person.id}</b>'
            f"<span>{format_clock(person.screen_time)} · {len(person.sightings)} views</span></div>"
        )
    return (
        '<h2>People</h2><section class="people">' + "".join(cards) + "</section>"
        '<p class="note">Grouped by face similarity within this video. These are not identities: '
        "nothing here says who anyone is, and the same person can be split into two groups.</p>"
    )


def _shot_grid(analysis: VideoAnalysis, limit: int = 48) -> str:
    frames: dict[int, Keyframe] = {}
    for frame in analysis.keyframes:
        # The brightest keyframe of a shot says the most about it.
        current = frames.get(frame.shot)
        if current is None or frame.brightness > current.brightness:
            frames[frame.shot] = frame

    shots = sorted(analysis.shots, key=lambda s: s.duration, reverse=True)[:limit]
    cards = []
    for shot in sorted(shots, key=lambda s: s.start):
        frame = frames.get(shot.index)
        picture = (
            f'<img src="{_jpeg_uri(frame.thumbnail)}" alt="Shot {shot.index + 1}">'
            if frame and frame.thumbnail else '<div class="blank"></div>'
        )
        entry = analysis.shot_labels[shot.index]
        caption = f'<div class="caption">{esc(entry.caption)}</div>' if entry.caption else ""
        tags = "".join(
            f'<span class="tag">{esc(label)}</span>' for label, _ in entry.labels.values()
        )
        cards.append(
            f'<article class="shot-card">{picture}<div class="when">{format_clock(shot.start)}'
            f" · {shot.duration:.0f} s</div>{caption}{tags}</article>"
        )
    heading = f"Longest {limit} of {len(analysis.shots)} shots" if len(analysis.shots) > limit else "Shots"
    return f'<h2>{heading}</h2><section class="shot-grid">{"".join(cards)}</section>'


def _transcript(analysis: VideoAnalysis) -> str:
    if not analysis.transcript:
        return ""
    lines = "".join(
        f'<div class="line"><time>{format_clock(s.start)}</time><p>{esc(s.text)}</p></div>'
        for s in analysis.transcript
    )
    language = f" · detected language: {esc(analysis.language)}" if analysis.language else ""
    return f'<h2>Speech{language}</h2><section class="transcript">{lines}</section>'


def render_video_report(analysis: VideoAnalysis) -> str:
    info = analysis.info
    name = Path(info.path).name
    duration = analysis.shots[-1].end if analysis.shots else info.duration
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    stats = [
        (format_clock(duration), "running time"),
        (str(len(analysis.shots)), "shots"),
        (str(len(analysis.keyframes)), "frames analysed"),
    ]
    if analysis.stages.get("faces"):
        stats.append((str(len(analysis.people)), "distinct faces"))
    stat_row = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)

    english = analysis.summary.get("en", "")
    persian = analysis.summary.get("fa", "")
    summary = (
        '<div class="summary-switch" role="group" aria-label="Summary language">'
        '<button type="button" data-lang="en" aria-pressed="true">EN</button>'
        '<button type="button" data-lang="fa" aria-pressed="false">فارسی</button></div>'
        f'<p class="summary" lang="en">{esc(english)}</p>'
        f'<p class="summary" lang="fa" hidden>{esc(persian)}</p>'
    )

    off = [stage for stage, on in analysis.stages.items() if not on]
    disabled = f" · not run: {', '.join(off)}" if off else ""
    problems = "".join(f'<p class="problem">{esc(item)}</p>' for item in analysis.problems)
    body = (
        f'<p class="eyebrow">Video analysis</p><h1>{esc(name)}</h1>'
        f'<div class="meta">{info.width}×{info.height} · {info.fps:.2f} fps · {esc(info.codec)}'
        f"{' · audio' if info.has_audio else ''}</div>"
        f'<div class="stats">{stat_row}</div>{summary}{_reel(analysis.keyframes)}'
        f"<h2>Timeline</h2>{_timeline(analysis, duration)}"
        f"{_bars(analysis)}{_people(analysis)}{_transcript(analysis)}{_shot_grid(analysis)}"
        f"{problems}"
        f'<div class="footer">Visual Intelligence Engine {__version__} · {stamp}{disabled}</div>'
    )
    return document(f"Video analysis: {name}", body, extra_css=_CSS, script=_SCRIPT)


def write_video_report(analysis: VideoAnalysis, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_video_report(analysis), encoding="utf-8")
    return path
