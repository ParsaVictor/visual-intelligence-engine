"""The Smart Gallery interface (Gradio).

No ``from __future__ import annotations`` here: Gradio reads the real annotation
``gr.SelectData`` to decide which handlers receive click events.
"""

import base64
import csv
import html
import inspect
import io
import re
import time
from pathlib import Path
from typing import Any

from vie import __version__
from vie.gallery import insights as gi
from vie.gallery.engine import MODES, Gallery, Result, annotated, face_crop
from vie.logging_setup import get_logger
from vie.pipeline import IMAGE_SUFFIXES

log = get_logger("gallery.app")

#: Exactly what indexing reads, so the picker never accepts a file it would skip.
UPLOAD_SUFFIXES = frozenset(IMAGE_SUFFIXES)

CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
.rtl, .rtl * { direction: rtl; text-align: right; }
#sg-head { padding: 22px 4px 6px; direction: rtl; text-align: right; }
#sg-head h1 { font-size: 32px; margin: 0 0 6px; letter-spacing: -.01em; }
#sg-head p { margin: 0; opacity: .72; }
#sg-head .ver { font: 12px ui-monospace, Consolas, monospace; opacity: .55; }
.sg { direction: rtl; text-align: right; font-size: 15px; }
.sg .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px; margin: 6px 0 22px; }
.sg .tile { border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: 16px 18px; background: var(--block-background-fill); }
.sg .tile b { display: block; font-size: 34px; line-height: 1.1; color: #e8912d; }
.sg .tile span { opacity: .78; }
.sg .tile small { display: block; opacity: .6; margin-top: 4px; }
.sg h2 { font-size: 20px; margin: 26px 0 10px; }
.sg .people { display: flex; flex-wrap: wrap; gap: 14px; }
.sg .person { width: 112px; text-align: center; }
.sg .person img { width: 104px; height: 104px; object-fit: cover; border-radius: 50%;
  border: 3px solid #e8912d55; }
.sg .person div { font-weight: 600; margin-top: 4px; text-align: center; }
.sg .person small { opacity: .65; display: block; text-align: center; }
.sg .group { border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: 12px 14px; margin-bottom: 10px; background: var(--block-background-fill); }
.sg .group header { display: flex; justify-content: space-between; font-weight: 600;
  margin-bottom: 8px; }
.sg .group header em { font-style: normal; opacity: .7; font-weight: 400; }
.sg .thumbs { display: flex; gap: 8px; flex-wrap: wrap; }
.sg .thumbs img { height: 92px; border-radius: 8px; }
.sg .bars { display: grid; gap: 6px; }
.sg .bar { display: grid; grid-template-columns: 180px 1fr 42px; align-items: center; gap: 10px; }
.sg .bar i { display: block; height: 10px; border-radius: 6px; background: #e8912d; }
.sg .bar s { text-decoration: none; opacity: .7; text-align: left; }
.sg .empty { opacity: .6; padding: 6px 0; }
.sg .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 860px) { .sg .cols { grid-template-columns: 1fr; } }
"""

HEADER = f"""
<div id="sg-head">
  <h1>گالری هوشمند</h1>
  <p>جستجو در عکس‌ها بر اساس محتوا: چهره، نام شخص، حیوان، غذا، مفهوم و عکس مشابه.
  همه‌چیز روی همین سیستم و بدون ارسال به اینترنت.</p>
  <div class="ver">Visual Intelligence Engine {__version__}</div>
</div>
"""

TABLE_HEADERS = ["رتبه", "فایل", "امتیاز", "چه چیزی پیدا شد", "همه‌ی موارد در عکس",
                 "کادر (x1,y1,x2,y2)", "مسیر"]


def everything_in(info: gi.Insights, file_name: str,
                  matched: tuple[tuple, str] | None = None) -> tuple[tuple[tuple, str], ...]:
    """Every box in a photo, labelled: the match first, then each person, animal, dish.

    A result for "cat" that also holds a person shows both, so nothing in the photo is
    silently left out of the picture.
    """
    boxes: list[tuple[tuple, str]] = [matched] if matched else []
    photo = info.photos.get(file_name)
    if photo is None:
        return tuple(boxes)

    def taken(box: tuple) -> bool:
        return any(gi._same_object(tuple(box), tuple(b)) for b, _ in boxes)

    for face in photo.faces:
        if face.person and not taken(face.box):
            boxes.append((face.box, info.people[face.person - 1].title))
    for animal in photo.animals:
        if not taken(animal.box):
            boxes.append((animal.box, animal.species or animal.group_fa))
    for box, _, _ in photo.dishes[:2]:
        if not taken(box):
            boxes.append((box, gi.dish_name(photo)))
    return tuple(boxes)


# ---- small helpers ----------------------------------------------------------


def _uri(image: Any, quality: int = 80) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _thumb(photo: gi.Photo, side: int = 220) -> str:
    try:
        return _uri(annotated(photo.file_path, (), side))
    except Exception:
        return ""


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("_")[:50] or "query"


def _fa_digits(value: Any) -> str:
    return gi.fa_digits(value)


def _rtl(text: str) -> str:
    """Start a caption right to left; Gradio lays captions out left to right."""
    return "‏" + text


# ---- overview ---------------------------------------------------------------


def overview_html(gallery: Gallery, people: bool = True, section: str = "all") -> str:
    """The overview page. ``section`` is "tiles", "details" (no tiles, no people) or "all".

    The live page shows the tiles, then a clickable people panel, then the details.
    """
    info = gallery.insights
    photos = info.photos
    species = info.species()
    dishes = info.dishes()
    named = sum(1 for p in info.people if p.name)
    tiles = [
        (len(photos), "عکس در گالری", ""),
        (len(info.people), "فرد متفاوت",
         f"{info.face_count} چهره در {len(info.with_people())} عکس"
         + (f"، {named} نام‌گذاری‌شده" if named else "")),
        (len(info.with_animals()), "عکس دارای حیوان", f"{len(species)} گونه متفاوت"),
        (len(info.with_food()), "عکس غذا", f"{len(dishes)} نوع غذا"),
        (len(info.by_label("subject")), "دسته‌ی موضوعی",
         f"{sum(1 for p in photos.values() if p.subject)} عکس دسته‌بندی شد"),
    ]
    parts = ['<div class="sg"><div class="tiles">']
    for value, label, small in tiles:
        parts.append(f'<div class="tile"><b>{_fa_digits(value)}</b><span>{label}</span>'
                     f'<small>{html.escape(_fa_digits(small))}</small></div>')
    parts.append("</div>")
    if section == "tiles":
        return "".join(parts) + "</div>"
    if section == "details":
        parts, people = ['<div class="sg">'], False

    if people:
        parts.append("<h2>👥 افراد</h2>")
    if people and info.people:
        parts.append('<div class="people">')
        for person in info.people:
            face = person.best_face
            try:
                picture = _uri(face_crop(photos[face.file_name].file_path, face.box), 85)
            except Exception:
                picture = ""
            parts.append(
                f'<div class="person"><img src="{picture}" alt=""><div>'
                f'{html.escape(person.title)}</div><small>{_fa_digits(len(person.photos))} عکس'
                f"</small></div>"
            )
        parts.append("</div>")
    elif people:
        parts.append('<div class="empty">هیچ چهره‌ای در گالری پیدا نشد.</div>')

    def groups(title: str, items: dict[str, list[str]], empty: str) -> None:
        parts.append(f"<h2>{title}</h2>")
        if not items:
            parts.append(f'<div class="empty">{empty}</div>')
            return
        for name, keys in items.items():
            thumbs = "".join(f'<img src="{_thumb(photos[k], 200)}" alt="">' for k in keys[:8])
            more = f" و {_fa_digits(len(keys) - 8)} عکس دیگر" if len(keys) > 8 else ""
            parts.append(
                f'<div class="group"><header><span>{html.escape(name)}</span>'
                f"<em>{_fa_digits(len(keys))} عکس{more}</em></header>"
                f'<div class="thumbs">{thumbs}</div></div>'
            )

    groups("🐾 حیوانات", species, "حیوانی در گالری پیدا نشد.")
    groups("🍽️ غذاها", dishes, "غذایی در گالری پیدا نشد.")

    def bars(title: str, items: dict[str, list[str]]) -> str:
        if not items:
            return f"<div><h2>{title}</h2><div class='empty'>—</div></div>"
        top = max(len(v) for v in items.values())
        rows = "".join(
            f'<div class="bar"><span>{html.escape(name)}</span>'
            f'<i style="width:{len(keys) / top * 100:.0f}%"></i>'
            f"<s>{_fa_digits(len(keys))}</s></div>"
            for name, keys in items.items()
        )
        return f'<div><h2>{title}</h2><div class="bars">{rows}</div></div>'

    parts.append('<div class="cols">')
    parts.append(bars("🏷️ موضوع اصلی عکس‌ها", info.by_label("subject")))
    parts.append(bars("📍 محیط", info.by_label("setting")))
    parts.append("</div></div>")
    return "".join(parts)


def overview_report(gallery: Gallery) -> Path:
    """The overview as one self-contained page, to send or present."""
    page = (
        '<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>گالری هوشمند – گزارش</title><style>"
        ":root{--border-color-primary:#dcdcd6;--block-background-fill:#fff}"
        "body{font-family:'Segoe UI',Tahoma,sans-serif;background:#f6f6f3;color:#15171a;"
        "margin:0;padding:24px}main{max-width:1200px;margin:0 auto}"
        "@media (prefers-color-scheme:dark){:root{--border-color-primary:#262b33;"
        "--block-background-fill:#12151a}body{background:#0b0d10;color:#eef0f2}}"
        f"{CSS}</style></head><body><main>{HEADER}{overview_html(gallery)}"
        f'<p class="sg" style="opacity:.6">مدل: {html.escape(gallery.insights.model)} – '
        f"{time.strftime('%Y-%m-%d %H:%M')}</p></main></body></html>"
    )
    path = gallery.directory / "reports" / "gallery-overview.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


# ---- results ----------------------------------------------------------------


def render_result(gallery: Gallery, result: Result,
                  show_boxes: bool = True) -> tuple[str, list, list, list[str]]:
    """Status line, picture grid, table and downloadable files for one search."""
    from vie.report.search import render_search_report

    info = gallery.insights
    rows: list[list[Any]] = []
    pictures: list[tuple[Any, str]] = []
    for rank, match in enumerate(result.matches, 1):
        what = gallery.describe(match, result.mode)
        score = gallery.score_text(match, result.mode)
        matched = (tuple(match.box), what) if match.box else None
        boxes = everything_in(info, match.file_name, matched) if show_boxes else ()
        tags = "، ".join(info.tags(match.file_name)) if match.file_name in info.photos else ""
        try:
            picture = annotated(match.file_path, boxes)
        except Exception:
            continue
        pictures.append((picture, _rtl(f"{_fa_digits(rank)}. {what} – {score} – "
                                       f"{Path(match.file_name).name}")))
        rows.append([rank, match.file_name, score, what, tags,
                     ",".join(map(str, match.box)) if match.box else "", match.file_path])

    title = MODES[result.mode]
    count = len(result.matches)
    if count:
        status = (f"### {title}: «{html.escape(result.query)}»\n\n"
                  f"**{_fa_digits(count)} عکس** پیدا شد، به ترتیب بهترین، در "
                  f"{_fa_digits(f'{result.seconds:.1f}')} ثانیه")
    else:
        status = (f"### {title}: «{html.escape(result.query)}»\n\n"
                  "هیچ عکسی از آستانه‌ی اطمینان عبور نکرد. یعنی سیستم مطمئن نیست چنین چیزی "
                  "در گالری باشد، و این بهتر از نشان‌دادن نتیجه‌ی غلط است.")
    if result.note:
        status += f"\n\n<small>{html.escape(result.note)}</small>"

    stamp = time.strftime("%H%M%S")
    base = gallery.directory / "reports" / f"{result.mode}-{_slug(result.query)}-{stamp}"
    base.parent.mkdir(parents=True, exist_ok=True)
    report = base.with_suffix(".html")
    report.write_text(
        render_search_report(result.matches, kind=result.mode, query=result.query),
        encoding="utf-8",
    )
    table = base.with_suffix(".csv")
    with table.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TABLE_HEADERS)
        writer.writerows(rows)
    return status, pictures, rows, [str(report), str(table)]


# ---- the app ----------------------------------------------------------------


def _accepted(target: Any, wanted: dict[str, Any]) -> dict[str, Any]:
    """Gradio 6 moved ``theme``, ``css`` and ``title`` from ``Blocks`` to ``launch``."""
    parameters = inspect.signature(target).parameters
    return {key: value for key, value in wanted.items() if key in parameters}


def _styling() -> dict[str, Any]:
    import gradio as gr

    from vie.demo import _build_theme

    return {"theme": _build_theme(gr), "css": CSS, "title": "گالری هوشمند"}


def _fa_error(exc: Exception) -> str:
    """What to tell the operator, in Persian. The details go to the log only."""
    text = str(exc)
    if any("؀" <= ch <= "ۿ" for ch in text):   # already written for the user
        return text
    low = text.lower()
    if "no face detected" in low:
        return "در این عکس چهره‌ی واضحی پیدا نشد؛ عکسی با چهره‌ی روشن و روبه‌رو انتخاب کنید."
    if any(k in low for k in ("could not decode", "cannot identify image", "truncated",
                              "decompression", "pi_heif", "heif")):
        return "این فایل قابل خواندن نیست؛ یک عکس jpg یا png دیگر انتخاب کنید."
    if isinstance(exc, FileNotFoundError) or "not found" in low:
        return "عکس پیدا نشد؛ لطفاً دوباره انتخابش کنید."
    if "database is locked" in low:
        return "گالری در حال به‌روزرسانی است؛ چند ثانیه بعد دوباره امتحان کنید."
    return "مشکلی پیش آمد؛ لطفاً دوباره امتحان کنید."


def _readable(path: Any) -> bool:
    """A query image PIL can open, checked before any model sees it."""
    from PIL import Image

    try:
        with Image.open(path) as handle:
            handle.verify()
        return True
    except Exception:
        return False


UNREADABLE = "این عکس قابل خواندن نیست (مثلاً HEIC آیفون یا فایل خراب)؛ لطفاً JPG یا PNG بدهید."


def build_app(gallery: Gallery) -> Any:
    import gradio as gr

    def SafeImage(**kwargs: Any) -> Any:  # noqa: N802 - used like the component it wraps
        """A ``gr.Image`` that never fails inside Gradio: a file PIL cannot read (HEIC,
        corrupt) reaches the guarded handler as its path, which then says so in Persian.

        Wrapping the instance, not subclassing: Gradio writes a ``.pyi`` stub beside the
        source of every component subclass.
        """
        image = gr.Image(**kwargs)
        original = image.preprocess

        def preprocess(payload: Any) -> Any:
            try:
                return original(payload)
            except Exception:
                log.warning("could not read a query image", exc_info=True)
                path = getattr(payload, "path", None)
                return str(path) if path else None

        image.preprocess = preprocess
        return image

    def info_now() -> gi.Insights:
        return gallery.insights

    def safe(fn: Any, n_outputs: int, message_first: bool = True) -> Any:
        """Run a handler; on any failure show a Persian line instead of a red toast.

        ``functools.wraps`` keeps the handler's signature visible: Gradio reads it to
        decide which handlers receive click data (``gr.SelectData``).
        """
        import functools

        @functools.wraps(fn)
        def wrapper(*args: Any) -> Any:
            try:
                return fn(*args)
            except Exception as exc:
                log.exception("handler failed")
                if n_outputs == 1:
                    return gr.update()
                first = f"⚠️ {html.escape(_fa_error(exc))}" if message_first else gr.update()
                return (first, *(gr.update(),) * (n_outputs - 1))
        return wrapper

    def failure(exc: Exception) -> tuple[str, list, list, None]:
        return f"### ⚠️ {html.escape(_fa_error(exc))}", [], [], None

    def run(mode: str, query: str = "", image: Any = None, top_k: int = 24,
            show_boxes: bool = True):
        try:
            if image and not _readable(image):
                return f"### ⚠️ {UNREADABLE}", [], [], None
            return render_result(gallery, gallery.search(mode, query, image, int(top_k)),
                                 bool(show_boxes))
        except Exception as exc:  # shown to the operator, who can simply try again
            log.exception("search failed")
            return failure(exc)

    def outputs() -> list[Any]:
        status = gr.Markdown(elem_classes="rtl")
        pictures = gr.Gallery(label="نتایج", columns=4, height=620, object_fit="contain",
                              allow_preview=True, show_label=True)
        table = gr.Dataframe(headers=TABLE_HEADERS, label="لیست نتایج", wrap=True,
                             interactive=False)
        files = gr.File(label="دانلود گزارش (HTML) و لیست (CSV)", file_count="multiple")
        return [status, pictures, table, files]

    def top_k_slider() -> Any:
        return gr.Slider(1, 100, value=24, step=1, label="حداکثر تعداد نتیجه")

    # ---- people: identified by one of their faces, never by a rank number ----

    def person_cards():
        cards = []
        for p in info_now().people:
            face = p.best_face
            try:
                picture = face_crop(info_now().photos[face.file_name].file_path, face.box, 220)
            except Exception:
                picture = None      # keep the card, so card index == person index
            cards.append((picture if picture is not None else blank(),
                          _rtl(f"{p.title} – {_fa_digits(len(p.photos))} عکس")))
        return cards

    def person_choices():
        return [(f"{p.title} ({_fa_digits(len(p.photos))} عکس)", gi.person_key(p))
                for p in info_now().people]

    dish_choices = [en for en, _ in gi.DISHES]

    with gr.Blocks(analytics_enabled=False, **_accepted(gr.Blocks, _styling())) as app:
        gr.HTML(HEADER)
        boxes_on = gr.Checkbox(value=True, label="🔲 کشیدن کادر دور افراد، حیوانات و غذا "
                               "(تیک را بردارید تا عکس‌ها بدون کادر نشان داده شوند)")

        with gr.Tab("📊 نمای کلی"):
            overview = gr.HTML(overview_html(gallery, section="tiles"))
            gr.Markdown("## 👥 افراد\nروی هر نفر کلیک کنید تا همین‌جا عکس‌هایش بیاید و "
                        "نامش را تعیین یا عوض کنید.", elem_classes="rtl")
            home_cards = gr.Gallery(value=person_cards(), label="افراد", columns=8,
                                    height=280, object_fit="cover", allow_preview=False)
            home_key = gr.State(None)
            with gr.Row():
                home_name = gr.Textbox(label="نام این شخص", scale=4,
                                       placeholder="ابتدا روی یک نفر کلیک کنید")
                home_save = gr.Button("💾 ذخیره نام", variant="primary", scale=1)
            home_msg = gr.Markdown(elem_classes="rtl")
            home_photos = gr.Gallery(label="عکس‌های این شخص", columns=4, height=520,
                                     object_fit="contain")
            overview_details = gr.HTML(overview_html(gallery, section="details"))
            with gr.Row():
                refresh = gr.Button("🔄 به‌روزرسانی آمار")
                export = gr.Button("⬇️ ساخت گزارش کامل گالری")
            export_file = gr.File(label="گزارش کامل گالری")

            def do_refresh():
                gallery.refresh()
                return (overview_html(gallery, section="tiles"),
                        overview_html(gallery, section="details"))

            refresh.click(safe(do_refresh, 2, message_first=False),
                          outputs=[overview, overview_details])
            export.click(safe(lambda: str(overview_report(gallery)), 1), outputs=export_file)

        with gr.Tab("🔎 جستجو"):
            gr.Markdown("هر نوع جستجو تب جدای خودش را دارد، تا معلوم باشد دنبال چه هستید. "
                        "ایندکس یک بار ساخته شده و جستجو فقط روی آن انجام می‌شود.",
                        elem_classes="rtl")
            with gr.Tab(MODES["face"]):
                gr.Markdown("یک عکس از شخص بدهید؛ همه‌ی عکس‌هایی که آن شخص در آن‌ها هست "
                            "پیدا می‌شود. اگر چند چهره در عکس باشد، بزرگ‌ترین چهره ملاک است.",
                            elem_classes="rtl")
                with gr.Row():
                    face_image = SafeImage(type="filepath", label="عکس شخص", height=260)
                    with gr.Column():
                        face_k = top_k_slider()
                        face_go = gr.Button("پیدا کن", variant="primary")
                face_out = outputs()
                face_go.click(lambda i, k, b: run("face", "", i, k, b),
                              [face_image, face_k, boxes_on], face_out)

            with gr.Tab(MODES["name"]):
                gr.Markdown("شخصی را که قبلا نام‌گذاری کرده‌اید با نامش پیدا کنید. "
                            "نام‌گذاری در همین تب (پایین) یا در تب «افراد» انجام می‌شود.",
                            elem_classes="rtl")
                with gr.Row():
                    name_box = gr.Dropdown(choices=gallery.names.names(), value=None,
                                           label="نام شخص", allow_custom_value=True)
                    with gr.Column():
                        name_k = top_k_slider()
                        name_go = gr.Button("پیدا کن", variant="primary")
                name_out = outputs()
                name_go.click(lambda n, k, b: run("name", n or "", None, k, b),
                              [name_box, name_k, boxes_on], name_out)
                with gr.Accordion("➕ ثبت شخص جدید: یک عکس و یک نام", open=False), gr.Row():
                    enroll_image = SafeImage(type="filepath", label="عکس شخص", height=220)
                    with gr.Column():
                        enroll_name = gr.Textbox(label="نام", placeholder="مثلا: علی رضایی")
                        enroll_go = gr.Button("ثبت نام", variant="primary")
                        enroll_msg = gr.Markdown(elem_classes="rtl")

            with gr.Tab(MODES["animal"]):
                gr.Markdown("نام حیوان را به انگلیسی بنویسید، یا از گونه‌هایی که در گالری "
                            "پیدا شده انتخاب کنید. گونه‌های شناخته‌شده بدون اجرای مدل و "
                            "مستقیم از ایندکس جواب داده می‌شوند.", elem_classes="rtl")
                with gr.Row():
                    animal_pick = gr.Dropdown(choices=info_now().species_names(), value=None,
                                              label="گونه‌های موجود در گالری")
                    animal_query = gr.Textbox(label="نام حیوان (انگلیسی)",
                                              placeholder="giant panda, cat, eagle, lion ...")
                    with gr.Column():
                        animal_k = top_k_slider()
                        animal_go = gr.Button("پیدا کن", variant="primary")
                animal_out = outputs()
                animal_pick.change(lambda v: v or "", animal_pick, animal_query)
                animal_go.click(lambda q, k, b: run("animal", q, None, k, b),
                                [animal_query, animal_k, boxes_on], animal_out)
                animal_query.submit(lambda q, k, b: run("animal", q, None, k, b),
                                    [animal_query, animal_k, boxes_on], animal_out)

            with gr.Tab(MODES["food"]):
                gr.Markdown("نام غذا را به انگلیسی بنویسید. غذاهای ایرانی هم پیدا می‌شوند، "
                            "مثل ghormeh sabzi یا kebab.", elem_classes="rtl")
                with gr.Row():
                    food_pick = gr.Dropdown(choices=dish_choices, value=None,
                                            label="غذاهای پیشنهادی")
                    food_query = gr.Textbox(label="نام غذا (انگلیسی)",
                                            placeholder="pizza, kebab, ghormeh sabzi ...")
                    with gr.Column():
                        food_k = top_k_slider()
                        food_go = gr.Button("پیدا کن", variant="primary")
                food_out = outputs()
                food_pick.change(lambda v: v or "", food_pick, food_query)
                food_go.click(lambda q, k, b: run("food", q, None, k, b),
                              [food_query, food_k, boxes_on], food_out)
                food_query.submit(lambda q, k, b: run("food", q, None, k, b),
                                  [food_query, food_k, boxes_on], food_out)

            with gr.Tab(MODES["text"]):
                gr.Markdown("هر چیزی را به انگلیسی توصیف کنید، مثل یک جمله. عکس با "
                            "بهترین ناحیه‌اش مقایسه می‌شود، پس سوژه‌های کوچک هم پیدا می‌شوند.",
                            elem_classes="rtl")
                with gr.Row():
                    text_query = gr.Textbox(label="توضیح (انگلیسی)",
                                            placeholder="a football player celebrating")
                    with gr.Column():
                        text_k = top_k_slider()
                        text_go = gr.Button("پیدا کن", variant="primary")
                gr.Examples(["a football player", "a castle", "a snowy mountain",
                             "a sunset over the sea", "a man with a dog", "animals in the wild"],
                            text_query, label="نمونه")
                text_out = outputs()
                text_go.click(lambda q, k, b: run("text", q, None, k, b),
                              [text_query, text_k, boxes_on], text_out)
                text_query.submit(lambda q, k, b: run("text", q, None, k, b),
                                  [text_query, text_k, boxes_on], text_out)

            with gr.Tab(MODES["similar"]):
                gr.Markdown("یک عکس بدهید؛ عکس‌های گالری که از نظر محتوا و صحنه به آن "
                            "شبیه‌اند، به ترتیب شباهت نشان داده می‌شوند.", elem_classes="rtl")
                with gr.Row():
                    similar_image = SafeImage(type="filepath", label="عکس نمونه", height=260)
                    with gr.Column():
                        similar_k = gr.Slider(1, 100, value=12, step=1,
                                              label="حداکثر تعداد نتیجه")
                        similar_go = gr.Button("پیدا کن", variant="primary")
                similar_out = outputs()
                similar_go.click(lambda i, k, b: run("similar", "", i, k, b),
                                 [similar_image, similar_k, boxes_on], similar_out)

        with gr.Tab("👥 افراد"):
            gr.Markdown("چهره‌های همه‌ی عکس‌ها خودکار به افراد گروه‌بندی شده‌اند. روی هر "
                        "شخص کلیک کنید تا عکس‌هایش را ببینید و برایش نام بگذارید.",
                        elem_classes="rtl")
            people_cards = gr.Gallery(value=person_cards(), label="افراد", columns=8,
                                      height=300, object_fit="cover", allow_preview=False)
            with gr.Row():
                person_pick = gr.Dropdown(choices=person_choices(), value=None,
                                          label="انتخاب شخص")
                person_name = gr.Textbox(label="نام این شخص",
                                         placeholder="مثلا: Cristiano Ronaldo")
                person_save = gr.Button("💾 ذخیره نام", variant="primary")
            person_msg = gr.Markdown(elem_classes="rtl")
            person_photos = gr.Gallery(label="عکس‌های این شخص", columns=4, height=560,
                                       object_fit="contain")

            def show_person(key, show_boxes=True):
                person = info_now().find_person(key)
                if person is None:
                    return [], "", ""
                items = []
                for photo_key in person.photos:
                    photo = info_now().photos.get(photo_key)
                    if photo is None:
                        continue
                    try:
                        boxes = everything_in(info_now(), photo_key) if show_boxes else ()
                        items.append((annotated(photo.file_path, boxes),
                                      Path(photo_key).name))
                    except Exception:
                        continue
                message = f"**{html.escape(person.title)}**: {_fa_digits(len(person.photos))} عکس"
                return items, person.name, message

            def pick_card(event: gr.SelectData):
                people = info_now().people
                return gi.person_key(people[event.index]) if event.index < len(people) else None

            def save_name(key, name):
                if not key:
                    return "ابتدا یک شخص را انتخاب کنید.", gr.update(), gr.update(), gr.update()
                if not (name or "").strip():
                    return "یک نام بنویسید.", gr.update(), gr.update(), gr.update()
                person = gallery.name_person(key, name)
                return (f"✅ نام «{html.escape(name.strip())}» ذخیره شد. حالا در تب جستجو با "
                        "این نام هم پیدا می‌شود.", person_cards(),
                        gr.update(choices=person_choices(), value=gi.person_key(person)),
                        gr.update(choices=gallery.names.names()))

            people_cards.select(safe(pick_card, 1), None, person_pick)
            person_pick.change(safe(show_person, 3, message_first=False),
                               [person_pick, boxes_on],
                               [person_photos, person_name, person_msg])
            person_save.click(safe(save_name, 4), [person_pick, person_name],
                              [person_msg, people_cards, person_pick, name_box])

            def enroll(image, name):
                if not image or not (name or "").strip():
                    return "یک عکس و یک نام لازم است.", gr.update(), gr.update(), gr.update()
                if not _readable(image):
                    return f"⚠️ {UNREADABLE}", gr.update(), gr.update(), gr.update()
                found = gallery.enroll(name, image)
                return (f"✅ «{html.escape(name.strip())}» ثبت شد و در "
                        f"{_fa_digits(found)} عکس پیدا شد.",
                        gr.update(choices=gallery.names.names(), value=name.strip()),
                        person_cards(), gr.update(choices=person_choices()))

            enroll_go.click(safe(enroll, 4), [enroll_image, enroll_name],
                            [enroll_msg, name_box, people_cards, person_pick])

        with gr.Tab("🗂️ مرور گالری"):
            gr.Markdown("گالری را بر اساس آنچه در عکس‌هاست مرور کنید.", elem_classes="rtl")
            browse_pick = gr.Dropdown(choices=browse_choices(gallery), label="نمایش",
                                      value="all")
            browse_status = gr.Markdown(elem_classes="rtl")
            browse_pictures = gr.Gallery(label="عکس‌ها", columns=5, height=640,
                                         object_fit="contain")
            browse_table = gr.Dataframe(headers=["فایل", "برچسب‌ها", "مسیر"],
                                        label="لیست", wrap=True, interactive=False)
            browse_pick.change(safe(lambda v, b: browse(gallery, v, b), 3),
                               [browse_pick, boxes_on],
                               [browse_status, browse_pictures, browse_table])
            app.load(safe(lambda: browse(gallery, "all"), 3), None,
                     [browse_status, browse_pictures, browse_table])

        with gr.Tab("➕ افزودن عکس"):
            gr.Markdown("عکس‌های جدید را انتخاب یا اینجا رها کنید. در پوشه‌ی گالری ذخیره و "
                        "پردازش می‌شوند و همه‌ی تب‌ها خودکار به‌روز می‌شوند. در این مدت "
                        "جستجوها با داده‌ی قبلی کار می‌کنند.", elem_classes="rtl")
            upload = gr.File(label="عکس‌ها (jpg، png، webp، jfif، bmp، tif)",
                             file_count="multiple", file_types=sorted(UPLOAD_SUFFIXES))
            upload_go = gr.Button("➕ افزودن به گالری و پردازش", variant="primary")
            upload_status = gr.HTML()
            upload_pictures = gr.Gallery(label="عکس‌های تازه اضافه‌شده", columns=5, height=360,
                                         object_fit="contain")

            def bar(done: int, total: int, text: str) -> str:
                share = min(100.0, done / total * 100) if total else 0
                return (f'<div class="sg"><div style="font-size:16px;margin-bottom:8px">{text}'
                        f'</div><div style="background:#8883;border-radius:8px;height:18px">'
                        f'<div style="width:{share:.0f}%;height:18px;border-radius:8px;'
                        f'background:#e8912d;transition:width .4s"></div></div>'
                        f'<div style="opacity:.7;margin-top:6px">{_fa_digits(done)} از '
                        f'{_fa_digits(total)} عکس – {_fa_digits(f"{share:.0f}")}٪</div></div>')

            upload_outputs = [upload_status, upload_pictures, overview, people_cards,
                              person_pick, browse_pick, animal_pick, name_box, home_cards,
                              overview_details, person_photos, person_name, person_msg]
            unchanged = (gr.update(),) * (len(upload_outputs) - 2)

            def add(files):
                import threading

                if not files:
                    yield (bar(0, 0, "ابتدا عکس انتخاب کنید."), [], *unchanged)
                    return
                paths = [str(getattr(f, "path", None) or getattr(f, "name", f)) for f in files]
                total = len(paths)
                state = {"done": 0, "total": total, "error": None, "stats": None}

                def progress(done, all_):
                    state["done"], state["total"] = done, all_

                def work():
                    try:
                        state["stats"] = gallery.add_photos(paths, progress)
                    except Exception as exc:  # shown in the page
                        log.exception("upload failed")
                        state["error"] = exc

                worker = threading.Thread(target=work, daemon=True)
                started = time.perf_counter()
                worker.start()
                while worker.is_alive():
                    seconds = time.perf_counter() - started
                    text = ("⏳ در حال پردازش: تشخیص چهره، حیوان، غذا و ساخت بردار هر عکس "
                            f"({_fa_digits(int(seconds))} ثانیه)")
                    yield (bar(state["done"], state["total"], text), [], *unchanged)
                    time.sleep(1)
                if state["error"] is not None:
                    yield (bar(0, total, f"⚠️ {html.escape(_fa_error(state['error']))}"),
                           [], *unchanged)
                    return
                try:
                    yield (bar(state["total"], state["total"], "🔄 به‌روزرسانی آمار و افراد..."),
                           [], *unchanged)
                    gallery.refresh()
                    stats = state["stats"]
                    fresh = []
                    for path in getattr(stats, "added_paths", [])[:20]:
                        try:
                            fresh.append(annotated(path, everything_in(
                                info_now(), Path(path).name)))
                        except Exception:
                            continue
                    text = (f"✅ {_fa_digits(stats.indexed)} عکس اضافه و پردازش شد در "
                            f"{_fa_digits(int(time.perf_counter() - started))} ثانیه. "
                            "همه‌ی تب‌ها به‌روز شدند.")
                    if getattr(stats, "duplicates", 0):
                        text += f" {_fa_digits(stats.duplicates)} عکس از قبل در گالری بود."
                    if getattr(stats, "unreadable", 0):
                        text += (f" {_fa_digits(stats.unreadable)} فایل قابل خواندن نبود و "
                                 "کنار گذاشته شد.")
                    yield (bar(stats.indexed, max(1, stats.indexed), text), fresh,
                           *fresh_page(), [], "", "")
                except Exception as exc:
                    log.exception("refresh after upload failed")
                    yield (bar(0, total, f"⚠️ {html.escape(_fa_error(exc))}"), [], *unchanged)

            upload_go.click(add, upload, upload_outputs)

        def fresh_page():
            """Every value fixed when the page was built, as the gallery is now."""
            cards = person_cards()
            return (overview_html(gallery, section="tiles"), cards,
                    gr.update(choices=person_choices(), value=None),
                    gr.update(choices=browse_choices(gallery)),
                    gr.update(choices=info_now().species_names()),
                    gr.update(choices=gallery.names.names()), cards,
                    overview_html(gallery, section="details"))

        # A reload or a second tab gets today's numbers, not the ones from start-up.
        app.load(safe(fresh_page, 8, message_first=False), None,
                 [overview, people_cards, person_pick, browse_pick, animal_pick, name_box,
                  home_cards, overview_details])

        # ---- the people panel on the first page ----

        def home_show(key, show_boxes):
            photos, name, message = show_person(key, show_boxes)
            return photos, name, message

        def home_save_name(key, name):
            message, cards, pick, names = save_name(key, name)
            return message, cards, cards, pick, names

        home_cards.select(safe(pick_card, 1), None, home_key).then(
            safe(home_show, 3, message_first=False), [home_key, boxes_on],
            [home_photos, home_name, home_msg])
        home_save.click(safe(home_save_name, 5), [home_key, home_name],
                        [home_msg, home_cards, people_cards, person_pick, name_box])
        home_name.submit(safe(home_save_name, 5), [home_key, home_name],
                         [home_msg, home_cards, people_cards, person_pick, name_box])
        # The switch re-draws what is on screen at once; searches use it on their next run.
        boxes_on.change(safe(home_show, 3, message_first=False), [home_key, boxes_on],
                        [home_photos, home_name, home_msg])
        boxes_on.change(safe(show_person, 3, message_first=False), [person_pick, boxes_on],
                        [person_photos, person_name, person_msg])
        boxes_on.change(safe(lambda v, b: browse(gallery, v, b), 3), [browse_pick, boxes_on],
                        [browse_status, browse_pictures, browse_table])

    return app


def blank(size: int = 220) -> Any:
    from PIL import Image

    return Image.new("RGB", (size, size), (60, 64, 72))


def browse_choices(gallery: Gallery) -> list[tuple[str, str]]:
    info = gallery.insights
    choices = [(f"همه‌ی عکس‌ها ({_fa_digits(len(info.photos))})", "all")]
    choices += [(f"👤 {p.title} ({_fa_digits(len(p.photos))})", f"person:{gi.person_key(p)}")
                for p in info.people]
    choices += [(f"🐾 {name} ({_fa_digits(len(keys))})", f"animal:{name}")
                for name, keys in info.species().items()]
    choices += [(f"🍽️ {name} ({_fa_digits(len(keys))})", f"food:{name}")
                for name, keys in info.dishes().items()]
    choices += [(f"🏷️ {name} ({_fa_digits(len(keys))})", f"subject:{name}")
                for name, keys in info.by_label("subject").items()]
    choices += [(f"📍 {name} ({_fa_digits(len(keys))})", f"setting:{name}")
                for name, keys in info.by_label("setting").items()]
    return choices


def browse(gallery: Gallery, choice: str, show_boxes: bool = True) -> tuple[str, list, list]:
    info = gallery.insights
    kind, _, value = (choice or "all").partition(":")
    if kind == "person":
        person = info.find_person(value)
        if person is None:
            return "### این شخص در فهرست فعلی نیست؛ دوباره انتخاب کنید.", [], []
        keys, title = person.photos, person.title
    elif kind == "animal":
        keys, title = info.species().get(value, []), value
    elif kind == "food":
        keys, title = info.dishes().get(value, []), value
    elif kind in {"subject", "setting"}:
        keys, title = info.by_label(kind).get(value, []), value
    else:
        keys, title = list(info.photos), "همه‌ی عکس‌ها"

    pictures, rows = [], []
    for key in keys:
        photo = info.photos.get(key)
        if photo is None:
            continue
        tags = "، ".join(info.tags(key))
        try:
            picture = annotated(photo.file_path,
                                everything_in(info, key) if show_boxes else ())
            pictures.append((picture, _rtl(f"{tags} – {Path(key).name}")))
        except Exception:
            continue
        rows.append([key, tags, photo.file_path])
    return f"### {html.escape(title)}: {_fa_digits(len(keys))} عکس", pictures, rows


def launch(gallery: Gallery, port: int = 7860, open_browser: bool = True,
           block: bool = True) -> Any:
    import os

    # Gradio checks its own address on start through httpx, which follows the Windows
    # system proxy (a VPN on 127.0.0.1:10808 here). Without this, the page would not
    # start whenever that VPN is off.
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key, ""),
                                                 "localhost,127.0.0.1,::1"]))
    app = build_app(gallery)
    return app.launch(
        server_name="127.0.0.1", server_port=port, inbrowser=open_browser,
        allowed_paths=[str(gallery.directory)], prevent_thread_lock=not block,
        **_accepted(app.launch, _styling()),
    )
