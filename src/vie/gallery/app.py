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

log = get_logger("gallery.app")

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

TABLE_HEADERS = ["رتبه", "فایل", "امتیاز", "چه چیزی پیدا شد", "کادر (x1,y1,x2,y2)", "مسیر"]


# ---- small helpers ----------------------------------------------------------


def _uri(image: Any, quality: int = 80) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _thumb(photo: gi.Photo, side: int = 220) -> str:
    try:
        return _uri(annotated(photo.file_path, (), side))
    except OSError:
        return ""


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("_")[:50] or "query"


def _fa_digits(value: Any) -> str:
    return gi.fa_digits(value)


def _rtl(text: str) -> str:
    """Start a caption right to left; Gradio lays captions out left to right."""
    return "‏" + text


# ---- overview ---------------------------------------------------------------


def overview_html(gallery: Gallery) -> str:
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

    parts.append("<h2>👥 افراد</h2>")
    if info.people:
        parts.append('<div class="people">')
        for person in info.people:
            face = person.best_face
            try:
                picture = _uri(face_crop(photos[face.file_name].file_path, face.box), 85)
            except OSError:
                picture = ""
            parts.append(
                f'<div class="person"><img src="{picture}" alt=""><div>'
                f'{html.escape(person.title)}</div><small>{_fa_digits(len(person.photos))} عکس'
                f"</small></div>"
            )
        parts.append("</div>")
    else:
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


def render_result(gallery: Gallery, result: Result) -> tuple[str, list, list, list[str]]:
    """Status line, picture grid, table and downloadable files for one search."""
    from vie.report.search import render_search_report

    rows: list[list[Any]] = []
    pictures: list[tuple[Any, str]] = []
    for rank, match in enumerate(result.matches, 1):
        what = gallery.describe(match, result.mode)
        score = gallery.score_text(match, result.mode)
        boxes = ((tuple(match.box), what),) if match.box else ()
        try:
            picture = annotated(match.file_path, boxes)
        except OSError:
            continue
        pictures.append((picture, _rtl(f"{_fa_digits(rank)}. {what} – {score} – "
                                       f"{Path(match.file_name).name}")))
        rows.append([rank, match.file_name, score, what,
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


def build_app(gallery: Gallery) -> Any:
    import gradio as gr

    def info_now() -> gi.Insights:
        return gallery.insights

    def failure(exc: Exception) -> tuple[str, list, list, None]:
        return f"### ⚠️ {html.escape(str(exc))}", [], [], None

    def run(mode: str, query: str = "", image: Any = None, top_k: int = 24):
        try:
            return render_result(gallery, gallery.search(mode, query, image, int(top_k)))
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

    info = gallery.insights
    species_choices = info.species_names()
    dish_choices = [en for en, _ in gi.DISHES]

    with gr.Blocks(analytics_enabled=False, **_accepted(gr.Blocks, _styling())) as app:
        gr.HTML(HEADER)

        with gr.Tab("📊 نمای کلی"):
            overview = gr.HTML(overview_html(gallery))
            with gr.Row():
                refresh = gr.Button("🔄 به‌روزرسانی آمار")
                export = gr.Button("⬇️ ساخت گزارش کامل گالری")
            export_file = gr.File(label="گزارش کامل گالری")

            def do_refresh():
                gallery.refresh()
                return overview_html(gallery)

            refresh.click(do_refresh, outputs=overview)
            export.click(lambda: str(overview_report(gallery)), outputs=export_file)

        with gr.Tab("🔎 جستجو"):
            gr.Markdown("هر نوع جستجو تب جدای خودش را دارد، تا معلوم باشد دنبال چه هستید. "
                        "ایندکس یک بار ساخته شده و جستجو فقط روی آن انجام می‌شود.",
                        elem_classes="rtl")
            with gr.Tab(MODES["face"]):
                gr.Markdown("یک عکس از شخص بدهید؛ همه‌ی عکس‌هایی که آن شخص در آن‌ها هست "
                            "پیدا می‌شود. اگر چند چهره در عکس باشد، بزرگ‌ترین چهره ملاک است.",
                            elem_classes="rtl")
                with gr.Row():
                    face_image = gr.Image(type="filepath", label="عکس شخص", height=260)
                    with gr.Column():
                        face_k = top_k_slider()
                        face_go = gr.Button("پیدا کن", variant="primary")
                face_out = outputs()
                face_go.click(lambda i, k: run("face", "", i, k), [face_image, face_k], face_out)

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
                name_go.click(lambda n, k: run("name", n or "", None, k),
                              [name_box, name_k], name_out)
                with gr.Accordion("➕ ثبت شخص جدید: یک عکس و یک نام", open=False), gr.Row():
                    enroll_image = gr.Image(type="filepath", label="عکس شخص", height=220)
                    with gr.Column():
                        enroll_name = gr.Textbox(label="نام", placeholder="مثلا: علی رضایی")
                        enroll_go = gr.Button("ثبت نام", variant="primary")
                        enroll_msg = gr.Markdown(elem_classes="rtl")

            with gr.Tab(MODES["animal"]):
                gr.Markdown("نام حیوان را به انگلیسی بنویسید، یا از گونه‌هایی که در گالری "
                            "پیدا شده انتخاب کنید. گونه‌های شناخته‌شده بدون اجرای مدل و "
                            "مستقیم از ایندکس جواب داده می‌شوند.", elem_classes="rtl")
                with gr.Row():
                    animal_pick = gr.Dropdown(choices=species_choices, value=None,
                                              label="گونه‌های موجود در گالری")
                    animal_query = gr.Textbox(label="نام حیوان (انگلیسی)",
                                              placeholder="giant panda, cat, eagle, lion ...")
                    with gr.Column():
                        animal_k = top_k_slider()
                        animal_go = gr.Button("پیدا کن", variant="primary")
                animal_out = outputs()
                animal_pick.change(lambda v: v or "", animal_pick, animal_query)
                animal_go.click(lambda q, k: run("animal", q, None, k),
                                [animal_query, animal_k], animal_out)
                animal_query.submit(lambda q, k: run("animal", q, None, k),
                                    [animal_query, animal_k], animal_out)

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
                food_go.click(lambda q, k: run("food", q, None, k), [food_query, food_k], food_out)
                food_query.submit(lambda q, k: run("food", q, None, k),
                                  [food_query, food_k], food_out)

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
                             "a sunset over the sea", "a red car", "animals in the wild"],
                            text_query, label="نمونه")
                text_out = outputs()
                text_go.click(lambda q, k: run("text", q, None, k), [text_query, text_k], text_out)
                text_query.submit(lambda q, k: run("text", q, None, k),
                                  [text_query, text_k], text_out)

            with gr.Tab(MODES["similar"]):
                gr.Markdown("یک عکس بدهید؛ عکس‌های گالری که از نظر محتوا و صحنه به آن "
                            "شبیه‌اند، به ترتیب شباهت نشان داده می‌شوند.", elem_classes="rtl")
                with gr.Row():
                    similar_image = gr.Image(type="filepath", label="عکس نمونه", height=260)
                    with gr.Column():
                        similar_k = gr.Slider(1, 100, value=12, step=1,
                                              label="حداکثر تعداد نتیجه")
                        similar_go = gr.Button("پیدا کن", variant="primary")
                similar_out = outputs()
                similar_go.click(lambda i, k: run("similar", "", i, k),
                                 [similar_image, similar_k], similar_out)

        with gr.Tab("👥 افراد"):
            gr.Markdown("چهره‌های همه‌ی عکس‌ها خودکار به افراد گروه‌بندی شده‌اند. روی هر "
                        "شخص کلیک کنید تا عکس‌هایش را ببینید و برایش نام بگذارید.",
                        elem_classes="rtl")

            def person_cards():
                return [
                    (face_crop(info_now().photos[p.best_face.file_name].file_path,
                               p.best_face.box, 220),
                     _rtl(f"{p.title} – {_fa_digits(len(p.photos))} عکس"))
                    for p in info_now().people
                ]

            def person_choices():
                return [(f"{p.title} ({_fa_digits(len(p.photos))} عکس)", p.id)
                        for p in info_now().people]

            people_cards = gr.Gallery(value=person_cards(), label="افراد", columns=8,
                                      height=300, object_fit="cover", allow_preview=False)
            with gr.Row():
                person_pick = gr.Dropdown(choices=person_choices(), value=None,
                                          label="انتخاب شخص")
                person_name = gr.Textbox(label="نام این شخص", placeholder="مثلا: Cristiano Ronaldo")
                person_save = gr.Button("💾 ذخیره نام", variant="primary")
            person_msg = gr.Markdown(elem_classes="rtl")
            person_photos = gr.Gallery(label="عکس‌های این شخص", columns=4, height=560,
                                       object_fit="contain")

            def show_person(person_id):
                person = next((p for p in info_now().people if p.id == person_id), None)
                if person is None:
                    return [], "", ""
                items = []
                for key in person.photos:
                    photo = info_now().photos[key]
                    boxes = tuple((f.box, person.title) for f in person.faces
                                  if f.file_name == key)
                    items.append((annotated(photo.file_path, boxes), Path(key).name))
                message = f"**{html.escape(person.title)}**: {_fa_digits(len(person.photos))} عکس"
                return items, person.name, message

            def pick_card(event: gr.SelectData):
                people = info_now().people
                return people[event.index].id if event.index < len(people) else None

            def save_name(person_id, name):
                if person_id is None:
                    return "ابتدا یک شخص را انتخاب کنید.", gr.update(), gr.update(), gr.update()
                if not (name or "").strip():
                    return "یک نام بنویسید.", gr.update(), gr.update(), gr.update()
                gallery.name_person(int(person_id), name)
                return (f"✅ نام «{html.escape(name.strip())}» ذخیره شد. حالا در تب جستجو با "
                        "این نام هم پیدا می‌شود.", person_cards(),
                        gr.update(choices=person_choices(), value=person_id),
                        gr.update(choices=gallery.names.names()))

            people_cards.select(pick_card, None, person_pick)
            person_pick.change(show_person, person_pick, [person_photos, person_name, person_msg])
            person_save.click(save_name, [person_pick, person_name],
                              [person_msg, people_cards, person_pick, name_box])

            def enroll(image, name):
                if not image or not (name or "").strip():
                    return "یک عکس و یک نام لازم است.", gr.update(), gr.update(), gr.update()
                try:
                    found = gallery.enroll(name, image)
                except Exception as exc:
                    return f"⚠️ {html.escape(str(exc))}", gr.update(), gr.update(), gr.update()
                return (f"✅ «{html.escape(name.strip())}» ثبت شد و در "
                        f"{_fa_digits(found)} عکس پیدا شد.",
                        gr.update(choices=gallery.names.names(), value=name.strip()),
                        person_cards(), gr.update(choices=person_choices()))

            enroll_go.click(enroll, [enroll_image, enroll_name],
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
            browse_pick.change(lambda v: browse(gallery, v), browse_pick,
                               [browse_status, browse_pictures, browse_table])
            app.load(lambda: browse(gallery, "all"), None,
                     [browse_status, browse_pictures, browse_table])

    return app


def browse_choices(gallery: Gallery) -> list[tuple[str, str]]:
    info = gallery.insights
    choices = [(f"همه‌ی عکس‌ها ({_fa_digits(len(info.photos))})", "all")]
    choices += [(f"👤 {p.title} ({_fa_digits(len(p.photos))})", f"person:{p.id}")
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


def browse(gallery: Gallery, choice: str) -> tuple[str, list, list]:
    info = gallery.insights
    kind, _, value = (choice or "all").partition(":")
    if kind == "person":
        person = next(p for p in info.people if str(p.id) == value)
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
        photo = info.photos[key]
        boxes: list[tuple[tuple, str]] = []
        if kind in {"person", "all"}:
            boxes += [(f.box, info.people[f.person - 1].title) for f in photo.faces
                      if f.person and (kind == "all" or str(f.person) == value)]
        if kind in {"animal", "all"}:
            boxes += [(a.box, a.species or a.group_fa) for a in photo.animals
                      if kind == "all" or gi.group_title(a) == value]
        if kind in {"food", "all"}:
            boxes += [(b, gi.dish_name(photo)) for b, _, _ in photo.dishes[:1]]
        tags = "، ".join(info.tags(key))
        try:
            picture = annotated(photo.file_path, tuple(boxes))
            pictures.append((picture, _rtl(f"{tags} – {Path(key).name}")))
        except OSError:
            continue
        rows.append([key, tags, photo.file_path])
    return f"### {html.escape(title)}: {_fa_digits(len(keys))} عکس", pictures, rows


def launch(gallery: Gallery, port: int = 7860, open_browser: bool = True,
           block: bool = True) -> Any:
    app = build_app(gallery)
    return app.launch(
        server_name="127.0.0.1", server_port=port, inbrowser=open_browser,
        allowed_paths=[str(gallery.directory)], prevent_thread_lock=not block,
        **_accepted(app.launch, _styling()),
    )
