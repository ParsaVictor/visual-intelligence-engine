"""What ``run_all.ipynb`` runs: index one gallery, then search it from a small panel.

The models, thresholds and search logic are the package's own, called the same way
``vie index`` and ``vie search`` call them. Only the folder handling and the panel are
added here, so results match the command line.
"""

from __future__ import annotations

import gc
import html
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_dataset

from vie import cli
from vie.config import Config
from vie.logging_setup import configure

MODELS = {
    "final": "google/siglip-so400m-patch14-384",
    "fast": "google/siglip-base-patch16-224",
}

KINDS = [
    ("👤 چهره (با یک عکس از شخص)", "face"),
    ("🦁 حیوان", "animal"),
    ("🍱 غذا", "food"),
    ("🌐 توضیح متنی", "text"),
    ("🎬 لحظه‌ای در ویدیو", "video"),
]
EXAMPLES = {
    "animal": "red panda",
    "food": "ghormeh sabzi",
    "text": "a man with a blue backpack",
    "video": "a goal being scored",
}


class Session:
    """One gallery: its config, its index, and models kept loaded between searches."""

    def __init__(self, gallery: str | Path, model: str = "final", name: str = "") -> None:
        folder = Path(str(gallery).strip().strip('"')).expanduser()
        if not str(gallery).strip():
            raise ValueError("آدرس گالری را در سلول تنظیمات، در GALLERY بنویسید.")
        if not folder.is_dir():
            raise FileNotFoundError(f"این پوشه پیدا نشد: {folder}")
        if model not in MODELS:
            raise ValueError(f"MODEL باید یکی از {sorted(MODELS)} باشد، نه {model!r}")

        self.folder = folder.resolve()
        self.contents = run_dataset.describe(self.folder)
        if not self.contents["images"] and not self.contents["videos"]:
            raise ValueError(f"هیچ عکس یا ویدیویی در این پوشه نیست: {self.folder}")

        slug = re.sub(r"[^\w.-]+", "_", name or self.folder.name)
        self.directory = run_dataset.ROOT / "data" / "runs" / f"{slug}-{model}"
        (self.directory / "reports").mkdir(parents=True, exist_ok=True)
        options = SimpleNamespace(
            folder=self.folder,
            models=run_dataset.ROOT / "data" / "models",
            model=MODELS[model],
            device="auto",
            batch_size=2,
            workers=2,
            no_multi_crop=False,
        )
        self.config_path = run_dataset.write_config(options, self.directory)
        self.config = Config.load(self.config_path)
        self._runtime: Any = None
        configure("WARNING")

        # Point the caches at data/models before anything imports huggingface_hub,
        # and copy rather than symlink: Windows refuses links without Developer Mode.
        from vie.cache import configure_cache

        configure_cache(self.config.model_cache)
        # Once the chosen model is on disk, run fully offline: nothing may wait on the
        # network. A first-ever run, with no model yet, stays online to download it.
        # This must happen before huggingface_hub, transformers or ultralytics import.
        snapshot = (Path(self.config.model_cache) / "huggingface" / "hub"
                    / f"models--{MODELS[model].replace('/', '--')}")
        if snapshot.is_dir():
            from vie.gallery.offline import go_offline

            go_offline()
        from huggingface_hub import file_download

        file_download.are_symlinks_supported = lambda cache_dir=None: False

    def describe(self) -> None:
        print(f"گالری: {self.folder}")
        print(f"مدل: {self.config.models['vision_language']}")
        print(f"عکس: {self.contents['images']}   ویدیو: {self.contents['videos']}   "
              f"حجم: {self.contents['megabytes']} MB")
        if self.contents["ignored_file_types"]:
            print(f"نادیده گرفته می‌شود: {self.contents['ignored_file_types']}")
        print(f"خروجی‌ها: {self.directory}")

    def runtime(self, _config: Config | None = None) -> Any:
        if self._runtime is None:
            self._runtime = cli._load_runtime(self.config)
        return self._runtime

    def release_models(self) -> None:
        """Indexing holds five models; searching usually needs one or two."""
        self._runtime = None
        gc.collect()

    # ---- indexing -----------------------------------------------------

    def index_images(self) -> Any:
        if not self.contents["images"]:
            print("عکسی برای ایندکس نیست.")
            return None

        from tqdm.auto import tqdm

        from vie.adapters import build_detectors
        from vie.models import SpeciesClassifier
        from vie.pipeline import build_index

        config = self.config
        bundle, _, encoder = self.runtime()
        detectors = build_detectors(bundle, config)
        bar = tqdm(total=self.contents["images"], desc="ایندکس عکس‌ها", unit="عکس")

        def progress(stats: Any) -> None:
            bar.total = max(1, stats.scanned - stats.skipped)
            bar.n = stats.indexed + stats.failed
            bar.refresh()

        started = time.perf_counter()
        stats = build_index(
            config,
            loader=detectors.loader,
            size_of=detectors.size_of,
            face_detector=detectors.face,
            animal_detector=detectors.animal,
            food_detector=detectors.food,
            embed_batch=encoder.embed_batch,
            food_gate=lambda vectors: encoder.food_gate_scores(vectors, config),
            classify_crops=SpeciesClassifier(bundle).classify,
            crop=detectors.crop,
            class_names=lambda: bundle.class_names,
            on_progress=progress,
        )
        bar.close()
        minutes = (time.perf_counter() - started) / 60
        print(f"{stats.summary()}   ({minutes:.1f} دقیقه)")
        if stats.failed:
            print(f"⚠️ {stats.failed} فایل خوانده نشد؛ بقیه سالم ایندکس شدند.")
        return stats

    def index_videos(self) -> Any:
        if not self.contents["videos"]:
            print("ویدیویی برای ایندکس نیست.")
            return None

        from vie.video.library import Stages, build_engine, index_videos

        bundle, _, encoder = self.runtime()
        engine = build_engine(self.config, bundle, encoder, Stages.from_config(self.config))
        stats = index_videos(self.config, engine)
        print(stats.summary())
        return stats

    def doctor(self) -> int:
        from vie.doctor import run_doctor

        return run_doctor(self.config)

    def open_gallery(self, port: int = 7860) -> Any:
        """Prepare the Smart Gallery and open it in the browser."""
        from IPython.display import HTML, display

        from vie.gallery.app import launch
        from vie.gallery.engine import Gallery

        self.release_models()
        gallery = Gallery(self.config)
        print("در حال آماده‌سازی گالری هوشمند (بار اول چند دقیقه طول می‌کشد)...")
        started = time.perf_counter()
        gallery.warm_up()
        info = gallery.insights
        print(f"آماده شد در {time.perf_counter() - started:.0f} ثانیه: "
              f"{len(info.photos)} عکس، {len(info.people)} فرد، "
              f"{len(info.with_animals())} عکس حیوان، {len(info.with_food())} عکس غذا")
        launch(gallery, port=port, open_browser=True, block=False)
        url = f"http://127.0.0.1:{port}"
        display(HTML(f'<p dir="rtl" style="font-size:16px">گالری هوشمند باز شد: '
                     f'<a href="{url}" target="_blank">{url}</a></p>'))
        return gallery

    # ---- search -------------------------------------------------------

    def search_images(self, kind: str, query: str = "", image: Any = None, top_k: int = 12) -> list:
        from vie.query import run_search

        request = SimpleNamespace(
            kind=kind, query=query, image=Path(image) if image else None, top_k=top_k
        )
        return run_search(request, self.config, self.runtime)

    def search_video(self, query: str = "", image: Any = None, top_k: int = 12) -> list:
        from vie.query import query_face_embedding
        from vie.video.store import VideoStore

        store = VideoStore(self.config)
        store.initialise()
        bundle, _, encoder = self.runtime()
        with store.connect() as conn:
            store.index.check_embedding_model(conn, self.config.models["vision_language"])
            if image:
                embedding = query_face_embedding(Path(image), self.config, bundle)
                return store.search_face(conn, embedding, top_k)
            moments = store.search_speech(conn, query, top_k)
            return moments + store.search_visual(conn, encoder.encode_text(query), top_k=top_k)

    def results_page(self, kind: str, query: str = "", image: Any = None, top_k: int = 12) -> tuple:
        """Run a search and return ``(html, saved report path, result count)``."""
        from vie.report.search import render_search_report

        shown = str(image) if image else query
        stamp = time.strftime("%H%M%S")
        safe = run_dataset.report_path(self.directory, kind, f"{Path(shown).stem or shown}-{stamp}")
        if kind == "video":
            from vie.report.theme import document
            from vie.video.understand import format_clock

            moments = self.search_video(query, image, top_k)
            rows = "".join(
                f"<tr><td>{html.escape(m.video_key)}</td><td>{format_clock(m.start)}</td>"
                f"<td>{format_clock(m.end)}</td><td>{m.score:.3f}</td>"
                f"<td>{html.escape(m.text or m.kind)}</td></tr>"
                for m in moments
            )
            page = document(
                f"Video search: {shown}",
                f"<h1>{html.escape(Path(shown).name if image else shown)}</h1><table>"
                "<tr><th>video</th><th>from</th><th>to</th><th>score</th><th>matched on</th></tr>"
                f"{rows}</table>",
            )
            (safe.with_suffix(".json")).write_text(
                json.dumps([asdict(m) for m in moments], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            count = len(moments)
        else:
            matches = self.search_images(kind, query, image, top_k)
            page = render_search_report(matches, kind=kind, query=shown)
            count = len(matches)
        safe.write_text(page, encoding="utf-8")
        return page, safe, count

    def panel(self) -> Any:
        """The last step of the notebook: choose what to look for, and see the results."""
        import ipywidgets as widgets
        from IPython.display import HTML, display

        kind = widgets.Dropdown(options=KINDS, value="text", description="دنبال چی؟")
        query = widgets.Text(description="جستجو:", layout=widgets.Layout(width="520px"))
        image = widgets.Text(description="مسیر عکس:", layout=widgets.Layout(width="520px"),
                             placeholder=r"D:\photos\person.jpg")
        top_k = widgets.IntSlider(value=12, min=1, max=60, description="تعداد نتیجه:")
        button = widgets.Button(description="جستجو", button_style="primary", icon="search")
        output = widgets.Output()

        def refresh(_: Any = None) -> None:
            current = kind.value
            image.layout.display = "" if current in {"face", "video"} else "none"
            query.layout.display = "none" if current == "face" else ""
            query.placeholder = EXAMPLES.get(current, "")

        def search(_: Any) -> None:
            button.disabled = True
            output.clear_output()
            with output:
                print("در حال جستجو... (بار اول هر نوع جستجو، بارگذاری مدل کمی طول می‌کشد)")
            started = time.perf_counter()
            try:
                page, saved, count = self.results_page(
                    kind.value, query.value.strip(), image.value.strip().strip('"') or None,
                    top_k.value,
                )
                output.clear_output()
                with output:
                    print(f"{count} نتیجه در {time.perf_counter() - started:.1f} ثانیه")
                    print(f"گزارش ذخیره شد: {saved}")
                    frame = html.escape(page, quote=True)
                    display(HTML(f'<iframe srcdoc="{frame}" style="width:100%;height:760px;'
                                 f'border:0;background:#fff"></iframe>'))
            except Exception as exc:  # shown in the panel, so it can simply be tried again
                output.clear_output()
                with output:
                    print(f"خطا: {exc}")
            finally:
                button.disabled = False

        kind.observe(refresh, names="value")
        button.on_click(search)
        refresh()
        box = widgets.VBox([kind, query, image, top_k, button, output])
        display(box)
        return box
