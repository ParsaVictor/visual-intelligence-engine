"""Search modes of the Smart Gallery, on top of the package's own search paths.

Face, species, dish and free-text search go through :func:`vie.query.run_search`
unchanged, so they return what ``vie search`` returns. Two modes are new:

* **by name**: faces the operator named, searched like a face query;
* **similar photos**: one photo against the stored whole-image vectors.

Crop vectors for the species and dish paths are cached on disk. Those paths encode
every candidate crop on each query, which on a CPU took two minutes per dish query.
The crops and the scoring are identical, so the results are too; only the repeat
work is gone.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vie.config import Config
from vie.database import Index
from vie.embeddings import from_blob, to_blob
from vie.gallery import insights as gi
from vie.logging_setup import get_logger
from vie.scoring import Match, best_per_image

log = get_logger("gallery")

Box = tuple[int, int, int, int]

REGIONS = {
    "whole": "کل تصویر", "center": "مرکز تصویر", "tl": "بالا چپ", "tr": "بالا راست",
    "bl": "پایین چپ", "br": "پایین راست",
}
MODES = {
    "face": "👤 چهره (با عکس)",
    "name": "🏷️ شخص (با نام)",
    "animal": "🐾 حیوان",
    "food": "🍽️ غذا",
    "text": "💡 مفهوم / توضیح آزاد",
    "similar": "🖼️ عکس‌های مشابه",
}


@dataclass
class Result:
    mode: str
    query: str
    matches: list[Match]
    seconds: float
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def prepare_model_cache(config: Config) -> None:
    """Point every model cache at the configured folder before anything loads.

    ``huggingface_hub`` reads its cache location once, on import, so this has to run
    first. Windows also refuses symlinks without Developer Mode, and the hub's own
    probe for that can pass while the real link still fails, so files are copied.
    """
    from vie.cache import configure_cache

    configure_cache(config.model_cache)
    from huggingface_hub import file_download

    file_download.are_symlinks_supported = lambda cache_dir=None: False


class CachedScorer:
    """``CropScorer`` with every crop vector kept on disk between queries."""

    def __init__(self, encoder: Any, path: Path) -> None:
        self.encoder = encoder
        self.path = Path(path)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS crops (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def key(crop: Any) -> str:
        digest = hashlib.blake2b(crop.tobytes(), digest_size=16)
        digest.update(f"{crop.size}{crop.mode}".encode())
        return digest.hexdigest()

    def vectors(self, crops: list[Any]) -> np.ndarray:
        keys = [self.key(c) for c in crops]
        with self._lock, self._connect() as conn:
            found = {}
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                marks = ",".join("?" for _ in chunk)
                found.update(
                    (k, from_blob(v)) for k, v in conn.execute(
                        f"SELECT key, vector FROM crops WHERE key IN ({marks})", chunk
                    )
                )
            missing = [i for i, k in enumerate(keys) if k not in found]
            if missing:
                encoded = self.encoder.encode_images([crops[i] for i in missing])
                for i, vector in zip(missing, encoded, strict=True):
                    found[keys[i]] = vector
                    conn.execute("INSERT OR REPLACE INTO crops VALUES (?, ?)",
                                 (keys[i], to_blob(vector)))
        return np.vstack([found[k] for k in keys])

    def score(self, crops: Any, prompts: Any) -> list[list[float]]:
        if not crops:
            return []
        images = self.vectors(list(crops))
        texts = self.encoder.encode_texts(list(prompts))
        return self.encoder.logits_from_similarity(images @ texts.T).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.encoder, name)


class Gallery:
    """One indexed gallery, its models, and every way to search it."""

    def __init__(self, config: Config) -> None:
        prepare_model_cache(config)
        self.config = config
        self.directory = Path(config.database_path).parent
        self.names = gi.NameBook(self.directory / "people.json", config.face.match_threshold)
        self._runtime: Any = None
        self._insights: gi.Insights | None = None
        self._lock = threading.RLock()

    # ---- models and overview -------------------------------------------

    def runtime(self, _config: Config | None = None) -> Any:
        with self._lock:
            if self._runtime is None:
                from vie.models import VisionLanguageEncoder, load_all

                bundle = load_all(self.config)
                encoder = VisionLanguageEncoder(bundle)
                scorer = CachedScorer(encoder, self.directory / "crop_cache.db")
                self._runtime = (bundle, scorer, encoder)
            return self._runtime

    @property
    def insights(self) -> gi.Insights:
        with self._lock:
            if self._insights is None:
                started = time.perf_counter()
                self._insights = gi.build(
                    self.config, self.runtime()[2], self.names, crop_vectors=self._face_vectors
                )
                log.info("gallery overview built in %.1f s", time.perf_counter() - started)
            return self._insights

    def _face_vectors(self, items: list[Any], kind: str) -> np.ndarray:
        """Vision-language vectors of face or animal crops, cached like every other crop."""
        index = Index(self.config.database_path)
        with index.connect() as conn:
            paths = dict(conn.execute("SELECT file_name, file_path FROM gallery_meta"))
        crop = face_crop if kind == "face" else box_crop
        crops = [crop(paths[item.file_name], item.box, 384) for item in items]
        return self.runtime()[1].vectors(crops)

    def refresh(self) -> gi.Insights:
        with self._lock:
            self._insights = None
            return self.insights

    def warm_up(self) -> None:
        """Pay every first-query cost now: models, the overview, the crop cache."""
        started = time.perf_counter()
        _ = self.insights
        _ = self.runtime()[0].face_app
        for mode, query in (("food", "food"), ("animal", "an animal")):
            try:
                self.search(mode, query)
            except Exception as exc:  # warming must never stop the gallery opening
                log.warning("warm-up %s failed: %s", mode, exc)
        log.info("gallery ready in %.1f s", time.perf_counter() - started)

    # ---- search ---------------------------------------------------------

    def _run(self, kind: str, query: str = "", image: Path | None = None,
             top_k: int | None = None) -> list[Match]:
        from vie.query import run_search

        request = SimpleNamespace(kind=kind, query=query, image=image, top_k=top_k)
        return run_search(request, self.config, self.runtime)

    def search(self, mode: str, query: str = "", image: str | Path | None = None,
               top_k: int = 24) -> Result:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        query = (query or "").strip()
        path = Path(image) if image else None
        if mode in {"face", "similar"} and path is None:
            raise ValueError("یک عکس انتخاب کنید")
        if mode not in {"face", "similar"} and not query:
            raise ValueError("عبارت جستجو را بنویسید")

        started = time.perf_counter()
        note = ""
        extra: dict[str, Any] = {}
        if mode == "face":
            matches = self._run("face", image=path, top_k=top_k)
            shown = path.name
        elif mode == "name":
            matches = self.search_name(query, top_k)
            shown = query
        elif mode == "similar":
            matches = self.search_similar(path, top_k)
            shown = path.name
        elif mode == "animal":
            note = self._animal_route(query)
            info = self.insights
            found = self._run("animal", query=query)
            matches = [m for m in found if not info.is_rejected_animal(m.file_name, m.box)]
            if len(matches) < len(found):
                note += (f" ({len(found) - len(matches)} کادر کنار گذاشته شد چون در بررسی دوم "
                         f"حیوان نبود)")
            group = info.animals_in_group(query)
            if group:
                # A box the second reading moved to another group (a calf ImageNet
                # called a dog at 27%) must not answer for this one.
                matches = [
                    m for m in matches
                    if (animal := info.animal_at(m.file_name, m.box)) is None
                    or animal.group.lower() == query.lower()
                ]
                # "dog" through the package's word list misses breeds such as vizsla;
                # the overview groups by ImageNet id range. Both are answered, so the
                # search finds every dog the overview counted.
                matches = best_per_image(matches + [
                    Match(a.file_name, self.insights.photos[a.file_name].file_path,
                          a.probability, box=a.box, label=a.species)
                    for a in group
                ])
                note = f"گروه «{query}»: هر گونه‌ای از این گروه که در ایندکس ثبت شده"
            matches = matches[:top_k]
            shown = query
        else:
            matches = self._run(mode, query=query, top_k=None if mode == "food" else top_k)
            if mode == "food":
                matches, dropped = self._only_food(matches)
                labelled = self.insights.photos_with_dish(query)
                if labelled:
                    # The overview's dish label sees the whole plate, which finds dishes
                    # the crop path misses (two of three plates of kebab).
                    photos = self.insights.photos
                    matches = best_per_image(matches + [
                        Match(key, photos[key].file_path, share, label=query)
                        for key, share in labelled
                    ])
                matches = matches[:top_k]
                if dropped:
                    note = (f"{dropped} عکس کنار گذاشته شد چون نه کادر غذا داشت و نه موضوع "
                            f"اصلی‌اش غذا بود")
            if mode == "text":
                matches, hidden = self._relevant_text(matches)
                if hidden:
                    note = (f"{hidden} عکس کم‌ارتباط پنهان شد (احتمال تطابق کمتر از "
                            f"{self._text_floor() * 100:g}٪)")
            shown = query
        return Result(mode, shown, matches, time.perf_counter() - started, note, extra)

    def _text_floor(self) -> float:
        """The calibrated probability below which a free-text result is noise.

        Measured on a real gallery with so400m: true hits read 1.7% to 79%, and the
        rest of the ranking 0.6% or less, mostly 0.0%. The base model is less sure of
        itself; a true "dog" read 0.2% there, so its floor is lower.
        """
        return 0.01 if "so400m" in self.config.models["vision_language"] else 0.002

    def _relevant_text(self, matches: list[Match]) -> tuple[list[Match], int]:
        """Hide the tail of a free-text ranking: it always has ``top_k`` entries."""
        floor = self._text_floor()
        kept = [m for m in matches if m.confidence is None or m.confidence >= floor]
        return kept, len(matches) - len(kept)

    def _only_food(self, matches: list[Match]) -> tuple[list[Match], int]:
        """Drop dish matches on photos that are plainly not of food.

        A photo the permissive food gate admitted with no food box is scored as a whole
        frame, against rivals that are other foods and an empty plate. Neither rival
        says "this is not food at all", so a cat could score 99% as sushi. A photo
        stays when the detector saw a dish or its main subject reads as food.
        """
        photos = self.insights.photos
        kept = [
            m for m in matches
            if m.file_name not in photos
            or gi.is_food_photo(photos[m.file_name])
            or photos[m.file_name].subject is None
        ]
        return kept, len(matches) - len(kept)

    def _animal_route(self, query: str) -> str:
        from vie.search.animal import choose_route

        index = Index(self.config.database_path)
        with index.connect() as conn:
            vocabulary = index.classifier_vocabulary(conn)
        if not vocabulary:
            return ""
        route = choose_route(query, vocabulary)
        if route.engine == "classifier":
            return (f"پاسخ مستقیم از ایندکس (طبقه‌بند گونه، {len(route.resolution.class_ids)} "
                    f"کلاس ImageNet)، بدون اجرای مدل")
        return "این گونه در واژگان طبقه‌بند نیست؛ با SigLIP روی کادرهای حیوان جستجو شد"

    def search_name(self, name: str, top_k: int) -> list[Match]:
        from vie.query import _ann_cache, _file_paths
        from vie.search import face as face_search
        from vie.vectorstore import VectorCatalog

        vectors = self.names.vectors(name)
        if not len(vectors):
            known = "، ".join(self.names.names()) or "هنوز کسی نام‌گذاری نشده"
            raise ValueError(f"نامی به اسم «{name}» ثبت نشده. نام‌های ثبت‌شده: {known}")
        index = Index(self.config.database_path)
        with index.connect() as conn:
            catalog = VectorCatalog.open(
                conn, "face_embeddings", self.config.search, _ann_cache(self.config)
            )
            paths = _file_paths(conn)
            scales = index.scales(conn)
        matches: list[Match] = []
        for vector in vectors:
            matches += face_search.rank_catalog(vector, catalog, paths, self.config)
        matches = [
            replace(m, box=gi.to_original(m.box, scales.get(m.file_name, 1.0)), label=name)
            for m in best_per_image(matches)
        ]
        return matches[:top_k]

    def search_similar(self, image: Path, top_k: int) -> list[Match]:
        from PIL import Image, ImageOps

        if not image.is_file():
            raise FileNotFoundError(f"عکس پیدا نشد: {image}")
        encoder = self.runtime()[2]
        with Image.open(image) as handle:
            picture = ImageOps.exif_transpose(handle).convert("RGB")
        query = encoder.encode_images([picture])[0]
        vectors = gi.whole_vectors(self.config)
        if not vectors:
            return []
        names = list(vectors)
        scores = np.vstack([vectors[n] for n in names]) @ query
        photos = self.insights.photos
        ranked = sorted(zip(names, scores, strict=True), key=lambda kv: -kv[1])[:top_k]
        return [
            Match(name, photos[name].file_path if name in photos else name, float(score),
                  label="شباهت کلی تصویر")
            for name, score in ranked
        ]

    # ---- naming ---------------------------------------------------------

    def name_person(self, person_id: int, name: str) -> None:
        person = next((p for p in self.insights.people if p.id == person_id), None)
        if person is None:
            raise ValueError(f"شخص {person_id} پیدا نشد")
        self.names.add(name, [f.embedding for f in person.faces])
        self.names.label(self.insights.people)

    def enroll(self, name: str, image: str | Path) -> int:
        """Add a named person from a photo of them; returns how many photos match."""
        from vie.query import query_face_embedding

        embedding = query_face_embedding(Path(image), self.config, self.runtime()[0])
        self.names.add(name, [embedding])
        self.names.label(self.insights.people)
        return len(self.search_name(name.strip(), 10_000))

    def forget(self, name: str) -> None:
        self.names.remove(name)
        self.names.label(self.insights.people)

    # ---- describing results --------------------------------------------

    def describe(self, match: Match, mode: str) -> str:
        """What matched, in words: the person, the species, the region."""
        info = self.insights
        if mode in {"face", "name"}:
            person = info.person_of(match.file_name, match.box)
            return person.title if person else (match.label or "چهره")
        if mode == "text":
            return f"بهترین ناحیه: {REGIONS.get(match.label or '', match.label or '')}"
        return match.label or ""

    def score_text(self, match: Match, mode: str) -> str:
        value = match.display_score
        if mode in {"face", "name"}:
            return f"{value:.2f} شباهت چهره"
        if mode == "similar":
            return f"{value * 100:.0f}% شباهت"
        if mode == "text":
            return f"{value * 100:.1f}% اطمینان"
        return f"{value * 100:.0f}% اطمینان"


# ---- pictures -------------------------------------------------------------


def _open(path: str, max_side: int) -> tuple[Any, tuple[int, int]]:
    """Decode as OpenCV does, with EXIF orientation applied, so stored boxes line up.

    Returns the picture and the full upright size the boxes refer to. JPEG draft mode
    decodes at a reduced scale directly, which is most of the cost of a 4K wallpaper.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        width, height = handle.size
        if handle.getexif().get(0x0112, 1) in (5, 6, 7, 8):
            width, height = height, width
        handle.draft("RGB", (max_side, max_side))
        image = ImageOps.exif_transpose(handle).convert("RGB")
    return image, (width, height)


@lru_cache(maxsize=512)
def annotated(path: str, boxes: tuple[tuple[Box, str], ...] = (), max_side: int = 720) -> Any:
    """A display copy of a photo with its boxes drawn and labelled."""
    from PIL import ImageDraw, ImageFont

    image, original = _open(path, max_side)
    image.thumbnail((max_side, max_side))
    scale = image.size[0] / original[0] if original[0] else 1.0
    if not boxes:
        return image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("tahoma.ttf", max(13, image.size[0] // 40))
    except OSError:
        font = ImageFont.load_default()
    width = max(2, image.size[0] // 200)
    for box, label in boxes:
        x1, y1, x2, y2 = (v * scale for v in box)
        draw.rectangle((x1, y1, x2, y2), outline="#ffb454", width=width)
        if label:
            text = _bidi(label)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            ty = y1 - (bottom - top) - 6 if y1 > (bottom - top) + 8 else y1 + 2
            draw.rectangle((x1, ty, x1 + right - left + 8, ty + bottom - top + 6), fill="#ffb454")
            draw.text((x1 + 4, ty + 1), text, fill="#1c1405", font=font)
    return image


@lru_cache(maxsize=256)
def face_crop(path: str, box: Box, size: int = 168) -> Any:
    """A square portrait around a face, with room for the hair and chin."""
    image, original = _open(path, 1600)
    scale = image.size[0] / original[0] if original[0] else 1.0
    x1, y1, x2, y2 = (v * scale for v in box)
    side = max(x2 - x1, y2 - y1) * 1.6
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, cx - side / 2)
    top = max(0, cy - side / 2)
    crop = image.crop((left, top, min(image.size[0], left + side), min(image.size[1], top + side)))
    crop.thumbnail((size, size))
    return crop


@lru_cache(maxsize=256)
def box_crop(path: str, box: Box, size: int = 384, padding: float = 0.15) -> Any:
    """A detected box with some context around it, as the species path crops."""
    image, original = _open(path, 1600)
    scale = image.size[0] / original[0] if original[0] else 1.0
    x1, y1, x2, y2 = (v * scale for v in box)
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    crop = image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y),
                       min(image.size[0], x2 + pad_x), min(image.size[1], y2 + pad_y)))
    crop.thumbnail((size, size))
    return crop


def _bidi(text: str) -> str:
    """Pillow draws text left to right; reverse Persian runs so they read correctly."""
    if not any("؀" <= ch <= "ۿ" for ch in text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text[::-1]
