"""The archive at a glance: people, animals, dishes and subjects, from the index.

Nothing here re-runs a detector. Faces, animal boxes, species predictions and food
boxes were stored at index time. Subjects, settings and dishes are zero-shot labels
scored against the stored whole-image vectors, so they cost one text encoding per
label and no image inference at all.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vie.config import Config
from vie.database import Index, parse_box
from vie.embeddings import from_blob
from vie.scoring import softmax
from vie.species import GENERIC_TERMS
from vie.video.people import Sighting, cluster_faces

Box = tuple[int, int, int, int]

#: ImageNet-1k orders its classes so that 0-397 are animals; the rest are objects.
LAST_ANIMAL_CLASS = 397

#: COCO ids of actual dishes. Cups, bowls and cutlery prove a table, not a meal.
COCO_DISHES = {
    46: "موز", 47: "سیب", 48: "ساندویچ", 49: "پرتقال", 50: "بروکلی",
    51: "هویج", 52: "هات‌داگ", 53: "پیتزا", 54: "دونات", 55: "کیک",
}

#: (prompt label, Persian name). Each group is scored as a softmax over its own
#: labels, as video labels are, so a group must be a sensible set of alternatives.
SUBJECTS: list[tuple[str, str]] = [
    ("a portrait of a person", "پرتره و چهره"),
    ("a group of people", "گروهی از افراد"),
    ("an athlete playing sport", "ورزش و ورزشکار"),
    ("an animal", "حیوان"),
    ("a plate of food or a meal", "غذا"),
    ("a natural landscape", "طبیعت و منظره"),
    ("a sunset or sunrise", "غروب و طلوع"),
    ("fire or flames", "آتش"),
    ("a castle or historic building", "قلعه و بنای تاریخی"),
    ("a city street or buildings", "شهر و ساختمان"),
    ("a car or other vehicle", "خودرو و وسیله نقلیه"),
    ("a cartoon, drawing or illustration", "کارتون و تصویرسازی"),
    ("a screenshot, text or a document", "متن و سند"),
    ("a product or everyday object", "شیء و محصول"),
]
SETTINGS: list[tuple[str, str]] = [
    ("indoors", "داخل ساختمان"),
    ("a forest or green nature", "جنگل و سبزه"),
    ("autumn leaves", "پاییز"),
    ("grass or a farm field", "چمنزار و مزرعه"),
    ("mountains", "کوهستان"),
    ("a beach or the sea", "ساحل و دریا"),
    ("snow and ice", "برف و یخ"),
    ("a desert", "بیابان"),
    ("a city", "شهر"),
    ("a stadium or sports field", "ورزشگاه"),
    ("the sky", "آسمان"),
    ("a plain studio background", "پس‌زمینه ساده"),
]
DISHES: list[tuple[str, str]] = [
    ("pizza", "پیتزا"), ("a burger", "همبرگر"), ("kebab", "کباب"),
    ("rice with stew", "برنج و خورش"), ("ghormeh sabzi", "قورمه‌سبزی"), ("pasta", "پاستا"),
    ("sushi", "سوشی"), ("salad", "سالاد"), ("soup", "سوپ"), ("bread", "نان"),
    ("a sandwich", "ساندویچ"), ("a hot dog", "هات‌داگ"),
    ("a baked potato", "سیب‌زمینی تنوری"), ("barbecue or grilled meat", "کباب و گوشت کبابی"),
    ("fried chicken", "مرغ سوخاری"), ("steak", "استیک"),
    ("french fries", "سیب‌زمینی سرخ‌کرده"), ("cake or dessert", "کیک و دسر"),
    ("ice cream", "بستنی"), ("fresh fruit", "میوه"), ("vegetables", "سبزیجات"),
    ("breakfast", "صبحانه"), ("a drink or coffee", "نوشیدنی و قهوه"),
    ("sauce or condiments", "سس و چاشنی"),
]
FOOD_SUBJECT = "a plate of food or a meal"
assert FOOD_SUBJECT in dict(SUBJECTS)
LABEL_SHARE = 0.40   # the same bar video labels use: share against the group's rivals
PROMPT = "a photo of {label}"

#: ImageNet-1k keeps related species in contiguous id ranges; ~120 dog breeds sit in
#: 151-268. Grouping by range, not by name, puts "vizsla" under dogs as well.
ANIMAL_GROUPS: list[tuple[int, int, str, str]] = [
    (151, 268, "dog", "سگ"), (269, 280, "wolf or fox", "گرگ و روباه"),
    (281, 285, "cat", "گربه"), (286, 293, "big cat", "گربه‌سان بزرگ"),
    (294, 297, "bear", "خرس"), (7, 24, "bird", "پرنده"), (80, 100, "bird", "پرنده"),
    (127, 146, "bird", "پرنده"), (365, 384, "monkey", "میمون و نخستی‌سانان"),
    (101, 101, "elephant", "فیل"), (385, 386, "elephant", "فیل"),
    (387, 387, "red panda", "پاندای قرمز"), (388, 388, "giant panda", "پاندا"),
    (0, 6, "fish", "ماهی"), (389, 397, "fish", "ماهی"), (52, 68, "snake", "مار"),
    (300, 326, "insect", "حشره"), (330, 332, "rabbit", "خرگوش"),
    (339, 339, "horse", "اسب"), (340, 340, "zebra", "گورخر"),
    (345, 347, "cow", "گاو و گاومیش"), (348, 350, "sheep or goat", "گوسفند و بز"),
    (147, 149, "whale or dolphin", "نهنگ و دلفین"), (351, 353, "deer", "آهو و گوزن"),
    (354, 354, "camel", "شتر"), (102, 102, "echidna", "خارپشت"),
]
UNKNOWN_ANIMAL = ("animal", "حیوان (گونه نامشخص)")

#: Open-vocabulary names for animals the classifier was unsure of:
#: (prompt, english group, persian group). The last two are rivals: a box that reads
#: as a person or an object was a detector false positive, not an unknown animal.
ANIMAL_PROMPTS: list[tuple[str, str, str]] = [
    ("a dog", "dog", "سگ"), ("a cat", "cat", "گربه"),
    ("a lion", "big cat", "گربه‌سان بزرگ"), ("a tiger", "big cat", "گربه‌سان بزرگ"),
    ("a leopard or cheetah", "big cat", "گربه‌سان بزرگ"), ("a bear", "bear", "خرس"),
    ("a giant panda", "giant panda", "پاندا"), ("a red panda", "red panda", "پاندای قرمز"),
    ("a wolf or a fox", "wolf or fox", "گرگ و روباه"), ("a bird", "bird", "پرنده"),
    ("an eagle or a hawk", "bird", "پرنده"), ("an owl", "bird", "پرنده"),
    ("a parrot", "bird", "پرنده"), ("a horse", "horse", "اسب"), ("a zebra", "zebra", "گورخر"),
    ("a cow or a calf", "cow", "گاو و گاومیش"), ("a raccoon", "raccoon", "راکون"),
    ("a whale or a dolphin", "whale or dolphin", "نهنگ و دلفین"),
    ("a sheep or a goat", "sheep or goat", "گوسفند و بز"),
    ("a deer", "deer", "آهو و گوزن"), ("an elephant", "elephant", "فیل"),
    ("a giraffe", "giraffe", "زرافه"), ("a camel", "camel", "شتر"),
    ("a monkey", "monkey", "میمون و نخستی‌سانان"), ("a rabbit", "rabbit", "خرگوش"),
    ("a squirrel", "squirrel", "سنجاب"), ("a fish", "fish", "ماهی"), ("a snake", "snake", "مار"),
    ("a turtle", "turtle", "لاک‌پشت"), ("an insect", "insect", "حشره"),
    ("a person", "", ""), ("an object or a vehicle", "", ""), ("food or a dish", "", ""),
]
ANIMAL_SHARE = 0.50
#: Below this, a classifier species is a guess the vision-language model may overrule.
CLASSIFIER_SURE = 0.50

#: Rivals for confirming that a detected face is a person. The face detector is kept
#: permissive on purpose (see the config), so it also fires on animals and objects.
#: Counting people needs the stricter reading; face search keeps every detection.
FACE_PROMPTS = [
    "a photo of a human face",
    "a photo of an animal",
    "a photo of food",
    "a photo of an object",
    "a cartoon or a drawing",
]
FACE_SHARE = 0.50


@dataclass
class Face:
    file_name: str
    box: Box                 # original image coordinates
    embedding: np.ndarray
    person: int = 0
    verified: bool = True    # confirmed to be a human face; see FACE_PROMPTS


@dataclass
class Animal:
    file_name: str
    box: Box
    species: str             # ImageNet name, or "" when no class was confident
    group: str               # English group, searchable: "dog", "giant panda"
    group_fa: str
    probability: float
    detector_confidence: float
    class_id: int = -1

    @property
    def display(self) -> str:
        return f"{self.group_fa} ({self.species})" if self.species else self.group_fa


@dataclass
class Photo:
    file_name: str
    file_path: str
    scale: float
    faces: list[Face] = field(default_factory=list)
    animals: list[Animal] = field(default_factory=list)
    dishes: list[tuple[Box, str, float]] = field(default_factory=list)  # detector
    subject: tuple[str, str, float] | None = None    # (english, persian, share)
    setting: tuple[str, str, float] | None = None
    dish: tuple[str, str, float] | None = None
    food_gate: bool = False


@dataclass
class Person:
    id: int
    faces: list[Face]
    centroid: np.ndarray
    name: str = ""

    @property
    def title(self) -> str:
        return self.name or f"شخص {fa_digits(self.id)}"

    @property
    def photos(self) -> list[str]:
        return list(dict.fromkeys(f.file_name for f in self.faces))

    @property
    def best_face(self) -> Face:
        def area(face: Face) -> int:
            x1, y1, x2, y2 = face.box
            return (x2 - x1) * (y2 - y1)

        return max(self.faces, key=area)


@dataclass
class Insights:
    photos: dict[str, Photo]
    people: list[Person]
    model: str
    #: Animal boxes the second reading found to be something else.
    rejected_animals: list[Animal] = field(default_factory=list)

    def is_rejected_animal(self, file_name: str, box: Box | None) -> bool:
        return box is not None and any(
            a.file_name == file_name and _same_object(a.box, box)
            for a in self.rejected_animals
        )

    def animal_at(self, file_name: str, box: Box | None) -> Animal | None:
        if box is None or file_name not in self.photos:
            return None
        return next(
            (a for a in self.photos[file_name].animals if _same_object(a.box, box)), None
        )

    def photos_with_dish(self, query: str) -> list[tuple[str, float]]:
        """Photos whose dish label is this query, as the overview labelled them."""
        wanted = re.sub(r"^(a|an|the)\s+", "", query.strip().lower())
        found = []
        for key, photo in self.photos.items():
            if photo.dish and is_food_photo(photo):
                label = re.sub(r"^(a|an|the)\s+", "", photo.dish[0].lower())
                if wanted == label or wanted in label.split(" or "):
                    found.append((key, photo.dish[2]))
        return found

    # ---- groupings ------------------------------------------------------

    @property
    def face_count(self) -> int:
        return sum(1 for p in self.photos.values() for f in p.faces if f.verified)

    def with_people(self) -> list[str]:
        return [k for k, p in self.photos.items() if any(f.verified for f in p.faces)]

    def with_animals(self) -> list[str]:
        return [k for k, p in self.photos.items() if p.animals]

    def with_food(self) -> list[str]:
        return [k for k, p in self.photos.items() if is_food_photo(p)]

    def species(self) -> dict[str, list[str]]:
        """``"Persian (english)"`` animal group -> photos, most common first."""
        found: dict[str, list[str]] = {}
        for key, photo in self.photos.items():
            for group in dict.fromkeys(group_title(a) for a in photo.animals):
                found.setdefault(group, []).append(key)
        return dict(sorted(found.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    def species_names(self) -> list[str]:
        """English names a species search understands: groups first, then species."""
        groups = [a.group for p in self.photos.values() for a in p.animals if a.species]
        names = [a.species for p in self.photos.values() for a in p.animals if a.species]
        return list(dict.fromkeys([*groups, *names]))

    def animals_in_group(self, group: str) -> list[Animal]:
        """Every stored animal of a group, as the overview counted them."""
        wanted = group.strip().lower()
        return [
            a for p in self.photos.values() for a in p.animals
            if a.species and a.group.lower() == wanted
        ]

    def dishes(self) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for key in self.with_food():
            found.setdefault(dish_name(self.photos[key]), []).append(key)
        return dict(sorted(found.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    def by_label(self, attribute: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for key, photo in self.photos.items():
            value = getattr(photo, attribute)
            if value:
                found.setdefault(value[1], []).append(key)
        return dict(sorted(found.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    def person_of(self, file_name: str, box: Box | None) -> Person | None:
        if box is None:
            return None
        for face in self.photos.get(file_name, Photo(file_name, "", 1.0)).faces:
            if _close(face.box, box) and face.person:
                return next((p for p in self.people if p.id == face.person), None)
        return None

    def tags(self, file_name: str) -> list[str]:
        """Short human-readable labels for one photo."""
        photo = self.photos[file_name]
        tags: list[str] = []
        names = [self.people[f.person - 1].title for f in photo.faces if f.person]
        tags += list(dict.fromkeys(names))
        tags += list(dict.fromkeys(a.display for a in photo.animals))
        if is_food_photo(photo):
            tags.append(dish_name(photo))
        if photo.subject:
            tags.append(photo.subject[1])
        if photo.setting:
            tags.append(photo.setting[1])
        return list(dict.fromkeys(tags))


def group_title(animal: Animal) -> str:
    return f"{animal.group_fa} ({animal.group})" if animal.species else animal.group_fa


def dish_name(photo: Photo) -> str:
    """The vision-language model's dish when it was sure, else the detector's, else food.

    COCO names only ten foods, so the detector calls most dishes a "sandwich" or a
    "cake"; the open-vocabulary label is the better name whenever it exists.
    """
    if photo.dish:
        return photo.dish[1]
    if photo.dishes:
        return max(photo.dishes, key=lambda d: d[2])[1]
    return "غذا (نوع نامشخص)"


def is_food_photo(photo: Photo) -> bool:
    """A dish the detector saw, or a photo whose main subject reads as food."""
    if photo.dishes:
        return True
    return bool(photo.subject and photo.subject[0] == FOOD_SUBJECT)


def fa_digits(value: Any) -> str:
    return str(value).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def _same_object(a: Box, b: Box) -> bool:
    """Each box's centre lies inside the other: one box, padded or not."""
    def centre_in(inner: Box, outer: Box) -> bool:
        cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
        return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

    return centre_in(a, b) and centre_in(b, a)


def _close(a: Box, b: Box, slack: int = 2) -> bool:
    return all(abs(x - y) <= slack for x, y in zip(a, b, strict=True))


def to_original(box: Box, scale: float) -> Box:
    """The same mapping ``vie.query`` applies, so boxes from both paths agree."""
    if scale == 1.0:
        return box
    x1, y1, x2, y2 = (int(round(v / scale)) for v in box)
    return x1, y1, x2, y2


def animal_group(class_id: int, species: str) -> tuple[str, str]:
    """``(english, persian)`` group of an ImageNet animal class."""
    for first, last, english, persian in ANIMAL_GROUPS:
        if first <= class_id <= last:
            return english, persian
    for generic, members in GENERIC_TERMS.items():
        for member in members:
            if re.search(rf"\b{re.escape(member.lower())}\b", species.lower()):
                return generic, species
    return species, species


# ---- loading --------------------------------------------------------------


def load_photos(config: Config) -> tuple[dict[str, Photo], list[str]]:
    """Every photo with what was stored for it at index time."""
    index = Index(config.database_path)
    with index.connect() as conn:
        vocabulary = index.classifier_vocabulary(conn) or []
        photos = {
            name: Photo(name, path, scale)
            for name, path, scale in conn.execute(
                "SELECT file_name, file_path, index_scale FROM gallery_meta ORDER BY file_name"
            )
        }
        for name, bbox, blob in conn.execute(
            "SELECT file_name, bbox, embedding FROM face_embeddings"
        ):
            photo = photos[name]
            photo.faces.append(
                Face(name, to_original(parse_box(bbox), photo.scale), from_blob(blob))
            )

        best: dict[tuple[str, str], tuple[int, float]] = {}
        for name, bbox, class_id, probability in conn.execute(
            "SELECT file_name, bbox, class_id, probability FROM animal_predictions"
        ):
            current = best.get((name, bbox))
            if current is None or probability > current[1]:
                best[(name, bbox)] = (class_id, probability)
        for name, bbox, confidence in conn.execute(
            "SELECT file_name, bbox, confidence FROM animal_boxes"
        ):
            if confidence < config.animal.search_confidence:
                continue
            class_id, probability = best.get((name, bbox), (-1, 0.0))
            confident = (
                0 <= class_id <= LAST_ANIMAL_CLASS
                and probability >= config.animal.classifier_threshold
                and class_id < len(vocabulary)
            )
            species = vocabulary[class_id] if confident else ""
            group, group_fa = animal_group(class_id, species) if species else UNKNOWN_ANIMAL
            photo = photos[name]
            photo.animals.append(Animal(
                name, to_original(parse_box(bbox), photo.scale), species, group, group_fa,
                probability, confidence, class_id if confident else -1,
            ))

        for name, bbox, confidence, class_id in conn.execute(
            "SELECT file_name, bbox, confidence, class_id FROM food_boxes"
        ):
            if class_id in COCO_DISHES and confidence >= config.food.search_confidence:
                photo = photos[name]
                photo.dishes.append(
                    (to_original(parse_box(bbox), photo.scale), COCO_DISHES[class_id], confidence)
                )
        for name, by_vlm in conn.execute("SELECT file_name, food_by_vlm FROM gallery_meta"):
            photos[name].food_gate = bool(by_vlm)
    return photos, vocabulary


def whole_vectors(config: Config) -> dict[str, np.ndarray]:
    index = Index(config.database_path)
    with index.connect() as conn:
        return {
            name: from_blob(blob)
            for name, blob in conn.execute(
                "SELECT file_name, embedding FROM clip_embeddings WHERE region = 'whole'"
            )
        }


# ---- zero-shot labels -----------------------------------------------------


def label_photos(
    vectors: dict[str, np.ndarray],
    labels: Sequence[tuple[str, str]],
    encode_texts: Callable[[list[str]], np.ndarray],
    to_logits: Callable[[np.ndarray], np.ndarray],
    share: float = LABEL_SHARE,
) -> dict[str, tuple[str, str, float]]:
    """Each photo's winning label, when it takes at least ``share`` of the group."""
    if not vectors:
        return {}
    names = list(vectors)
    matrix = np.vstack([vectors[n] for n in names])
    texts = encode_texts([PROMPT.format(label=en) for en, _ in labels])
    logits = to_logits(matrix @ texts.T)
    result: dict[str, tuple[str, str, float]] = {}
    for name, row in zip(names, logits, strict=True):
        probabilities = softmax([float(v) for v in row])
        top = int(np.argmax(probabilities))
        if probabilities[top] >= share:
            result[name] = (labels[top][0], labels[top][1], probabilities[top])
    return result


# ---- people ---------------------------------------------------------------


def cluster_people(photos: dict[str, Photo], config: Config) -> list[Person]:
    """Group every face in the archive into people.

    The video module's average-linkage clustering, with the photo standing in for the
    frame: two faces in one photo are never merged, because one person cannot appear
    twice in the same picture.
    """
    faces = [face for photo in photos.values() for face in photo.faces if face.verified]
    if not faces:
        return []
    order = {name: i for i, name in enumerate(photos)}
    sightings = [Sighting(float(order[f.file_name]), f.box, f.embedding) for f in faces]
    lookup = {id(s): f for s, f in zip(sightings, faces, strict=True)}
    groups = cluster_faces(sightings, config.video.face_cluster_threshold, merge_gap=0.0)

    people: list[Person] = []
    for group in groups:
        members = [lookup[id(s)] for s in group.sightings]
        centroid = np.mean([m.embedding for m in members], axis=0)
        people.append(Person(0, members, centroid / (np.linalg.norm(centroid) or 1.0)))
    people.sort(key=lambda p: (-len(p.photos), -len(p.faces)))
    for number, person in enumerate(people, 1):
        person.id = number
        for face in person.faces:
            face.person = number
    return people


class NameBook:
    """Names given to people, stored as face embeddings beside the index.

    A name is attached to faces, not to a cluster number, so it survives re-indexing
    and a new clustering. Naming is always the operator's act; nothing is guessed.
    """

    def __init__(self, path: Path, threshold: float) -> None:
        self.path = Path(path)
        self.threshold = threshold
        self.entries: dict[str, list[list[float]]] = {}
        if self.path.is_file():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def names(self) -> list[str]:
        return sorted(self.entries)

    def add(self, name: str, embeddings: Sequence[np.ndarray]) -> None:
        name = name.strip()
        if not name:
            raise ValueError("نام خالی است")
        stored = self.entries.setdefault(name, [])
        stored.extend([float(v) for v in np.asarray(e, dtype=np.float32).ravel()]
                      for e in embeddings)
        self.save()

    def remove(self, name: str) -> None:
        self.entries.pop(name, None)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False), encoding="utf-8")

    def vectors(self, name: str) -> np.ndarray:
        return np.asarray(self.entries.get(name, []), dtype=np.float32)

    def identify(self, embedding: np.ndarray) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for name in self.entries:
            score = float(np.max(self.vectors(name) @ embedding))
            if score >= self.threshold and (best is None or score > best[1]):
                best = (name, score)
        return best

    def label(self, people: list[Person]) -> None:
        for person in people:
            found = self.identify(person.centroid)
            person.name = found[0] if found else ""


# ---- everything -----------------------------------------------------------


#: ``crop_vectors(items, kind)``: one vector per face (kind "face") or box ("box").
CropVectors = Callable[[list[Any], str], np.ndarray]


def verify_faces(
    photos: dict[str, Photo],
    crop_vectors: CropVectors,
    encode_texts: Callable[[list[str]], np.ndarray],
    to_logits: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Mark each detected face that the vision-language model also reads as a person."""
    faces = [f for photo in photos.values() for f in photo.faces]
    if not faces:
        return
    logits = to_logits(crop_vectors(faces, "face") @ encode_texts(FACE_PROMPTS).T)
    for face, row in zip(faces, logits, strict=True):
        face.verified = softmax([float(v) for v in row])[0] >= FACE_SHARE


def check_animals(
    photos: dict[str, Photo],
    crop_vectors: CropVectors,
    encode_texts: Callable[[list[str]], np.ndarray],
    to_logits: Callable[[np.ndarray], np.ndarray],
) -> list[Animal]:
    """Read every animal box a second time, from its own crop. Returns those dropped.

    Three cases, each seen on a real gallery:

    * a box that reads as food, a person or an object was a detector error (a hot dog
      came back from ImageNet as a dog breed), so it leaves the overview;
    * an animal the classifier could not name gets the vision-language model's name
      (ImageNet has no raccoon and no panda cub);
    * a classifier guess under ``CLASSIFIER_SURE`` that the vision-language model
      clearly contradicts is replaced (a calf at 27% "dog").
    """
    animals = [a for photo in photos.values() for a in photo.animals]
    if not animals:
        return []
    prompts = [f"a photo of {prompt}" for prompt, _, _ in ANIMAL_PROMPTS]
    logits = to_logits(crop_vectors(animals, "box") @ encode_texts(prompts).T)
    rejected: list[Animal] = []
    for animal, row in zip(animals, logits, strict=True):
        probabilities = softmax([float(v) for v in row])
        top = int(np.argmax(probabilities))
        _, group, group_fa = ANIMAL_PROMPTS[top]
        if probabilities[top] < ANIMAL_SHARE:
            continue
        if not group:
            rejected.append(animal)
        elif not animal.species or (
            animal.probability < CLASSIFIER_SURE and animal.group != group
        ):
            animal.species, animal.group, animal.group_fa = group, group, group_fa
            animal.probability = probabilities[top]
    dropped = {id(a) for a in rejected}
    for photo in photos.values():
        photo.animals = [a for a in photo.animals if id(a) not in dropped]
    return rejected


#: Kept for callers of the earlier name.
name_unknown_animals = check_animals


def build(
    config: Config,
    encoder: Any,
    names: NameBook,
    crop_vectors: CropVectors | None = None,
) -> Insights:
    photos, _ = load_photos(config)
    rejected: list[Animal] = []
    if crop_vectors is not None:
        verify_faces(photos, crop_vectors, encoder.encode_texts, encoder.logits_from_similarity)
        rejected = check_animals(
            photos, crop_vectors, encoder.encode_texts, encoder.logits_from_similarity
        )
    vectors = whole_vectors(config)
    for attribute, labels, subset in (
        ("subject", SUBJECTS, vectors),
        ("setting", SETTINGS, vectors),
    ):
        for name, value in label_photos(
            subset, labels, encoder.encode_texts, encoder.logits_from_similarity
        ).items():
            setattr(photos[name], attribute, value)
    food = {k: vectors[k] for k, p in photos.items() if is_food_photo(p) and k in vectors}
    for name, value in label_photos(
        food, DISHES, encoder.encode_texts, encoder.logits_from_similarity, share=0.30
    ).items():
        photos[name].dish = value

    people = cluster_people(photos, config)
    names.label(people)
    return Insights(photos, people, config.models["vision_language"], rejected)
