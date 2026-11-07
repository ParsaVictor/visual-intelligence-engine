"""The Smart Gallery's own logic: grouping, labelling, people and naming.

The search paths themselves are the package's and tested elsewhere. What is pinned
here is what the gallery adds on top, with no model loaded.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vie.database import Index
from vie.gallery import insights as gi
from vie.scoring import Match


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


# ---- animals --------------------------------------------------------------


@pytest.mark.parametrize(("class_id", "species", "group"), [
    (211, "vizsla", "dog"),               # a breed the word list "dog" does not name
    (159, "Rhodesian ridgeback", "dog"),
    (281, "tabby", "cat"),
    (388, "giant panda", "giant panda"),
    (387, "lesser panda", "red panda"),
    (22, "bald eagle", "bird"),
])
def test_animal_groups_follow_imagenet_ranges(class_id, species, group) -> None:
    assert gi.animal_group(class_id, species)[0] == group


def test_unknown_animals_are_named_and_false_boxes_dropped() -> None:
    photo = gi.Photo("a.jpg", "a.jpg", 1.0)
    photo.animals = [
        gi.Animal("a.jpg", (0, 0, 10, 10), "", *gi.UNKNOWN_ANIMAL, 0.1, 0.9),
        gi.Animal("a.jpg", (20, 20, 30, 30), "", *gi.UNKNOWN_ANIMAL, 0.1, 0.9),
    ]
    prompts = [p for p, _, _ in gi.ANIMAL_PROMPTS]
    panda, person = prompts.index("a giant panda"), prompts.index("a person")

    def crops(items, kind):
        assert kind == "box"
        return np.eye(len(prompts), dtype=np.float32)[[panda, person]]

    gi.name_unknown_animals(
        {"a.jpg": photo}, crops, lambda texts: np.eye(len(prompts), dtype=np.float32),
        lambda similarity: similarity * 20,
    )
    assert [(a.group, a.group_fa) for a in photo.animals] == [("giant panda", "پاندا")]


# ---- food -----------------------------------------------------------------


def test_dish_name_prefers_the_open_vocabulary_label() -> None:
    photo = gi.Photo("a.jpg", "a.jpg", 1.0, dishes=[((0, 0, 9, 9), "sandwich", 0.8)])
    assert gi.dish_name(photo) == "sandwich"
    photo.dish = ("kebab", "کباب", 0.7)
    assert gi.dish_name(photo) == "کباب"


def test_food_photo_needs_a_dish_or_a_food_subject() -> None:
    photo = gi.Photo("a.jpg", "a.jpg", 1.0, food_gate=True)
    assert not gi.is_food_photo(photo)
    photo.subject = (gi.FOOD_SUBJECT, "غذا", 0.9)
    assert gi.is_food_photo(photo)


def test_food_results_on_photos_that_are_not_food_are_dropped() -> None:
    """A cat scored 99% as sushi; the gallery must not show it."""
    from vie.gallery.engine import Gallery

    cat = gi.Photo("cat.jpg", "cat.jpg", 1.0, subject=("an animal", "حیوان", 0.9))
    sushi = gi.Photo("sushi.jpg", "sushi.jpg", 1.0, subject=(gi.FOOD_SUBJECT, "غذا", 0.9))
    import threading

    gallery = Gallery.__new__(Gallery)          # no models: only the filter is exercised
    gallery._lock = threading.RLock()
    gallery._insights = gi.Insights({"cat.jpg": cat, "sushi.jpg": sushi}, [], "m")
    kept, dropped = gallery._only_food(
        [Match("cat.jpg", "cat.jpg", 0.99), Match("sushi.jpg", "sushi.jpg", 0.98)]
    )
    assert [m.file_name for m in kept] == ["sushi.jpg"]
    assert dropped == 1


# ---- zero-shot labels -----------------------------------------------------


def test_a_label_needs_a_clear_share_of_its_group() -> None:
    labels = [("a", "A"), ("b", "B"), ("c", "C")]
    vectors = {"clear.jpg": _unit(1, 0, 0), "torn.jpg": _unit(1, 1, 1)}
    found = gi.label_photos(
        vectors, labels, lambda texts: np.eye(3, dtype=np.float32), lambda s: s * 10
    )
    assert found["clear.jpg"][:2] == ("a", "A")
    assert "torn.jpg" not in found          # a three-way tie is not an answer


# ---- people ---------------------------------------------------------------


def _photos(*faces: tuple[str, np.ndarray]) -> dict[str, gi.Photo]:
    photos: dict[str, gi.Photo] = {}
    for number, (name, vector) in enumerate(faces):
        photo = photos.setdefault(name, gi.Photo(name, name, 1.0))
        photo.faces.append(gi.Face(name, (number, 0, number + 50, 50), vector))
    return photos


def test_same_person_across_photos_is_one_person(config) -> None:
    alice, bob = _unit(1, 0, 0), _unit(0, 1, 0)
    photos = _photos(("1.jpg", alice), ("2.jpg", alice), ("2.jpg", bob), ("3.jpg", bob))
    people = gi.cluster_people(photos, config)
    assert sorted(sorted(p.photos) for p in people) == [["1.jpg", "2.jpg"], ["2.jpg", "3.jpg"]]


def test_two_faces_in_one_photo_are_never_one_person(config) -> None:
    twin = _unit(1, 0, 0)
    people = gi.cluster_people(_photos(("1.jpg", twin), ("1.jpg", twin)), config)
    assert len(people) == 2


def test_unverified_faces_are_not_counted_as_people(config) -> None:
    photos = _photos(("1.jpg", _unit(1, 0, 0)), ("2.jpg", _unit(0, 1, 0)))
    photos["2.jpg"].faces[0] = replace(photos["2.jpg"].faces[0], verified=False)
    people = gi.cluster_people(photos, config)
    assert [p.photos for p in people] == [["1.jpg"]]
    assert gi.Insights(photos, people, "m").face_count == 1


def test_names_attach_to_faces_and_survive_reclustering(tmp_path, config) -> None:
    alice = _unit(1, 0, 0)
    book = gi.NameBook(tmp_path / "people.json", config.face.match_threshold)
    book.add("Alice", [alice])

    reopened = gi.NameBook(tmp_path / "people.json", config.face.match_threshold)
    people = gi.cluster_people(_photos(("1.jpg", alice), ("2.jpg", _unit(0, 1, 0))), config)
    reopened.label(people)
    named = {p.photos[0]: p.name for p in people}
    assert named == {"1.jpg": "Alice", "2.jpg": ""}


def test_an_empty_name_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        gi.NameBook(tmp_path / "people.json", 0.46).add("  ", [_unit(1, 0)])


# ---- loading from a real index --------------------------------------------


def test_boxes_are_mapped_back_to_the_original_image(tmp_path, config) -> None:
    index = Index(tmp_path / "index.db")
    index.initialise()
    with index.connect() as conn:
        index.upsert_image(conn, file_name="a.jpg", file_path="a.jpg", hash_="h", size=1,
                           has_face=True, has_animal=True, has_food=False, index_scale=0.5)
        index.add_face(conn, "a.jpg", (10, 10, 50, 50), _unit(1, 0, 0))
        index.add_animal_box(conn, "a.jpg", (100, 100, 200, 200), 0.9)
        index.add_animal_prediction(conn, "a.jpg", (100, 100, 200, 200), 211, 0.8)
        index.set_classifier_vocabulary(conn, "convnext_tiny", [f"c{i}" for i in 211 * [0]]
                                        + ["vizsla"])
    photos, _ = gi.load_photos(replace(config, database_path=tmp_path / "index.db"))
    photo = photos["a.jpg"]
    assert photo.faces[0].box == (20, 20, 100, 100)
    assert photo.animals[0].box == (200, 200, 400, 400)
    assert (photo.animals[0].species, photo.animals[0].group) == ("vizsla", "dog")


# ---- second reading of animals, and search consistency ---------------------


def test_second_reading_drops_food_and_overrules_an_unsure_guess() -> None:
    photo = gi.Photo("a.jpg", "a.jpg", 1.0)
    hot_dog = gi.Animal("a.jpg", (0, 0, 10, 10), "vizsla", "dog", "سگ", 0.89, 0.9, 211)
    calf = gi.Animal("a.jpg", (20, 20, 30, 30), "vizsla", "dog", "سگ", 0.27, 0.9, 211)
    sure_dog = gi.Animal("a.jpg", (40, 40, 50, 50), "vizsla", "dog", "سگ", 0.95, 0.9, 211)
    photo.animals = [hot_dog, calf, sure_dog]
    prompts = [p for p, _, _ in gi.ANIMAL_PROMPTS]
    reads = [prompts.index("food or a dish"), prompts.index("a cow or a calf"),
             prompts.index("a cat")]

    rejected = gi.check_animals(
        {"a.jpg": photo}, lambda items, kind: np.eye(len(prompts), dtype=np.float32)[reads],
        lambda texts: np.eye(len(prompts), dtype=np.float32), lambda s: s * 20,
    )
    assert rejected == [hot_dog]
    assert [a.group for a in photo.animals] == ["cow", "dog"]   # a sure 95% stays a dog


def test_a_padded_box_is_the_same_animal() -> None:
    photo = gi.Photo("a.jpg", "a.jpg", 1.0)
    photo.animals = [gi.Animal("a.jpg", (100, 100, 200, 200), "x", "cow", "گاو", 0.9, 0.9)]
    info = gi.Insights({"a.jpg": photo}, [], "m")
    assert info.animal_at("a.jpg", (85, 85, 215, 215)).group == "cow"
    assert info.animal_at("a.jpg", (300, 300, 400, 400)) is None


def test_dish_queries_match_the_overview_label() -> None:
    plate = gi.Photo("k.jpg", "k.jpg", 1.0, subject=(gi.FOOD_SUBJECT, "غذا", 0.9),
                     dish=("kebab", "کباب", 0.8))
    grill = gi.Photo("g.jpg", "g.jpg", 1.0, subject=(gi.FOOD_SUBJECT, "غذا", 0.9),
                     dish=("barbecue or grilled meat", "کباب و گوشت کبابی", 0.7))
    info = gi.Insights({"k.jpg": plate, "g.jpg": grill}, [], "m")
    assert info.photos_with_dish("Kebab") == [("k.jpg", 0.8)]
    assert info.photos_with_dish("a hot dog") == []
    assert info.photos_with_dish("grilled meat") == [("g.jpg", 0.7)]


def test_free_text_tail_below_the_floor_is_hidden() -> None:
    import threading

    from vie.gallery.engine import Gallery

    gallery = Gallery.__new__(Gallery)
    gallery._lock = threading.RLock()
    gallery.config = SimpleNamespace(models={"vision_language": "google/siglip-so400m"})
    kept, hidden = gallery._relevant_text([
        Match("castle.jpg", "c", 0.2, confidence=0.71),
        Match("burger.jpg", "b", 0.05, confidence=0.0004),
    ])
    assert [m.file_name for m in kept] == ["castle.jpg"]
    assert hidden == 1


# ---- several subjects in one photo ------------------------------------------


def _scene_reads(*labels: str) -> tuple[np.ndarray, int]:
    """One region vector per label, each reading clearly as that scene prompt."""
    prompts = [p for p, _, _ in gi.SCENE_PROMPTS]
    rows = [prompts.index(label) for label in labels]
    return np.eye(len(prompts), dtype=np.float32)[rows], len(prompts)


def _scene(photo: gi.Photo, *labels: str) -> None:
    vectors, size = _scene_reads(*labels)
    regions = ["whole", "tl", "tr", "bl", "br", "center"][: len(labels)]
    gi.scene_check(
        {photo.file_name: photo}, {photo.file_name: list(zip(regions, vectors, strict=True))},
        lambda texts: np.eye(size, dtype=np.float32), lambda s: s * 20, layout="1+2x2+center",
    )


def _photo_on_disk(tmp_path, name: str = "p.png") -> gi.Photo:
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (400, 200)).save(path)
    return gi.Photo(name, str(path), 1.0)


def test_a_cat_the_detector_missed_beside_a_person_is_found(tmp_path) -> None:
    """The reported case: the person was found and the cat was not."""
    photo = _photo_on_disk(tmp_path)
    photo.faces = [gi.Face(photo.file_name, (10, 10, 60, 60), _unit(1, 0, 0), person=1)]
    _scene(photo, "a person", "a person", "a cat")          # whole, tl, tr
    assert [(a.group, a.source) for a in photo.animals] == [("cat", "scene")]
    assert photo.animals[0].box == (200, 0, 400, 100)        # the top-right quadrant
    assert photo.person_scene


def test_an_animal_the_detector_boxed_is_not_counted_twice(tmp_path) -> None:
    photo = _photo_on_disk(tmp_path)
    photo.animals = [gi.Animal(photo.file_name, (250, 20, 350, 90), "tabby", "cat", "گربه",
                               0.9, 0.9, 281)]
    _scene(photo, "a dog", "a dog", "a dog")                 # a misreading of that cat
    assert [a.group for a in photo.animals] == ["cat"]


def test_unclear_regions_add_nothing(tmp_path) -> None:
    photo = _photo_on_disk(tmp_path)
    prompts = [p for p, _, _ in gi.SCENE_PROMPTS]
    torn = np.zeros((1, len(prompts)), dtype=np.float32)
    torn[0, [prompts.index("a cat"), prompts.index("a dog")]] = 1.0   # 50/50
    gi.scene_check({photo.file_name: photo}, {photo.file_name: [("whole", torn[0])]},
                   lambda texts: np.eye(len(prompts), dtype=np.float32), lambda s: s * 20)
    assert photo.animals == [] and not photo.person_scene


def test_a_region_read_as_food_makes_it_a_food_photo(tmp_path) -> None:
    photo = _photo_on_disk(tmp_path)
    _scene(photo, "a person", "food or a dish")
    assert gi.is_food_photo(photo)


def test_results_box_everything_in_the_photo() -> None:
    from vie.gallery.app import everything_in

    photo = gi.Photo("p.jpg", "p.jpg", 1.0)
    photo.faces = [gi.Face("p.jpg", (10, 10, 60, 60), _unit(1, 0, 0), person=1)]
    photo.animals = [gi.Animal("p.jpg", (200, 0, 400, 100), "cat", "cat", "گربه", 0.8, 0.0,
                               -1, "scene")]
    person = gi.Person(1, photo.faces, _unit(1, 0, 0))
    info = gi.Insights({"p.jpg": photo}, [person], "m")
    boxes = everything_in(info, "p.jpg", ((200, 0, 400, 100), "cat"))
    assert [label for _, label in boxes] == ["cat", "شخص ۱"]     # the cat is not repeated
    assert info.tags("p.jpg")[:2] == ["شخص ۱", "گربه (cat)"]
