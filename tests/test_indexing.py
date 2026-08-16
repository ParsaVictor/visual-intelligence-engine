"""Regression tests for P1-2: the face/food cross-verification must actually run."""

from __future__ import annotations

from vie.indexing import STAGE_ORDER, analyse_image


class RecordingFaceDetector:
    def __init__(self, faces, log):
        self.faces, self.log = faces, log

    def detect(self, image):
        self.log.append("face")
        return self.faces


class RecordingBoxDetector:
    def __init__(self, boxes, log, name):
        self.boxes, self.log, self.name = boxes, log, name

    def detect(self, image, classes, confidence):
        self.log.append(self.name)
        return [(b, c) for b, c in self.boxes if c >= confidence]


def _run(config, faces=(), animals=(), foods=()):
    log: list[str] = []
    analysis = analyse_image(
        object(),
        (1000, 1000),
        config,
        face_detector=RecordingFaceDetector(list(faces), log),
        animal_detector=RecordingBoxDetector(list(animals), log, "animal"),
        food_detector=RecordingBoxDetector(list(foods), log, "food"),
    )
    return analysis, log


def test_faces_are_detected_before_food(config) -> None:
    """The original ran food before face, which is why the guard never fired."""
    _, log = _run(config)
    assert log.index("face") < log.index("food")


def test_documented_stage_order(config) -> None:
    _, log = _run(config)
    assert log == list(STAGE_ORDER)


def test_food_box_on_a_face_is_rejected(config) -> None:
    """The exact case the original always accepted."""
    faces = [((100, 100, 300, 300), "emb")]
    foods = [((150, 150, 250, 250), 0.9)]
    analysis, _ = _run(config, faces=faces, foods=foods)
    assert analysis.food_boxes == []
    assert analysis.rejected_food_boxes == [(150, 150, 250, 250)]
    assert analysis.has_food is False


def test_food_box_away_from_a_face_is_kept(config) -> None:
    faces = [((100, 100, 300, 300), "emb")]
    foods = [((600, 600, 800, 800), 0.9)]
    analysis, _ = _run(config, faces=faces, foods=foods)
    assert analysis.food_boxes == [(600, 600, 800, 800)]
    assert analysis.has_food is True


def test_food_is_kept_when_there_are_no_faces(config) -> None:
    analysis, _ = _run(config, foods=[((150, 150, 250, 250), 0.9)])
    assert analysis.has_food is True


def test_tiny_food_box_is_dropped(config) -> None:
    analysis, _ = _run(config, foods=[((0, 0, 10, 10), 0.9)])
    assert analysis.food_boxes == []


def test_tiny_animal_box_is_dropped(config) -> None:
    analysis, _ = _run(config, animals=[((0, 0, 10, 10), 0.9)])
    assert analysis.has_animal is False


def test_animal_gate_uses_the_permissive_threshold(config) -> None:
    """A detection between the gate and search thresholds must still be indexed,
    otherwise the search pass can never reach it."""
    conf = (config.animal.gate_confidence + config.animal.search_confidence) / 2
    analysis, _ = _run(config, animals=[((0, 0, 200, 200), conf)])
    assert analysis.has_animal is True


def test_below_gate_confidence_is_not_indexed(config) -> None:
    below = config.animal.gate_confidence - 0.05
    analysis, _ = _run(config, animals=[((0, 0, 200, 200), below)])
    assert analysis.has_animal is False


def test_faces_are_recorded_for_storage(config) -> None:
    faces = [((10, 10, 50, 50), "emb-a"), ((60, 60, 100, 100), "emb-b")]
    analysis, _ = _run(config, faces=faces)
    assert analysis.has_face is True
    assert len(analysis.faces) == 2
