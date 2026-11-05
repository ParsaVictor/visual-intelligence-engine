"""The same system, with real weights on real public images.

Deselected by default (see ``addopts`` in pyproject.toml). Run with::

    pytest -m models

Uses scikit-image's bundled sample photographs: real, redistributable, and with no
identifiable private individuals among those selected.
"""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from vie.config import Config

pytestmark = pytest.mark.models

FAST_MODEL = "google/siglip-base-patch16-224"


@pytest.fixture(scope="module")
def sample_gallery(tmp_path_factory) -> Path:
    data = pytest.importorskip("skimage.data")
    from PIL import Image

    root = tmp_path_factory.mktemp("gallery")
    for name, array in (
        ("cat.png", data.chelsea()),
        ("coffee.png", data.coffee()),
        ("rocket.png", data.rocket()),
    ):
        Image.fromarray(array).save(root / name)
    return root


@pytest.fixture(scope="module")
def real_config(sample_gallery, tmp_path_factory) -> Config:
    import yaml

    from vie.config import DEFAULT_CONFIG_PATH

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["models"]["vision_language"] = FAST_MODEL
    raw["paths"] = {
        "gallery": str(sample_gallery),
        "database": str(tmp_path_factory.mktemp("db") / "index.db"),
        "model_cache": "",
    }
    raw["indexing"]["batch_size"] = 2
    raw["runtime"]["low_memory"] = bool(os.environ.get("VIE_LOW_MEMORY"))
    config = Config.from_dict(raw)
    config.validate()
    return config


@pytest.fixture(scope="module")
def encoder(real_config):
    from vie.models import VisionLanguageEncoder, load_all

    return VisionLanguageEncoder(load_all(real_config))


def test_encoder_vectors_are_unit_length(encoder, sample_gallery) -> None:
    import numpy as np
    from PIL import Image

    vector = encoder.encode_images([Image.open(sample_gallery / "cat.png").convert("RGB")])[0]
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-3


def test_siglip_retrieves_the_right_image(encoder, sample_gallery) -> None:
    """If tokenizer padding or colour order is wrong, this is where it shows."""
    from PIL import Image

    names = ["cat.png", "coffee.png", "rocket.png"]
    images = encoder.encode_images(
        [Image.open(sample_gallery / n).convert("RGB") for n in names]
    )
    for query, expected in (
        ("a photo of a cat", "cat.png"),
        ("a cup of coffee", "coffee.png"),
        ("a rocket launching into the sky", "rocket.png"),
    ):
        scores = images @ encoder.encode_text(query)
        assert names[int(scores.argmax())] == expected, (query, scores.tolist())


def test_contrastive_scoring_accepts_a_hit_and_rejects_a_miss(
    encoder, sample_gallery, real_config
) -> None:
    from PIL import Image

    from vie.vlm import hypothesis_score

    cat = Image.open(sample_gallery / "cat.png").convert("RGB")
    prompts_for = real_config.animal.prompts_for
    hit = hypothesis_score(encoder.score([cat], prompts_for("cat"))[0], real_config.score_mode)
    miss = hypothesis_score(encoder.score([cat], prompts_for("zebra"))[0], real_config.score_mode)
    print(f"cat={hit:.3f} zebra={miss:.3f} threshold={real_config.animal.match_threshold}")
    assert hit > miss
    assert hit >= real_config.animal.match_threshold > miss


def test_food_gate_separates_food_from_not_food(encoder, sample_gallery, real_config) -> None:
    from PIL import Image

    names = ["coffee.png", "rocket.png"]
    vectors = encoder.encode_images(
        [Image.open(sample_gallery / n).convert("RGB") for n in names]
    )
    coffee, rocket = encoder.food_gate_scores(vectors, real_config)
    print(f"coffee={coffee:.3f} rocket={rocket:.3f} "
          f"threshold={real_config.food.clip_gate_threshold}")
    # The gate is deliberately permissive: an image it rejects can never be found, and a
    # borderline image it admits is still filtered by the search step. So the rocket may
    # pass; what must hold is that food scores higher and is never shut out.
    assert coffee > rocket
    assert coffee >= real_config.food.clip_gate_threshold


def test_calibrated_text_score_is_a_probability(encoder) -> None:
    high, low = encoder.calibrated(0.4), encoder.calibrated(-0.1)
    assert 0.0 <= low < high <= 1.0


def test_species_classifier_top1_on_a_real_cat(real_config, sample_gallery) -> None:
    from PIL import Image

    from vie.models import SpeciesClassifier, load_all

    bundle = load_all(real_config)
    top = SpeciesClassifier(bundle).classify(
        [Image.open(sample_gallery / "cat.png").convert("RGB")], 5
    )[0]
    names = [bundle.class_names[i].lower() for i, _ in top]
    assert any("cat" in n or "tabby" in n for n in names), names


def _full_stack_available() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) for m in ("insightface", "ultralytics", "onnxruntime"))


@pytest.mark.skipif(not _full_stack_available(), reason="needs insightface and ultralytics")
def test_full_cli_pipeline_on_real_models(real_config, tmp_path, capsys) -> None:
    """Every model, through the real command line, index then all four searches."""
    import yaml

    from vie import cli
    from vie.query import run_search

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(real_config.raw), encoding="utf-8")

    assert cli.main(["index", "--config", str(config_path), "--workers", "2"]) == 0
    assert "indexed=3" in capsys.readouterr().out

    for kind, query in (("text", "a cat"), ("animal", "cat"), ("animal", "quokka"),
                        ("food", "coffee")):
        code = cli.main(["search", kind, "--query", query, "--config", str(config_path)])
        assert code in {0, 1}, (kind, query)
        print(kind, query, capsys.readouterr().out)

    def load(cfg):
        return cli._load_runtime(cfg)

    # No sample image holds a face, which must be a clean error rather than a crash.
    probe = Path(real_config.gallery_path) / "cat.png"
    with pytest.raises(ValueError):
        run_search(SimpleNamespace(kind="face", image=probe, query="", top_k=None),
                   real_config, load)
    shutil.rmtree(tmp_path, ignore_errors=True)
