from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vie.config import DEFAULT_CONFIG_PATH, Config


@pytest.fixture()
def raw_config() -> dict:
    with Path(DEFAULT_CONFIG_PATH).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture()
def config(raw_config: dict) -> Config:
    return Config.from_dict(raw_config)
