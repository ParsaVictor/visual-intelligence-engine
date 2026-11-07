"""Download every model weight once, so later runs need no internet.

    python scripts/fetch_models.py
    python scripts/fetch_models.py --skip-optional

Weights go to ``data/models/`` (ignored by git). Each detector is also loaded after its
download, which proves it works on this machine, then released before the next one so
the peak stays low. The two large SigLIP checkpoints are only downloaded.
"""

from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path

import yaml

from vie.cache import configure_cache
from vie.config import DEFAULT_CONFIG_PATH, Config

ROOT = Path(__file__).resolve().parents[1]
SIGLIP_FILES = ["*.json", "*.txt", "*.model", "model.safetensors"]


def config_for(cache: Path) -> Config:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"]["model_cache"] = str(cache)
    return Config.from_dict(raw)


def copy_instead_of_symlink() -> None:
    """Windows refuses symlinks without Developer Mode (WinError 1314).

    huggingface_hub's own probe can pass while the real link still fails, so tell it
    outright that links are unsupported. It then copies each file into place.
    """
    from huggingface_hub import file_download

    file_download.are_symlinks_supported = lambda cache_dir=None: False


def snapshot(repo: str, patterns: list[str] | None = None) -> None:
    # Imported late: huggingface_hub reads its cache location on import.
    from huggingface_hub import snapshot_download

    copy_instead_of_symlink()
    print(snapshot_download(repo, allow_patterns=patterns))


def load_and_release(config: Config, attribute: str) -> None:
    from vie.models import load_all

    bundle = load_all(config)
    getattr(bundle, attribute)
    del bundle
    gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, default=ROOT / "data" / "models")
    parser.add_argument("--skip-optional", action="store_true",
                        help="skip the speech and caption models, which are off by default")
    args = parser.parse_args()

    cache = args.models.resolve()
    configure_cache(cache)
    config = config_for(cache)

    steps = [
        ("SigLIP base", lambda: snapshot("google/siglip-base-patch16-224", SIGLIP_FILES)),
        ("ConvNeXt classifier", lambda: load_and_release(config, "classifier")),
        ("MegaDetector", lambda: load_and_release(config, "animal_detector")),
        ("RT-DETR-X", lambda: load_and_release(config, "object_detector")),
        ("InsightFace buffalo_l", lambda: load_and_release(config, "face_app")),
        ("SigLIP so400m", lambda: snapshot("google/siglip-so400m-patch14-384", SIGLIP_FILES)),
    ]
    if not args.skip_optional:
        steps += [
            ("Whisper small", lambda: snapshot("Systran/faster-whisper-small")),
            ("BLIP captions", lambda: snapshot("Salesforce/blip-image-captioning-base")),
        ]

    failed = []
    for name, step in steps:
        print(f"\n=== {name}", flush=True)
        started = time.perf_counter()
        try:
            step()
            print(f"ok: {name} ({time.perf_counter() - started:.0f} s)", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print(f"FAILED: {name}", flush=True)

    print("\nall models ready" if not failed else f"\nfailed: {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
