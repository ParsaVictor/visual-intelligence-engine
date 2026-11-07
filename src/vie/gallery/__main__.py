"""Open the Smart Gallery for an indexed gallery.

    python -m vie.gallery --config data/runs/photos-final/config.yaml
"""

from __future__ import annotations

import argparse
import sys

from vie.config import Config
from vie.logging_setup import configure, get_logger

log = get_logger("gallery")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vie.gallery", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--config", required=True, help="the config of an indexed run")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-warm-up", action="store_true",
                        help="open at once; the first search of each kind is then slower")
    args = parser.parse_args(argv)
    configure("INFO")

    from vie.gallery.app import launch
    from vie.gallery.engine import Gallery

    gallery = Gallery(Config.load(args.config))
    if not args.no_warm_up:
        log.info("loading models and preparing the gallery (first time takes a few minutes)")
        gallery.warm_up()
    launch(gallery, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
