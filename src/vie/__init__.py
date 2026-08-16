"""Visual Intelligence Engine — multimodal search over an image archive."""

__version__ = "0.1.0"

from vie.config import Config, ConfigError
from vie.scoring import Match, best_per_image, contrastive_score

__all__ = ["Config", "ConfigError", "Match", "best_per_image", "contrastive_score", "__version__"]
