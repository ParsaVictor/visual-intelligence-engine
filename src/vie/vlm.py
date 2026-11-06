"""Vision-language scoring, for either CLIP or SigLIP.

Why the two are not interchangeable
-----------------------------------
Both map images and text into a shared space, but they are *trained* with
different objectives, and that changes how their raw logits should be read.

**CLIP** uses a softmax over a batch: for one image it asks "which of these N
captions is the right one?" Its logits are only meaningful *relative to each
other*, and the answer always sums to 1 across the candidates. That is exactly
why the contrastive-prompt trick works: supply rival prompts and the softmax
becomes a real posterior.

**SigLIP** replaces that with an independent sigmoid per image-text pair: "is
this caption true for this image, yes or no?" Each pair is scored on its own,
with a learned temperature *and a learned bias* baked into the logit. There is
no normalisation across candidates.

The practical consequence: running softmax over SigLIP logits throws away the
bias term that makes the score absolute, and running sigmoid over CLIP logits
produces a number with no calibration at all. Same-looking tensors, different
meaning, which is why :class:`vie.config.Config` refuses a mismatched pairing
and why both recognition paths score through :func:`hypothesis_score` here.

Model choice
------------
======================================  =======  ==================
model                                   params   ImageNet 0-shot
======================================  =======  ==================
``openai/clip-vit-large-patch14``       428 M    ~75.5 %
``google/siglip-base-patch16-224``      203 M    ~76.2 %
``google/siglip-so400m-patch14-384``    877 M    ~83.2 %
======================================  =======  ==================

Figures are the published ones, not measured here. ``so400m`` is the default.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

SOFTMAX = "softmax"
SIGMOID = "sigmoid"


def is_siglip(model_name: str) -> bool:
    return "siglip" in model_name.lower()


def is_multilingual(model_name: str) -> bool:
    """SigLIP 2 was trained on multilingual WebLI; SigLIP and CLIP on English."""
    return "siglip2" in model_name.lower()


def has_right_to_left(text: str) -> bool:
    """Persian, Arabic or Hebrew script anywhere in ``text``."""
    return any("\u0590" <= ch <= "\u08ff" or "\ufb1d" <= ch <= "\ufeff" for ch in text)


def default_score_mode(model_name: str) -> str:
    return SIGMOID if is_siglip(model_name) else SOFTMAX


def _softmax(values: Sequence[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _sigmoid(value: float) -> float:
    # Split by sign to avoid overflow on large-magnitude logits.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


def similarity_to_probability(cosine: float, logit_scale: float, logit_bias: float) -> float:
    """SigLIP's own calibrated reading of a cosine, as trained.

    This is what makes the free-text path honest: a raw SigLIP cosine of 0.2 says
    little to a reader, while the model's sigmoid says how likely the caption is
    to be true of the image. Only meaningful for models with a learned bias.
    """
    return _sigmoid(cosine * logit_scale + logit_bias)


def hypothesis_score(logits: Sequence[float], mode: str) -> float:
    """Score the first prompt against the rest, in [0, 1].

    ``softmax``  posterior of the hypothesis among rivals (CLIP).
    ``sigmoid``  the sigmoid of the logit margin over the strongest rival (SigLIP).

    The margin is a measured choice (``scripts/calibrate.py``). SigLIP's absolute
    probability ranks just as well, but it is a poor thing to threshold: on real
    photos the best cut was 0.05 for dishes and 0.003 for animals, and a correct "cat"
    typically scored 2%. Its learned bias makes every prompt improbable by default,
    and the bias cancels in a difference, leaving a scale that holds across domains.
    """
    if not logits:
        raise ValueError("logits must not be empty")
    if len(logits) == 1:
        raise ValueError(
            "contrastive scoring needs at least one rival prompt; with a single "
            "candidate the score is meaningless"
        )

    if mode == SOFTMAX:
        return _softmax(list(logits))[0]
    if mode == SIGMOID:
        return _sigmoid(logits[0] - max(logits[1:]))
    raise ValueError(f"unknown score mode {mode!r}")


def gate_score(
    logits: Sequence[float], n_positive: int, mode: str
) -> float:
    """How much more this looks like the positive prompts than the negative ones.

    Used by the food gate. Positive means the image looks more like food than not.
    ``sigmoid`` mode returns the logit margin, which is unbounded; ``softmax`` returns
    a probability gap in [-1, 1]. Thresholds are therefore specific to the mode.
    """
    if n_positive <= 0 or n_positive >= len(logits):
        raise ValueError(
            f"need at least one positive and one negative prompt, "
            f"got {n_positive} positives out of {len(logits)}"
        )

    if mode == SIGMOID:
        return max(logits[:n_positive]) - max(logits[n_positive:])
    if mode == SOFTMAX:
        probs = _softmax(list(logits))
        return max(probs[:n_positive]) - max(probs[n_positive:])
    raise ValueError(f"unknown score mode {mode!r}")
