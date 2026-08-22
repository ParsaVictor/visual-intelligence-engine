"""Vision-language scoring, for either CLIP or SigLIP.

Why the two are not interchangeable
-----------------------------------
Both map images and text into a shared space, but they are *trained* with
different objectives, and that changes how their raw logits should be read.

**CLIP** uses a softmax over a batch: for one image it asks "which of these N
captions is the right one?" Its logits are only meaningful *relative to each
other*, and the answer always sums to 1 across the candidates. That is exactly
why the contrastive-prompt trick works — supply rival prompts and the softmax
becomes a real posterior.

**SigLIP** replaces that with an independent sigmoid per image-text pair: "is
this caption true for this image, yes or no?" Each pair is scored on its own,
with a learned temperature *and a learned bias* baked into the logit. There is
no normalisation across candidates.

The practical consequence: running softmax over SigLIP logits throws away the
bias term that makes the score absolute, and running sigmoid over CLIP logits
produces a number with no calibration at all. Same-looking tensors, different
meaning — which is why :class:`vie.config.Config` refuses a mismatched pairing.

What SigLIP buys
----------------
Because the sigmoid loss needs no batch-wide normalisation, SigLIP trains
efficiently at small batch sizes and reaches better accuracy per parameter:

======================================  =======  ==================
model                                   params   ImageNet 0-shot
======================================  =======  ==================
``openai/clip-vit-large-patch14``       428 M    ~75.5 %
``google/siglip-base-patch16-224``      203 M    ~76.2 %
``google/siglip-so400m-patch14-384``    877 M    ~83.2 %
======================================  =======  ==================

Figures are the published ones, not measured here. The base SigLIP is the
default: it beats the CLIP this project started with while being roughly half
its size, so both quality and speed improve at once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

SOFTMAX = "softmax"
SIGMOID = "sigmoid"


def is_siglip(model_name: str) -> bool:
    return "siglip" in model_name.lower()


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


def hypothesis_score(logits: Sequence[float], mode: str) -> float:
    """Score the first prompt against the rest.

    ``softmax``  posterior of the hypothesis among rivals (CLIP).
    ``sigmoid``  the hypothesis' own probability, damped when a rival scores
                 higher — SigLIP's per-pair probability is absolute, so the
                 rivals act as a veto rather than as normalisation.
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
        probs = [_sigmoid(v) for v in logits]
        hypothesis, rivals = probs[0], probs[1:]
        best_rival = max(rivals)
        if best_rival >= hypothesis:
            # A rival explains the image at least as well: reject rather than
            # report a high absolute probability for the wrong reason.
            return hypothesis * (hypothesis / (hypothesis + best_rival))
        return hypothesis

    raise ValueError(f"unknown score mode {mode!r}")


def gate_score(
    logits: Sequence[float], n_positive: int, mode: str
) -> float:
    """Margin between the best positive and the best negative prompt.

    Used by the food gate: several ways of saying "this is food" against
    several ways of saying "this is not". Returns a value in [-1, 1] where
    positive means the image looks more like food than not.
    """
    if n_positive <= 0 or n_positive >= len(logits):
        raise ValueError(
            f"need at least one positive and one negative prompt, "
            f"got {n_positive} positives out of {len(logits)}"
        )

    if mode == SIGMOID:
        probs = [_sigmoid(v) for v in logits]
    elif mode == SOFTMAX:
        probs = _softmax(list(logits))
    else:
        raise ValueError(f"unknown score mode {mode!r}")

    return max(probs[:n_positive]) - max(probs[n_positive:])
