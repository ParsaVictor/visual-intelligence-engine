"""Device and dtype resolution.

The original notebook mixed three idioms for the same check::

    if device.type == "cuda":   # line 91  — correct
    if device.type == "cuda":   # line 259 — correct
    if device == "cuda":        # line 643 — always False

``device`` is a ``torch.device`` object, and ``torch.device("cuda") == "cuda"``
evaluates to **False** (verified on torch 2.8). The third check therefore never
fired, so on a CUDA run the CLIP weights were cast to half precision while the
pixel values were left as float32 — which raises::

    RuntimeError: expected scalar type Half but found Float

Resolving device and dtype in one place, and always deriving the dtype from the
resolved device, removes the possibility of the two disagreeing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Runtime:
    """Resolved execution context."""

    device: Any          # torch.device
    use_half: bool

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def autocast_dtype(self) -> Any:
        import torch

        return torch.float16 if self.use_half else torch.float32

    def cast_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move a processor's output to the device with a matching dtype.

        Only floating-point tensors are cast; ``input_ids`` and attention masks
        are integer tensors and must stay integral.
        """
        import torch

        out: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if self.use_half and value.is_floating_point():
                    value = value.half()
            out[key] = value
        return out

    def prepare_model(self, model: Any) -> Any:
        """Move a model to the device and match the runtime dtype."""
        model = model.to(self.device)
        if self.use_half:
            model = model.half()
        return model.eval()


def resolve(preference: str = "auto", half_precision: bool = True) -> Runtime:
    """Resolve a device preference into a concrete runtime.

    Half precision is only ever enabled on CUDA: fp16 on CPU is slower than
    fp32 and unsupported for several ops.
    """
    import torch

    name = preference
    if preference == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no CUDA device is available")

    return Runtime(device=device, use_half=bool(half_precision) and device.type == "cuda")
