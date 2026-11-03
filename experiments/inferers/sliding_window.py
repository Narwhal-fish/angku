"""MONAI ``sliding_window_inference`` registered as inferer ``sliding_window``.

Requires the ``medical`` extra. The name is always registered so a missing install gives a
clear hint instead of "unknown inferer", and param typos fail without MONAI.

Tiles the input into ``roi_size`` windows (padding a volume smaller than the roi), runs the
model on ``sw_batch_size`` windows at a time and blends overlapping predictions (``constant``
= plain average, ``gaussian`` = centre-weighted). Uses no RNG: same weights, same output.
Windows and the stitched output stay on the input's device.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from torch import Tensor, nn

from angku.registry import INFERERS, Params


class SlidingWindowParams(Params):
    # Required: the window must fit what the model was trained on (usually preprocessing.roi_size).
    roi_size: list[int] = Field(min_length=1, max_length=3)
    sw_batch_size: int = Field(default=1, ge=1)
    overlap: float = Field(default=0.25, ge=0, lt=1)
    mode: Literal["constant", "gaussian"] = "constant"
    sigma_scale: float = Field(default=0.125, gt=0)
    padding_mode: Literal["constant", "reflect", "replicate", "circular"] = "constant"
    cval: float = 0.0

    @field_validator("roi_size")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if any(r < 1 for r in v):
            raise ValueError(f"roi_size must be positive, got {v}")
        return v


@INFERERS.register("sliding_window", params=SlidingWindowParams)
def build_sliding_window(p: SlidingWindowParams):
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as e:
        raise ImportError("inferer 'sliding_window' needs MONAI: uv pip install -e \".[medical]\"") from e

    def infer(model: nn.Module, x: Tensor) -> Tensor:
        if len(p.roi_size) != x.ndim - 2:
            raise ValueError(
                f"sliding_window: roi_size has {len(p.roi_size)} dims but the input has {x.ndim - 2} spatial dims"
            )
        out = sliding_window_inference(
            x,
            roi_size=p.roi_size,
            sw_batch_size=p.sw_batch_size,
            predictor=model,
            overlap=p.overlap,
            mode=p.mode,
            sigma_scale=p.sigma_scale,
            padding_mode=p.padding_mode,
            cval=p.cval,
        )
        return out.as_tensor() if hasattr(out, "as_tensor") else out

    return infer
