"""A minimal N-D conv net registered as ``tiny_cnn``.

Framework-only (torch), no domain-specific dependencies. Its job is to be the
smallest thing that exercises the platform end to end: registry -> build ->
forward -> (later) train. Works for 2D or 3D inputs.
"""

from __future__ import annotations

from pydantic import Field
from torch import nn

from mlops.registry import MODELS, Params


class TinyCNNParams(Params):
    in_channels: int = Field(gt=0)
    out_channels: int = Field(gt=0)
    hidden: int = Field(default=8, gt=0)
    spatial_dims: int = Field(default=3, ge=1, le=3)


@MODELS.register("tiny_cnn", params=TinyCNNParams)
def build_tiny_cnn(p: TinyCNNParams) -> nn.Module:
    conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[p.spatial_dims]
    return nn.Sequential(
        conv(p.in_channels, p.hidden, kernel_size=3, padding=1),
        nn.ReLU(),
        conv(p.hidden, p.out_channels, kernel_size=1),
    )
