"""A minimal N-D conv net registered as ``tiny_cnn``.

Framework-only (torch), no domain-specific dependencies. Its job is to be the
smallest thing that exercises the platform end to end: registry -> build ->
forward -> (later) train. Works for 2D or 3D inputs.
"""

from __future__ import annotations

from pydantic import Field
from torch import nn

from angku.registry import MODELS, Params


class TinyCNNParams(Params):
    in_channels: int = Field(gt=0)
    out_channels: int = Field(gt=0)
    hidden: int = Field(default=8, gt=0)
    spatial_dims: int = Field(default=3, ge=1, le=3)
    # Element-wise dropout after the hidden activation. 0 (default) adds no layer at all, so the
    # state_dict keys and existing runs are unchanged; > 0 is what `angku predict --mc-samples` needs.
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


@MODELS.register("tiny_cnn", params=TinyCNNParams)
def build_tiny_cnn(p: TinyCNNParams) -> nn.Module:
    conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[p.spatial_dims]
    layers: list[nn.Module] = [conv(p.in_channels, p.hidden, kernel_size=3, padding=1), nn.ReLU()]
    if p.dropout > 0:
        layers.append(nn.Dropout(p.dropout))
    layers.append(conv(p.hidden, p.out_channels, kernel_size=1))
    return nn.Sequential(*layers)
