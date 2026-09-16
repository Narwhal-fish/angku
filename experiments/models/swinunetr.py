"""SwinUNETR (MONAI) registered as model ``swinunetr``.

Requires the ``medical`` extra: ``uv pip install -e ".[medical]"``.
The name is always registered so the platform can report a clear install hint
at build time instead of an unhelpful "unknown model".

Param names mirror ``monai.networks.nets.SwinUNETR.__init__`` in MONAI 1.6.0
(no ``img_size`` argument any more). Input spatial dims must be divisible by
``patch_size ** 5`` (32 for the default patch_size=2).
"""

from __future__ import annotations

from pydantic import Field
from torch import nn

from mlops.registry import MODELS, Params


class SwinUNETRParams(Params):
    in_channels: int = Field(gt=0)
    out_channels: int = Field(gt=0)
    feature_size: int = Field(default=24, gt=0)
    patch_size: int = Field(default=2, gt=0)
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    window_size: int = Field(default=7, gt=0)
    drop_rate: float = Field(default=0.0, ge=0, le=1)
    attn_drop_rate: float = Field(default=0.0, ge=0, le=1)
    dropout_path_rate: float = Field(default=0.0, ge=0, le=1)
    norm_name: str = "instance"
    use_checkpoint: bool = False
    spatial_dims: int = Field(default=3, ge=2, le=3)
    use_v2: bool = False


@MODELS.register("swinunetr", params=SwinUNETRParams)
def build_swinunetr(p: SwinUNETRParams) -> nn.Module:
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as e:
        raise ImportError(
            "model 'swinunetr' needs MONAI: uv pip install -e \".[medical]\""
        ) from e
    return SwinUNETR(**p.model_dump())
