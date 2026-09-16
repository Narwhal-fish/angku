"""MONAI ``DiceCELoss`` registered as loss ``dice_ce``.

Requires the ``medical`` extra. The name is always registered so a missing
install gives a clear hint instead of "unknown loss".

Runner contract is logits ``(N, C, *spatial)`` and integer targets
``(N, *spatial)``; MONAI wants the target with a channel dim ``(N, 1, *spatial)``
when ``to_onehot_y`` is set, so the wrapper adds it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from torch import Tensor, nn

from mlops.registry import LOSSES, Params


class DiceCEParams(Params):
    include_background: bool = True
    to_onehot_y: bool = False
    sigmoid: bool = False
    softmax: bool = False
    squared_pred: bool = False
    jaccard: bool = False
    reduction: Literal["mean", "sum", "none"] = "mean"
    smooth_nr: float = Field(default=1e-5, ge=0)
    smooth_dr: float = Field(default=1e-5, ge=0)
    batch: bool = False
    lambda_dice: float = Field(default=1.0, ge=0)
    lambda_ce: float = Field(default=1.0, ge=0)
    label_smoothing: float = Field(default=0.0, ge=0, le=1)


class _DiceCE(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        return self.inner(logits, target)


@LOSSES.register("dice_ce", params=DiceCEParams)
def build_dice_ce(p: DiceCEParams) -> nn.Module:
    try:
        from monai.losses import DiceCELoss
    except ImportError as e:
        raise ImportError("loss 'dice_ce' needs MONAI: uv pip install -e \".[medical]\"") from e
    return _DiceCE(DiceCELoss(**p.model_dump()))
