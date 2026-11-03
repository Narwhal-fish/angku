"""``torch.nn.CrossEntropyLoss`` registered as loss ``cross_entropy``.

On CUDA with deterministic algorithms on it computes the same loss without ``nll_loss``, which
has no deterministic CUDA kernel for spatial targets (see ``_cross_entropy.py``); elsewhere it
is exactly ``nn.CrossEntropyLoss``.

Framework-only (torch). Expects logits of shape ``(N, C, *spatial)`` and
integer class targets of shape ``(N, *spatial)``, i.e. the standard
segmentation / classification contract.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import Field
from torch import nn

from angku.registry import LOSSES, Params
from experiments.losses._cross_entropy import CrossEntropy


class CrossEntropyParams(Params):
    # Per-class rescaling weights, length must equal the number of classes.
    weight: list[float] | None = None
    ignore_index: int = -100
    label_smoothing: float = Field(default=0.0, ge=0, le=1)
    reduction: Literal["mean", "sum", "none"] = "mean"


@LOSSES.register("cross_entropy", params=CrossEntropyParams)
def build_cross_entropy(p: CrossEntropyParams) -> nn.Module:
    weight = torch.tensor(p.weight, dtype=torch.float32) if p.weight is not None else None
    return CrossEntropy(
        weight=weight,
        ignore_index=p.ignore_index,
        label_smoothing=p.label_smoothing,
        reduction=p.reduction,
    )
