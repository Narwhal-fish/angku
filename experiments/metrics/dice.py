"""Per-class Dice registered as metric ``dice``.

MONAI ``DiceMetric`` semantics (argmax the logits, one-hot both sides, Dice per
sample per class, ``ignore_empty``), implemented in torch so it needs no extra and
returns per-sample values the runner can sum across batches and ranks.

Keys are ``dice/<class>`` (class index, or ``class_names[index]``), plus
``dice/mean`` (per sample, mean over the reported classes that are defined) when
more than one class is reported. NaN means undefined for that sample: with
``ignore_empty`` a class absent from the label says nothing about the model.
"""

from __future__ import annotations

import torch
from pydantic import field_validator
from torch import Tensor

from mlops.registry import METRICS, Params


class DiceParams(Params):
    include_background: bool = True
    ignore_empty: bool = True
    # One name per model output channel (background included even when not reported).
    class_names: list[str] | None = None

    @field_validator("class_names")
    @classmethod
    def _unique(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(set(v)) != len(v):
            raise ValueError(f"class_names must be unique, got {v}")
        return v


@METRICS.register("dice", params=DiceParams)
def build_dice(p: DiceParams):
    def dice(logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        num_classes = logits.shape[1]
        names = p.class_names if p.class_names is not None else [str(c) for c in range(num_classes)]
        if len(names) != num_classes:
            raise ValueError(f"dice: class_names has {len(names)} entries but the model outputs {num_classes} classes")
        spatial = tuple(range(2, logits.ndim))
        pred = torch.nn.functional.one_hot(logits.argmax(1), num_classes).movedim(-1, 1)
        true = torch.nn.functional.one_hot(target.long(), num_classes).movedim(-1, 1)
        inter = (pred * true).sum(spatial).double()
        denom = (pred.sum(spatial) + true.sum(spatial)).double()
        # Nothing predicted and nothing labelled: perfect, unless empty labels are ignored.
        score = torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))
        if p.ignore_empty:
            score = score.masked_fill(true.sum(spatial) == 0, float("nan"))
        first = 0 if p.include_background else 1
        out = {f"dice/{names[c]}": score[:, c] for c in range(first, num_classes)}
        if num_classes - first > 1:
            out["dice/mean"] = score[:, first:].nanmean(dim=1)
        return out

    return dice
