"""Keep-best: which epoch's weights to keep besides the final ones. Pure data + one pure function.

A launch option, never in any id: choosing which epoch to keep does not change training.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict


class BestCriterion(BaseModel):
    """``val/dice/mean:max`` = keep the weights of the epoch with the highest ``val/dice/mean``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    mode: Literal["min", "max"]

    @classmethod
    def parse(cls, text: str) -> BestCriterion:
        metric, sep, mode = text.rpartition(":")
        if not sep or not metric or mode not in ("min", "max"):
            raise ValueError(f"keep-best must look like METRIC:min|max (e.g. val/loss:min), got {text!r}")
        return cls(metric=metric, mode=mode)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return f"{self.metric}:{self.mode}"


class BestRecord(BaseModel):
    """Which epoch ``best.pt`` holds. In the manifest of a FINISHED run started with keep-best."""

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    metric: str
    mode: Literal["min", "max"]
    epoch: int | None  # None: the metric was never defined (NaN every epoch): no best.pt
    value: float | None


def best_epoch(history: Sequence[dict[str, float]], criterion: BestCriterion) -> tuple[int, float] | None:
    """``(epoch, value)`` of the best epoch, 1-based. Strictly better wins, so a tie keeps the
    earlier epoch; NaN (undefined) never wins. None when no epoch has a defined value."""
    best: tuple[int, float] | None = None
    for epoch, row in enumerate(history, start=1):
        v = row.get(criterion.metric)
        if v is None or math.isnan(v):
            continue
        if best is None or (v < best[1] if criterion.mode == "min" else v > best[1]):
            best = (epoch, v)
    return best


def best_record(history: Sequence[dict[str, float]], criterion: BestCriterion) -> BestRecord:
    found = best_epoch(history, criterion)
    epoch, value = found if found is not None else (None, None)
    return BestRecord(metric=criterion.metric, mode=criterion.mode, epoch=epoch, value=value)
