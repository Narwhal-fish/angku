"""``rand_flip``: MONAI ``RandFlipd(keys=[image, label])`` semantics on a batch.

With probability ``prob`` per sample, flip along every axis in ``spatial_axis``
(all spatial axes when ``None``). Image and label always flip together.
"""

from __future__ import annotations

import torch
from pydantic import Field, field_validator
from torch import Tensor

from experiments.augmentations._batch import check_axes, coin, n_spatial
from mlops.registry import AUGMENTATIONS, Params


class RandFlipParams(Params):
    prob: float = Field(default=0.1, ge=0, le=1)
    # MONAI accepts an int or a sequence; None means every spatial axis.
    spatial_axis: int | list[int] | None = None

    @field_validator("spatial_axis")
    @classmethod
    def _as_list(cls, v):
        if isinstance(v, int):
            return [v]
        if v is not None and len(set(v)) != len(v):
            raise ValueError("spatial_axis must not repeat an axis")
        return v


@AUGMENTATIONS.register("rand_flip", params=RandFlipParams)
def build_rand_flip(p: RandFlipParams):
    def aug(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        nsp = n_spatial(x, y)
        axes = list(range(nsp)) if p.spatial_axis is None else list(p.spatial_axis)
        check_axes(axes, nsp, "spatial_axis")
        do = coin(x.shape[0], p.prob, x.device)
        if not bool(do.any()):
            return x, y
        x_dims = [a + 2 for a in axes]
        y_dims = [a + 1 for a in axes]
        # torch.where broadcasts the per-sample mask over the remaining dims.
        mx = do.view(-1, *([1] * (x.ndim - 1)))
        my = do.view(-1, *([1] * (y.ndim - 1)))
        return torch.where(mx, torch.flip(x, x_dims), x), torch.where(my, torch.flip(y, y_dims), y)

    return aug
