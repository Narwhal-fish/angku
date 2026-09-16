"""``rand_rotate90``: MONAI ``RandRotate90d(keys=[image, label])`` semantics on a batch.

With probability ``prob`` per sample, rotate by ``k * 90`` degrees in the plane
``spatial_axes`` with ``k`` drawn uniformly from ``1..max_k``. The plane must
be square so every sample in the batch keeps the same shape.
"""

from __future__ import annotations

import torch
from pydantic import Field, field_validator
from torch import Tensor

from experiments.augmentations._batch import check_axes, coin, n_spatial
from mlops.registry import AUGMENTATIONS, Params


class RandRotate90Params(Params):
    prob: float = Field(default=0.1, ge=0, le=1)
    max_k: int = Field(default=3, ge=1)
    spatial_axes: tuple[int, int] = (0, 1)

    @field_validator("spatial_axes")
    @classmethod
    def _distinct(cls, v):
        if v[0] == v[1]:
            raise ValueError("spatial_axes must name two different axes")
        return v


@AUGMENTATIONS.register("rand_rotate90", params=RandRotate90Params)
def build_rand_rotate90(p: RandRotate90Params):
    a, b = p.spatial_axes

    def aug(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        nsp = n_spatial(x, y)
        check_axes([a, b], nsp, "spatial_axes")
        if x.shape[a + 2] != x.shape[b + 2]:
            raise ValueError(
                f"rand_rotate90 needs a square plane; spatial_axes {p.spatial_axes} have sizes "
                f"{x.shape[a + 2]} x {x.shape[b + 2]}"
            )
        do = coin(x.shape[0], p.prob, x.device)
        if not bool(do.any()):
            return x, y
        ks = torch.randint(1, p.max_k + 1, (x.shape[0],)).tolist()
        xs, ys = [], []
        for i, (flag, k) in enumerate(zip(do.tolist(), ks, strict=True)):
            if flag:
                xs.append(torch.rot90(x[i], k, dims=(a + 1, b + 1)))
                ys.append(torch.rot90(y[i], k, dims=(a, b)))
            else:
                xs.append(x[i])
                ys.append(y[i])
        return torch.stack(xs), torch.stack(ys)

    return aug
