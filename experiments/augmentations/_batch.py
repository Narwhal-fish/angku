"""Shared helpers for batch augmentations.

Runner contract: ``x`` is ``(N, C, *spatial)`` and ``y`` is ``(N, *spatial)``.
Spatial axis ``i`` (MONAI numbering, 0-based over the spatial dims) is
``x`` dim ``i + 2`` and ``y`` dim ``i + 1``.

Randomness is drawn from torch's global RNG, which ``run_experiment`` seeds
before building anything. MONAI's own transforms carry an unseeded numpy
``RandomState``, which would break "same seed -> same run"; that is why these
mirror the MONAI semantics instead of wrapping ``RandFlipd`` / ``RandRotate90d``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def n_spatial(x: Tensor, y: Tensor) -> int:
    if x.ndim < 3 or y.ndim != x.ndim - 1 or x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[1:]:
        raise ValueError(f"expected x (N, C, *spatial) and y (N, *spatial); got x {tuple(x.shape)}, y {tuple(y.shape)}")
    return x.ndim - 2


def check_axes(axes: list[int], nsp: int, name: str) -> None:
    bad = [a for a in axes if not 0 <= a < nsp]
    if bad:
        raise ValueError(f"{name} {bad} out of range for a {nsp}-D volume (valid: 0..{nsp - 1})")


def coin(n: int, prob: float, device: torch.device) -> Tensor:
    """Per-sample Bernoulli(prob) decisions, drawn on the CPU generator for reproducibility."""
    return (torch.rand(n) < prob).to(device)
