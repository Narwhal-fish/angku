"""Synthetic N-D "blob" segmentation data, for smoke tests and CI.

Each sample is a volume with one solid sphere of a random class on a noisy
background; the target is the class mask. Fully deterministic given ``seed``,
so two runs of the same spec see identical data.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from mlops.domain.experiment import Dataset
from mlops.ports.data import DataError, DataSplits


class SyntheticBlobs:
    """Serves exactly one dataset name (default ``synthetic-blobs``)."""

    def __init__(
        self,
        *,
        n_train: int = 8,
        n_val: int = 4,
        shape: tuple[int, ...] = (8, 8, 8),
        in_channels: int = 1,
        num_classes: int = 2,
        seed: int = 0,
        name: str = "synthetic-blobs",
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 (background + at least one blob class)")
        self.n_train = n_train
        self.n_val = n_val
        self.shape = tuple(shape)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seed = seed
        self.name = name

    def load(self, dataset: Dataset, preprocessing: dict[str, Any]) -> DataSplits:
        if dataset.name != self.name:
            raise DataError(f"SyntheticBlobs serves {self.name!r}, not {dataset.name!r}")
        if preprocessing:
            raise DataError(f"SyntheticBlobs has no preprocessing options, got {sorted(preprocessing)}")
        g = torch.Generator().manual_seed(self.seed)
        train = self._make(self.n_train, g)
        val = self._make(self.n_val, g) if self.n_val else None
        return DataSplits(train=train, val=val)

    def _make(self, n: int, g: torch.Generator) -> TensorDataset:
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        grids = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in self.shape], indexing="ij")
        for _ in range(n):
            cls = int(torch.randint(1, self.num_classes, (1,), generator=g))
            center = [float(torch.randint(0, s, (1,), generator=g)) for s in self.shape]
            radius = min(self.shape) / 4
            dist2 = sum((grid - c) ** 2 for grid, c in zip(grids, center, strict=True))
            mask = dist2 <= radius**2
            y = torch.where(mask, torch.tensor(cls), torch.tensor(0)).long()
            x = mask.float().unsqueeze(0).repeat(self.in_channels, *([1] * len(self.shape)))
            x = x + 0.1 * torch.randn(x.shape, generator=g)
            xs.append(x)
            ys.append(y)
        return TensorDataset(torch.stack(xs), torch.stack(ys))
