"""Port: where training data comes from.

The runner never knows how a dataset is stored (NIfTI on disk, DVC, lakeFS,
synthetic). It asks a ``DataProvider`` for the dataset pinned in the spec and
gets back torch ``Dataset`` objects whose items are ``(inputs, targets)``
tensor pairs. Preprocessing is the provider's job: it receives the free-form
``preprocessing`` section verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from mlops.domain.experiment import Dataset

# One training example: (inputs, targets). Batching stacks each side.
Sample = tuple[Tensor, Tensor]


class DataError(Exception):
    """The provider cannot serve the requested dataset."""


@dataclass(frozen=True)
class DataSplits:
    train: TorchDataset[Sample]
    val: TorchDataset[Sample] | None = None


class DataProvider(Protocol):
    def load(self, dataset: Dataset, preprocessing: dict[str, Any]) -> DataSplits:
        """Return the splits for ``dataset`` at its pinned snapshot.

        Must raise ``DataError`` if it does not serve ``dataset.name`` or the
        snapshot is unavailable, rather than silently substituting other data.
        """
        ...
