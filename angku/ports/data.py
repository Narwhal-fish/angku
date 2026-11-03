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

from angku.domain.experiment import Dataset
from angku.domain.split import SplitAssignment

# One training example: (inputs, targets). Batching stacks each side.
Sample = tuple[Tensor, Tensor]


class DataError(Exception):
    """The provider cannot serve the requested dataset."""


@dataclass(frozen=True)
class DataSplits:
    train: TorchDataset[Sample]
    val: TorchDataset[Sample] | None = None
    # Only with ``spec.split`` asking for one. Training never reads it: it is for ``predict``.
    test: TorchDataset[Sample] | None = None


class DataProvider(Protocol):
    def snapshot(self, dataset: Dataset) -> str | None:
        """Content id (``sha256:<64hex>``) of what ``load`` would serve, or None if the
        provider cannot attest to its data. Independent of where the files live."""
        ...

    def cases(self, dataset: Dataset) -> list[str]:
        """Stable ids of every case ``load`` can serve (needed only for ``spec.split``; a provider
        without this method serves its own split and refuses a spec that sets one)."""
        ...

    def load(self, dataset: Dataset, preprocessing: dict[str, Any], split: SplitAssignment | None = None) -> DataSplits:
        """Return the splits for ``dataset`` at its pinned snapshot: the cases ``split`` names
        (ids from ``cases``) or, when None, the provider's own split. Called with ``split`` only
        when the spec sets one.

        Must raise ``DataError`` if it does not serve ``dataset.name`` or the
        snapshot is unavailable, rather than silently substituting other data.
        """
        ...
