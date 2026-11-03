"""Which case goes to train, val and test: a pure function of the case ids and ``spec.split``.

The provider lists its cases (``DataProvider.cases``); this module decides, so every provider
splits the same way and the bundle can record the outcome (``split.json``).

Order: ``sorted`` = by case id; ``shuffled`` = by sha256(seed, case id). Hashing rather than
``random.shuffle`` keeps the assignment independent of the Python version and of the order the
provider lists its cases in. Test is taken from the front of that order, then val (holdout) or
the folds (kfold) from the rest. Counts: ``count_of`` (round half up, at least 1 unless 0).
Each part comes back sorted by case id.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from angku.domain.errors import DomainError
from angku.domain.experiment import Split


class SplitError(DomainError):
    """The cases cannot be split as the spec asks (too few, duplicates)."""


class SplitAssignment(BaseModel):
    """Case ids per part. Stored in the bundle as ``split.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: list[str]
    val: list[str]
    test: list[str] = []

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def count_of(fraction: float, n: int) -> int:
    """``fraction`` of ``n`` cases, rounded half up; a non-zero fraction gets at least one case."""
    if fraction == 0:
        return 0
    return max(1, math.floor(fraction * n + 0.5))


def _ordered(case_ids: Sequence[str], split: Split) -> list[str]:
    ids = sorted(case_ids)
    if split.order == "sorted":
        return ids
    return sorted(ids, key=lambda c: hashlib.sha256(f"{split.seed}\0{c}".encode()).hexdigest())


def assign_split(case_ids: Sequence[str], split: Split) -> SplitAssignment:
    if not case_ids:
        raise SplitError("the dataset has no cases to split")
    if len(set(case_ids)) != len(case_ids):
        dupes = sorted({c for c in case_ids if case_ids.count(c) > 1})
        raise SplitError(f"duplicate case ids: {', '.join(dupes[:5])}")
    n = len(case_ids)
    order = _ordered(case_ids, split)
    n_test = count_of(split.test, n)
    test, rest = order[:n_test], order[n_test:]
    if split.kind == "holdout":
        assert split.val is not None
        n_val = count_of(split.val, n)
        val, train = rest[:n_val], rest[n_val:]
    else:
        assert split.k is not None and split.fold is not None
        if len(rest) < split.k:
            raise SplitError(f"{len(rest)} cases left after test: fewer cases than k={split.k} folds")
        size, extra = divmod(len(rest), split.k)
        start = split.fold * size + min(split.fold, extra)
        stop = start + size + (1 if split.fold < extra else 0)
        val, train = rest[start:stop], rest[:start] + rest[stop:]
    if not train:
        raise SplitError(f"{n} cases: test {len(test)} + val {len(val)} leave no training case")
    return SplitAssignment(train=sorted(train), val=sorted(val), test=sorted(test))
