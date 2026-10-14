"""Medical Segmentation Decathlon Task09 (spleen CT), registered as dataset ``msd-task09-spleen``.

Requires the ``medical`` extra (MONAI + nibabel). The name is always registered so a
missing install gives a clear hint instead of "unknown dataset".

Provider config (operational, never in any id): ``root`` = the directory that holds
``Task09_Spleen/`` (``--data-root`` / ``MLOPS_DATA_ROOT`` on the CLI).

The yaml's ``preprocessing`` section (in the fingerprint) is validated by
``SpleenPreprocessing``. Deterministic steps follow the MONAI spleen tutorial and run
once, at ``load``: orient RAS -> resample to ``spacing`` -> clip+scale
``intensity_range`` to [0, 1] -> crop to body foreground -> pad up to ``roi_size``.
They use MONAI transforms (deterministic, so MONAI's own RNG never matters).

Random train crops follow ``RandCropByPosNegLabeld`` semantics but draw from torch's
global RNG, like the augmentations, so the run seed decides them: with probability
``pos / (pos + neg)`` the crop is centred on a random label voxel, otherwise on a
uniformly random voxel. Val items are one fixed crop centred on the label centroid
(stand-in until full-volume sliding-window evaluation, roadmap item 8).

Split: cases sorted by file name, the last ``num_val`` are validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import Field, ValidationError
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from mlops.domain.experiment import Dataset
from mlops.ports.data import DataError, DataSplits, Sample
from mlops.registry import DATASETS, Params

NAME = "msd-task09-spleen"
TASK_DIR = "Task09_Spleen"

Vec3f = tuple[float, float, float]
Vec3i = tuple[int, int, int]


class SpleenConfig(Params):
    root: Path


class SpleenPreprocessing(Params):
    spacing: Vec3f = (1.5, 1.5, 2.0)
    intensity_range: tuple[float, float] = (-57.0, 164.0)
    roi_size: Vec3i = (96, 96, 96)
    num_val: int = Field(default=9, ge=1)
    pos: float = Field(default=1.0, ge=0)
    neg: float = Field(default=1.0, ge=0)


@dataclass(frozen=True)
class Case:
    name: str
    image: Tensor  # (1, D, H, W) float32 in [0, 1]
    label: Tensor  # (D, H, W) long
    foreground: Tensor  # (N, 3) indices of label > 0


@DATASETS.register(NAME, params=SpleenConfig)
def build_msd_spleen(p: SpleenConfig) -> MsdSpleen:
    return MsdSpleen(p.root)


class MsdSpleen:
    def __init__(self, root: Path | str) -> None:
        self.task = Path(root) / TASK_DIR

    def load(self, dataset: Dataset, preprocessing: dict[str, Any]) -> DataSplits:
        if dataset.name != NAME:
            raise DataError(f"MsdSpleen serves {NAME!r}, not {dataset.name!r}")
        try:
            p = SpleenPreprocessing.model_validate(preprocessing)
        except ValidationError as e:
            raise DataError(f"invalid preprocessing for {NAME!r}:\n{e}") from e
        if p.pos + p.neg <= 0:
            raise DataError("preprocessing.pos + preprocessing.neg must be > 0")

        pairs = self._list_cases()
        if p.num_val >= len(pairs):
            raise DataError(f"num_val={p.num_val} leaves no training case ({len(pairs)} cases in {self.task})")
        cases = _preprocess(pairs, p)
        n_train = len(cases) - p.num_val
        return DataSplits(
            train=RandomCrops(cases[:n_train], p.roi_size, pos=p.pos, neg=p.neg),
            val=CentredCrops(cases[n_train:], p.roi_size),
        )

    def _list_cases(self) -> list[tuple[str, Path, Path]]:
        index = self.task / "dataset.json"
        if not index.is_file():
            raise DataError(f"{NAME}: no {index} (expected the MSD {TASK_DIR} layout under the data root)")
        entries = json.loads(index.read_text())["training"]
        pairs = []
        for e in entries:
            image, label = self.task / e["image"], self.task / e["label"]
            for f in (image, label):
                if not f.is_file():
                    raise DataError(f"{NAME}: dataset.json lists {f}, which does not exist")
            pairs.append((image.name.removesuffix(".nii.gz"), image, label))
        return sorted(pairs, key=lambda t: t[1].name)


def _preprocess(pairs: list[tuple[str, Path, Path]], p: SpleenPreprocessing) -> list[Case]:
    try:
        from monai import transforms as T
    except ImportError as e:
        raise ImportError(f"dataset {NAME!r} needs MONAI + nibabel: uv pip install -e \".[medical]\"") from e

    keys = ["image", "label"]
    pipeline = T.Compose([
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=keys),
        # labels pinned: MONAI is changing its default, which must not silently move our voxels
        T.Orientationd(keys=keys, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        T.Spacingd(keys=keys, pixdim=p.spacing, mode=("bilinear", "nearest"), dtype=torch.float32),
        T.ScaleIntensityRanged(
            keys="image", a_min=p.intensity_range[0], a_max=p.intensity_range[1], b_min=0.0, b_max=1.0, clip=True
        ),
        T.CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        T.SpatialPadd(keys=keys, spatial_size=p.roi_size),
    ])
    cases = []
    for name, image, label in pairs:
        d = pipeline({"image": str(image), "label": str(label)})
        img = d["image"].as_tensor().float().contiguous()
        lbl = d["label"].as_tensor()[0].round().long().contiguous()
        cases.append(Case(name=name, image=img, label=lbl, foreground=torch.nonzero(lbl > 0)))
    return cases


def _crop(case: Case, centre: Tensor, roi: Vec3i) -> Sample:
    size = case.label.shape
    start = [min(max(int(c) - r // 2, 0), s - r) for c, r, s in zip(centre.tolist(), roi, size)]
    sl = tuple(slice(a, a + r) for a, r in zip(start, roi))
    return case.image[(slice(None), *sl)].clone(), case.label[sl].clone()


class RandomCrops(TorchDataset[Sample]):
    """One random ``roi`` crop per case per access; draws from torch's global RNG."""

    def __init__(self, cases: list[Case], roi: Vec3i, *, pos: float, neg: float) -> None:
        self.cases, self.roi = cases, roi
        self.p_pos = pos / (pos + neg)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int) -> Sample:
        case = self.cases[i]
        fg = case.foreground
        if len(fg) and torch.rand(()) < self.p_pos:
            centre = fg[torch.randint(len(fg), ())]
        else:
            centre = torch.stack([torch.randint(s, ()) for s in case.label.shape])
        return _crop(case, centre, self.roi)


class CentredCrops(TorchDataset[Sample]):
    """One fixed ``roi`` crop per case, centred on the label centroid (volume centre if empty)."""

    def __init__(self, cases: list[Case], roi: Vec3i) -> None:
        self.cases, self.roi = cases, roi

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int) -> Sample:
        case = self.cases[i]
        if len(case.foreground):
            centre = case.foreground.float().mean(0).round().long()
        else:
            centre = torch.tensor(case.label.shape) // 2
        return _crop(case, centre, self.roi)
