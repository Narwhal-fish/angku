"""Batch augmentations: MONAI RandFlipd / RandRotate90d semantics, torch RNG, on-device."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from angku.config import load_experiment_yaml
from angku.registry import AUGMENTATIONS, RegistryError, discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load():
    discover("experiments")


def _batch(n=3, spatial=(4, 4, 4)):
    x = torch.arange(n * 2 * 4**3, dtype=torch.float32).reshape(n, 2, *spatial)
    y = torch.arange(n * 4**3).reshape(n, *spatial)
    return x, y


def test_baseline_yaml_augmentations_resolve():
    spec = load_experiment_yaml(CONFIGS / "swinunetr_baseline.yaml")
    augs = [AUGMENTATIONS.build(a.name, a.params) for a in spec.augmentation]
    assert len(augs) == 2
    x, y = _batch()
    for aug in augs:
        x2, y2 = aug(x, y)
        assert x2.shape == x.shape and y2.shape == y.shape


# --- rand_flip ---


def test_rand_flip_prob_one_flips_image_and_label_together():
    aug = AUGMENTATIONS.build("rand_flip", {"prob": 1.0, "spatial_axis": [0]})
    x, y = _batch()
    x2, y2 = aug(x, y)
    assert torch.equal(x2, torch.flip(x, dims=[2]))  # spatial axis 0 == x dim 2
    assert torch.equal(y2, torch.flip(y, dims=[1]))  # == y dim 1


def test_rand_flip_multiple_axes_and_default_all():
    x, y = _batch()
    x2, y2 = AUGMENTATIONS.build("rand_flip", {"prob": 1.0, "spatial_axis": [0, 2]})(x, y)
    assert torch.equal(x2, torch.flip(x, dims=[2, 4]))
    assert torch.equal(y2, torch.flip(y, dims=[1, 3]))
    x3, y3 = AUGMENTATIONS.build("rand_flip", {"prob": 1.0})(x, y)  # None -> every spatial axis
    assert torch.equal(x3, torch.flip(x, dims=[2, 3, 4]))
    assert torch.equal(y3, torch.flip(y, dims=[1, 2, 3]))


def test_rand_flip_prob_zero_is_identity():
    x, y = _batch()
    x2, y2 = AUGMENTATIONS.build("rand_flip", {"prob": 0.0})(x, y)
    assert torch.equal(x2, x) and torch.equal(y2, y)


def test_rand_flip_is_per_sample_and_seeded_by_torch():
    aug = AUGMENTATIONS.build("rand_flip", {"prob": 0.5, "spatial_axis": [0]})
    x, y = _batch(n=64)
    torch.manual_seed(0)
    x2, _ = aug(x, y)
    flipped = [bool(torch.equal(x2[i], torch.flip(x[i], dims=[1]))) for i in range(64)]
    assert 0 < sum(flipped) < 64, "with prob=0.5 some samples flip and some do not"
    torch.manual_seed(0)
    x3, _ = aug(x, y)
    assert torch.equal(x2, x3), "same torch seed -> same decisions (reproducibility contract)"


def test_rand_flip_does_not_touch_input_in_place():
    x, y = _batch()
    x0, y0 = x.clone(), y.clone()
    AUGMENTATIONS.build("rand_flip", {"prob": 1.0})(x, y)
    assert torch.equal(x, x0) and torch.equal(y, y0)


def test_rand_flip_rejects_typo_and_bad_axis():
    with pytest.raises(RegistryError, match="probb"):
        AUGMENTATIONS.build("rand_flip", {"probb": 1.0})
    with pytest.raises(RegistryError):
        AUGMENTATIONS.build("rand_flip", {"prob": 1.5})
    aug = AUGMENTATIONS.build("rand_flip", {"prob": 1.0, "spatial_axis": [3]})
    with pytest.raises(ValueError, match="spatial_axis"):
        aug(*_batch())  # 3-D volume has axes 0..2 only


# --- rand_rotate90 ---


def test_rand_rotate90_prob_one_k_one():
    aug = AUGMENTATIONS.build("rand_rotate90", {"prob": 1.0, "max_k": 1})
    x, y = _batch()
    x2, y2 = aug(x, y)
    assert torch.equal(x2, torch.rot90(x, 1, dims=(2, 3)))  # default spatial_axes (0, 1)
    assert torch.equal(y2, torch.rot90(y, 1, dims=(1, 2)))


def test_rand_rotate90_custom_plane():
    aug = AUGMENTATIONS.build("rand_rotate90", {"prob": 1.0, "max_k": 1, "spatial_axes": [1, 2]})
    x, y = _batch()
    x2, y2 = aug(x, y)
    assert torch.equal(x2, torch.rot90(x, 1, dims=(3, 4)))
    assert torch.equal(y2, torch.rot90(y, 1, dims=(2, 3)))


def test_rand_rotate90_k_is_drawn_per_sample_in_1_to_max_k():
    aug = AUGMENTATIONS.build("rand_rotate90", {"prob": 1.0, "max_k": 3})
    x, y = _batch(n=64)
    torch.manual_seed(1)
    x2, y2 = aug(x, y)
    seen = set()
    for i in range(64):
        ks = [k for k in (1, 2, 3) if torch.equal(x2[i], torch.rot90(x[i], k, dims=(1, 2)))]
        assert len(ks) == 1, "each sample rotated by exactly one k"
        assert torch.equal(y2[i], torch.rot90(y[i], ks[0], dims=(0, 1))), "label rotated identically"
        seen.add(ks[0])
    assert seen == {1, 2, 3}


def test_rand_rotate90_prob_zero_is_identity():
    x, y = _batch()
    x2, y2 = AUGMENTATIONS.build("rand_rotate90", {"prob": 0.0})(x, y)
    assert torch.equal(x2, x) and torch.equal(y2, y)


def test_rand_rotate90_non_square_plane_is_refused():
    aug = AUGMENTATIONS.build("rand_rotate90", {"prob": 1.0})
    x = torch.zeros(2, 1, 4, 6, 4)
    y = torch.zeros(2, 4, 6, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="square"):
        aug(x, y)


def test_rand_rotate90_rejects_typo_and_bad_params():
    with pytest.raises(RegistryError, match="max_kk"):
        AUGMENTATIONS.build("rand_rotate90", {"max_kk": 3})
    with pytest.raises(RegistryError):
        AUGMENTATIONS.build("rand_rotate90", {"max_k": 0})
    with pytest.raises(RegistryError):
        AUGMENTATIONS.build("rand_rotate90", {"spatial_axes": [0, 0]})
