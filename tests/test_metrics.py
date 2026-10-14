"""Validation metrics: per-sample values, NaN = undefined for that sample."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from mlops.config import load_experiment_yaml
from mlops.registry import METRICS, RegistryError, discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load():
    discover("experiments")


def _logits(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Confident logits whose argmax is ``pred`` (N, *spatial) -> (N, C, *spatial)."""
    return torch.nn.functional.one_hot(pred, num_classes).movedim(-1, 1).float() * 10


def test_baseline_yaml_metrics_resolve():
    spec = load_experiment_yaml(CONFIGS / "swinunetr_baseline.yaml")
    metrics = [METRICS.build(m.name, m.params) for m in spec.metrics]
    target = torch.randint(0, 2, (2, 4, 4, 4))
    out = metrics[0](_logits(target, 2), target)
    assert set(out) == {"dice/spleen"}
    assert out["dice/spleen"].shape == (2,)


def test_dice_param_typo_rejected():
    with pytest.raises(RegistryError, match="include_backgroud"):
        METRICS.build("dice", {"include_backgroud": False})


def test_dice_perfect_and_disjoint_predictions():
    dice = METRICS.build("dice", {})
    target = torch.zeros(2, 4, 4, dtype=torch.long)
    target[:, :2] = 1
    perfect = dice(_logits(target, 2), target)
    assert torch.equal(perfect["dice/1"], torch.ones(2))
    assert torch.equal(perfect["dice/0"], torch.ones(2))
    swapped = dice(_logits(1 - target, 2), target)
    assert torch.equal(swapped["dice/1"], torch.zeros(2))


def test_dice_partial_overlap_value():
    dice = METRICS.build("dice", {"include_background": False})
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2] = 1  # 8 voxels
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    pred[0, 1:3] = 1  # 8 voxels, 4 overlap -> 2*4/16
    out = dice(_logits(pred, 2), target)
    assert out["dice/1"].item() == pytest.approx(0.5)
    assert set(out) == {"dice/1"}  # one class reported -> no redundant mean


def test_dice_mean_over_classes_when_several():
    dice = METRICS.build("dice", {"include_background": False})
    target = torch.tensor([[0, 1, 2, 2]])
    pred = torch.tensor([[0, 1, 1, 2]])  # class 1: 2/3, class 2: 2/3
    out = dice(_logits(pred, 3), target)
    assert set(out) == {"dice/1", "dice/2", "dice/mean"}
    assert out["dice/mean"].item() == pytest.approx(2 / 3)


def test_dice_empty_ground_truth_is_nan_by_default():
    # MONAI ignore_empty=True: a class absent from the label says nothing about the model.
    dice = METRICS.build("dice", {"include_background": False})
    target = torch.zeros(1, 4, dtype=torch.long)
    out = dice(_logits(target, 2), target)
    assert math.isnan(out["dice/1"].item())


def test_dice_empty_ground_truth_scored_when_not_ignored():
    dice = METRICS.build("dice", {"include_background": False, "ignore_empty": False})
    target = torch.zeros(2, 4, dtype=torch.long)
    pred = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]])
    out = dice(_logits(pred, 2), target)
    assert out["dice/1"].tolist() == [1.0, 0.0]  # empty vs empty = 1, false positive = 0


def test_dice_mean_ignores_undefined_classes():
    dice = METRICS.build("dice", {"include_background": False})
    target = torch.tensor([[0, 1, 1, 0]])  # class 2 absent -> NaN, mean is class 1 only
    out = dice(_logits(target, 3), target)
    assert math.isnan(out["dice/2"].item())
    assert out["dice/mean"].item() == 1.0


def test_dice_class_names_label_the_keys():
    dice = METRICS.build("dice", {"class_names": ["background", "spleen"]})
    target = torch.randint(0, 2, (1, 4, 4))
    assert set(dice(_logits(target, 2), target)) == {"dice/background", "dice/spleen", "dice/mean"}


def test_dice_class_names_must_match_channels():
    dice = METRICS.build("dice", {"class_names": ["background", "spleen", "liver"]})
    target = torch.randint(0, 2, (1, 4, 4))
    with pytest.raises(ValueError, match="class_names has 3 entries but the model outputs 2 classes"):
        dice(_logits(target, 2), target)


def test_dice_class_names_must_be_unique():
    with pytest.raises(RegistryError, match="unique"):
        METRICS.build("dice", {"class_names": ["a", "a"]})


def test_dice_matches_monai_dice_metric():
    pytest.importorskip("monai", reason="medical extra not installed")
    from monai.metrics import DiceMetric

    g = torch.Generator().manual_seed(0)
    target = torch.randint(0, 3, (4, 6, 6, 6), generator=g)
    target[0][target[0] == 2] = 0  # sample 0 has no class 2 -> NaN in both
    logits = torch.randn(4, 3, 6, 6, 6, generator=g)

    ours = METRICS.build("dice", {"include_background": False})(logits, target)
    ref = DiceMetric(include_background=False, reduction="none", ignore_empty=True)
    pred_1h = torch.nn.functional.one_hot(logits.argmax(1), 3).movedim(-1, 1)
    true_1h = torch.nn.functional.one_hot(target, 3).movedim(-1, 1)
    expected = ref(pred_1h, true_1h)  # (N, C-1)
    for c in (1, 2):
        assert torch.allclose(ours[f"dice/{c}"].float(), expected[:, c - 1].float(), equal_nan=True)
