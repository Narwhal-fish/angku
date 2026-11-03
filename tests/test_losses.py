"""Loss registrations from the experiments package."""

from __future__ import annotations

import pytest
import torch

from angku.registry import LOSSES, RegistryError, discover


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments.losses")


def test_cross_entropy_registered_and_computes():
    loss_fn = LOSSES.build("cross_entropy", {})
    logits = torch.zeros(2, 3, 4, 4)  # uniform prediction over 3 classes
    target = torch.zeros(2, 4, 4, dtype=torch.long)
    assert torch.isclose(loss_fn(logits, target), torch.tensor(3.0).log())


def test_cross_entropy_params_forwarded():
    loss_fn = LOSSES.build(
        "cross_entropy",
        {"weight": [1.0, 2.0], "ignore_index": 255, "label_smoothing": 0.1, "reduction": "sum"},
    )
    assert torch.equal(loss_fn.weight, torch.tensor([1.0, 2.0]))
    assert loss_fn.ignore_index == 255
    assert loss_fn.label_smoothing == 0.1
    assert loss_fn.reduction == "sum"


def test_cross_entropy_param_typo_rejected():
    with pytest.raises(RegistryError, match="label_smothing"):
        LOSSES.build("cross_entropy", {"label_smothing": 0.1})


def test_cross_entropy_bad_reduction_rejected():
    with pytest.raises(RegistryError, match="reduction"):
        LOSSES.build("cross_entropy", {"reduction": "avg"})


# --- dice_ce (MONAI): only runs when the `medical` extra is installed ---


def test_dice_ce_registered_even_without_monai():
    assert "dice_ce" in LOSSES


def test_dice_ce_param_typo_rejected_without_building():
    with pytest.raises(RegistryError, match="softmaxx"):
        LOSSES.build("dice_ce", {"softmaxx": True})


def test_dice_ce_from_baseline_yaml_computes_and_backprops():
    pytest.importorskip("monai", reason="medical extra not installed")
    from pathlib import Path

    from angku.config import load_experiment_yaml

    spec = load_experiment_yaml(Path(__file__).parent.parent / "experiments/configs/swinunetr_baseline.yaml")
    loss_fn = LOSSES.build(spec.loss.name, spec.loss.params)

    # runner contract: logits (N, C, *spatial), integer targets (N, *spatial)
    logits = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 8, 8, 8))
    loss = loss_fn(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_dice_ce_perfect_prediction_is_near_zero():
    pytest.importorskip("monai", reason="medical extra not installed")
    loss_fn = LOSSES.build("dice_ce", {"to_onehot_y": True, "softmax": True})
    target = torch.randint(0, 2, (1, 8, 8, 8))
    logits = torch.nn.functional.one_hot(target, 2).permute(0, 4, 1, 2, 3).float() * 50
    assert loss_fn(logits, target) < 1e-3


def test_dice_ce_lambda_weights_forwarded():
    pytest.importorskip("monai", reason="medical extra not installed")
    loss_fn = LOSSES.build("dice_ce", {"to_onehot_y": True, "softmax": True, "lambda_dice": 0.0, "lambda_ce": 1.0})
    logits = torch.zeros(1, 3, 4, 4)  # 2-D works too
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    assert torch.isclose(loss_fn(logits, target), torch.tensor(3.0).log())


# --- cross-entropy without nll_loss: CUDA has no deterministic nll_loss2d kernel ---

_CE_CASES = [
    {},
    {"reduction": "sum"},
    {"reduction": "none"},
    {"weight": [0.3, 1.7, 1.0]},
    {"ignore_index": 1},
    {"label_smoothing": 0.1},
    {"weight": [0.3, 1.7, 1.0], "ignore_index": 2, "label_smoothing": 0.2},
    {"weight": [0.3, 1.7, 1.0], "ignore_index": 0, "label_smoothing": 0.2, "reduction": "sum"},
    {"weight": [0.3, 1.7, 1.0], "ignore_index": 2, "label_smoothing": 0.2, "reduction": "none"},
]


@pytest.mark.parametrize("shape", [(4,), (2, 5, 6), (2, 4, 4, 4)], ids=["classes", "2d", "3d"])
@pytest.mark.parametrize("kw", _CE_CASES, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "default")
def test_one_hot_cross_entropy_matches_pytorch(kw, shape):
    from experiments.losses._cross_entropy import one_hot_cross_entropy

    torch.manual_seed(0)
    logits = torch.randn(shape[0], 3, *shape[1:], dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 3, (shape[0], *shape[1:]))
    args = dict(kw)
    if "weight" in args:
        args["weight"] = torch.tensor(args["weight"], dtype=torch.float64)
    ref = torch.nn.functional.cross_entropy(logits, target, **args)
    (g_ref,) = torch.autograd.grad(ref.sum(), logits)
    got = one_hot_cross_entropy(logits, target, **args)
    (g_got,) = torch.autograd.grad(got.sum(), logits)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(g_got, g_ref)


def test_cross_entropy_on_cpu_is_exactly_pytorch():
    """The one-hot path is CUDA-only: CPU runs (and every earlier bundle) stay bit-for-bit."""
    torch.manual_seed(0)
    logits, target = torch.randn(2, 2, 4, 4, 4), torch.randint(0, 2, (2, 4, 4, 4))
    params = {"weight": [0.5, 2.0], "label_smoothing": 0.1}
    got = LOSSES.build("cross_entropy", params)(logits, target)
    ref = torch.nn.CrossEntropyLoss(weight=torch.tensor(params["weight"]), label_smoothing=0.1)(logits, target)
    assert torch.equal(got, ref)


def test_dice_ce_uses_the_deterministic_cross_entropy_and_is_unchanged_on_cpu():
    pytest.importorskip("monai", reason="medical extra not installed")
    from monai.losses import DiceCELoss

    from experiments.losses._cross_entropy import CrossEntropy

    params = {"to_onehot_y": True, "softmax": True, "label_smoothing": 0.1}
    loss_fn = LOSSES.build("dice_ce", params)
    assert isinstance(loss_fn.inner.cross_entropy, CrossEntropy)
    torch.manual_seed(0)
    logits, target = torch.randn(2, 2, 4, 4, 4), torch.randint(0, 2, (2, 4, 4, 4))
    assert torch.equal(loss_fn(logits, target), DiceCELoss(**params)(logits, target.unsqueeze(1)))
