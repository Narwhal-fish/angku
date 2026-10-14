"""Inferers: how the model is applied to a validation volume (``spec.inference``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from mlops.config import load_experiment_yaml
from mlops.registry import INFERERS, MODELS, RegistryError, discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load():
    discover("experiments")


class _FixedInput(nn.Module):
    """Refuses any input whose spatial size is not ``roi``: proves the inferer tiles."""

    def __init__(self, roi: tuple[int, ...], out_channels: int = 2) -> None:
        super().__init__()
        self.roi = roi
        self.conv = nn.Conv3d(1, out_channels, 3, padding=1)

    def forward(self, x):
        assert tuple(x.shape[2:]) == self.roi, f"window {tuple(x.shape[2:])} != roi {self.roi}"
        return self.conv(x)


# --- registration & params (no MONAI needed) ---


def test_sliding_window_registered_even_without_monai():
    assert "sliding_window" in INFERERS


def test_sliding_window_param_typo_rejected():
    with pytest.raises(RegistryError, match="overlapp"):
        INFERERS.build("sliding_window", {"roi_size": [4, 4, 4], "overlapp": 0.5})


@pytest.mark.parametrize("bad", [
    {"roi_size": [0, 4, 4]},
    {"roi_size": [4, 4, 4], "overlap": 1.0},
    {"roi_size": [4, 4, 4], "sw_batch_size": 0},
    {"roi_size": [4, 4, 4], "mode": "triangle"},
    {},
])
def test_sliding_window_bad_params_rejected(bad):
    with pytest.raises(RegistryError):
        INFERERS.build("sliding_window", bad)


# --- behaviour (MONAI) ---


def test_sliding_window_tiles_a_volume_larger_than_the_roi():
    pytest.importorskip("monai", reason="medical extra not installed")
    infer = INFERERS.build("sliding_window", {"roi_size": [4, 4, 4], "sw_batch_size": 3, "overlap": 0.25})
    model = _FixedInput((4, 4, 4)).eval()
    x = torch.randn(1, 1, 10, 9, 7)  # not a multiple of the roi on any axis
    with torch.no_grad():
        out = infer(model, x)
    assert out.shape == (1, 2, 10, 9, 7)


def test_sliding_window_with_roi_equal_to_the_volume_is_the_plain_forward():
    pytest.importorskip("monai", reason="medical extra not installed")
    infer = INFERERS.build("sliding_window", {"roi_size": [6, 6, 6], "overlap": 0.0})
    model = MODELS.build("tiny_cnn", {"in_channels": 1, "out_channels": 2, "hidden": 4, "spatial_dims": 3}).eval()
    x = torch.randn(2, 1, 6, 6, 6)
    with torch.no_grad():
        assert torch.allclose(infer(model, x), model(x), atol=1e-6)


def test_sliding_window_pads_a_volume_smaller_than_the_roi():
    pytest.importorskip("monai", reason="medical extra not installed")
    infer = INFERERS.build("sliding_window", {"roi_size": [8, 8, 8]})
    with torch.no_grad():
        out = infer(_FixedInput((8, 8, 8)).eval(), torch.randn(1, 1, 5, 8, 6))
    assert out.shape == (1, 2, 5, 8, 6)


def test_sliding_window_gaussian_is_deterministic():
    pytest.importorskip("monai", reason="medical extra not installed")
    infer = INFERERS.build("sliding_window", {"roi_size": [4, 4, 4], "overlap": 0.5, "mode": "gaussian"})
    torch.manual_seed(0)
    model = _FixedInput((4, 4, 4)).eval()
    x = torch.randn(1, 1, 9, 9, 9)
    with torch.no_grad():
        assert torch.equal(infer(model, x), infer(model, x))


def test_roi_dims_must_match_the_volume():
    pytest.importorskip("monai", reason="medical extra not installed")
    infer = INFERERS.build("sliding_window", {"roi_size": [4, 4]})
    with pytest.raises(ValueError, match="roi_size has 2 dims but the input has 3 spatial dims"):
        infer(_FixedInput((4, 4, 4)), torch.randn(1, 1, 8, 8, 8))


def test_baseline_yaml_inference_resolves():
    spec = load_experiment_yaml(CONFIGS / "swinunetr_baseline.yaml")
    assert spec.inference is not None and spec.inference.name == "sliding_window"
    INFERERS.validate_params(spec.inference.name, spec.inference.params)
