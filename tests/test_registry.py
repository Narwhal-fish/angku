"""Registry behaviour + the model registrations from the experiments package."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mlops.config import load_experiment_yaml
from mlops.registry import MODELS, Params, Registry, RegistryError, discover

CONFIGS = Path(__file__).parent.parent / "experiments/configs"


@pytest.fixture(scope="module", autouse=True)
def _load_experiments():
    discover("experiments.models")


# --- generic registry semantics (throwaway registry, no torch involved) ---


class _P(Params):
    width: int = 8


def test_register_get_build():
    reg: Registry[dict] = Registry("thing")

    @reg.register("a", params=_P)
    def make_a(p: _P) -> dict:
        return {"width": p.width}

    assert "a" in reg and reg.names() == ["a"]
    assert reg.build("a", {}) == {"width": 8}
    assert reg.build("a", {"width": 3}) == {"width": 3}


def test_duplicate_registration_is_an_error():
    reg: Registry[int] = Registry("thing")
    reg.register("a", params=_P)(lambda p: 1)
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("a", params=_P)(lambda p: 2)


def test_unknown_name_lists_known_ones():
    reg: Registry[int] = Registry("thing")
    reg.register("a", params=_P)(lambda p: 1)
    with pytest.raises(RegistryError, match=r"unknown thing 'zzz'; registered: a"):
        reg.get("zzz")


def test_param_typo_rejected():
    reg: Registry[int] = Registry("thing")
    reg.register("a", params=_P)(lambda p: 1)
    with pytest.raises(RegistryError, match="widht"):
        reg.build("a", {"widht": 3})


def test_params_schema_must_subclass_params():
    reg: Registry[int] = Registry("thing")
    with pytest.raises(RegistryError):
        reg.register("a", params=dict)  # type: ignore[arg-type]


# --- discovery of experiments/models ---


def test_discover_registers_models():
    assert "tiny_cnn" in MODELS
    assert "swinunetr" in MODELS  # name is registered even without MONAI installed


def test_discover_is_idempotent():
    discover("experiments.models")  # second call must not raise "already registered"
    assert MODELS.names() == ["swinunetr", "tiny_cnn"]


# --- generic model, driven by experiment.yaml (core path, torch only) ---


def test_build_tiny_cnn_from_yaml_and_forward():
    spec = load_experiment_yaml(CONFIGS / "tiny_cnn_smoke.yaml")
    model = MODELS.build(spec.model.name, spec.model.params)
    p = spec.model.params
    x = torch.zeros(1, p["in_channels"], 8, 8, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, p["out_channels"], 8, 8, 8)


def test_tiny_cnn_param_typo_rejected():
    with pytest.raises(RegistryError, match="hiden"):
        MODELS.build("tiny_cnn", {"in_channels": 1, "out_channels": 2, "hiden": 4})


def test_tiny_cnn_dropout_is_a_layer_only_when_asked_for():
    plain = MODELS.build("tiny_cnn", {"in_channels": 1, "out_channels": 2})
    assert not any(isinstance(m, torch.nn.Dropout) for m in plain.modules())
    assert list(plain.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias"]  # unchanged
    model = MODELS.build("tiny_cnn", {"in_channels": 1, "out_channels": 2, "dropout": 0.3})
    assert [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)] == [0.3]
    assert model(torch.zeros(1, 1, 4, 4, 4)).shape == (1, 2, 4, 4, 4)
    with pytest.raises(RegistryError, match="dropout"):
        MODELS.build("tiny_cnn", {"in_channels": 1, "out_channels": 2, "dropout": 1.0})


# --- SwinUNETR: only runs when the `medical` extra is installed ---


def test_swinunetr_param_typo_rejected_without_building():
    # schema validation happens before the factory, so no MONAI needed here
    with pytest.raises(RegistryError, match="feature_sizee"):
        MODELS.build("swinunetr", {"in_channels": 1, "out_channels": 2, "feature_sizee": 48})


def test_swinunetr_removed_img_size_is_rejected():
    # MONAI >= 1.5 dropped img_size; make sure old configs fail loudly.
    with pytest.raises(RegistryError, match="img_size"):
        MODELS.build("swinunetr", {"in_channels": 1, "out_channels": 2, "img_size": [96, 96, 96]})


def test_build_swinunetr_from_yaml_and_forward():
    pytest.importorskip("monai", reason="medical extra not installed")
    pytest.importorskip("einops", reason="medical extra not installed")
    spec = load_experiment_yaml(CONFIGS / "swinunetr_baseline.yaml")
    model = MODELS.build(spec.model.name, spec.model.params)
    assert type(model).__name__ == "SwinUNETR"

    p = spec.model.params
    # dims must be divisible by 32; 64 keeps the bottleneck at 2^3 so InstanceNorm is happy
    x = torch.zeros(1, p["in_channels"], 64, 64, 64)
    with torch.no_grad():
        y = model.eval()(x)
    assert y.shape == (1, p["out_channels"], 64, 64, 64)
