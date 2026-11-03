"""Load and validate experiment.yaml into an ExperimentSpec.

This is the only place that knows about YAML. Validation itself lives in the
domain models (extra="forbid", field validators); this module just feeds them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from angku.domain.experiment import ExperimentSpec


class ConfigError(Exception):
    """experiment.yaml failed to parse or validate."""


def load_experiment_dict(data: dict[str, Any]) -> ExperimentSpec:
    try:
        return ExperimentSpec.model_validate(data)
    except ValidationError as e:
        raise ConfigError(str(e)) from e


def load_experiment_yaml(path: str | Path) -> ExperimentSpec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return load_experiment_dict(raw)
    except ConfigError as e:
        raise ConfigError(f"{path}:\n{e}") from e
