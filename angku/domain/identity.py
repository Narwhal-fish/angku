"""ExperimentID and RunID computation.

ExperimentID = sha256(canonical JSON of the scientific fingerprint).
RunID        = sha256(canonical JSON of {experiment_id, resources, attempt}).

Canonical JSON = json.dumps(..., sort_keys=True, separators=(",", ":")) so that
key order in the YAML file never affects the hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from angku.domain.errors import UnpinnedInputError
from angku.domain.experiment import ExperimentSpec, Resources


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_experiment_id(spec: ExperimentSpec) -> str:
    """Fingerprint of the scientific experiment. Raises if any input is unpinned."""
    missing = spec.unpinned_inputs()
    if missing:
        raise UnpinnedInputError(missing)
    return _sha256(canonical_json(spec.fingerprint()))


def compute_run_id(experiment_id: str, resources: Resources, attempt: int = 1) -> str:
    """One concrete execution of an experiment on specific hardware."""
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt}")
    payload = {
        "experiment_id": experiment_id,
        "resources": resources.model_dump(),
        "attempt": attempt,
    }
    return _sha256(canonical_json(payload))
