"""The run bundle: everything needed to reproduce a run, laid out in an ArtifactStore.

    spec.json       the full ExperimentSpec (meta and tracking included, so it can be re-run as-is)
    manifest.json   RunManifest: ids, attempt, status, per-epoch history
    model.pt        final model ``state_dict`` (absent when the run FAILED)

JSON rather than YAML on purpose: ``mlops.config`` is the only YAML-aware
layer, and a bundle must be readable without it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch

from mlops.domain.experiment import ExperimentSpec
from mlops.domain.manifest import RunManifest
from mlops.ports.artifacts import ArtifactStore

SPEC = "spec.json"
MANIFEST = "manifest.json"
MODEL = "model.pt"


@dataclass(frozen=True)
class Bundle:
    spec: ExperimentSpec
    manifest: RunManifest
    model_state: dict[str, Any] | None


def save_bundle(
    store: ArtifactStore,
    *,
    spec: ExperimentSpec,
    manifest: RunManifest,
    model_state: Mapping[str, Any] | None,
) -> None:
    run_id = manifest.run_id
    store.put(run_id, SPEC, spec.model_dump_json(indent=2).encode())
    store.put(run_id, MANIFEST, manifest.model_dump_json(indent=2).encode())
    if model_state is not None:
        buf = io.BytesIO()
        torch.save(dict(model_state), buf)
        store.put(run_id, MODEL, buf.getvalue())


def load_bundle(store: ArtifactStore, run_id: str) -> Bundle:
    spec = ExperimentSpec.model_validate_json(store.get(run_id, SPEC))
    manifest = RunManifest.model_validate_json(store.get(run_id, MANIFEST))
    model_state = None
    if MODEL in store.list(run_id):
        model_state = torch.load(
            io.BytesIO(store.get(run_id, MODEL)), map_location="cpu", weights_only=True
        )
    return Bundle(spec=spec, manifest=manifest, model_state=model_state)
