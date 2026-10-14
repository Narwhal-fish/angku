"""Verify a run bundle: re-run it and compare with what the bundle recorded.

The bundle promises that ``run_experiment(bundle.spec, attempt=bundle.manifest.attempt)``
yields the same run_id, history and weights. ``reproduce_run`` executes that
promise and reports which parts held. The original bundle is evidence and is
never written to: the replay lands in a ``scratch`` store supplied by the caller.

Only FINISHED bundles can be verified (a FAILED one has no weights). A replay
that itself blows up propagates as-is: a broken environment is not a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mlops.application.bundle import Bundle, load_bundle
from mlops.application.runner import run_experiment
from mlops.ports.artifacts import ArtifactStore
from mlops.ports.data import DataProvider
from mlops.ports.tracking import Tracker

CHECKS = ("run_id", "history", "weights")


class ReproduceError(Exception):
    """The bundle cannot be verified (e.g. it records a FAILED run)."""


@dataclass(frozen=True)
class Mismatch:
    check: str  # one of CHECKS
    detail: str


@dataclass(frozen=True)
class ReproduceReport:
    run_id: str  # as the bundle names it
    experiment_id: str
    replay_run_id: str  # what re-running the bundle actually computed
    checks: dict[str, bool]
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "replay_run_id": self.replay_run_id,
            "checks": dict(self.checks),
            "mismatches": [{"check": m.check, "detail": m.detail} for m in self.mismatches],
        }


def reproduce_run(
    store: ArtifactStore,
    run_id: str,
    *,
    data: DataProvider,
    tracker: Tracker,
    scratch: ArtifactStore,
    device: str | None = None,
    atol: float = 0.0,
) -> ReproduceReport:
    """Re-run the bundle at ``run_id`` into ``scratch`` and compare.

    ``atol=0`` (default) demands bit-exact weights, the promise within one
    launcher/environment; a positive ``atol`` switches to ``allclose`` for
    cross-launcher checks.
    """
    original = load_bundle(store, run_id)
    if original.manifest.status != "FINISHED" or original.model_state is None:
        raise ReproduceError(
            f"run {run_id} is {original.manifest.status}: nothing to reproduce (no weights)"
        )

    result = run_experiment(
        original.spec,
        data=data,
        tracker=tracker,
        artifacts=scratch,
        attempt=original.manifest.attempt,
        device=device,
    )
    replay = load_bundle(scratch, result.run_id)

    mismatches: list[Mismatch] = []
    mismatches += _check_run_id(original, replay)
    mismatches += _check_history(original, replay)
    mismatches += _check_weights(original, replay, atol)
    failed = {m.check for m in mismatches}
    return ReproduceReport(
        run_id=run_id,
        experiment_id=original.manifest.experiment_id,
        replay_run_id=result.run_id,
        checks={c: c not in failed for c in CHECKS},
        mismatches=mismatches,
    )


def _check_run_id(original: Bundle, replay: Bundle) -> list[Mismatch]:
    a, b = original.manifest.run_id, replay.manifest.run_id
    if a == b:
        return []
    return [Mismatch("run_id", f"bundle names run {a} but its spec + attempt {original.manifest.attempt} yield {b}")]


def _check_history(original: Bundle, replay: Bundle) -> list[Mismatch]:
    a, b = original.manifest.history, replay.manifest.history
    if len(a) != len(b):
        return [Mismatch("history", f"bundle has {len(a)} epochs, replay produced {len(b)}")]
    out: list[Mismatch] = []
    for epoch, (ea, eb) in enumerate(zip(a, b)):
        if set(ea) != set(eb):
            out.append(Mismatch("history", f"epoch {epoch}: metrics {sorted(ea)} vs {sorted(eb)}"))
            continue
        for k in sorted(ea):
            if ea[k] != eb[k]:
                out.append(Mismatch("history", f"epoch {epoch}: {k} {ea[k]!r} vs replay {eb[k]!r}"))
    return out


def _check_weights(original: Bundle, replay: Bundle, atol: float) -> list[Mismatch]:
    a, b = original.model_state or {}, replay.model_state or {}
    out: list[Mismatch] = []
    for k in sorted(set(a) ^ set(b)):
        side = "bundle" if k in a else "replay"
        out.append(Mismatch("weights", f"{k}: only in {side}"))
    for k in sorted(set(a) & set(b)):
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            out.append(Mismatch("weights", f"{k}: {tuple(ta.shape)}/{ta.dtype} vs {tuple(tb.shape)}/{tb.dtype}"))
            continue
        same = torch.equal(ta, tb) if atol == 0 else torch.allclose(ta, tb, atol=atol, rtol=0)
        if not same:
            diff = (ta.float() - tb.float()).abs().max().item()
            out.append(Mismatch("weights", f"{k}: max |diff| = {diff:.3g} (atol {atol:g})"))
    return out
