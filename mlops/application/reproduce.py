"""Verify a run bundle: re-run it and compare with what the bundle recorded.

The bundle promises that ``run_experiment(bundle.spec, attempt=bundle.manifest.attempt)``
yields the same run_id, history and weights. ``reproduce_run`` executes that
promise and reports which parts held. The original bundle is evidence and is
never written to: the replay lands in a ``scratch`` store supplied by the caller.

Only FINISHED bundles can be verified (a FAILED one has no weights). A replay
that itself blows up propagates as-is: a broken environment is not a mismatch.

With a ``probe`` the replay records its environment too, and ``environment`` lists what
differs from the original's (library versions, GPU, threads, code state): not a check,
the likely explanation when weights or history do not match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from mlops.application.bundle import Bundle, load_bundle
from mlops.application.diff import Change, change_lines, diff_environments
from mlops.application.runner import run_experiment
from mlops.domain.best import BestCriterion
from mlops.ports.artifacts import ArtifactStore
from mlops.ports.data import DataProvider
from mlops.ports.environment import EnvironmentProbe
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
    # Original vs replay environment (informational; empty = nothing differs or nothing recorded).
    environment: list[Change] = field(default_factory=list)

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
            "environment": [c.to_dict() for c in self.environment],
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
    probe: EnvironmentProbe | None = None,
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
        probe=probe,
        # keep-best is a launch option, not in the spec: replay it so best.pt is checked too
        keep_best=(BestCriterion(metric=original.manifest.best.metric, mode=original.manifest.best.mode)
                   if original.manifest.best is not None else None),
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
        environment=diff_environments(original.manifest, replay.manifest) if probe is not None else [],
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
            if ea[k] != eb[k] and not (math.isnan(ea[k]) and math.isnan(eb[k])):
                out.append(Mismatch("history", f"epoch {epoch}: {k} {ea[k]!r} vs replay {eb[k]!r}"))
    return out


def _check_weights(original: Bundle, replay: Bundle, atol: float) -> list[Mismatch]:
    out = _compare_states(original.model_state or {}, replay.model_state or {}, atol, "")
    if original.best_state is not None or replay.best_state is not None:
        out += _compare_states(original.best_state or {}, replay.best_state or {}, atol, "best.pt ")
    return out


def _compare_states(a: dict[str, Any], b: dict[str, Any], atol: float, prefix: str) -> list[Mismatch]:
    out: list[Mismatch] = []
    for k in sorted(set(a) ^ set(b)):
        side = "bundle" if k in a else "replay"
        out.append(Mismatch("weights", f"{prefix}{k}: only in {side}"))
    for k in sorted(set(a) & set(b)):
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            out.append(Mismatch("weights", f"{prefix}{k}: {tuple(ta.shape)}/{ta.dtype} vs {tuple(tb.shape)}/{tb.dtype}"))
            continue
        same = torch.equal(ta, tb) if atol == 0 else torch.allclose(ta, tb, atol=atol, rtol=0)
        if not same:
            diff = (ta.float() - tb.float()).abs().max().item()
            out.append(Mismatch("weights", f"{prefix}{k}: max |diff| = {diff:.3g} (atol {atol:g})"))
    return out


def reproduce_lines(name: str, r: ReproduceReport) -> list[str]:
    """What `mlops reproduce` prints (and the UI shows): verdict, per-check result, mismatches,
    and the environment differences as the likely explanation."""
    out = [
        f"{name}: {'REPRODUCED' if r.ok else 'NOT REPRODUCED'}",
        f"  experiment_id  {r.experiment_id}",
        f"  run_id         {r.run_id}",
        *(f"  {check:<14} {'ok' if passed else 'MISMATCH'}" for check, passed in r.checks.items()),
        *(f"    {m.check}: {m.detail}" for m in r.mismatches),
    ]
    if r.environment:
        heading = "environment differences (likely cause):" if not r.ok else "environment differences (did not matter):"
        out += change_lines(f"  {heading}", r.environment)
    return out
