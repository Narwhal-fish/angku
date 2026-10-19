"""`predict_run`: apply a finished run's model to one part of its data and store the result.

The run is evidence and is only read. Its spec says which data, preprocessing, split, inferer
and metrics; the data is verified against the pinned snapshot and re-split, and the split must
equal the bundle's ``split.json`` (otherwise "test" would not be the cases the run never saw).
Each case is evaluated on its own (batch 1, the spec's inferer), with the model in eval mode
and torch seeded like the run, so the same inputs give bit-identical numbers.

Result layout, next to the runs under the same artifact root (no ``spec.json``, so it is never
listed as a run):

    <result_id>/result.json          PredictionRecord
    <result_id>/report.md            derived from result.json
    <result_id>/predictions/<case>.pt   argmax labels (uint8), only with save_predictions
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mlops.application.bundle import BEST, MODEL, Bundle, load_bundle
from mlops.application.report import format_number
from mlops.application.runner import (
    Inferer,
    Metric,
    load_splits,
    resolve_device,
    resolve_split,
    seed_everything,
    verify_data,
)
from mlops.domain.environment import record_environment
from mlops.domain.prediction import (
    CaseResult,
    Member,
    Part,
    PredictionRecord,
    Weights,
    compute_result_id,
)
from mlops.ports.artifacts import ArtifactError, ArtifactStore
from mlops.ports.data import DataProvider
from mlops.ports.environment import EnvironmentProbe
from mlops.registry import INFERERS, LOSSES, METRICS, MODELS

RESULT = "result.json"
REPORT = "report.md"
PREDICTIONS = "predictions"

VAL_NOTE = ("val was seen during training (val/loss every epoch, keep-best chose its epoch on it): "
            "these numbers are optimistic; use a test split for an unbiased estimate")


class PredictError(Exception):
    """The run cannot be predicted with as asked (not finished, no such weights or part, split changed)."""


def predict_run(
    store: ArtifactStore,
    run_id: str,
    *,
    data: DataProvider,
    part: Part = "test",
    weights: Weights | None = None,
    device: str | None = None,
    probe: EnvironmentProbe | None = None,
    save_predictions: bool = False,
    out: ArtifactStore | None = None,
) -> PredictionRecord:
    """Evaluate ``run_id``'s model on ``part`` and write the result into ``out`` (default: ``store``).
    ``weights`` None = ``best.pt`` when the run kept one, else ``model.pt``."""
    bundle = load_bundle(store, run_id)
    if bundle.manifest.status != "FINISHED":
        raise PredictError(f"run {run_id[:12]} is {bundle.manifest.status}: only a FINISHED run has weights")
    weights = weights or (BEST if bundle.best_state is not None else MODEL)
    state = bundle.best_state if weights == BEST else bundle.model_state
    if state is None:
        raise PredictError(f"run {run_id[:12]} has no {weights} (keep-best was not asked for)")
    spec = bundle.spec

    data_check = verify_data(spec, data)
    cases, dataset = _part(bundle, data, part)
    dev = resolve_device(spec.resources, device)
    environment = record_environment(spec, probe.capture(spec)) if probe is not None else None

    seed_everything(spec.training.seed, deterministic=spec.training.deterministic)
    model: nn.Module = MODELS.build(spec.model.name, spec.model.params)
    model.load_state_dict(state)
    model.to(dev).eval()
    loss_fn = LOSSES.build(spec.loss.name, spec.loss.params)
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(dev)
    metrics: list[Metric] = [METRICS.build(m.name, m.params) for m in spec.metrics]
    infer: Inferer | None = (
        INFERERS.build(spec.inference.name, spec.inference.params) if spec.inference is not None else None
    )

    target = out if out is not None else store
    member = Member(run_id=bundle.manifest.run_id, experiment_id=bundle.manifest.experiment_id,
                    weights=weights, epoch=_epoch_of(bundle, weights))
    method: dict[str, Any] = {}
    result_id = compute_result_id(kind="predict", members=[member], part=part,
                                  dataset_snapshot=spec.dataset.snapshot, method=method)
    rows: list[CaseResult] = []
    for i, case in enumerate(cases):
        x, y = dataset[i]
        logits, row = _evaluate(model, x.unsqueeze(0).to(dev), y.unsqueeze(0).to(dev), loss_fn, metrics, infer)
        rows.append(CaseResult(case=case, metrics=row))
        if save_predictions:
            buf = io.BytesIO()
            torch.save(logits.argmax(dim=1)[0].to(torch.uint8).cpu(), buf)
            target.put(result_id, f"{PREDICTIONS}/{case}.pt", buf.getvalue())

    record = PredictionRecord(
        result_id=result_id, name=spec.meta.name, members=[member], part=part,
        dataset=spec.dataset.name, data_snapshot=data_check, method=method, device=str(dev),
        cases=rows, metrics=mean_metrics(rows), environment=environment,
        notes=[VAL_NOTE] if part == "val" else [],
    )
    target.put(result_id, RESULT, record.model_dump_json(indent=2).encode())
    target.put(result_id, REPORT, render_prediction_report(record).encode())
    return record


def _part(bundle: Bundle, data: DataProvider, part: Part) -> tuple[list[str], Any]:
    """(case ids, dataset) of ``part``, after checking the split still matches the bundle's."""
    spec = bundle.spec
    assignment = resolve_split(spec, data)
    if assignment is not None and bundle.split is not None and assignment != bundle.split:
        raise PredictError(
            "the data splits differently now than when the run trained (split.json): "
            "the cases it never saw are no longer known"
        )
    if part == "test" and (assignment is None or not assignment.test):
        raise PredictError(
            "this run has no test split (set `split.test` in the spec and train again), "
            "or predict on val with part 'val'"
        )
    splits = load_splits(spec, data, assignment)
    dataset = splits.test if part == "test" else splits.val
    if dataset is None:
        raise PredictError(f"the data provider served no {part} split for this run")
    if assignment is not None:
        cases = list(getattr(assignment, part))
    else:  # the provider's own split names no cases
        cases = [f"case-{i:03d}" for i in range(len(dataset))]
    if len(cases) != len(dataset):
        raise PredictError(f"{part}: {len(cases)} case ids but the provider served {len(dataset)} items")
    return cases, dataset


def _epoch_of(bundle: Bundle, weights: Weights) -> int | None:
    if weights == BEST:
        return bundle.manifest.best.epoch if bundle.manifest.best is not None else None
    return len(bundle.manifest.history) or None


@torch.no_grad()
def _evaluate(model: nn.Module, x: Tensor, y: Tensor, loss_fn: Any, metrics: Sequence[Metric],
              infer: Inferer | None) -> tuple[Tensor, dict[str, float]]:
    logits = infer(model, x) if infer is not None else model(x)
    row = {"loss": float(loss_fn(logits, y))}
    for metric in metrics:
        for key, values in metric(logits, y).items():
            if key in row:
                raise PredictError(f"two metrics both produce {key!r}")
            row[key] = float(values[0])
    return logits, row


def mean_metrics(rows: Sequence[CaseResult]) -> dict[str, float]:
    """Per key, the mean over the cases where it is defined; NaN if no case defines it."""
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r.metrics if k not in keys]
    out = {}
    for k in keys:
        values = [r.metrics[k] for r in rows if k in r.metrics and not math.isnan(r.metrics[k])]
        out[k] = sum(values) / len(values) if values else math.nan
    return out


def load_result(store: ArtifactStore, result_id: str) -> PredictionRecord:
    return PredictionRecord.model_validate_json(store.get(result_id, RESULT))


def list_results(store: ArtifactStore) -> list[PredictionRecord]:
    """Every result under the root (directories holding ``result.json``), by name then id."""
    out = []
    for rid in store.runs():
        try:
            if RESULT in store.list(rid):
                out.append(load_result(store, rid))
        except (ArtifactError, ValueError):
            continue
    return sorted(out, key=lambda r: (r.name, r.part, r.result_id))


# -- text ------------------------------------------------------------------------


def _members_line(r: PredictionRecord) -> str:
    return ", ".join(
        f"{m.run_id[:12]} {m.weights}" + (f" (epoch {m.epoch})" if m.epoch is not None else "") for m in r.members
    )


def prediction_lines(r: PredictionRecord) -> list[str]:
    """What `mlops predict` prints."""
    out = [
        f"{r.name}: predicted {r.part} ({len(r.cases)} case{'s' if len(r.cases) != 1 else ''})",
        f"  result_id  {r.result_id}",
        f"  run        {_members_line(r)}",
        "  " + "  ".join(f"{k}={format_number(v)}" for k, v in r.metrics.items()),
    ]
    out += [f"  note: {n}" for n in r.notes]
    return out


def result_line(r: PredictionRecord) -> str:
    """One line of `mlops results`."""
    metrics = "  ".join(f"{k}={format_number(v)}" for k, v in r.metrics.items())
    return f"{r.result_id[:12]}  {r.name}  {r.kind} {r.part} ({len(r.cases)})  {_members_line(r)}  {metrics}"


def render_prediction_report(r: PredictionRecord) -> str:
    """``report.md`` of a result: a pure function of the record."""
    keys = list(r.metrics)
    lines = [
        f"# {r.name} — {r.kind} on {r.part}",
        "",
        "| field | value |",
        "|---|---|",
        f"| result_id | `{r.result_id}` |",
        f"| runs | {_members_line(r)} |",
        f"| part | {r.part} ({len(r.cases)} cases) |",
        f"| data | {r.dataset}: {r.data_snapshot.status} ({r.data_snapshot.detail or 'matches the pin'}) |",
        f"| device | `{r.device}` |",
    ]
    for n in r.notes:
        lines += ["", f"> {n}"]
    lines += ["", "## Mean over cases", "", "| metric | mean |", "|---|---|"]
    lines += [f"| {k} | {_cell(r.metrics[k])} |" for k in keys]
    lines += ["", "## Per case", "", "| case | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    lines += [f"| {c.case} | " + " | ".join(_cell(c.metrics.get(k, math.nan)) for k in keys) + " |" for c in r.cases]
    return "\n".join(lines) + "\n"


def _cell(v: float) -> str:
    return "—" if math.isnan(v) else format_number(v)
