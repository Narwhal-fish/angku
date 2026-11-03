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
    <result_id>/uncertainty/<case>.pt   {"entropy", "variance"} maps (float32), MC dropout + save_predictions

MC dropout (``mc_samples``): the model stays in eval mode except its dropout layers, each case is
predicted ``mc_samples`` times with torch seeded to ``mc_seed + case index`` (so a case's samples
do not depend on the cases before it), the prediction is the mean softmax (like a soft vote) and
the per-case uncertainty is the voxel mean of the predictive entropy and of the variance of the
probabilities across samples.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
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
    RESULT_FORMAT_VERSION,
    CaseResult,
    Kind,
    Member,
    Part,
    PredictionRecord,
    Vote,
    Weights,
    compute_result_id,
    parse_result,
)
from mlops.ports.artifacts import ArtifactError, ArtifactStore
from mlops.ports.data import DataProvider
from mlops.ports.environment import EnvironmentProbe
from mlops.registry import INFERERS, LOSSES, METRICS, MODELS

RESULT = "result.json"
REPORT = "report.md"
PREDICTIONS = "predictions"
UNCERTAINTY = "uncertainty"

VAL_NOTE = (
    "val was seen during training (val/loss every epoch, keep-best chose its epoch on it): "
    "these numbers are optimistic; use a test split for an unbiased estimate"
)


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
    mc_samples: int | None = None,
    mc_seed: int = 0,
) -> PredictionRecord:
    """Evaluate ``run_id``'s model on ``part`` and write the result into ``out`` (default: ``store``).
    ``weights`` None = ``best.pt`` when the run kept one, else ``model.pt``. ``mc_samples`` = MC
    dropout with that many forward passes per case (the model must have dropout)."""
    mc = None
    if mc_samples is not None:
        if mc_samples < 2:
            raise PredictError(f"MC dropout needs at least 2 samples per case, got {mc_samples}")
        mc = {"samples": mc_samples, "seed": mc_seed}
    return _predict(
        store,
        [run_id],
        kind="predict",
        vote=None,
        data=data,
        part=part,
        weights=weights,
        device=device,
        probe=probe,
        save_predictions=save_predictions,
        out=out,
        mc=mc,
    )


def ensemble_runs(
    store: ArtifactStore,
    run_ids: Sequence[str],
    *,
    vote: Vote,
    data: DataProvider,
    part: Part = "test",
    weights: Weights | None = None,
    device: str | None = None,
    probe: EnvironmentProbe | None = None,
    save_predictions: bool = False,
    out: ArtifactStore | None = None,
) -> PredictionRecord:
    """Merge the predictions of several runs on the same cases. ``soft``: average the softmax
    probabilities, then argmax (loss = the loss of that average). ``hard``: each run votes its
    argmax per voxel, most votes wins, a tie goes to the tied class with the higher average
    probability (no loss: votes are not probabilities). The runs must be comparable (same data,
    preprocessing, cases of ``part``, loss, metrics, inferer; models may differ) or nothing runs.
    The result does not depend on the order the runs are given in."""
    if len(set(run_ids)) != len(run_ids):
        raise PredictError("a run is listed twice: each member votes once")
    return _predict(
        store,
        sorted(run_ids),
        kind="ensemble",
        vote=vote,
        data=data,
        part=part,
        weights=weights,
        device=device,
        probe=probe,
        save_predictions=save_predictions,
        out=out,
    )


@dataclass
class _Loaded:
    bundle: Bundle
    member: Member
    state: dict[str, Any]


def _load(store: ArtifactStore, run_id: str, weights: Weights | None) -> _Loaded:
    bundle = load_bundle(store, run_id)
    if bundle.manifest.status != "FINISHED":
        raise PredictError(f"run {run_id[:12]} is {bundle.manifest.status}: only a FINISHED run has weights")
    chosen: Weights = weights or (BEST if bundle.best_state is not None else MODEL)
    state = bundle.best_state if chosen == BEST else bundle.model_state
    if state is None:
        raise PredictError(f"run {run_id[:12]} has no {chosen} (keep-best was not asked for)")
    member = Member(
        run_id=bundle.manifest.run_id,
        experiment_id=bundle.manifest.experiment_id,
        weights=chosen,
        epoch=_epoch_of(bundle, chosen),
    )
    return _Loaded(bundle, member, state)


# What every member must share for their predictions to be merged (and compared) case by case.
_COMPARABLE = ("dataset", "preprocessing", "loss", "metrics", "inference")


def _check_comparable(members: list[_Loaded]) -> None:
    first = members[0].bundle.spec.model_dump(include=set(_COMPARABLE))
    for m in members[1:]:
        spec = m.bundle.spec.model_dump(include=set(_COMPARABLE))
        for section in _COMPARABLE:
            if first[section] != spec[section]:
                raise PredictError(
                    f"runs {members[0].member.run_id[:12]} and {m.member.run_id[:12]} differ in `{section}`: "
                    "their predictions cannot be merged case by case"
                )


def _predict(
    store: ArtifactStore,
    run_ids: Sequence[str],
    *,
    kind: Kind,
    vote: Vote | None,
    data: DataProvider,
    part: Part,
    weights: Weights | None,
    device: str | None,
    probe: EnvironmentProbe | None,
    save_predictions: bool,
    out: ArtifactStore | None,
    mc: dict[str, int] | None = None,
) -> PredictionRecord:
    if not run_ids:
        raise PredictError("no run to predict with")
    members = [_load(store, r, weights) for r in run_ids]
    _check_comparable(members)
    spec = members[0].bundle.spec

    data_check = verify_data(spec, data)
    parts = [_part(m.bundle, data, part) for m in members]
    cases, dataset = parts[0]
    for m, (other, _) in zip(members[1:], parts[1:], strict=True):
        if other != cases:
            raise PredictError(
                f"runs {members[0].member.run_id[:12]} and {m.member.run_id[:12]} have different {part} cases "
                "(k-fold members share only the test cases)"
            )
    dev = resolve_device(spec.resources, device)
    environment = record_environment(spec, probe.capture(spec)) if probe is not None else None

    seed_everything(spec.training.seed, deterministic=spec.training.deterministic)
    models = []
    for m in members:
        model: nn.Module = MODELS.build(m.bundle.spec.model.name, m.bundle.spec.model.params)
        model.load_state_dict(m.state)
        models.append(model.to(dev).eval())
    if mc is not None and enable_dropout(models[0]) == 0:
        raise PredictError(
            f"run {members[0].member.run_id[:12]}'s model has no dropout (p > 0): MC dropout would give "
            "the same prediction every sample; train with dropout (e.g. tiny_cnn `dropout: 0.2`)"
        )
    loss_fn = LOSSES.build(spec.loss.name, spec.loss.params)
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(dev)
    metrics: list[Metric] = [METRICS.build(m.name, m.params) for m in spec.metrics]
    infer: Inferer | None = (
        INFERERS.build(spec.inference.name, spec.inference.params) if spec.inference is not None else None
    )

    target = out if out is not None else store
    method: dict[str, Any] = {"vote": vote} if vote is not None else {}
    if mc is not None:
        method["mc_dropout"] = mc
    result_id = compute_result_id(
        kind=kind, members=[m.member for m in members], part=part, dataset_snapshot=spec.dataset.snapshot, method=method
    )
    rows: list[CaseResult] = []
    member_rows: list[list[CaseResult]] = [[] for _ in members]
    for i, case in enumerate(cases):
        x, y = dataset[i]
        x, y = x.unsqueeze(0).to(dev), y.unsqueeze(0).to(dev)
        maps: tuple[Tensor, Tensor] | None = None
        if mc is not None:
            torch.manual_seed(mc["seed"] + i)
            samples = [_logits(models[0], x, infer) for _ in range(mc["samples"])]
            combined, with_loss = combine(samples, "soft")
            maps = uncertainty(torch.stack([lg.softmax(dim=1) for lg in samples]))
            logits = [combined]
        else:
            logits = [_logits(model, x, infer) for model in models]
            combined, with_loss = combine(logits, vote)
        row = _row(combined, y, loss_fn if with_loss else None, metrics)
        if maps is not None:
            row |= {"uncertainty/entropy": float(maps[0].mean()), "uncertainty/variance": float(maps[1].mean())}
        rows.append(CaseResult(case=case, metrics=row))
        if kind == "ensemble":
            for j, lg in enumerate(logits):
                member_rows[j].append(CaseResult(case=case, metrics=_row(lg, y, loss_fn, metrics)))
        if save_predictions:
            buf = io.BytesIO()
            torch.save(combined.argmax(dim=1)[0].to(torch.uint8).cpu(), buf)
            target.put(result_id, f"{PREDICTIONS}/{case}.pt", buf.getvalue())
            if maps is not None:
                buf = io.BytesIO()
                torch.save({"entropy": maps[0][0].float().cpu(), "variance": maps[1][0].float().cpu()}, buf)
                target.put(result_id, f"{UNCERTAINTY}/{case}.pt", buf.getvalue())

    record = PredictionRecord(
        format_version=RESULT_FORMAT_VERSION,
        result_id=result_id,
        kind=kind,
        name=spec.meta.name,
        members=[m.member for m in members],
        part=part,
        dataset=spec.dataset.name,
        data_snapshot=data_check,
        method=method,
        device=str(dev),
        cases=rows,
        metrics=mean_metrics(rows),
        environment=environment,
        member_metrics={m.member.run_id: mean_metrics(r) for m, r in zip(members, member_rows, strict=True)}
        if kind == "ensemble"
        else {},
        notes=[VAL_NOTE] if part == "val" else [],
    )
    target.put(result_id, RESULT, record.model_dump_json(indent=2).encode())
    target.put(result_id, REPORT, render_prediction_report(record).encode())
    return record


def combine(logits: Sequence[Tensor], vote: Vote | None) -> tuple[Tensor, bool]:
    """(what the metrics see, whether a loss is defined for it). One member and no vote = its logits.
    ``soft``: log of the mean softmax (softmax of it gives the mean back, so loss and argmax-based
    metrics see the averaged probabilities). ``hard``: one-hot of the vote winner; ties broken by
    the mean probability, then by the lower class index."""
    if vote is None:
        assert len(logits) == 1
        return logits[0], True
    probs = torch.stack([lg.softmax(dim=1) for lg in logits]).mean(dim=0)
    if vote == "soft":
        return probs.clamp_min(1e-12).log(), True
    classes = probs.shape[1]
    votes = sum(torch.nn.functional.one_hot(lg.argmax(dim=1), classes) for lg in logits).movedim(-1, 1)
    # votes differ by >= 1 where they differ; 2 * votes + prob (< 1) lets the probability decide only ties.
    winner = (2 * votes.to(probs.dtype) + probs).argmax(dim=1)
    return torch.nn.functional.one_hot(winner, classes).movedim(-1, 1).to(probs.dtype), False


def enable_dropout(model: nn.Module) -> int:
    """Put only the dropout layers in train mode (BatchNorm & co. stay in eval); returns how many
    of them actually drop something (p > 0)."""
    active = 0
    for m in model.modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.train()
            active += m.p > 0
    return active


@torch.no_grad()
def uncertainty(probs: Tensor) -> tuple[Tensor, Tensor]:
    """``probs`` (samples, N, C, *spatial) -> per-voxel (N, *spatial): predictive entropy of the
    mean probabilities (nats) and the variance across samples (population), averaged over classes."""
    mean = probs.mean(dim=0)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(dim=1)
    variance = probs.var(dim=0, correction=0).mean(dim=1)
    return entropy, variance


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
def _logits(model: nn.Module, x: Tensor, infer: Inferer | None) -> Tensor:
    return infer(model, x) if infer is not None else model(x)


@torch.no_grad()
def _row(logits: Tensor, y: Tensor, loss_fn: Any | None, metrics: Sequence[Metric]) -> dict[str, float]:
    row = {"loss": float(loss_fn(logits, y))} if loss_fn is not None else {}
    for metric in metrics:
        for key, values in metric(logits, y).items():
            if key in row:
                raise PredictError(f"two metrics both produce {key!r}")
            row[key] = float(values[0])
    return row


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
    return parse_result(store.get(result_id, RESULT))


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


def _what(r: PredictionRecord) -> str:
    if r.kind == "ensemble":
        return f"ensemble ({r.method.get('vote')} vote)"
    if "mc_dropout" in r.method:
        return f"predict with MC dropout ({r.method['mc_dropout']['samples']} samples)"
    return r.kind


def _metric_cells(metrics: dict[str, float]) -> str:
    return "  ".join(f"{k}={format_number(v)}" for k, v in metrics.items())


def prediction_lines(r: PredictionRecord) -> list[str]:
    """What `angku predict` / `angku ensemble` print."""
    n = f"{len(r.cases)} case{'s' if len(r.cases) != 1 else ''}"
    if r.kind == "ensemble":
        head = f"{r.name}: {_what(r)} of {len(r.members)} runs on {r.part} ({n})"
    else:
        head = f"{r.name}: predicted {r.part} ({n})"
        if "mc_dropout" in r.method:
            mc = r.method["mc_dropout"]
            head += f" with MC dropout ({mc['samples']} samples, seed {mc['seed']})"
    out = [
        head,
        f"  result_id  {r.result_id}",
        f"  run{'s' if len(r.members) > 1 else ' '}       {_members_line(r)}",
        f"  {_metric_cells(r.metrics)}",
    ]
    if r.member_metrics:
        out.append("  each run alone:")
        out += [f"    {run[:12]}  {_metric_cells(m)}" for run, m in r.member_metrics.items()]
    out += [f"  note: {n}" for n in r.notes]
    return out


def result_line(r: PredictionRecord) -> str:
    """One line of `angku results`."""
    what = f"{_what(r)} {r.part} ({len(r.cases)})"
    return f"{r.result_id[:12]}  {r.name}  {what}  {_members_line(r)}  {_metric_cells(r.metrics)}"


def render_prediction_report(r: PredictionRecord) -> str:
    """``report.md`` of a result: a pure function of the record."""
    keys = list(r.metrics)
    lines = [
        f"# {r.name} — {_what(r)} on {r.part}",
        "",
        "| field | value |",
        "|---|---|",
        f"| result_id | `{r.result_id}` |",
        f"| runs | {_members_line(r)} |",
        f"| part | {r.part} ({len(r.cases)} cases) |",
        f"| data | {r.dataset}: {r.data_snapshot.status} ({r.data_snapshot.detail or 'matches the pin'}) |",
        f"| device | `{r.device}` |",
    ]
    if "mc_dropout" in r.method:
        mc = r.method["mc_dropout"]
        lines.append(f"| method | MC dropout: {mc['samples']} samples per case, seed {mc['seed']} + case index |")
    for n in r.notes:
        lines += ["", f"> {n}"]
    lines += ["", "## Mean over cases", "", "| metric | mean |", "|---|---|"]
    lines += [f"| {k} | {_cell(r.metrics[k])} |" for k in keys]
    if "mc_dropout" in r.method:
        lines += [
            "",
            "`uncertainty/entropy`: predictive entropy of the mean probabilities (nats), "
            "`uncertainty/variance`: variance of the probabilities across samples; both are "
            "voxel means per case. Metrics and loss are of the mean probabilities.",
        ]
    if r.member_metrics:
        member_keys = list(next(iter(r.member_metrics.values())))
        lines += [
            "",
            "## Each run alone",
            "",
            "Same cases, each run's own prediction: did the vote help?",
            "",
            "| run | " + " | ".join(member_keys) + " |",
            "|---|" + "---|" * len(member_keys),
        ]
        lines += [
            f"| {run[:12]} | " + " | ".join(_cell(m.get(k, math.nan)) for k in member_keys) + " |"
            for run, m in r.member_metrics.items()
        ]
    lines += ["", "## Per case", "", "| case | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    lines += [f"| {c.case} | " + " | ".join(_cell(c.metrics.get(k, math.nan)) for k in keys) + " |" for c in r.cases]
    return "\n".join(lines) + "\n"


def _cell(v: float) -> str:
    return "—" if math.isnan(v) else format_number(v)
