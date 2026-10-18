"""What the UI's tables show, as plain strings. Pure, and never imports textual.

Numbers, verdicts and metric summaries come from ``application.report`` so a screen and the
run's ``report.md`` never disagree.
"""

from __future__ import annotations

import math

from mlops.application.listing import ExperimentGroup, RunSummary
from mlops.application.report import (
    actual_rows,
    check_rows,
    data_verdict,
    final_metric_rows,
    format_number,
    metric_names,
    record_verdict,
    sparkline,
)
from mlops.application.runner import RunResult
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.manifest import RunManifest

EXPERIMENT_COLUMNS = ("NAME", "EXPERIMENT", "RUNS", "FINISHED", "AGREE")


def experiment_row(g: ExperimentGroup) -> tuple[str, ...]:
    if g.metrics_agree is None:
        agree = "-"
    elif g.metrics_agree:
        agree = "yes" if g.atol == 0 else f"within {g.atol:g}"
    else:
        agree = "DIFFER"
    return (g.name, g.experiment_id[:12], str(len(g.runs)), str(g.finished), agree)


def main_metric(runs: list[RunSummary]) -> str | None:
    """The metric the Runs screen shows: the keep-best one, else ``val/loss``, else the first logged."""
    for r in runs:
        if r.best is not None:
            return r.best.metric
    names = [k for r in runs for k in r.final_metrics]
    if "val/loss" in names:
        return "val/loss"
    return names[0] if names else None


def run_columns(metric: str | None) -> tuple[str, ...]:
    return ("RUN", "ATTEMPT", "STATUS", "EPOCHS", *((metric,) if metric else ()), "NOTES")


def run_row(r: RunSummary, metric: str | None) -> tuple[str, ...]:
    epochs = f"{r.epochs_done if r.epochs_done is not None else '?'}/{r.epochs_total or '?'}"
    notes = []
    if r.best is not None and r.best.epoch is not None:
        notes.append(f"best {r.best.metric}@{r.best.epoch}")
    if r.resumes:
        notes.append(f"resumed x{r.resumes}")
    if r.resumable:
        notes.append("resumable")
    if r.error:
        notes.append(r.error)
    value = (format_number(r.final_metrics.get(metric)),) if metric else ()
    attempt = str(r.attempt) if r.attempt is not None else "?"
    return (r.run_id[:12], attempt, r.status, epochs, *value, "; ".join(notes))


def overview_rows(s: RunSummary, *, spec: ExperimentSpec | None, manifest: RunManifest | None,
                  now: float | None = None) -> list[tuple[str, str]]:
    """``now`` (unix seconds) ages a RUNNING run's heartbeat."""
    rows = [("name", s.name)]
    if spec is not None and spec.meta.description:
        rows.append(("description", spec.meta.description))
    rows += [
        ("experiment_id", s.experiment_id),
        ("run_id", s.run_id),
        ("status", s.status),
        ("attempt", str(s.attempt) if s.attempt is not None else "?"),
        ("epochs", f"{s.epochs_done if s.epochs_done is not None else '?'} / {s.epochs_total or '?'}"),
        ("workers", str(s.workers) if s.workers is not None else "?"),
        ("device", manifest.device if manifest is not None else "—"),
    ]
    if s.status == "RUNNING" and s.updated is not None and now is not None:
        rows.append(("progress", f"epoch {s.epochs_done} / {s.epochs_total or '?'}, "
                                 f"heartbeat {max(0, now - s.updated):.0f}s ago"))
    if s.resumes and manifest is not None:
        rows.append(("resumed", ", ".join(f"after epoch {p.epoch}" for p in manifest.resumes)))
    if s.resumable:
        rows.append(("resumable", f"yes: mlops resume {s.run_id} --artifacts <DIR>"))
    if s.best is not None:
        b = s.best
        rows.append(("best", f"{b.metric} ({b.mode}) never defined" if b.epoch is None
                     else f"{b.metric} ({b.mode}) = {format_number(b.value)} at epoch {b.epoch}"))
    if manifest is not None:
        rows += [("environment", record_verdict(manifest.environment)), ("data", data_verdict(manifest))]
    else:
        missing = "recorded when the run ends" if s.status == "RUNNING" else "no manifest"
        rows += [("environment", missing), ("data", missing)]
    if s.error:
        rows.append(("error", s.error))
    return rows


METRIC_COLUMNS = ("METRIC", "FINAL", "MIN", "MAX", "CURVE")


def metric_rows(history: list[dict[str, float]]) -> list[tuple[str, str, str, str, str]]:
    """The report's final/min/max per metric + a sparkline of its defined values."""
    out = []
    for row in final_metric_rows(history):
        values = [e[row[0]] for e in history if row[0] in e and not math.isnan(e[row[0]])]
        out.append((*row, sparkline(values) if values else ""))
    return out


def history_rows(history: list[dict[str, float]]) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """(header, one row per epoch)."""
    names = metric_names(history)
    rows = [(str(i), *(format_number(e.get(k)) for k in names)) for i, e in enumerate(history, start=1)]
    return ("epoch", *names), rows


CHECK_COLUMNS = ("PIN", "STATUS", "DETAIL")
ACTUAL_COLUMNS = ("FIELD", "VALUE")


def environment_rows(manifest: RunManifest | None) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """(checks, what it actually ran on); Markdown code spans dropped."""
    if manifest is None:
        return [], []
    actual = actual_rows(manifest.environment, manifest.device) if manifest.environment is not None else []
    return check_rows(manifest), [(k, v.replace("`", "")) for k, v in actual]


def resume_lines(name: str, result: RunResult) -> list[str]:
    out = [f"{name}: FINISHED", f"  run_id         {result.run_id}"]
    if result.resumes:
        out.append(f"  resumed after epoch {result.resumes[-1].epoch}")
    return [*out, f"  epochs         {len(result.history)}"]
