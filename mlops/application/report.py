"""Human-readable run report, rendered from the run bundle.

The bundle is the source of truth, not the tracker: ``render_report`` is a
pure function of ``Bundle`` (no clock, no environment lookups), so the same
bundle always renders the same bytes and a report can be regenerated or diffed
at any time. ``report.md`` stored next to the bundle is derived data;
``spec.json`` / ``manifest.json`` / ``model.pt`` remain the evidence.

Markdown on purpose: readable as plain text in a terminal, rendered by any git
host, and diffable between runs.
"""

from __future__ import annotations

import json
import math
from typing import Any

from mlops.application.bundle import Bundle, load_bundle
from mlops.domain.best import BestRecord
from mlops.domain.environment import EnvironmentRecord
from mlops.domain.experiment import Split
from mlops.domain.manifest import ResumePoint, RunManifest
from mlops.ports.artifacts import ArtifactStore

REPORT = "report.md"

_SPARKS = "▁▂▃▄▅▆▇█"


def write_report(store: ArtifactStore, run_id: str) -> str:
    """Render the bundle at ``run_id`` and store it as ``report.md``. Returns the Markdown."""
    md = render_report(load_bundle(store, run_id))
    store.put(run_id, REPORT, md.encode())
    return md


def render_report(bundle: Bundle) -> str:
    sections = [
        _header(bundle),
        _final_metrics(bundle),
        _curves(bundle),
        _experiment(bundle),
        _resources(bundle),
        _model(bundle),
        _environment(bundle),
        _reproduce(bundle),
    ]
    return "\n\n".join(sections) + "\n"


# -- sections ------------------------------------------------------------------


def _header(b: Bundle) -> str:
    s, m = b.spec, b.manifest
    lines = [f"# {s.meta.name} — {m.status}", ""]
    if s.meta.description:
        lines += [s.meta.description, ""]
    rows = [
        ("experiment_id", f"`{m.experiment_id}`"),
        ("run_id", f"`{m.run_id}`"),
        ("attempt", str(m.attempt)),
        ("status", m.status),
        ("device", f"`{m.device}`"),
        ("epochs completed", f"{len(m.history)} / {s.training.epochs}"),
    ]
    if s.meta.tags:
        rows.append(("tags", ", ".join(s.meta.tags)))
    if m.resumes:
        rows.append(("resumed", ", ".join(_resume_point(r) for r in m.resumes)))
    if m.best is not None:
        rows.append(("best", _best(m.best)))
    if m.split is not None:
        rows.append(("split", split_counts(m.split) + ", cases in `split.json`"))
    rows.append(("environment", _env_verdict(b)))
    rows.append(("data", _data_verdict(b)))
    return "\n".join(lines) + "\n" + _table(("field", "value"), rows)


def split_rule(split: Split | None) -> str:
    """How ``spec.split`` cuts the cases, in one line."""
    if split is None:
        return "the data provider's own split"
    order = f"shuffled (seed {split.seed})" if split.order == "shuffled" else "sorted by case id"
    test = f", test {format_number(split.test)}" if split.test else ""
    if split.kind == "holdout":
        return f"holdout: val {format_number(split.val or 0)}{test}, {order}"
    return f"{split.k}-fold, fold {split.fold}{test}, {order}"


def split_counts(counts: dict[str, int]) -> str:
    """``manifest.split`` as ``train 27, val 8, test 6`` (test only when there is one)."""
    return ", ".join(f"{part} {n}" for part, n in counts.items() if n or part != "test")


def _best(best: BestRecord) -> str:
    if best.epoch is None:
        return f"`{best.metric}` ({best.mode}) never defined: no `best.pt`"
    return f"`{best.metric}` ({best.mode}) = {format_number(best.value)} at epoch {best.epoch}, `best.pt`"


def _final_metrics(b: Bundle) -> str:
    history = b.manifest.history
    out = "## Final metrics\n\n"
    if not history:
        return out + "No epoch completed."
    return out + _table(("metric", "final", "min", "max"), final_metric_rows(history))


def final_metric_rows(history: list[dict[str, float]]) -> list[tuple[str, str, str, str]]:
    """(metric, final, "min (epoch e)", "max (epoch e)") per metric, as the report prints them."""
    rows = []
    for k in metric_names(history):
        # NaN = undefined that epoch (e.g. Dice of a class absent from every val label).
        series = [(i, e[k]) for i, e in enumerate(history, start=1) if k in e and not math.isnan(e[k])]
        if not series:
            rows.append((k, "—", "—", "—"))
            continue
        lo = min(series, key=lambda p: p[1])
        hi = max(series, key=lambda p: p[1])
        rows.append((
            k,
            format_number(history[-1].get(k)),
            f"{format_number(lo[1])} (epoch {lo[0]})",
            f"{format_number(hi[1])} (epoch {hi[0]})",
        ))
    return rows


def _curves(b: Bundle) -> str:
    history = b.manifest.history
    out = "## Training curves\n\n"
    if not history:
        return out + "No epoch completed."
    names = metric_names(history)
    sparks = []
    for k in names:
        values = [e[k] for e in history if k in e and not math.isnan(e[k])]
        if values:
            sparks.append(f"    {k:<{max(map(len, names))}}  {sparkline(values)}  "
                          f"{format_number(values[0])} → {format_number(values[-1])}")
    rows = [(str(i), *(format_number(e.get(k)) for k in names)) for i, e in enumerate(history, start=1)]
    return out + "\n".join(sparks) + "\n\n" + _table(("epoch", *names), rows)


def _experiment(b: Bundle) -> str:
    s = b.spec
    t = s.training
    rows = [
        ("code.repo", s.code.repo),
        ("code.commit", f"`{s.code.commit}`"),
        ("environment.image", f"`{s.environment.image}`"),
        ("environment.image_digest", f"`{s.environment.image_digest}`"),
        ("dataset", f"{s.dataset.name}" + (f" ({s.dataset.version})" if s.dataset.version else "")),
        ("dataset.snapshot", f"`{s.dataset.snapshot}`"),
        ("preprocessing", _json(s.preprocessing)),
        ("model", f"{s.model.name} {_json(s.model.params)}"),
        ("loss", f"{s.loss.name} {_json(s.loss.params)}"),
        *(
            [(f"metric {i}", f"{m.name} {_json(m.params)}") for i, m in enumerate(s.metrics, start=1)]
            or [("metrics", "val/loss only")]
        ),
        (
            ("inference", f"{s.inference.name} {_json(s.inference.params)}")
            if s.inference is not None
            else ("inference", "model(x) on each val batch")
        ),
        ("split", split_rule(s.split)),
        ("optimizer", t.optimizer),
        ("learning_rate", format_number(t.learning_rate)),
        ("epochs", str(t.epochs)),
        ("batch_size (global)", str(t.batch_size)),
        ("seed", str(t.seed)),
        ("deterministic", str(t.deterministic).lower()),
    ]
    if s.augmentation:
        for i, a in enumerate(s.augmentation, start=1):
            rows.append((f"augmentation {i}", f"{a.name} {_json(a.params)}"))
    else:
        rows.append(("augmentation", "none"))
    intro = "Everything below is in the ExperimentID fingerprint, except the image tag and dataset version label."
    return "## Experiment\n\n" + intro + "\n\n" + _table(("field", "value"), rows)


def _resources(b: Bundle) -> str:
    r = b.spec.resources
    rows = [
        ("workers (world size)", str(r.workers)),
        ("gpu per worker", str(r.gpu)),
        ("cpu per worker", str(r.cpu)),
        ("memory per worker", r.memory),
        ("batch per worker", str(b.spec.per_worker_batch_size())),
    ]
    return "## Resources\n\nPart of the RunID, not the ExperimentID.\n\n" + _table(("field", "value"), rows)


def _model(b: Bundle) -> str:
    out = "## Model\n\n"
    if b.model_state is None:
        return out + f"No weights: the run {b.manifest.status}, so `model.pt` was not written."
    n = sum(t.numel() for t in b.model_state.values())
    dtypes = sorted({str(t.dtype).removeprefix("torch.") for t in b.model_state.values()})
    rows = [("parameters", f"{n:,}"), ("tensors", str(len(b.model_state))), ("dtypes", ", ".join(dtypes))]
    return out + _table(("field", "value"), rows)


def _resume_point(r: ResumePoint) -> str:
    out = f"after epoch {r.epoch}"
    if r.environment is not None:
        out += f" (environment {record_verdict(r.environment)})"
    return out


def _env_verdict(b: Bundle) -> str:
    return record_verdict(b.manifest.environment)


def record_verdict(record: EnvironmentRecord | None) -> str:
    if record is None:
        return "not recorded"
    if record.verified:
        return "verified"
    bad = [f"{name} {c.status}" for name, c in record.checks.items() if c.status != "ok"]
    return "not verified (" + ", ".join(bad) + ")"


def _data_verdict(b: Bundle) -> str:
    return data_verdict(b.manifest)


def data_verdict(manifest: RunManifest) -> str:
    check = manifest.data_snapshot
    if check is None:
        return "not recorded"
    return "verified" if check.status == "ok" else f"not verified (dataset.snapshot {check.status})"


def _checks(b: Bundle) -> str:
    return _table(("pin", "status", "detail"), check_rows(b.manifest))


def check_rows(manifest: RunManifest) -> list[tuple[str, str, str]]:
    """(pin, status, detail) for every recorded check, the data snapshot last."""
    checks = dict(manifest.environment.checks) if manifest.environment else {}
    if manifest.data_snapshot is not None:
        checks["dataset.snapshot"] = manifest.data_snapshot
    return [(n, c.status, c.detail) for n, c in checks.items()]


def _environment(b: Bundle) -> str:
    s, record = b.spec, b.manifest.environment
    out = "## Environment\n\n"
    data = "\n\n" + _checks(b) if b.manifest.data_snapshot is not None else ""
    if record is None:
        rows = [
            ("code.commit", f"`{s.code.commit}`"),
            ("image", f"`{s.environment.image}`"),
            ("image_digest", f"`{s.environment.image_digest}`"),
            ("device (actual)", f"`{b.manifest.device}`"),
        ]
        note = (
            "Not recorded for this run: commit and image digest are only **declared** by the spec "
            "and not verified against what actually ran."
        )
        return out + note + "\n\n" + _table(("field", "value"), rows) + data

    verdict = (
        "The code and image this run ran on match the spec's pins."
        if record.verified
        else "**Not verified**: the run cannot be shown to have run exactly what the spec pins."
    )
    checks = _checks(b)
    actual = actual_rows(record, b.manifest.device)
    return out + verdict + "\n\n" + checks + "\n\nActually ran on:\n\n" + _table(("field", "value"), actual)


def actual_rows(record: EnvironmentRecord, device: str) -> list[tuple[str, str]]:
    """What the run actually ran on, ids in Markdown code spans."""
    a = record.actual
    return [
        ("python", a.python),
        ("platform", a.platform),
        ("device", f"`{device}`"),
        ("gpus", ", ".join(a.gpus) or "none"),
        ("cuda / cudnn", f"{a.cuda or '—'} / {a.cudnn or '—'}"),
        ("torch threads", str(a.torch_threads)),
        ("git HEAD", f"`{a.code.head}`" if a.code.head else "—"),
        ("image_digest", f"`{a.image_digest}`" if a.image_digest else "—"),
        *((name, version) for name, version in sorted(a.packages.items())),
    ]


def _reproduce(b: Bundle) -> str:
    return (
        "## Reproduce\n\n"
        "Re-run this bundle and compare run_id, history and weights:\n\n"
        f"    mlops reproduce {b.manifest.run_id} --artifacts <DIR>"
    )


# -- helpers -------------------------------------------------------------------


def metric_names(history: list[dict[str, float]]) -> list[str]:
    names: list[str] = []
    for e in history:
        names += [k for k in e if k not in names]
    return names


def sparkline(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARKS[0] * len(values)
    top = len(_SPARKS) - 1
    return "".join(_SPARKS[min(top, int((v - lo) / (hi - lo) * top))] for v in values)


def format_number(v: float | None) -> str:
    """Missing or undefined (NaN) values render as a dash."""
    return "—" if v is None or math.isnan(v) else f"{v:.6g}"


def _json(v: Any) -> str:
    return f"`{json.dumps(v, sort_keys=True)}`"


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    def line(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    return "\n".join([line(header), "|" + "|".join("---" for _ in header) + "|", *map(line, rows)])
