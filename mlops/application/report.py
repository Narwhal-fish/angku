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
from typing import Any

from mlops.application.bundle import Bundle, load_bundle
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
    return "\n".join(lines) + "\n" + _table(("field", "value"), rows)


def _final_metrics(b: Bundle) -> str:
    history = b.manifest.history
    out = "## Final metrics\n\n"
    if not history:
        return out + "No epoch completed."
    rows = []
    for k in _metric_names(history):
        series = [(i, e[k]) for i, e in enumerate(history, start=1) if k in e]
        lo = min(series, key=lambda p: p[1])
        hi = max(series, key=lambda p: p[1])
        final = history[-1].get(k)
        rows.append((
            k,
            _num(final) if final is not None else "—",
            f"{_num(lo[1])} (epoch {lo[0]})",
            f"{_num(hi[1])} (epoch {hi[0]})",
        ))
    return out + _table(("metric", "final", "min", "max"), rows)


def _curves(b: Bundle) -> str:
    history = b.manifest.history
    out = "## Training curves\n\n"
    if not history:
        return out + "No epoch completed."
    names = _metric_names(history)
    sparks = []
    for k in names:
        values = [e[k] for e in history if k in e]
        sparks.append(f"    {k:<{max(map(len, names))}}  {_sparkline(values)}  "
                      f"{_num(values[0])} → {_num(values[-1])}")
    rows = [(str(i), *(_num(e[k]) if k in e else "—" for k in names)) for i, e in enumerate(history, start=1)]
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
        ("optimizer", t.optimizer),
        ("learning_rate", _num(t.learning_rate)),
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


def _environment(b: Bundle) -> str:
    s = b.spec
    rows = [
        ("code.commit", f"`{s.code.commit}`"),
        ("image", f"`{s.environment.image}`"),
        ("image_digest", f"`{s.environment.image_digest}`"),
        ("device (actual)", f"`{b.manifest.device}`"),
    ]
    note = (
        "Commit and image digest are **declared** by the spec and not verified against what "
        "actually ran; library versions and GPU model are not recorded yet."
    )
    return "## Environment\n\n" + note + "\n\n" + _table(("field", "value"), rows)


def _reproduce(b: Bundle) -> str:
    return (
        "## Reproduce\n\n"
        "Re-run this bundle and compare run_id, history and weights:\n\n"
        f"    mlops reproduce {b.manifest.run_id} --artifacts <DIR>"
    )


# -- helpers -------------------------------------------------------------------


def _metric_names(history: list[dict[str, float]]) -> list[str]:
    names: list[str] = []
    for e in history:
        names += [k for k in e if k not in names]
    return names


def _sparkline(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARKS[0] * len(values)
    top = len(_SPARKS) - 1
    return "".join(_SPARKS[min(top, int((v - lo) / (hi - lo) * top))] for v in values)


def _num(v: float) -> str:
    return f"{v:.6g}"


def _json(v: Any) -> str:
    return f"`{json.dumps(v, sort_keys=True)}`"


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    def line(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    return "\n".join([line(header), "|" + "|".join("---" for _ in header) + "|", *map(line, rows)])
