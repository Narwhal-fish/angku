"""`mlops ui`: a terminal browser over an artifacts dir (roadmap 13, phase A: read-only).

    mlops/ui/rows.py   pure: RunSummary / ExperimentGroup / manifest -> table cells (no textual import)
    mlops/ui/app.py    Textual: Experiments -> Runs -> Run (tabs Overview, Metrics, Environment, Report)

Keys: Enter drills down, Esc goes back (or clears the filter), / filters, Tab switches tabs, q quits.
The UI only reads the store; it re-reads it every ``refresh`` seconds.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.application.bundle import CHECKPOINT, MANIFEST, SPEC
from mlops.application.listing import ExperimentGroup, RunSummary
from mlops.application.report import REPORT, final_metric_rows, sparkline
from mlops.cli import main
from mlops.domain.best import BestRecord
from mlops.ui.rows import (
    EXPERIMENT_COLUMNS,
    experiment_row,
    history_rows,
    main_metric,
    metric_rows,
    overview_rows,
    run_columns,
    run_row,
)

ROOT = Path(__file__).parent.parent
SMOKE = ROOT / "experiments/configs/tiny_cnn_smoke.yaml"


def _summary(**kw) -> RunSummary:
    base = dict(run_id="a" * 64, experiment_id="e" * 64, name="exp", status="FINISHED", attempt=1,
                epochs_done=3, epochs_total=3, final_metrics={"train/loss": 0.5, "val/loss": 0.25})
    return RunSummary(**{**base, **kw})


# --- rows (pure) ---


def test_experiment_row():
    one = ExperimentGroup("e" * 64, "exp", [_summary()])
    same = ExperimentGroup("e" * 64, "exp", [_summary(), _summary(run_id="b" * 64, attempt=2)])
    differ = ExperimentGroup("e" * 64, "exp", [_summary(), _summary(run_id="b" * 64, final_metrics={"val/loss": 0.3})])
    close = ExperimentGroup("e" * 64, "exp", [_summary(), _summary(run_id="b" * 64, final_metrics={
        "train/loss": 0.5, "val/loss": 0.255})], atol=0.01)
    assert EXPERIMENT_COLUMNS == ("NAME", "EXPERIMENT", "RUNS", "FINISHED", "AGREE")
    assert experiment_row(one) == ("exp", "e" * 12, "1", "1", "-")
    assert experiment_row(same)[-1] == "yes"
    assert experiment_row(differ)[-1] == "DIFFER"
    assert experiment_row(close)[-1] == "within 0.01"


def test_main_metric_prefers_keep_best_then_val_loss():
    best = BestRecord(metric="val/dice/mean", mode="max", epoch=2, value=0.7)
    assert main_metric([_summary(), _summary(best=best)]) == "val/dice/mean"
    assert main_metric([_summary()]) == "val/loss"
    assert main_metric([_summary(final_metrics={"train/loss": 1.0})]) == "train/loss"
    assert main_metric([_summary(status="INTERRUPTED", final_metrics={})]) is None


def test_run_row_and_columns():
    assert run_columns("val/loss") == ("RUN", "ATTEMPT", "STATUS", "EPOCHS", "val/loss", "NOTES")
    assert run_columns(None) == ("RUN", "ATTEMPT", "STATUS", "EPOCHS", "NOTES")
    best = BestRecord(metric="val/loss", mode="min", epoch=2, value=0.2)
    assert run_row(_summary(best=best, resumes=1), "val/loss") == (
        "a" * 12, "1", "FINISHED", "3/3", "0.25", "best val/loss@2; resumed x1")
    interrupted = _summary(status="INTERRUPTED", attempt=None, epochs_done=None, final_metrics={}, resumable=True)
    assert run_row(interrupted, "val/loss") == ("a" * 12, "?", "INTERRUPTED", "?/3", "—", "resumable")
    assert run_row(_summary(final_metrics={"val/loss": math.nan}), None) == ("a" * 12, "1", "FINISHED", "3/3", "")


def test_metric_rows_match_the_report():
    history = [{"loss": 3.0, "acc": 0.1}, {"loss": 2.0, "acc": 0.4}, {"loss": 1.0, "acc": 0.3}]
    rows = metric_rows(history)
    assert [r[:4] for r in rows] == final_metric_rows(history)
    assert rows[0][4] == sparkline([3.0, 2.0, 1.0])
    assert history_rows(history) == (("epoch", "loss", "acc"), [("1", "3", "0.1"), ("2", "2", "0.4"), ("3", "1", "0.3")])
    assert metric_rows([]) == [] and history_rows([]) == (("epoch",), [])


def test_overview_without_manifest():
    s = _summary(status="INTERRUPTED", attempt=None, epochs_done=None, final_metrics={}, resumable=True)
    rows = dict(overview_rows(s, spec=None, manifest=None))
    assert rows["status"] == "INTERRUPTED"
    assert rows["run_id"] == "a" * 64
    assert rows["resumable"] == f"yes: mlops resume {'a' * 64} --artifacts <DIR>"
    assert rows["environment"] == "no manifest"


def test_core_and_rows_never_import_textual():
    code = (
        "import sys, mlops.domain, mlops.config, mlops.registry, mlops.ports, mlops.application, "
        "mlops.application.listing, mlops.application.report, mlops.ui.rows, mlops.cli; "
        "assert 'textual' not in sys.modules, 'textual imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# --- app ---


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> Path:
    """Two FINISHED attempts + one INTERRUPTED run of the smoke spec, one FAILED run of another."""
    base = tmp_path_factory.mktemp("ui")
    art = base / "artifacts"
    for attempt in ("1", "2"):
        assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art), "--attempt", attempt]) == 0
    data = yaml.safe_load(SMOKE.read_text())
    data["meta"]["name"] = "broken-cnn"
    data["model"]["params"] = {"in_channels": 3, "out_channels": 2, "hidden": 4, "spatial_dims": 3}
    (base / "broken.yaml").write_text(yaml.safe_dump(data))
    assert main(["run", str(base / "broken.yaml"), "--tracker", "inmemory", "--artifacts", str(art)]) == 1
    finished = next(d for d in art.iterdir() if json.loads((d / MANIFEST).read_text())["status"] == "FINISHED")
    killed = art / ("f" * 64)
    killed.mkdir()
    shutil.copy(finished / SPEC, killed / SPEC)
    (killed / CHECKPOINT).write_bytes(b"not read by the ui")
    return art


def _rows(table) -> list[tuple]:
    return [tuple(str(c) for c in table.get_row_at(i)) for i in range(table.row_count)]


def _app(art: Path, refresh: float = 60.0):
    from mlops.ui.app import MlopsApp

    return MlopsApp(LocalArtifactStore(art), label=str(art), refresh=refresh)


def _drive(art: Path, steps, *, refresh: float = 60.0) -> None:
    pytest.importorskip("textual")

    async def go():
        app = _app(art, refresh)
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            await steps(app, pilot)

    asyncio.run(go())


def test_experiments_screen_lists_one_row_per_experiment(artifacts):
    async def steps(app, pilot):
        from textual.widgets import DataTable

        rows = {r[0]: r for r in _rows(app.screen.query_one(DataTable))}
        assert set(rows) == {"tiny-cnn-smoke", "broken-cnn"}
        assert rows["tiny-cnn-smoke"][2:] == ("3", "2", "yes")
        assert rows["broken-cnn"][2:] == ("1", "0", "-")

    _drive(artifacts, steps)


def test_enter_drills_down_and_escape_goes_back(artifacts):
    async def steps(app, pilot):
        from textual.widgets import DataTable

        from mlops.ui.app import ExperimentsScreen, RunScreen, RunsScreen

        table = app.screen.query_one(DataTable)
        table.move_cursor(row=[r[0] for r in _rows(table)].index("tiny-cnn-smoke"))
        await pilot.press("enter")
        assert isinstance(app.screen, RunsScreen)
        runs = app.screen.query_one(DataTable)
        assert [str(c.label) for c in runs.ordered_columns][4] == "val/loss"
        assert sorted(r[2] for r in _rows(runs)) == ["FINISHED", "FINISHED", "INTERRUPTED"]
        await pilot.press("enter")
        assert isinstance(app.screen, RunScreen)
        await pilot.press("escape")
        assert isinstance(app.screen, RunsScreen)
        await pilot.press("escape")
        assert isinstance(app.screen, ExperimentsScreen)
        await pilot.press("escape")  # the root stays
        assert isinstance(app.screen, ExperimentsScreen)

    _drive(artifacts, steps)


def _finished_run_id(art: Path) -> str:
    return next(d.name for d in sorted(art.iterdir())
                if (d / MANIFEST).exists() and json.loads((d / MANIFEST).read_text())["status"] == "FINISHED")


def test_run_screen_tabs(artifacts):
    run_id = _finished_run_id(artifacts)

    async def steps(app, pilot):
        from textual.widgets import DataTable, Markdown, TabbedContent

        from mlops.ui.app import RunScreen

        app.push_screen(RunScreen(run_id))
        await pilot.pause()
        overview = dict(_rows(app.screen.query_one("#overview", DataTable)))
        assert overview["run_id"] == run_id and overview["status"] == "FINISHED"
        tabs = app.screen.query_one(TabbedContent)
        assert tabs.active == "tab-overview"
        await pilot.press("tab")
        assert tabs.active == "tab-metrics"
        assert "val/loss" in [r[0] for r in _rows(app.screen.query_one("#metrics", DataTable))]
        assert _rows(app.screen.query_one("#history", DataTable))[0][0] == "1"
        await pilot.press("tab")
        assert tabs.active == "tab-environment"
        assert "code.commit" in [r[0] for r in _rows(app.screen.query_one("#checks", DataTable))]
        assert all("`" not in c for r in _rows(app.screen.query_one("#actual", DataTable)) for c in r)
        await pilot.press("tab")
        assert tabs.active == "tab-report"
        assert app.screen.query_one("#report", Markdown).source == (artifacts / run_id / REPORT).read_text()
        await pilot.press("tab")
        assert tabs.active == "tab-overview"

    _drive(artifacts, steps)


def test_run_screen_of_an_interrupted_run(artifacts):
    async def steps(app, pilot):
        from textual.widgets import DataTable, Markdown

        from mlops.ui.app import RunScreen

        app.push_screen(RunScreen("f" * 64))
        await pilot.pause()
        overview = dict(_rows(app.screen.query_one("#overview", DataTable)))
        assert overview["status"] == "INTERRUPTED"
        assert overview["resumable"].startswith("yes")
        assert app.screen.query_one("#metrics", DataTable).row_count == 0
        assert "no report" in app.screen.query_one("#report", Markdown).source

    _drive(artifacts, steps)


def test_filter_narrows_rows_and_escape_clears_it(artifacts):
    async def steps(app, pilot):
        from textual.widgets import DataTable

        table = app.screen.query_one(DataTable)
        await pilot.press("slash", "q", "backspace", "b", "r", "o")  # q types, it does not quit
        assert app.is_running
        assert [r[0] for r in _rows(table)] == ["broken-cnn"]
        await pilot.press("enter")  # back to the table, filter kept
        assert [r[0] for r in _rows(table)] == ["broken-cnn"]
        await pilot.press("escape")
        assert table.row_count == 2

    _drive(artifacts, steps)


def test_rereads_the_store(artifacts, tmp_path):
    art = tmp_path / "artifacts"
    shutil.copytree(artifacts, art)

    async def steps(app, pilot):
        from textual.widgets import DataTable

        table = app.screen.query_one(DataTable)
        assert table.row_count == 2
        shutil.rmtree(next(d for d in art.iterdir() if "broken" in (d / SPEC).read_text()))
        await pilot.pause(0.3)
        assert [r[0] for r in _rows(table)] == ["tiny-cnn-smoke"]

    _drive(art, steps, refresh=0.05)


# --- CLI ---


def test_cli_ui_starts_the_app(artifacts, monkeypatch):
    pytest.importorskip("textual")
    from mlops.ui.app import MlopsApp

    seen = {}
    monkeypatch.setattr(MlopsApp, "run", lambda self: seen.update(label=self.label, refresh=self.refresh_every))
    assert main(["ui", "--artifacts", str(artifacts), "--refresh", "5"]) == 0
    assert seen == {"label": str(artifacts), "refresh": 5.0}


def test_cli_ui_without_textual_is_a_clean_error(artifacts, monkeypatch, capsys):
    for name in [n for n in sys.modules if n == "textual" or n.startswith("textual.")] or ["textual"]:
        monkeypatch.setitem(sys.modules, name, None)  # cached submodules too, or `textual.app` still imports
    monkeypatch.delitem(sys.modules, "mlops.ui.app", raising=False)
    assert main(["ui", "--artifacts", str(artifacts)]) == 2
    assert "'.[tui]'" in capsys.readouterr().err


def test_cli_ui_missing_dir_is_exit_2(tmp_path, capsys):
    assert main(["ui", "--artifacts", str(tmp_path / "nope")]) == 2
    assert "nope" in capsys.readouterr().err
