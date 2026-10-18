"""`mlops ui`: a terminal browser over an artifacts dir (roadmap 13; phase A browse, phase B actions).

    mlops/ui/rows.py   pure: RunSummary / ExperimentGroup / manifest -> table cells (no textual import)
    mlops/ui/app.py    Textual: Experiments -> Runs -> Run (tabs Overview, Metrics, Environment, Report)

Keys: Enter drills down, Esc goes back (or clears the filter), / filters, Tab switches tabs, q quits.
Phase B (Runs / Run screens): space marks a run, d diffs the two marked runs, r reproduces and
R resumes the run under the cursor after a confirmation, in a background worker. The browser itself
only reads the store; it re-reads it every ``refresh`` seconds. The actions are injected
(``UiActions``) by the CLI, which owns the adapters.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

from mlops.adapters.local_artifacts import LocalArtifactStore
from mlops.application.bundle import CHECKPOINT, MANIFEST, SPEC
from mlops.application.listing import ExperimentGroup, RunSummary
from mlops.application.report import REPORT, final_metric_rows, sparkline
from mlops.application.reproduce import ReproduceReport
from mlops.application.runner import RunResult
from mlops.cli import CliError, _ui_actions, build_parser, main
from mlops.domain.best import BestRecord
from mlops.domain.manifest import ResumePoint
from mlops.ui.actions import UiActions
from mlops.ui.rows import (
    EXPERIMENT_COLUMNS,
    experiment_row,
    history_rows,
    main_metric,
    metric_rows,
    overview_rows,
    resume_lines,
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


def test_resume_lines():
    result = RunResult(experiment_id="e" * 64, run_id="a" * 64, device="cpu", history=[{"loss": 1.0}] * 3,
                       resumes=[ResumePoint(epoch=2)])
    assert resume_lines("exp", result) == [
        "exp: FINISHED", f"  run_id         {'a' * 64}", "  resumed after epoch 2", "  epochs         3"]


def test_core_and_rows_never_import_textual():
    code = (
        "import sys, mlops.domain, mlops.config, mlops.registry, mlops.ports, mlops.application, "
        "mlops.application.listing, mlops.application.report, mlops.ui.rows, mlops.ui.actions, mlops.cli; "
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


def _app(art: Path, refresh: float = 60.0, actions: UiActions | None = None):
    from mlops.ui.app import MlopsApp

    return MlopsApp(LocalArtifactStore(art), label=str(art), refresh=refresh, actions=actions)


def _drive(art: Path, steps, *, refresh: float = 60.0, actions: UiActions | None = None) -> list[str]:
    """Run ``steps(app, pilot)`` headless; returns every notification the app showed."""
    pytest.importorskip("textual")
    notes: list[str] = []

    async def go():
        app = _app(art, refresh, actions)
        notify = app.notify

        def record(message, **kw):
            notes.append(str(message))
            return notify(message, **kw)

        app.notify = record
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            await steps(app, pilot)

    asyncio.run(go())
    return notes


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
        assert [str(c.label) for c in runs.ordered_columns][1:] == list(run_columns("val/loss"))  # [0] = mark
        assert sorted(r[3] for r in _rows(runs)) == ["FINISHED", "FINISHED", "INTERRUPTED"]
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
    monkeypatch.setattr(MlopsApp, "run", lambda self: seen.update(
        label=self.label, refresh=self.refresh_every, actions=self.actions, atol=self.atol))
    assert main(["ui", "--artifacts", str(artifacts), "--refresh", "5", "--atol", "0.1"]) == 0
    assert seen["label"] == str(artifacts) and seen["refresh"] == 5.0 and seen["atol"] == 0.1
    assert isinstance(seen["actions"], UiActions)


def test_cli_ui_without_textual_is_a_clean_error(artifacts, monkeypatch, capsys):
    for name in [n for n in sys.modules if n == "textual" or n.startswith("textual.")] or ["textual"]:
        monkeypatch.setitem(sys.modules, name, None)  # cached submodules too, or `textual.app` still imports
    monkeypatch.delitem(sys.modules, "mlops.ui.app", raising=False)
    assert main(["ui", "--artifacts", str(artifacts)]) == 2
    assert "'.[tui]'" in capsys.readouterr().err


def test_cli_ui_missing_dir_is_exit_2(tmp_path, capsys):
    assert main(["ui", "--artifacts", str(tmp_path / "nope")]) == 2
    assert "nope" in capsys.readouterr().err


# --- phase B: actions ---


class FakeActions:
    def __init__(self, *, fail: bool = False, gate: threading.Event | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail
        self.gate = gate

    def reproduce(self, run_id: str) -> ReproduceReport:
        self.calls.append(("reproduce", run_id))
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail:
            raise RuntimeError("boom")
        return ReproduceReport(run_id=run_id, experiment_id="e" * 64, replay_run_id=run_id,
                               checks={"run_id": True, "history": True, "weights": True})

    def resume(self, run_id: str) -> RunResult:
        self.calls.append(("resume", run_id))
        return RunResult(experiment_id="e" * 64, run_id=run_id, device="cpu", history=[{"loss": 1.0}],
                         resumes=[ResumePoint(epoch=1)])

    def ui(self) -> UiActions:
        return UiActions(reproduce=self.reproduce, resume=self.resume)


async def _to_runs(app, pilot):
    from textual.widgets import DataTable

    table = app.screen.query_one(DataTable)
    table.move_cursor(row=[r[0] for r in _rows(table)].index("tiny-cnn-smoke"))
    await pilot.press("enter")


def _cursor_to(app, status: str) -> str:
    """Put the Runs cursor on the first run with ``status``; returns its full run_id."""
    from textual.widgets import DataTable

    table = app.screen.query_one(DataTable)
    row = [r[3] for r in _rows(table)].index(status)
    table.move_cursor(row=row)
    return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value


async def _settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


def test_mark_two_runs_and_diff_them(artifacts):
    async def steps(app, pilot):
        from textual.widgets import DataTable

        from mlops.ui.app import DiffScreen, RunsScreen

        await _to_runs(app, pilot)
        table = app.screen.query_one(DataTable)
        table.move_cursor(row=0)
        await pilot.press("space", "down", "space")
        assert [r[0] for r in _rows(table)][:2] == ["*", "*"]
        await pilot.press("d")
        assert isinstance(app.screen, DiffScreen)
        assert "same experiment: fingerprints identical" in "\n".join(app.screen.lines)
        await pilot.press("escape")
        assert isinstance(app.screen, RunsScreen)

    _drive(artifacts, steps, actions=FakeActions().ui())


def test_diff_needs_two_marks_and_a_third_drops_the_oldest(artifacts):
    async def steps(app, pilot):
        from textual.widgets import DataTable

        from mlops.ui.app import RunsScreen

        await _to_runs(app, pilot)
        table = app.screen.query_one(DataTable)
        table.move_cursor(row=0)
        await pilot.press("space", "d")
        assert isinstance(app.screen, RunsScreen)
        await pilot.press("down", "space", "down", "space")
        assert [r[0] for r in _rows(table)] == [" ", "*", "*"]
        await pilot.press("space")  # unmark
        assert [r[0] for r in _rows(table)] == [" ", "*", " "]

    notes = _drive(artifacts, steps, actions=FakeActions().ui())
    assert any("mark two runs" in n.lower() for n in notes)


def test_reproduce_asks_first_then_runs_in_the_background(artifacts):
    fake = FakeActions()

    async def steps(app, pilot):
        from mlops.ui.app import ConfirmScreen, ResultScreen, RunsScreen

        await _to_runs(app, pilot)
        run_id = _cursor_to(app, "FINISHED")
        await pilot.press("r")
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("n")
        assert fake.calls == []
        await pilot.press("r", "y")
        await _settle(app, pilot)
        assert fake.calls == [("reproduce", run_id)]
        assert isinstance(app.screen, ResultScreen)
        assert "tiny-cnn-smoke: REPRODUCED" in app.screen.lines
        await pilot.press("escape")
        assert isinstance(app.screen, RunsScreen)
        assert app.busy is None

    _drive(artifacts, steps, actions=fake.ui())


def test_reproduce_refuses_a_run_that_did_not_finish(artifacts):
    fake = FakeActions()

    async def steps(app, pilot):
        from mlops.ui.app import RunsScreen

        await _to_runs(app, pilot)
        _cursor_to(app, "INTERRUPTED")
        await pilot.press("r")
        assert isinstance(app.screen, RunsScreen)

    notes = _drive(artifacts, steps, actions=fake.ui())
    assert fake.calls == [] and any("only FINISHED" in n for n in notes)


def test_resume_a_resumable_run(artifacts):
    fake = FakeActions()

    async def steps(app, pilot):
        from mlops.ui.app import ConfirmScreen, ResultScreen

        await _to_runs(app, pilot)
        _cursor_to(app, "FINISHED")
        await pilot.press("R")
        assert not isinstance(app.screen, ConfirmScreen)  # a finished run is not resumable
        _cursor_to(app, "INTERRUPTED")
        await pilot.press("R", "y")
        await _settle(app, pilot)
        assert fake.calls == [("resume", "f" * 64)]
        assert isinstance(app.screen, ResultScreen)
        assert "tiny-cnn-smoke: FINISHED" in app.screen.lines

    notes = _drive(artifacts, steps, actions=fake.ui())
    assert any("not resumable" in n for n in notes)


def test_run_screen_has_the_same_keys(artifacts):
    fake = FakeActions()
    run_id = _finished_run_id(artifacts)

    async def steps(app, pilot):
        from mlops.ui.app import RunScreen

        app.push_screen(RunScreen(run_id))
        await pilot.pause()
        await pilot.press("r", "y")
        await _settle(app, pilot)

    _drive(artifacts, steps, actions=fake.ui())
    assert fake.calls == [("reproduce", run_id)]


def test_a_failing_action_is_reported(artifacts):
    async def steps(app, pilot):
        await _to_runs(app, pilot)
        _cursor_to(app, "FINISHED")
        await pilot.press("r", "y")
        await _settle(app, pilot)
        assert app.busy is None

    notes = _drive(artifacts, steps, actions=FakeActions(fail=True).ui())
    assert any("boom" in n for n in notes)


def test_one_action_at_a_time(artifacts):
    gate = threading.Event()
    fake = FakeActions(gate=gate)

    async def steps(app, pilot):
        await _to_runs(app, pilot)
        _cursor_to(app, "FINISHED")
        await pilot.press("r", "y")
        await pilot.pause()
        assert app.busy is not None
        await pilot.press("r")
        gate.set()
        await _settle(app, pilot)

    notes = _drive(artifacts, steps, actions=fake.ui())
    assert len(fake.calls) == 1 and any("already running" in n for n in notes)


def test_without_actions_the_keys_say_so(artifacts):
    async def steps(app, pilot):
        await _to_runs(app, pilot)
        _cursor_to(app, "FINISHED")
        await pilot.press("r")

    notes = _drive(artifacts, steps)
    assert any("not available" in n for n in notes)


def test_cli_ui_actions_reproduce_and_resume_for_real(tmp_path):
    art = tmp_path / "artifacts"
    assert main(["run", str(SMOKE), "--tracker", "inmemory", "--artifacts", str(art)]) == 0
    (finished,) = [d.name for d in art.iterdir()]
    data = yaml.safe_load(SMOKE.read_text())
    data["training"]["epochs"] = 2
    (tmp_path / "two.yaml").write_text(yaml.safe_dump(data))
    assert main(["run", str(tmp_path / "two.yaml"), "--tracker", "inmemory", "--artifacts", str(art),
                 "--checkpoint-every", "1", "--keep-checkpoint"]) == 0
    (killed,) = [d.name for d in art.iterdir() if d.name != finished]
    for name in (MANIFEST, "model.pt", REPORT):  # as if killed after the last checkpoint
        (art / killed / name).unlink()

    args = build_parser().parse_args(["ui", "--artifacts", str(art), "--tracker", "inmemory"])
    actions = _ui_actions(args, LocalArtifactStore(art))
    assert actions.reproduce(finished).ok
    result = actions.resume(killed)
    assert result.run_id == killed and result.resumes[-1].epoch == 2
    assert json.loads((art / killed / MANIFEST).read_text())["status"] == "FINISHED"
    assert (art / killed / REPORT).exists()
    with pytest.raises(CliError, match="already FINISHED"):
        actions.resume(killed)
