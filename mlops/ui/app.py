"""``angku ui``: a terminal browser over an artifact store. Needs the ``tui`` extra.

Experiments -> Runs -> Run (tabs Overview, Metrics, Environment, Report). Enter drills down,
Esc goes back (or clears the filter first), / filters the list, Tab cycles the run's tabs.
Browsing is read-only: it goes through ``application.listing`` / ``report`` and every screen
re-reads the store every ``refresh`` seconds, redrawing only what changed.

RayJobs (j on the Experiments screen): our RayJobs in the namespace via kubectl, fetched in a
thread worker and re-fetched every max(refresh, 5) s; Enter shows one, l its driver log, o opens its
run when the bundle is in this artifacts dir. kubectl errors are shown, never raised.

Actions (Runs and Run screens): space marks a run, d diffs the two marked ones, r reproduces and
R resumes the run under the cursor. r and R ask first, then run ``UiActions`` (injected by the
CLI) in a thread worker, one at a time; the result opens in a dialog, a failure is a notification.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Markdown, Static, TabbedContent, TabPane
from textual.worker import Worker, WorkerState

from mlops.adapters.kuberay import RayJobStatus
from mlops.application.bundle import MANIFEST, SPEC, load_bundle
from mlops.application.diff import diff_lines, diff_runs
from mlops.application.listing import RunSummary, group_by_experiment, list_runs
from mlops.application.progress import PROGRESS, read_progress
from mlops.application.report import REPORT, render_report
from mlops.application.reproduce import ReproduceReport, reproduce_lines
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.manifest import RunManifest, parse_manifest
from mlops.ports.artifacts import ArtifactError, ArtifactStore
from mlops.ui.actions import UiActions
from mlops.ui.rows import (
    ACTUAL_COLUMNS,
    CHECK_COLUMNS,
    EXPERIMENT_COLUMNS,
    METRIC_COLUMNS,
    RAYJOB_COLUMNS,
    environment_rows,
    experiment_row,
    history_rows,
    main_metric,
    metric_rows,
    overview_rows,
    rayjob_detail_rows,
    rayjob_row,
    resume_lines,
    run_columns,
    run_row,
)

Rows = list[tuple[str, tuple[str, ...]]]  # (row key, cells)


def _last_line(error: BaseException | None) -> str:
    """kubectl's stderr repeats its retries; the last line says what went wrong."""
    lines = [line for line in str(error).splitlines() if line.strip()]
    return lines[-1] if lines else repr(error)


def _fill(table: DataTable[Any], columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    table.clear(columns=True)
    table.add_columns(*columns)
    for cells in rows:
        table.add_row(*cells)


class TableScreen(Screen[None]):
    """A filterable list that keeps its cursor on the same row across re-reads."""

    BINDINGS = [Binding("slash", "filter", "Filter"), Binding("escape", "back", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._filter = ""
        self._shown: tuple[tuple[str, ...], Rows] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(cursor_type="row")
        yield Input(placeholder="filter", id="filter")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_rows()
        self.set_interval(self.app.refresh_every, self.refresh_rows)
        self.query_one(DataTable).focus()

    def columns(self, runs: list[RunSummary]) -> tuple[str, ...]:
        raise NotImplementedError

    def rows(self, runs: list[RunSummary]) -> Rows:
        raise NotImplementedError

    def open(self, key: str) -> None:
        raise NotImplementedError

    def refresh_rows(self) -> None:
        runs = self.app.runs()
        columns = self.columns(runs)
        needle = self._filter.lower()
        rows = [(k, c) for k, c in self.rows(runs) if needle in " ".join(c).lower()]
        if (columns, rows) == self._shown:
            return
        table = self.query_one(DataTable)
        current = None
        if table.row_count:
            current = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        table.clear(columns=columns != (self._shown or ((), []))[0])
        if not table.columns:
            table.add_columns(*columns)
        for key, cells in rows:
            table.add_row(*cells, key=key)
        if current in {k for k, _ in rows}:
            table.move_cursor(row=table.get_row_index(current))
        self._shown = (columns, rows)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value is not None:
            self.open(event.row_key.value)

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.add_class("active")
        box.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter = event.value
        self.refresh_rows()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.query_one(DataTable).focus()

    def action_back(self) -> None:
        box = self.query_one("#filter", Input)
        if self._filter or box.has_class("active"):
            box.value = ""
            box.remove_class("active")
            self._filter = ""
            self.refresh_rows()
            self.query_one(DataTable).focus()
        elif len(self.app.screen_stack) > 1:
            self.app.pop_screen()


class ExperimentsScreen(TableScreen):
    BINDINGS = [Binding("j", "rayjobs", "RayJobs")]

    def action_rayjobs(self) -> None:
        self.app.open_rayjobs()

    def on_mount(self) -> None:
        self.sub_title = f"Experiments  ({self.app.label})"
        super().on_mount()

    def columns(self, runs: list[RunSummary]) -> tuple[str, ...]:
        return EXPERIMENT_COLUMNS

    def rows(self, runs: list[RunSummary]) -> Rows:
        return [(g.experiment_id, experiment_row(g)) for g in group_by_experiment(runs)]

    def open(self, key: str) -> None:
        self.app.push_screen(RunsScreen(key))


RUN_ACTIONS = [Binding("r", "reproduce", "Reproduce"), Binding("R", "resume", "Resume")]


class RunsScreen(TableScreen):
    BINDINGS = [Binding("space", "mark", "Mark"), Binding("d", "diff", "Diff marked"), *RUN_ACTIONS]

    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id
        self.marked: list[str] = []  # at most two run_ids, oldest first

    def _mine(self, runs: list[RunSummary]) -> list[RunSummary]:
        return [r for r in runs if r.experiment_id == self.experiment_id]

    def on_mount(self) -> None:
        mine = self._mine(self.app.runs())
        name = mine[0].name if mine else self.experiment_id[:12]
        self.sub_title = f"Runs: {name}  {self.experiment_id[:12]}"
        super().on_mount()

    def columns(self, runs: list[RunSummary]) -> tuple[str, ...]:
        return ("", *run_columns(main_metric(self._mine(runs))))

    def rows(self, runs: list[RunSummary]) -> Rows:
        mine = self._mine(runs)
        metric = main_metric(mine)
        return [(r.run_id, ("*" if r.run_id in self.marked else " ", *run_row(r, metric))) for r in mine]

    def open(self, key: str) -> None:
        self.app.push_screen(RunScreen(key))

    def cursor_run(self) -> str | None:
        table = self.query_one(DataTable)
        if not table.row_count:
            return None
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value

    def action_mark(self) -> None:
        run_id = self.cursor_run()
        if run_id is None:
            return
        if run_id in self.marked:
            self.marked.remove(run_id)
        else:
            self.marked = [*self.marked, run_id][-2:]
        self.refresh_rows()

    def action_diff(self) -> None:
        if len(self.marked) != 2:
            self.notify("Mark two runs with space, then press d.", severity="warning")
            return
        self.app.push_screen(DiffScreen(*self.marked))

    def action_reproduce(self) -> None:
        if (run_id := self.cursor_run()) is not None:
            self.app.request("reproduce", run_id)

    def action_resume(self) -> None:
        if (run_id := self.cursor_run()) is not None:
            self.app.request("resume", run_id)


class RunScreen(Screen[None]):
    TABS = ("tab-overview", "tab-metrics", "tab-environment", "tab-report")
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("tab", "next_tab", "Next tab", priority=True),
        Binding("shift+tab", "prev_tab", "Prev tab", priority=True),
        *RUN_ACTIONS,
    ]

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
        self._loaded: tuple[Any, ...] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-overview"):
            with TabPane("Overview", id="tab-overview"):
                yield DataTable(id="overview", cursor_type="row")
            with TabPane("Metrics", id="tab-metrics"):
                yield DataTable(id="metrics", cursor_type="row")
                yield DataTable(id="history", cursor_type="row")
            with TabPane("Environment", id="tab-environment"):
                yield DataTable(id="checks", cursor_type="row")
                yield DataTable(id="actual", cursor_type="row")
            with TabPane("Report", id="tab-report"):
                with VerticalScroll():
                    yield Markdown(id="report")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"Run {self.run_id[:12]}"
        self.load()
        self.set_interval(self.app.refresh_every, self.load)

    def load(self) -> None:
        store: ArtifactStore = self.app.store
        try:
            files = sorted(store.list(self.run_id))
        except ArtifactError:
            files = []
        manifest_bytes = store.get(self.run_id, MANIFEST) if MANIFEST in files else None
        progress = read_progress(store, self.run_id) if PROGRESS in files and manifest_bytes is None else None
        if (files, manifest_bytes, progress) == self._loaded:
            return
        self._loaded = (files, manifest_bytes, progress)
        summary = next((r for r in self.app.runs() if r.run_id == self.run_id), None)
        if summary is None:  # gone, or not a run (no spec.json)
            _fill(
                self.query_one("#overview", DataTable),
                ("FIELD", "VALUE"),
                [("run_id", self.run_id), ("status", "not found in the store")],
            )
            return
        spec = manifest = None
        if summary.status != "UNREADABLE":
            spec = ExperimentSpec.model_validate_json(store.get(self.run_id, SPEC))
            manifest = parse_manifest(manifest_bytes) if manifest_bytes else None
        _fill(
            self.query_one("#overview", DataTable),
            ("FIELD", "VALUE"),
            overview_rows(summary, spec=spec, manifest=manifest, now=time.time()),
        )
        if manifest is not None:
            history = manifest.history
        elif summary.status == "RUNNING" and progress is not None:
            history = progress.history  # this process's epochs so far (a resume starts empty)
        else:
            history = []
        _fill(self.query_one("#metrics", DataTable), METRIC_COLUMNS, metric_rows(history))
        _fill(self.query_one("#history", DataTable), *history_rows(history))
        checks, actual = environment_rows(manifest)
        _fill(self.query_one("#checks", DataTable), CHECK_COLUMNS, checks)
        _fill(self.query_one("#actual", DataTable), ACTUAL_COLUMNS, actual)
        self.query_one("#report", Markdown).update(self._report(summary, files, manifest))

    def _report(self, summary: RunSummary, files: list[str], manifest: RunManifest | None) -> str:
        store: ArtifactStore = self.app.store
        if REPORT in files:
            return store.get(self.run_id, REPORT).decode()
        if manifest is not None:
            try:
                return render_report(load_bundle(store, self.run_id))
            except ArtifactError as e:
                return f"Could not render the report: {e}"
        if summary.status == "RUNNING":
            return "The run is still running; its report is written when it ends."
        return f"No manifest yet, so no report ({summary.status})."

    def _step(self, by: int) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = self.TABS[(self.TABS.index(tabs.active) + by) % len(self.TABS)]

    def action_next_tab(self) -> None:
        self._step(1)

    def action_prev_tab(self) -> None:
        self._step(-1)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_reproduce(self) -> None:
        self.app.request("reproduce", self.run_id)

    def action_resume(self) -> None:
        self.app.request("resume", self.run_id)


class RayJobsScreen(Screen[None]):
    BINDINGS = [Binding("escape", "back", "Back"), Binding("l", "logs", "Logs"), Binding("o", "open_run", "Open run")]

    def __init__(self) -> None:
        super().__init__()
        self.jobs: dict[str, RayJobStatus] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="error", markup=False)
        yield DataTable(cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "RayJobs"
        self.fetch()
        self.set_interval(max(self.app.refresh_every, 5.0), self.fetch)
        self.query_one(DataTable).focus()

    def fetch(self) -> None:
        assert self.app.actions is not None and self.app.actions.rayjobs is not None
        self.run_worker(self.app.actions.rayjobs, group="rayjobs", thread=True, exclusive=True, exit_on_error=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.group != "rayjobs":
            return
        error = self.query_one("#error", Static)
        if event.state == WorkerState.SUCCESS:
            error.update("")
            self._show(event.worker.result)
        elif event.state == WorkerState.ERROR:
            error.update(f"kubectl failed: {_last_line(event.worker.error)}")
            self.notify(f"kubectl failed: {_last_line(event.worker.error)}", severity="error")

    def _show(self, jobs: list[RayJobStatus]) -> None:
        self.jobs = {j.name: j for j in jobs}
        here = {r.run_id for r in self.app.runs()}
        table = self.query_one(DataTable)
        current = self.cursor_job()
        table.clear(columns=True)
        table.add_columns(*RAYJOB_COLUMNS)
        for j in jobs:
            table.add_row(*rayjob_row(j, in_store=j.run_id in here), key=j.name)
        if current in self.jobs:
            table.move_cursor(row=table.get_row_index(current))

    def cursor_job(self) -> str | None:
        table = self.query_one(DataTable)
        if not table.row_count:
            return None
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value in self.jobs:
            self.app.push_screen(RayJobScreen(self.jobs[event.row_key.value]))

    def action_logs(self) -> None:
        if (name := self.cursor_job()) is not None:
            self.app.open_logs(name)

    def action_open_run(self) -> None:
        if (name := self.cursor_job()) is not None:
            self.app.open_run(self.jobs[name].run_id)

    def action_back(self) -> None:
        self.app.pop_screen()


class RayJobScreen(Screen[None]):
    BINDINGS = [Binding("escape", "back", "Back"), Binding("l", "logs", "Logs"), Binding("o", "open_run", "Open run")]

    def __init__(self, job: RayJobStatus) -> None:
        super().__init__()
        self.job = job

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"RayJob {self.job.name}"
        _fill(self.query_one(DataTable), ("FIELD", "VALUE"), rayjob_detail_rows(self.job))

    def action_logs(self) -> None:
        self.app.open_logs(self.job.name)

    def action_open_run(self) -> None:
        self.app.open_run(self.job.run_id)

    def action_back(self) -> None:
        self.app.pop_screen()


class LogsScreen(Screen[None]):
    """The tail of a RayJob's driver log, re-fetched every max(refresh, 5) s."""

    BINDINGS = [Binding("escape", "back", "Back")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.job_name = name
        self.text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(id="log", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"Logs {self.job_name}"
        self.fetch()
        self.set_interval(max(self.app.refresh_every, 5.0), self.fetch)

    def fetch(self) -> None:
        assert self.app.actions is not None and self.app.actions.rayjob_logs is not None
        self.run_worker(
            partial(self.app.actions.rayjob_logs, self.job_name),
            group="logs",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.group != "logs":
            return
        if event.state == WorkerState.SUCCESS:
            self.text = event.worker.result
        elif event.state == WorkerState.ERROR:
            self.text = f"kubectl failed: {_last_line(event.worker.error)}"
        else:
            return
        self.query_one("#log", Static).update(self.text)
        self.query_one(VerticalScroll).scroll_end(animate=False)

    def action_back(self) -> None:
        self.app.pop_screen()


class DiffScreen(Screen[None]):
    """What `angku diff` prints for the two runs."""

    BINDINGS = [Binding("escape", "back", "Back")]

    def __init__(self, run_a: str, run_b: str) -> None:
        super().__init__()
        self.run_a, self.run_b = run_a, run_b
        self.lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(id="diff")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"Diff {self.run_a[:12]} .. {self.run_b[:12]}"
        store = self.app.store
        try:
            self.lines = diff_lines(diff_runs(store, self.run_a, store, self.run_b, atol=self.app.atol))
        except ArtifactError as e:
            self.lines = [f"cannot diff: {e}"]
        self.query_one("#diff", Static).update("\n".join(self.lines))

    def action_back(self) -> None:
        self.app.pop_screen()


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "answer(True)", "Yes"),
        Binding("n", "answer(False)", "No"),
        Binding("escape", "answer(False)", "No"),
    ]

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.question)
            with Horizontal(classes="buttons"):
                yield Button("Yes (y)", id="yes", variant="primary")
                yield Button("No (n)", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, yes: bool) -> None:
        self.dismiss(yes)


class ResultScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close"), Binding("enter", "close", "Close")]

    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self.lines = lines

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            with VerticalScroll():
                yield Static("\n".join(self.lines), markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Close (esc)", id="close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()


class MlopsApp(App[None]):
    TITLE = "angku"
    CSS = """
    #filter { dock: bottom; display: none; }
    #filter.active { display: block; }
    #metrics, #checks { height: auto; margin-bottom: 1; }
    ConfirmScreen, ResultScreen { align: center middle; }
    .dialog {
        width: 90%; max-width: 110; height: auto; max-height: 90%;
        border: thick $accent; padding: 1 2; background: $surface;
    }
    .dialog VerticalScroll { height: auto; max-height: 30; }
    .buttons { height: auto; margin-top: 1; }
    .dialog Button { margin-right: 2; }
    """
    BINDINGS = [Binding("q", "quit", "Quit"), Binding("question_mark", "toggle_help", "Help")]

    def __init__(
        self,
        store: ArtifactStore,
        *,
        label: str,
        refresh: float = 2.0,
        experiment: str | None = None,
        actions: UiActions | None = None,
        atol: float = 0.0,
    ) -> None:
        super().__init__()
        self.store = store
        self.label = label
        self.refresh_every = refresh
        self.experiment = experiment
        self.actions = actions
        self.atol = atol
        self.busy: str | None = None  # "reproduce <run>" while an action runs

    def get_default_screen(self) -> Screen[Any]:
        return ExperimentsScreen()

    def runs(self) -> list[RunSummary]:
        return list_runs(self.store, experiment=self.experiment)

    def action_toggle_help(self) -> None:
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()

    # -- navigation shared by screens ---------------------------------------------

    def open_rayjobs(self) -> None:
        if self.actions is None or self.actions.rayjobs is None:
            self.notify("RayJobs are not available here.", severity="warning")
            return
        self.push_screen(RayJobsScreen())

    def open_logs(self, name: str) -> None:
        if self.actions is None or self.actions.rayjob_logs is None:
            self.notify("RayJob logs are not available here.", severity="warning")
            return
        self.push_screen(LogsScreen(name))

    def open_run(self, run_id: str | None) -> None:
        if run_id is None or run_id not in {r.run_id for r in self.runs()}:
            self.notify(f"run {(run_id or '?')[:12]} is not in this artifacts dir.", severity="warning")
            return
        self.push_screen(RunScreen(run_id))

    # -- actions ----------------------------------------------------------------

    def request(self, kind: str, run_id: str) -> None:
        """Check, ask, then run ``kind`` ("reproduce" | "resume") on ``run_id`` in the background."""
        if self.actions is None:
            self.notify(f"{kind} is not available here.", severity="warning")
            return
        if self.busy is not None:
            self.notify(f"{self.busy} is already running; one action at a time.", severity="warning")
            return
        run = next((r for r in self.runs() if r.run_id == run_id), None)
        if run is None:
            self.notify(f"run {run_id[:12]} is not in the store.", severity="error")
            return
        if kind == "reproduce" and run.status != "FINISHED":
            self.notify(f"{run_id[:12]} is {run.status}: only FINISHED runs can be reproduced.", severity="warning")
            return
        if kind == "resume" and not run.resumable:
            self.notify(f"{run_id[:12]} is {run.status} and not resumable.", severity="warning")
            return
        epochs = run.epochs_total or "?"
        question = (
            f"Reproduce {run.name} {run_id[:12]}? This re-runs its {epochs} epoch(s) on this machine."
            if kind == "reproduce"
            else f"Resume {run.name} {run_id[:12]} from its checkpoint ({run.epochs_done or '?'}/{epochs} epochs)?"
        )

        def answered(yes: bool | None) -> None:
            if yes:
                self._start(kind, run_id, run.name)

        self.push_screen(ConfirmScreen(question), answered)

    def _start(self, kind: str, run_id: str, name: str) -> None:
        assert self.actions is not None
        self.busy = f"{kind} {run_id[:12]}"
        self.notify(f"{self.busy} started.")
        fn = self.actions.reproduce if kind == "reproduce" else self.actions.resume
        self.run_worker(partial(fn, run_id), name=name, group=kind, thread=True, exit_on_error=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        worker = event.worker
        if worker.group not in ("reproduce", "resume"):
            return
        if event.state == WorkerState.SUCCESS:
            self.busy = None
            result = worker.result
            lines = (
                reproduce_lines(worker.name, result)
                if isinstance(result, ReproduceReport)
                else resume_lines(worker.name, result)
            )
            self.push_screen(ResultScreen(lines))
        elif event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            self.notify(f"{self.busy} failed: {worker.error}", severity="error", timeout=30)
            self.busy = None
