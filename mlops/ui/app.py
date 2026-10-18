"""``mlops ui``: a terminal browser over an artifact store. Needs the ``tui`` extra.

Experiments -> Runs -> Run (tabs Overview, Metrics, Environment, Report). Enter drills down,
Esc goes back (or clears the filter first), / filters the list, Tab cycles the run's tabs.
Read-only: it goes through ``application.listing`` / ``report`` and never writes the store;
every screen re-reads it every ``refresh`` seconds and redraws only what changed.
"""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Markdown, TabbedContent, TabPane

from mlops.application.bundle import MANIFEST, SPEC, load_bundle
from mlops.application.listing import RunSummary, group_by_experiment, list_runs
from mlops.application.report import REPORT, render_report
from mlops.domain.experiment import ExperimentSpec
from mlops.domain.manifest import RunManifest
from mlops.ports.artifacts import ArtifactError, ArtifactStore
from mlops.ui.rows import (
    ACTUAL_COLUMNS,
    CHECK_COLUMNS,
    EXPERIMENT_COLUMNS,
    METRIC_COLUMNS,
    environment_rows,
    experiment_row,
    history_rows,
    main_metric,
    metric_rows,
    overview_rows,
    run_columns,
    run_row,
)

Rows = list[tuple[str, tuple[str, ...]]]  # (row key, cells)


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
    def on_mount(self) -> None:
        self.sub_title = f"Experiments  ({self.app.label})"
        super().on_mount()

    def columns(self, runs: list[RunSummary]) -> tuple[str, ...]:
        return EXPERIMENT_COLUMNS

    def rows(self, runs: list[RunSummary]) -> Rows:
        return [(g.experiment_id, experiment_row(g)) for g in group_by_experiment(runs)]

    def open(self, key: str) -> None:
        self.app.push_screen(RunsScreen(key))


class RunsScreen(TableScreen):
    def __init__(self, experiment_id: str) -> None:
        super().__init__()
        self.experiment_id = experiment_id

    def _mine(self, runs: list[RunSummary]) -> list[RunSummary]:
        return [r for r in runs if r.experiment_id == self.experiment_id]

    def on_mount(self) -> None:
        mine = self._mine(self.app.runs())
        name = mine[0].name if mine else self.experiment_id[:12]
        self.sub_title = f"Runs: {name}  {self.experiment_id[:12]}"
        super().on_mount()

    def columns(self, runs: list[RunSummary]) -> tuple[str, ...]:
        return run_columns(main_metric(self._mine(runs)))

    def rows(self, runs: list[RunSummary]) -> Rows:
        mine = self._mine(runs)
        metric = main_metric(mine)
        return [(r.run_id, run_row(r, metric)) for r in mine]

    def open(self, key: str) -> None:
        self.app.push_screen(RunScreen(key))


class RunScreen(Screen[None]):
    TABS = ("tab-overview", "tab-metrics", "tab-environment", "tab-report")
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("tab", "next_tab", "Next tab", priority=True),
        Binding("shift+tab", "prev_tab", "Prev tab", priority=True),
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
        files = sorted(store.list(self.run_id))
        manifest_bytes = store.get(self.run_id, MANIFEST) if MANIFEST in files else None
        if (files, manifest_bytes) == self._loaded:
            return
        self._loaded = (files, manifest_bytes)
        summary = next((r for r in self.app.runs() if r.run_id == self.run_id), None)
        if summary is None:  # gone, or not a run (no spec.json)
            _fill(self.query_one("#overview", DataTable), ("FIELD", "VALUE"),
                  [("run_id", self.run_id), ("status", "not found in the store")])
            return
        spec = manifest = None
        if summary.status != "UNREADABLE":
            spec = ExperimentSpec.model_validate_json(store.get(self.run_id, SPEC))
            manifest = RunManifest.model_validate_json(manifest_bytes) if manifest_bytes else None
        _fill(self.query_one("#overview", DataTable), ("FIELD", "VALUE"),
              overview_rows(summary, spec=spec, manifest=manifest))
        history = manifest.history if manifest is not None else []
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


class MlopsApp(App[None]):
    TITLE = "mlops"
    CSS = """
    #filter { dock: bottom; display: none; }
    #filter.active { display: block; }
    #metrics, #checks { height: auto; margin-bottom: 1; }
    """
    BINDINGS = [Binding("q", "quit", "Quit"), Binding("question_mark", "toggle_help", "Help")]

    def __init__(self, store: ArtifactStore, *, label: str, refresh: float = 2.0,
                 experiment: str | None = None) -> None:
        super().__init__()
        self.store = store
        self.label = label
        self.refresh_every = refresh
        self.experiment = experiment

    def get_default_screen(self) -> Screen[Any]:
        return ExperimentsScreen()

    def runs(self) -> list[RunSummary]:
        return list_runs(self.store, experiment=self.experiment)

    def action_toggle_help(self) -> None:
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()
