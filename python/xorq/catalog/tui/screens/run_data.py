from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static


class RunDataScreen(Screen):
    """Full-screen parquet data viewer pushed on top of the catalog screen."""

    BINDINGS = (
        ("q", "go_back", "Back"),
        ("escape", "go_back", "Back"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("h", "scroll_left", "Left"),
        ("l", "scroll_right", "Right"),
    )

    def __init__(self, parquet_path: str, title: str):
        super().__init__()
        self._parquet_path = parquet_path
        self._title = title

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="run-data-container"):
            yield Static("", id="run-data-status")
            yield DataTable(id="run-data-table")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self._title
        table = self.query_one("#run-data-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.loading = True
        self._load_parquet_data()

    @work(thread=True, exit_on_error=False)
    def _load_parquet_data(self) -> None:
        import pyarrow.parquet as pq  # noqa: PLC0415

        try:
            pf = pq.ParquetFile(self._parquet_path)
            total_rows = pf.metadata.num_rows
            file_size = Path(self._parquet_path).stat().st_size
            # Read first 500 rows for preview
            arrow_table = pf.read().slice(0, 500)
            df = arrow_table.to_pandas()
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(str(round(v, 2)) if isinstance(v, float) else str(v) for v in row)
                for row in df.itertuples(index=False)
            )
            self.app.call_from_thread(
                self._render_data, columns, rows, total_rows, len(columns), file_size
            )
        except Exception as e:
            self.app.call_from_thread(self._render_error, str(e))

    def _render_data(self, columns, rows, total_rows, num_cols, file_size) -> None:
        size_mb = file_size / (1024 * 1024)
        self.query_one("#run-data-status", Static).update(
            f" {total_rows:,} rows \u00b7 {num_cols} columns \u00b7 {size_mb:.1f} MB"
        )
        table = self.query_one("#run-data-table", DataTable)
        table.clear(columns=True)
        table.loading = False
        for col in columns:
            table.add_column(col, key=col)
        for i, row in enumerate(rows):
            table.add_row(*row, key=str(i))

    def _render_error(self, message: str) -> None:
        self.query_one("#run-data-status", Static).update(f" Error: {message}")
        self.query_one("#run-data-table", DataTable).loading = False

    def _focused_widget(self) -> DataTable:
        focused = self.app.focused
        if isinstance(focused, DataTable):
            return focused
        return self.query_one("#run-data-table", DataTable)

    def action_cursor_down(self) -> None:
        self._focused_widget().action_cursor_down()

    def action_cursor_up(self) -> None:
        self._focused_widget().action_cursor_up()

    def action_scroll_left(self) -> None:
        self._focused_widget().action_scroll_left()

    def action_scroll_right(self) -> None:
        self._focused_widget().action_scroll_right()

    def action_go_back(self) -> None:
        self.app.pop_screen()
