from datetime import datetime
from functools import cache

from attr import frozen
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static


REFRESH_INTERVAL = 2

COLUMNS = ("KIND", "ALIAS", "HASH", "OUTPUT", "TAGS")


@frozen
class CatalogRowData:
    kind: str = "expr"
    alias: str = ""
    hash: str = ""
    column_count: int | None = None
    tags: tuple[str, ...] = ()

    @property
    @cache
    def output_display(self) -> str:
        match self.column_count:
            case None:
                return "?"
            case int(n):
                return f"{n} cols"
            case _:
                return "?"

    @property
    @cache
    def tags_display(self) -> str:
        return ", ".join(self.tags) if self.tags else ""

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.kind,
            self.alias,
            self.hash,
            self.output_display,
            self.tags_display,
        )


def snapshot_catalog(catalog):
    alias_lookup = {ca.catalog_entry.name: ca.alias for ca in catalog.catalog_aliases}
    return tuple(
        CatalogRowData(
            kind="expr",
            alias=alias_lookup.get(entry.name, ""),
            hash=entry.name,
            column_count=_safe_column_count(entry),
        )
        for entry in catalog.catalog_entries
    )


def _safe_column_count(entry):
    try:
        return len(entry.expr.columns)
    except Exception:
        return None


class CatalogTUI(App):
    TITLE = "xorq catalog"
    BINDINGS = (
        ("q", "quit", "Quit"),
        ("r", "force_refresh", "Refresh"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("h", "cursor_left", "Left"),
        ("l", "cursor_right", "Right"),
    )
    CSS = """
    #table-container {
        height: 1fr;
    }
    DataTable {
        height: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self, catalog):
        super().__init__()
        self._catalog = catalog

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="table-container"):
            yield DataTable()
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        for col in COLUMNS:
            table.add_column(col, key=col)
        self._do_refresh()
        self.set_interval(REFRESH_INTERVAL, self._do_refresh)

    def _do_refresh(self) -> None:
        table = self.query_one(DataTable)
        cursor_row = table.cursor_row
        rows = snapshot_catalog(self._catalog)
        table.clear()
        for row_data in rows:
            table.add_row(*row_data.row, key=row_data.hash)
        if cursor_row is not None and len(rows) > 0:
            table.move_cursor(row=min(cursor_row, len(rows) - 1))
        stamp = datetime.now().strftime("%H:%M:%S")
        self.query_one("#status-bar", Static).update(
            f" {len(rows)} entries | refreshed {stamp} | every {REFRESH_INTERVAL}s"
        )

    def action_force_refresh(self) -> None:
        self._do_refresh()

    def action_cursor_down(self) -> None:
        self.query_one(DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(DataTable).action_cursor_up()

    def action_cursor_left(self) -> None:
        self.query_one(DataTable).action_scroll_left()

    def action_cursor_right(self) -> None:
        self.query_one(DataTable).action_scroll_right()
