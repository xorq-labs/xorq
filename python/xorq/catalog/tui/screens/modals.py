from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Input, RadioButton, RadioSet, Static, TextArea

from xorq.catalog.tui.models import ComposeConfig, RunConfig, SinkConfig


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ActionModalBase(Screen):
    """Base modal screen with standard cancel/confirm flow.

    Subclasses override compose() for their UI and _build_result()
    to construct the dismissal value.
    """

    BINDINGS = (("escape", "cancel", "Cancel"),)

    def __init__(self, entry_name: str, expr_hash: str):
        super().__init__()
        self._entry_name = entry_name
        self._expr_hash = expr_hash

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Run (existing behavior, refactored to extend ActionModalBase)
# ---------------------------------------------------------------------------


_CACHE_OPTIONS = (
    ("snapshot", "snapshot"),
    ("source", "source"),
    ("none", "no cache"),
    ("ttl_snapshot", "ttl snapshot"),
)


class RunOptionsScreen(ActionModalBase):
    """Lightweight modal for selecting cache strategy. ctrl+r=run, Esc=cancel."""

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+r", "confirm", "Run"),
    )

    def compose(self) -> ComposeResult:
        with Vertical(id="run-options-container"):
            yield Static(
                f" run {self._entry_name}  [dim]ctrl+r=run  esc=cancel[/]",
                id="run-options-title",
            )
            with RadioSet(id="cache-strategy"):
                for i, (_, label) in enumerate(_CACHE_OPTIONS):
                    yield RadioButton(label, value=(i == 0))
            with Horizontal(id="ttl-row"):
                yield Static(" ttl:", id="ttl-label")
                yield Input(placeholder="300", id="ttl-input", restrict=r"^\d*$")

    def on_mount(self) -> None:
        self.query_one("#ttl-row").display = False

    def action_confirm(self) -> None:
        self._do_confirm()

    @on(RadioSet.Changed, "#cache-strategy")
    def _on_cache_strategy_changed(self, event: RadioSet.Changed) -> None:
        selected_idx = event.radio_set.pressed_index
        cache_type = _CACHE_OPTIONS[selected_idx][0]
        self.query_one("#ttl-row").display = cache_type == "ttl_snapshot"

    def _do_confirm(self) -> None:
        radio_set = self.query_one("#cache-strategy", RadioSet)
        selected_idx = radio_set.pressed_index
        cache_type = _CACHE_OPTIONS[selected_idx][0]

        ttl = None
        if cache_type == "ttl_snapshot":
            ttl_text = self.query_one("#ttl-input", Input).value.strip()
            ttl = int(ttl_text) if ttl_text else 300

        self.dismiss(
            RunConfig(
                entry_name=self._entry_name,
                expr_hash=self._expr_hash,
                cache_type=cache_type,
                ttl=ttl,
            )
        )


# ---------------------------------------------------------------------------
# Sink (stub — will be filled when SinkNode is implemented)
# ---------------------------------------------------------------------------


_SINK_STRATEGY_OPTIONS = (
    ("append", "append"),
    ("merge", "merge"),
)


class SinkOptionsScreen(ActionModalBase):
    """Modal for configuring a sink target (Snowflake, Parquet, etc.).

    Stub: shows table name and strategy inputs. Full implementation
    arrives with SinkNode (RFC 0001).
    """

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+r", "confirm", "Sink"),
    )

    def compose(self) -> ComposeResult:
        with Vertical(id="sink-options-container"):
            yield Static(
                f" sink {self._entry_name}  [dim]ctrl+r=run  esc=cancel[/]",
                id="sink-options-title",
            )
            yield Static(" table name:", id="sink-table-label")
            yield Input(placeholder="my_table", id="sink-table-input")
            with RadioSet(id="sink-strategy"):
                for i, (_, label) in enumerate(_SINK_STRATEGY_OPTIONS):
                    yield RadioButton(label, value=(i == 0))
            yield Static(" unique key (comma-sep):", id="sink-key-label")
            yield Input(placeholder="id", id="sink-key-input")
            yield Static(" incremental column:", id="sink-incr-label")
            yield Input(placeholder="", id="sink-incr-input")

    def action_confirm(self) -> None:
        radio_set = self.query_one("#sink-strategy", RadioSet)
        selected_idx = radio_set.pressed_index
        strategy = _SINK_STRATEGY_OPTIONS[selected_idx][0]

        table_name = self.query_one("#sink-table-input", Input).value.strip()
        if not table_name:
            return

        raw_key = self.query_one("#sink-key-input", Input).value.strip()
        unique_key = tuple(k.strip() for k in raw_key.split(",") if k.strip())
        incr_col = self.query_one("#sink-incr-input", Input).value.strip() or None

        self.dismiss(
            SinkConfig(
                entry_name=self._entry_name,
                expr_hash=self._expr_hash,
                target_backend="default",
                table_name=table_name,
                unique_key=unique_key,
                strategy=strategy,
                incremental_column=incr_col,
            )
        )


# ---------------------------------------------------------------------------
# Serve (stub — will be filled when Flight TUI lifecycle is added)
# ---------------------------------------------------------------------------


class ServeOptionsScreen(ActionModalBase):
    """Modal for starting a Flight gRPC server for an entry.

    Stub: shows host/port inputs. Full implementation arrives with
    Flight server lifecycle management.
    """

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+r", "confirm", "Start"),
    )

    def compose(self) -> ComposeResult:
        with Vertical(id="serve-options-container"):
            yield Static(
                f" serve {self._entry_name}  [dim]ctrl+r=start  esc=cancel[/]",
                id="serve-options-title",
            )
            yield Static(" host:", id="serve-host-label")
            yield Input(placeholder="0.0.0.0", id="serve-host-input")
            yield Static(" port:", id="serve-port-label")
            yield Input(placeholder="8815", id="serve-port-input", restrict=r"^\d*$")

    def action_confirm(self) -> None:
        # Stub: dismiss with a dict until a proper ServeConfig model exists
        host = self.query_one("#serve-host-input", Input).value.strip() or "0.0.0.0"
        port_text = self.query_one("#serve-port-input", Input).value.strip()
        port = int(port_text) if port_text else 8815
        self.dismiss(
            {
                "entry_name": self._entry_name,
                "expr_hash": self._expr_hash,
                "host": host,
                "port": port,
            }
        )


# ---------------------------------------------------------------------------
# Action Chooser (when multiple actions are available for an entry)
# ---------------------------------------------------------------------------


class ActionChooserScreen(ActionModalBase):
    """Modal that lets the user pick an action type when multiple are available.

    Dismisses with the chosen action string (e.g., "run", "sink", "serve")
    or None on cancel.
    """

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+r", "confirm", "Select"),
    )

    def __init__(self, entry_name: str, expr_hash: str, actions: tuple[str, ...]):
        super().__init__(entry_name, expr_hash)
        self._actions = actions

    def compose(self) -> ComposeResult:
        with Vertical(id="action-chooser-container"):
            yield Static(
                f" {self._entry_name}  [dim]ctrl+r=select  esc=cancel[/]",
                id="action-chooser-title",
            )
            with RadioSet(id="action-choices"):
                for i, action in enumerate(self._actions):
                    yield RadioButton(action, value=(i == 0))

    def action_confirm(self) -> None:
        radio_set = self.query_one("#action-choices", RadioSet)
        selected_idx = radio_set.pressed_index
        self.dismiss(self._actions[selected_idx])


# ---------------------------------------------------------------------------
# Confirm (generic destructive-action confirmation)
# ---------------------------------------------------------------------------


class ConfirmScreen(Screen):
    """Confirmation modal for destructive actions (delete entry, remove alias).

    Dismisses with True on confirm, None on cancel.
    """

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+d", "confirm", "Confirm"),
    )

    def __init__(self, title: str, message: str):
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-container"):
            yield Static(self._title, id="confirm-title")
            yield Static(self._message, id="confirm-message")
            yield Static(
                " [dim]ctrl+d=confirm  esc=cancel[/]",
                id="confirm-hint",
            )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        self.dismiss(True)


# ---------------------------------------------------------------------------
# Compose (select entries + optional code + alias -> build & catalog)
# ---------------------------------------------------------------------------


class ComposeScreen(Screen):
    """Modal for composing catalog entries with optional inline code.

    Left pane: available catalog entries. Enter appends the highlighted
    entry to the chain (same entry can be added multiple times).

    Right pane top: the composition chain (ordered) with schema validation.
    Backspace removes the last entry. First entry = source, rest = transforms.

    Right pane bottom: code editor + alias input.

    ctrl+r to build & catalog, Esc to cancel.
    """

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+r", "confirm", "Compose"),
        ("backspace", "remove_last", "Remove"),
    )

    def __init__(
        self,
        available_entries: tuple[tuple[str, str, str, dict, dict], ...],
    ):
        """available_entries: (display_name, kind, hash, schema_in, schema_out)."""
        super().__init__()
        self._available = available_entries
        self._schema_map: dict[str, tuple[dict, dict]] = {
            h: (si, so) for _, _, h, si, so in available_entries
        }
        self._chain: list[tuple[str, str, str]] = []  # (name, kind, hash)
        self._chain_valid = False

    def compose(self) -> ComposeResult:
        with Vertical(id="compose-container"):
            yield Static(
                " compose  [dim]Enter=add  Backspace=remove  ctrl+r=build  esc=cancel[/]",
                id="compose-title",
            )
            with Horizontal(id="compose-body"):
                with Vertical(id="compose-left"):
                    yield Static(" catalog entries", id="entries-label")
                    yield DataTable(id="compose-entries-table")
                with Vertical(id="compose-right"):
                    yield Static(" chain", id="chain-label")
                    yield DataTable(id="compose-chain-table")
                    yield Static("", id="compose-validation")
                    yield Static(
                        " code [dim](optional, use `source`)[/]", id="code-label"
                    )
                    yield TextArea(id="compose-code", language="python")
                    with Horizontal(id="compose-alias-row"):
                        yield Static(" alias:", id="alias-label")
                        yield Input(placeholder="optional", id="compose-alias-input")

    def on_mount(self) -> None:
        entries_table = self.query_one("#compose-entries-table", DataTable)
        entries_table.cursor_type = "row"
        entries_table.zebra_stripes = True
        entries_table.add_column("NAME", key="name")
        entries_table.add_column("KIND", key="kind")
        for display_name, kind, entry_hash, _, _ in self._available:
            entries_table.add_row(display_name, kind, key=entry_hash)

        chain_table = self.query_one("#compose-chain-table", DataTable)
        chain_table.cursor_type = "row"
        chain_table.zebra_stripes = True
        chain_table.add_column("", key="status")
        chain_table.add_column("#", key="idx")
        chain_table.add_column("ROLE", key="role")
        chain_table.add_column("NAME", key="name")
        chain_table.add_column("KIND", key="kind")

    @on(DataTable.RowSelected, "#compose-entries-table")
    def _on_entry_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None:
            return
        entry_hash = str(event.row_key.value)
        match next((e for e in self._available if e[2] == entry_hash), None):
            case None:
                return
            case (name, kind, h, _, _):
                self._chain.append((name, kind, h))
        self._render_chain()

    def action_remove_last(self) -> None:
        focused = self.app.focused
        entries_table = self.query_one("#compose-entries-table", DataTable)
        chain_table = self.query_one("#compose-chain-table", DataTable)
        if focused not in (entries_table, chain_table):
            return
        if self._chain:
            self._chain.pop()
            self._render_chain()

    def _validate_chain(self) -> tuple[tuple[str, ...], str]:
        """Validate kind and schema compatibility of the chain.

        Returns (status_icons, message) where status_icons has one
        icon per chain entry: check for valid, X for invalid.

        Rules:
        - Source (position 1): any kind is valid
        - Transforms (position 2+): must be unbound_expr kind
        - Schema: each transform's schema_in must be a subset of the
          previous step's schema_out
        """
        if not self._chain:
            return (), ""

        statuses = ["\u2713"]  # source is always valid
        errors: list[str] = []

        _, _, first_hash = self._chain[0]
        _, current_schema_out = self._schema_map.get(first_hash, ({}, {}))

        for _i, (name, kind, entry_hash) in enumerate(self._chain[1:], start=1):
            schema_in, schema_out = self._schema_map.get(entry_hash, ({}, {}))

            # Transforms must be unbound_expr
            if kind != "unbound_expr":
                statuses.append("\u2717")
                errors.append(
                    f"  {name}: must be unbound_expr to use as transform (got {kind})"
                )
                current_schema_out = schema_out
                continue

            # Schema: current output must be superset of transform's input
            missing = tuple(
                col for col, typ in schema_in.items() if col not in current_schema_out
            )
            mismatched = tuple(
                col
                for col, typ in schema_in.items()
                if col in current_schema_out
                and str(current_schema_out[col]) != str(typ)
            )

            match (missing, mismatched):
                case ((), ()):
                    statuses.append("\u2713")
                case _:
                    statuses.append("\u2717")
                    for col in missing:
                        errors.append(f"  {name}: missing column '{col}'")
                    for col in mismatched:
                        errors.append(
                            f"  {name}: type mismatch on '{col}' "
                            f"({current_schema_out[col]} vs {schema_in[col]})"
                        )

            current_schema_out = schema_out

        self._chain_valid = all(s == "\u2713" for s in statuses)
        message = "\n".join(errors) if errors else ""
        return tuple(statuses), message

    def _render_chain(self) -> None:
        statuses, validation_msg = self._validate_chain()

        table = self.query_one("#compose-chain-table", DataTable)
        table.clear()
        for i, (name, kind, _) in enumerate(self._chain):
            role = "source" if i == 0 else "transform"
            icon = statuses[i] if i < len(statuses) else "?"
            table.add_row(icon, str(i + 1), role, name, kind, key=str(i))

        chain_label = self.query_one("#chain-label", Static)
        validation = self.query_one("#compose-validation", Static)
        match self._chain:
            case []:
                chain_label.update(" chain [dim](empty)[/]")
                validation.update("")
                self._chain_valid = False
            case _:
                names = " \u2192 ".join(n for n, _, _ in self._chain)
                chain_label.update(f" {names}")
                match validation_msg:
                    case "":
                        validation.update(" [green]\u2713 schemas compatible[/]")
                    case msg:
                        validation.update(f" [red]\u2717 schema errors:[/]\n{msg}")

    def action_confirm(self) -> None:
        if not self._chain:
            return
        if not self._chain_valid:
            self.query_one("#compose-validation", Static).update(
                " [red]cannot compose \u2014 fix schema errors first[/]"
            )
            return
        entries = tuple(name for name, _, _ in self._chain)

        code_text = self.query_one("#compose-code", TextArea).text.strip()
        code = code_text if code_text else None

        alias_text = self.query_one("#compose-alias-input", Input).value.strip()
        alias = alias_text if alias_text else None

        self.dismiss(ComposeConfig(entries=entries, code=code, alias=alias))

    def action_cancel(self) -> None:
        self.dismiss(None)
