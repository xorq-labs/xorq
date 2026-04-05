import re
import threading
import zipfile
from datetime import datetime
from functools import cached_property, lru_cache
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import yaml12
from attr import field, frozen
from attr.validators import instance_of, optional
from pygments.style import Style as PygmentsStyle
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)
from rich.syntax import Syntax
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    RichLog,
    Static,
    Tree,
)

from xorq.catalog.catalog import CatalogEntry


DEFAULT_REFRESH_INTERVAL = 10


class XorqSQLStyle(PygmentsStyle):
    """Pygments style using xorq brand colors for SQL syntax highlighting."""

    background_color = "#0a2a2e"
    styles = {
        Token: "#C1F0FF",
        Keyword: "bold #C1F0FF",
        Keyword.DML: "bold #C1F0FF",
        Keyword.DDL: "bold #C1F0FF",
        Name: "#C1F0FF",
        Name.Builtin: "#7ED4C8",
        String: "#2BBE75",
        String.Single: "#2BBE75",
        Number: "#F5CA2C",
        Number.Integer: "#F5CA2C",
        Number.Float: "#F5CA2C",
        Comment: "italic #4AA8EC",
        Comment.Single: "italic #4AA8EC",
        Operator: "#5abfb5",
        Punctuation: "#7ED4C8",
    }


XORQ_DARK = Theme(
    name="xorq-dark",
    primary="#C1F0FF",
    secondary="#4AA8EC",
    warning="#F5CA2C",
    error="#FF4757",
    success="#2BBE75",
    accent="#C1F0FF",
    foreground="#C1F0FF",
    background="#05181A",
    surface="#0a2a2e",
    panel="#0f3338",
    dark=True,
)

COLUMNS = ("KIND", "ALIAS", "HASH")

SCHEMA_PREVIEW_COLUMNS = ("NAME", "TYPE")

REVISION_COLUMNS = ("STATUS", "HASH", "COLUMNS", "CACHED", "DATE")

GIT_LOG_COLUMNS = ("HASH", "DATE", "MESSAGE")


def _format_cached(value: bool | None) -> str:
    match value:
        case True:
            return "●"
        case False:
            return "○"
        case _:
            return "—"


@frozen
class CatalogRowData:
    entry: CatalogEntry = field(repr=False)
    aliases: tuple[str, ...] = field(factory=tuple, validator=instance_of(tuple))

    @cached_property
    def cached(self) -> bool | None:
        parquet_cache_paths = self.entry.parquet_cache_paths
        if parquet_cache_paths:
            return all(Path(p).exists() for p in parquet_cache_paths)
        return None

    @property
    def kind(self) -> str:
        return str(self.entry.kind)

    @property
    def hash(self) -> str:
        return self.entry.name

    @property
    def backends(self) -> tuple[str, ...]:
        return self.entry.backends

    @property
    def schema_in(self) -> tuple[tuple[str, str], ...] | None:
        si = self.entry.metadata.schema_in
        return tuple(si.items()) if si is not None else None

    @property
    def schema_out(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.entry.metadata.schema_out.items())

    @cached_property
    def aliases_display(self) -> str:
        return ", ".join(self.aliases) if self.aliases else ""

    @cached_property
    def backends_display(self) -> str:
        return ", ".join(sorted(set(self.backends))) if self.backends else ""

    @property
    def cached_display(self) -> str:
        return _format_cached(self.cached)

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.aliases_display, self.hash)

    @cached_property
    def sqls(self) -> tuple[tuple[str, str, str], ...]:
        """((name, engine, sql), ...) for all queries in the expression plan."""
        return self.entry.metadata.sql_queries

    @cached_property
    def lineage_dag(self) -> dict | None:
        dag = self.entry.metadata.lineage
        if isinstance(dag, dict) and dag.get("nodes"):
            return dag
        return None

    @cached_property
    def cache_info_text(self) -> str:
        match self.cached:
            case None:
                return "— unknown"
            case True:
                return f"● cached  {self.entry.parquet_cache_paths[0]}"
            case _:
                return "○ uncached"

    @cached_property
    def tags(self) -> tuple[str, ...]:
        """Extract tag labels from the lineage DAG."""
        dag = self.lineage_dag
        if not dag:
            return ()
        return tuple(
            tag_name
            for n in dag.get("nodes", ())
            if n.get("op") in _TAG_OPS
            and isinstance(n.get("tag"), dict)
            and (tag_name := n["tag"].get("tag", ""))
        )

    @cached_property
    def composed_from_hashes(self) -> frozenset[str]:
        return frozenset(
            d["entry_name"]
            for d in self.entry.metadata.composed_from
            if "entry_name" in d
        )

    @property
    def row_key(self) -> str:
        return self.hash


@frozen
class GitLogRowData:
    hash: str = field(default="", validator=instance_of(str))
    date: str = field(default="", validator=instance_of(str))
    message: str = field(default="", validator=instance_of(str))

    @property
    def row(self) -> tuple[str, ...]:
        return (self.hash, self.date, self.message)


@frozen
class RevisionRowData:
    hash: str = field(default="", validator=instance_of(str))
    column_count: int | None = field(default=None, validator=optional(instance_of(int)))
    cached: bool | None = field(default=None, validator=optional(instance_of(bool)))
    commit_date: str = field(default="", validator=instance_of(str))
    is_current: bool = field(default=False, validator=instance_of(bool))

    @cached_property
    def cached_display(self) -> str:
        return _format_cached(self.cached)

    @cached_property
    def status_display(self) -> str:
        return "CURRENT →" if self.is_current else ""

    @cached_property
    def columns_display(self) -> str:
        match self.column_count:
            case None:
                return "?"
            case int(n):
                return f"{n} cols"
            case _:
                return "?"

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.status_display,
            self.hash,
            self.columns_display,
            self.cached_display,
            self.commit_date,
        )


def _entry_info(entry) -> tuple[int | None, bool | None]:
    parquet_cache_paths = entry.parquet_cache_paths
    cached = (
        all(Path(p).exists() for p in parquet_cache_paths)
        if parquet_cache_paths
        else None
    )
    return len(entry.columns), cached


def _load_catalog_row(entry, aliases=()) -> CatalogRowData:
    return CatalogRowData(entry=entry, aliases=aliases)


@lru_cache(maxsize=1)
def _catalog_list_cached(catalog, yaml_mtime: float) -> tuple:
    """Compute catalog entry list; auto-invalidates when yaml mtime changes."""
    return tuple(catalog.list())


def _get_catalog_list(catalog) -> tuple:
    """Return catalog entry list, recomputing only when the YAML file has changed."""
    yaml_mtime = catalog.catalog_yaml.yaml_path.stat().st_mtime
    return _catalog_list_cached(catalog, yaml_mtime)


@lru_cache(maxsize=1)
def _catalog_aliases_cached(catalog, yaml_mtime: float) -> tuple:
    """Compute catalog aliases; auto-invalidates when yaml mtime changes."""
    return tuple(catalog.catalog_aliases)


def _get_catalog_aliases(catalog) -> tuple:
    """Return catalog aliases, recomputing only when the YAML file has changed."""
    yaml_mtime = catalog.catalog_yaml.yaml_path.stat().st_mtime
    return _catalog_aliases_cached(catalog, yaml_mtime)


@lru_cache(maxsize=1)
def _build_alias_multimap(
    catalog_aliases,
) -> dict[str, tuple[str, ...]]:
    from itertools import groupby  # noqa: PLC0415
    from operator import attrgetter  # noqa: PLC0415

    key = attrgetter("catalog_entry.name")
    sorted_aliases = sorted(catalog_aliases, key=key)
    return {
        name: tuple(sorted(ca.alias for ca in group))
        for name, group in groupby(sorted_aliases, key=key)
    }


def _build_git_log_rows(repo, max_count=100) -> tuple[GitLogRowData, ...]:
    return tuple(
        GitLogRowData(
            hash=commit.hexsha[:12],
            date=datetime.fromtimestamp(commit.committed_date).strftime(
                "%Y-%m-%d %H:%M"
            ),
            message=commit.message.strip().split("\n")[0],
        )
        for commit in repo.iter_commits(max_count=max_count)
    )


_TAG_OPS = frozenset({"Tag", "HashingTag"})


def _populate_lineage_tree(tree_widget: Tree, dag: dict) -> None:
    """Populate lineage tree from a lineage DAG, showing only relation nodes.

    Tag and HashingTag nodes are filtered out (they're displayed separately
    from sidecar metadata).
    """
    tree_widget.clear()

    nodes_by_id: dict[str, dict] = {}
    relation_ids: set[str] = set()
    for n in dag.get("nodes", ()):
        nodes_by_id[n["id"]] = n
        if "schema" in n and n.get("op") not in _TAG_OPS:
            relation_ids.add(n["id"])

    inputs_map: dict[str, list[str]] = {}
    for edge in dag.get("edges", ()):
        inputs_map.setdefault(edge["target"], []).append(edge["source"])

    root_id = dag.get("root")
    if not root_id or root_id not in nodes_by_id:
        tree_widget.root.set_label("(empty)")
        return

    def _relation_inputs(node_id: str, seen: set[str]) -> list[str]:
        result: list[str] = []
        for inp in inputs_map.get(node_id, []):
            if inp in seen:
                continue
            seen.add(inp)
            if inp in relation_ids:
                result.append(inp)
            else:
                result.extend(_relation_inputs(inp, seen))
        return result

    def _label(node: dict) -> str:
        op = node.get("op", "?")
        name = node.get("name", "")
        text = node.get("label") or op
        ncols = len(node.get("schema", {}))
        if name and name not in text:
            text = f"{text} ({name})"
        if ncols:
            text = f"{text}  [dim]{ncols} cols[/dim]"
        return text

    def _add_schema_leaves(branch, node: dict) -> None:
        for col_name, col_info in node.get("schema", {}).items():
            dtype = col_info.get("dtype", "?")
            nullable = col_info.get("nullable", True)
            null_tag = "" if nullable else " [dim]NOT NULL[/dim]"
            branch.add_leaf(f"[dim]{col_name}[/dim]  {dtype}{null_tag}")

    visited: set[str] = set()

    def _build(node_id: str, parent) -> None:
        node = nodes_by_id.get(node_id)
        if node is None:
            return
        label = _label(node)
        if node_id in visited:
            parent.add_leaf(f"[dim]↻ {label}[/dim]")
            return
        visited.add(node_id)
        rel_inputs = _relation_inputs(node_id, set())
        branch = parent.add(label, expand=bool(rel_inputs))
        _add_schema_leaves(branch, node)
        for inp in rel_inputs:
            _build(inp, branch)

    # Lineage tree — relation nodes only, no tags.
    if root_id in relation_ids:
        root_node = nodes_by_id[root_id]
        tree_widget.root.set_label(_label(root_node))
        _add_schema_leaves(tree_widget.root, root_node)
        visited.add(root_id)
    else:
        tree_widget.root.set_label("[dim](expression)[/dim]")
    for inp in _relation_inputs(root_id, set()):
        _build(inp, tree_widget.root)
    tree_widget.root.expand()


def _populate_tags_tree(tree_widget: Tree, dag: dict) -> None:
    """Populate a tree showing all Tag/HashingTag nodes grouped by tag name."""
    tree_widget.clear()

    if not dag:
        tree_widget.root.set_label("(no tags)")
        return

    tag_nodes = [
        n
        for n in dag.get("nodes", ())
        if n.get("op") in _TAG_OPS and isinstance(n.get("tag"), dict)
    ]

    if not tag_nodes:
        tree_widget.root.set_label("(no tags)")
        return

    tree_widget.root.set_label(f"Tags ({len(tag_nodes)})")
    for n in tag_nodes:
        tag_meta = n["tag"]
        tag_name = tag_meta.get("tag", n.get("op", "?"))
        op = n.get("op", "Tag")
        ncols = len(n.get("schema", {}))
        label = f"[bold]{tag_name}[/bold]"
        if op == "HashingTag":
            label = f"[#5abfb5]{label}[/]"
        if ncols:
            label = f"{label}  [dim]{ncols} cols[/dim]"
        branch = tree_widget.root.add(label, expand=True)
        for k, v in tag_meta.items():
            if k == "tag":
                continue
            branch.add_leaf(f"[dim]{k}:[/dim] {v}")
    tree_widget.root.expand()


def _populate_full_lineage_tree(tree_widget: Tree, dag: dict) -> None:
    """Populate lineage tree showing ALL nodes including Tag/HashingTag."""
    tree_widget.clear()

    nodes_by_id: dict[str, dict] = {}
    schema_ids: set[str] = set()
    for n in dag.get("nodes", ()):
        nodes_by_id[n["id"]] = n
        if "schema" in n:
            schema_ids.add(n["id"])

    inputs_map: dict[str, list[str]] = {}
    for edge in dag.get("edges", ()):
        inputs_map.setdefault(edge["target"], []).append(edge["source"])

    root_id = dag.get("root")
    if not root_id or root_id not in nodes_by_id:
        tree_widget.root.set_label("(empty)")
        return

    def _label(node: dict) -> str:
        op = node.get("op", "?")
        name = node.get("name", "")
        ncols = len(node.get("schema", {}))

        if op in _TAG_OPS and isinstance(node.get("tag"), dict):
            tag_name = node["tag"].get("tag", "")
            alias = node["tag"].get("alias", "")
            text = f"[#5abfb5]{op}[/] ({tag_name}"
            if alias:
                text = f"{text}: {alias}"
            text = f"{text})"
        else:
            text = node.get("label") or op
            if name and name not in text:
                text = f"{text} ({name})"

        if ncols:
            text = f"{text}  [dim]{ncols} cols[/dim]"
        return text

    def _schema_inputs(node_id: str, seen: set[str]) -> list[str]:
        result: list[str] = []
        for inp in inputs_map.get(node_id, []):
            if inp in seen:
                continue
            seen.add(inp)
            if inp in schema_ids:
                result.append(inp)
            else:
                result.extend(_schema_inputs(inp, seen))
        return result

    visited: set[str] = set()

    def _build(node_id: str, parent) -> None:
        node = nodes_by_id.get(node_id)
        if node is None:
            return
        label = _label(node)
        if node_id in visited:
            parent.add_leaf(f"[dim]\u21bb {label}[/dim]")
            return
        visited.add(node_id)
        children = _schema_inputs(node_id, set())
        branch = parent.add(label, expand=bool(children))
        # Add schema leaves for non-tag nodes
        if node.get("op") not in _TAG_OPS:
            for col_name, col_info in node.get("schema", {}).items():
                dtype = col_info.get("dtype", "?")
                nullable = col_info.get("nullable", True)
                null_tag = "" if nullable else " [dim]NOT NULL[/dim]"
                branch.add_leaf(f"[dim]{col_name}[/dim]  {dtype}{null_tag}")
        # Add tag metadata as leaves for tag nodes
        elif isinstance(node.get("tag"), dict):
            for k, v in node["tag"].items():
                if k == "tag":
                    continue
                branch.add_leaf(f"[dim]{k}:[/dim] {v}")
        for inp in children:
            _build(inp, branch)

    if root_id in schema_ids:
        root_node = nodes_by_id[root_id]
        tree_widget.root.set_label(_label(root_node))
        visited.add(root_id)
    else:
        tree_widget.root.set_label("[dim](expression)[/dim]")
    for inp in _schema_inputs(root_id, set()):
        _build(inp, tree_widget.root)
    tree_widget.root.expand()


def _render_sql_dag(sqls: tuple[tuple[str, str, str], ...]) -> str:
    """Render multiple SQL queries as a topologically-sorted DAG."""
    name_to_sql = {name: (engine, sql) for name, engine, sql in sqls}
    # build dependency graph: name → set of names it depends on
    deps = {
        name: frozenset(
            ref
            for ref in re.findall(r'FROM "([a-f0-9]{20,})"', sql)
            if ref in name_to_sql
        )
        for name, (_, sql) in name_to_sql.items()
    }
    try:
        order = list(TopologicalSorter(deps).static_order())
    except CycleError:
        order = list(name_to_sql)

    parts = tuple(
        segment
        for i, name in enumerate(order)
        for engine, sql in (name_to_sql[name],)
        for label in (("main" if name == "main" else name[:12]),)
        for segment in (
            (f"-- [{label}] ({engine})\n{sql}", "  ↓")
            if i < len(order) - 1
            else (f"-- [{label}] ({engine})\n{sql}",)
        )
    )
    return "\n\n".join(parts)


def _revision_pair(i, rev_entry, commit):
    exists = rev_entry.exists()
    col_count, cached = _entry_info(rev_entry) if exists else (None, None)
    row = RevisionRowData(
        hash=rev_entry.name,
        column_count=col_count,
        cached=cached,
        commit_date=datetime.fromtimestamp(commit.committed_date).strftime(
            "%Y-%m-%d %H:%M"
        ),
        is_current=(i == 0),
    )
    return row, (rev_entry, commit, exists)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


@frozen
class _TogglePanelState:
    visible: bool = field(default=False, validator=instance_of(bool))
    loaded: bool = field(default=False, validator=instance_of(bool))
    entry_hash: str | None = field(default=None, validator=optional(instance_of(str)))


def _is_bindable(
    source_schema_out: tuple[tuple[str, str], ...],
    transform_schema_in: tuple[tuple[str, str], ...] | None,
) -> bool:
    """Check if transform's schema_in is a subset of source's schema_out."""
    if transform_schema_in is None:
        return False
    source_cols = dict(source_schema_out)
    return all(source_cols.get(col) == dtype for col, dtype in transform_schema_in)


class ComposeScreen(ModalScreen):
    """Chain-building compose modal: Enter adds entries, Backspace removes, ctrl+r builds.

    Left pane: all catalog entries. Enter appends the highlighted entry to
    the chain. First entry = source, rest = transforms.

    Right pane: the ordered chain with schema validation (checkmarks / X).
    Backspace removes the last entry.

    ctrl+r to build & catalog, Escape to cancel.
    """

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+r", "confirm", "Compose"),
        ("backspace", "remove_last", "Remove"),
        ("j", "cursor_down", "\u2193"),
        ("k", "cursor_up", "\u2191"),
    )

    def __init__(
        self,
        row_cache: dict[str, CatalogRowData],
        initial_source: CatalogRowData | None = None,
    ) -> None:
        super().__init__()
        self._row_cache = row_cache
        self._initial_source = initial_source
        self._chain: list[CatalogRowData] = []
        self._chain_valid = False

    def compose(self) -> ComposeResult:
        with Vertical(id="compose-dialog"):
            yield Static(
                " compose  [dim]Enter=add  Backspace=remove"
                "  ctrl+r=build  esc=cancel[/]",
                id="compose-title",
            )
            with Horizontal(id="compose-split"):
                with Vertical(id="compose-source-panel"):
                    yield DataTable(id="compose-entries-table")
                with Vertical(id="compose-transform-panel"):
                    yield DataTable(id="compose-chain-table")
                    yield Static("", id="compose-validation")
            yield Static("", id="compose-status")

    def on_mount(self) -> None:
        entries_table = self.query_one("#compose-entries-table", DataTable)
        entries_table.cursor_type = "row"
        entries_table.zebra_stripes = True
        entries_table.add_column("NAME", key="name")
        entries_table.add_column("KIND", key="kind")

        for rd in sorted(self._row_cache.values(), key=lambda r: r.sort_key):
            entries_table.add_row(
                rd.aliases_display or rd.hash[:12],
                rd.kind,
                key=rd.row_key,
            )

        chain_table = self.query_one("#compose-chain-table", DataTable)
        chain_table.cursor_type = "row"
        chain_table.zebra_stripes = True
        chain_table.add_column("", key="status")
        chain_table.add_column("#", key="idx")
        chain_table.add_column("ROLE", key="role")
        chain_table.add_column("NAME", key="name")

        self.query_one("#compose-source-panel").border_title = "Catalog Entries"
        self.query_one("#compose-transform-panel").border_title = "Chain"

        if self._initial_source is not None:
            self._chain.append(self._initial_source)
            self._render_chain()

        self._update_status()
        entries_table.focus()

    # --- Add / Remove entries ---

    @on(DataTable.RowSelected, "#compose-entries-table")
    def _on_entry_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None:
            return
        rd = self._row_cache.get(str(event.row_key.value))
        if rd is not None:
            self._chain.append(rd)
            self._render_chain()

    def action_remove_last(self) -> None:
        if self._chain:
            self._chain.pop()
            self._render_chain()

    # --- Validation ---

    def _validate_chain(self) -> tuple[tuple[str, ...], str]:
        if not self._chain:
            return (), ""

        statuses = ["\u2713"]
        errors: list[str] = []
        current_schema = dict(self._chain[0].schema_out)

        for _i, rd in enumerate(self._chain[1:], start=1):
            if rd.schema_in is None:
                statuses.append("\u2717")
                errors.append(
                    f"  {rd.aliases_display or rd.hash[:12]}: not a transform"
                )
                current_schema = dict(rd.schema_out)
                continue

            missing = [c for c, t in rd.schema_in if c not in current_schema]
            mismatched = [
                c
                for c, t in rd.schema_in
                if c in current_schema and current_schema[c] != t
            ]

            if missing or mismatched:
                statuses.append("\u2717")
                for c in missing:
                    errors.append(
                        f"  {rd.aliases_display or rd.hash[:12]}: missing '{c}'"
                    )
                for c in mismatched:
                    errors.append(
                        f"  {rd.aliases_display or rd.hash[:12]}: type mismatch '{c}'"
                    )
            else:
                statuses.append("\u2713")

            current_schema = dict(rd.schema_out)

        self._chain_valid = all(s == "\u2713" for s in statuses)
        return tuple(statuses), "\n".join(errors)

    def _render_chain(self) -> None:
        statuses, validation_msg = self._validate_chain()

        chain_table = self.query_one("#compose-chain-table", DataTable)
        chain_table.clear()
        for i, rd in enumerate(self._chain):
            role = "source" if i == 0 else "transform"
            icon = statuses[i] if i < len(statuses) else "?"
            chain_table.add_row(
                icon,
                str(i + 1),
                role,
                rd.aliases_display or rd.hash[:12],
                key=str(i),
            )

        validation = self.query_one("#compose-validation", Static)
        if not self._chain:
            validation.update(" [dim](empty chain)[/]")
            self._chain_valid = False
        elif validation_msg:
            validation.update(f" [red]\u2717 schema errors:[/]\n{validation_msg}")
        else:
            names = " \u2192 ".join(
                rd.aliases_display or rd.hash[:12] for rd in self._chain
            )
            validation.update(f" [green]\u2713[/] {names}")

        self._update_status()

    def _update_status(self) -> None:
        n = len(self._chain)
        label = "empty" if n == 0 else f"{n} step{'s' if n != 1 else ''}"
        self.query_one("#compose-status", Static).update(f" Chain: {label}")

    # --- Navigation ---

    def _focused_table(self) -> DataTable | None:
        focused = self.app.focused
        return focused if isinstance(focused, DataTable) else None

    def action_cursor_down(self) -> None:
        t = self._focused_table()
        if t:
            t.action_cursor_down()

    def action_cursor_up(self) -> None:
        t = self._focused_table()
        if t:
            t.action_cursor_up()

    def action_cancel(self) -> None:
        self.dismiss(None)

    # --- Confirm ---

    def action_confirm(self) -> None:
        if not self._chain:
            return
        if not self._chain_valid:
            self.query_one("#compose-validation", Static).update(
                " [red]fix schema errors before composing[/]"
            )
            return
        entry_names = tuple(rd.aliases_display or rd.hash for rd in self._chain)
        self.dismiss(entry_names)


class DataViewScreen(Screen):
    """VisiData + Vim inspired spreadsheet view for a catalog entry."""

    BINDINGS = (
        ("q", "go_back", "Back"),
        ("h", "cursor_left", "\u2190"),
        ("j", "cursor_down", "\u2193"),
        ("k", "cursor_up", "\u2191"),
        ("l", "cursor_right", "\u2192"),
        ("g", "go_top", "Top"),
        ("G", "go_bottom", "Bottom"),
        ("s", "toggle_stats", "Stats"),
        ("c", "open_compose", "Compose"),
        ("left_square_bracket", "sort_asc", "Sort \u2191"),
        ("right_square_bracket", "sort_desc", "Sort \u2193"),
        ("colon", "command_mode", "Cmd"),
        ("slash", "search_mode", "Search"),
    )

    def __init__(
        self,
        row_data: CatalogRowData,
        composed_expr=None,
        label: str | None = None,
    ) -> None:
        super().__init__()
        self._row_data = row_data
        self._entry = row_data.entry
        self._current_expr = composed_expr
        self._current_label = label or row_data.aliases_display or row_data.hash[:12]
        self._tracking_alias: str | None = (
            row_data.aliases[0] if row_data.aliases else None
        )
        self._head_limit = 50
        self._sort_column: str | None = None
        self._sort_ascending = True
        self._stats_visible = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="dv-data-panel"):
            yield DataTable(id="dv-data-table")
        with Vertical(id="dv-stats-panel"):
            yield DataTable(id="dv-stats-table")
        yield Input(placeholder=":", id="dv-command-input", disabled=True)
        yield Static("", id="dv-status-bar")
        yield Footer()

    def on_mount(self) -> None:
        data_table = self.query_one("#dv-data-table", DataTable)
        data_table.cursor_type = "cell"
        data_table.zebra_stripes = True
        data_table.loading = True

        stats_table = self.query_one("#dv-stats-table", DataTable)
        stats_table.cursor_type = "row"
        stats_table.zebra_stripes = True

        self.query_one(
            "#dv-data-panel"
        ).border_title = f"Data \u2014 {self._current_label}"
        stats_panel = self.query_one("#dv-stats-panel")
        stats_panel.border_title = "Summary Statistics"
        stats_panel.display = False

        self._update_status("Loading\u2026")
        self._load_initial_data()

    def _update_status(self, text: str) -> None:
        self.query_one("#dv-status-bar", Static).update(f" {text}")

    def _update_status_detail(self, row_count: int, col_count: int) -> None:
        sort_info = ""
        if self._sort_column:
            direction = "ASC" if self._sort_ascending else "DESC"
            sort_info = f" \u00b7 sort: {self._sort_column} {direction}"
        self._update_status(
            f"{self._current_label} \u00b7 {row_count} rows \u00d7 {col_count} cols"
            f" \u00b7 head({self._head_limit}){sort_info}"
        )

    # --- Data Loading ---

    @work(thread=True, exit_on_error=False)
    def _load_initial_data(self) -> None:
        try:
            if self._current_expr is None:
                self._current_expr = self._entry.expr
            self._execute_and_render(self._current_expr)
        except Exception as e:
            self.app.call_from_thread(self._render_error, str(e))

    def _execute_and_render(self, expr) -> None:
        """Execute expr.head() and send results to the main thread for rendering."""
        if self._sort_column:
            from xorq.vendor import ibis  # noqa: PLC0415

            sort_key = (
                ibis.asc(self._sort_column)
                if self._sort_ascending
                else ibis.desc(self._sort_column)
            )
            expr = expr.order_by(sort_key)
        df = expr.head(self._head_limit).execute()
        columns = tuple(str(c) for c in df.columns)
        rows = tuple(
            tuple(str(round(v, 4)) if isinstance(v, float) else str(v) for v in row)
            for row in df.itertuples(index=False)
        )
        self.app.call_from_thread(self._render_data, columns, rows, len(df))

    def _render_data(self, columns: tuple, rows: tuple, total_rows: int) -> None:
        with self.app.batch_update():
            data_table = self.query_one("#dv-data-table", DataTable)
            data_table.clear(columns=True)
            data_table.loading = False
            for col in columns:
                data_table.add_column(col, key=col)
            for i, row in enumerate(rows):
                data_table.add_row(*row, key=str(i))
            self._update_status_detail(total_rows, len(columns))
            self.query_one(
                "#dv-data-panel"
            ).border_subtitle = f"{total_rows} rows \u00d7 {len(columns)} cols"

    def _render_error(self, message: str) -> None:
        self.query_one("#dv-data-table", DataTable).loading = False
        self._update_status(f"Error: {message}")

    # --- Navigation ---

    def action_cursor_left(self) -> None:
        self.query_one("#dv-data-table", DataTable).action_cursor_left()

    def action_cursor_down(self) -> None:
        self.query_one("#dv-data-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#dv-data-table", DataTable).action_cursor_up()

    def action_cursor_right(self) -> None:
        self.query_one("#dv-data-table", DataTable).action_cursor_right()

    def action_go_top(self) -> None:
        dt = self.query_one("#dv-data-table", DataTable)
        if dt.row_count:
            dt.move_cursor(row=0)

    def action_go_bottom(self) -> None:
        dt = self.query_one("#dv-data-table", DataTable)
        if dt.row_count:
            dt.move_cursor(row=dt.row_count - 1)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def key_escape(self) -> None:
        cmd_input = self.query_one("#dv-command-input", Input)
        if not cmd_input.disabled and cmd_input.has_focus:
            self._dismiss_command()
            return
        self.action_go_back()

    # --- Sort ---

    def _get_current_column_name(self) -> str | None:
        dt = self.query_one("#dv-data-table", DataTable)
        if not dt.columns:
            return None
        col_keys = list(dt.columns.keys())
        idx = dt.cursor_column
        if 0 <= idx < len(col_keys):
            return str(col_keys[idx])
        return None

    def action_sort_asc(self) -> None:
        col = self._get_current_column_name()
        if col and self._current_expr is not None:
            self._sort_column = col
            self._sort_ascending = True
            self._reload_data()

    def action_sort_desc(self) -> None:
        col = self._get_current_column_name()
        if col and self._current_expr is not None:
            self._sort_column = col
            self._sort_ascending = False
            self._reload_data()

    def _reload_data(self) -> None:
        self.query_one("#dv-data-table", DataTable).loading = True
        self._update_status("Loading\u2026")
        self._do_reload()

    @work(thread=True, exit_on_error=False)
    def _do_reload(self) -> None:
        try:
            self._execute_and_render(self._current_expr)
        except Exception as e:
            self.app.call_from_thread(self._render_error, str(e))

    # --- Stats ---

    def action_toggle_stats(self) -> None:
        self._stats_visible = not self._stats_visible
        self.query_one("#dv-stats-panel").display = self._stats_visible
        if self._stats_visible and self._current_expr is not None:
            self._load_describe()

    @work(thread=True, exit_on_error=False)
    def _load_describe(self) -> None:
        try:
            df = self._current_expr.describe().execute()
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(str(round(v, 4)) if isinstance(v, float) else str(v) for v in row)
                for row in df.itertuples(index=False)
            )
            self.app.call_from_thread(self._render_stats, columns, rows)
        except Exception as e:
            self.app.call_from_thread(self._render_stats_error, str(e))

    def _render_stats(self, columns: tuple, rows: tuple) -> None:
        with self.app.batch_update():
            stats_table = self.query_one("#dv-stats-table", DataTable)
            stats_table.clear(columns=True)
            for col in columns:
                stats_table.add_column(col, key=col)
            for i, row in enumerate(rows):
                stats_table.add_row(*row, key=str(i))
            self.query_one("#dv-stats-panel").border_subtitle = f"{len(rows)} columns"

    def _render_stats_error(self, message: str) -> None:
        self.query_one("#dv-stats-panel").border_subtitle = f"Error: {message}"

    # --- Command Mode ---

    def action_command_mode(self) -> None:
        cmd_input = self.query_one("#dv-command-input", Input)
        cmd_input.disabled = False
        cmd_input.display = True
        cmd_input.value = ""
        cmd_input.focus()

    def action_search_mode(self) -> None:
        cmd_input = self.query_one("#dv-command-input", Input)
        cmd_input.disabled = False
        cmd_input.display = True
        cmd_input.value = "/"
        cmd_input.focus()

    def _dismiss_command(self) -> None:
        cmd_input = self.query_one("#dv-command-input", Input)
        cmd_input.display = False
        cmd_input.disabled = True
        self.query_one("#dv-data-table", DataTable).focus()

    @on(Input.Submitted, "#dv-command-input")
    def _on_command_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self._dismiss_command()
        if text.startswith("/"):
            self._do_search(text[1:])
        elif text:
            self._execute_command(text)

    def _execute_command(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        match cmd:
            case "compose":
                self.action_open_compose()
            case "code":
                self._cmd_code(arg.strip())
            case "head":
                self._cmd_head(arg.strip())
            case "describe":
                self._cmd_describe()
            case "q" | "quit":
                self.action_go_back()
            case _:
                self._update_status(f"Unknown command: {cmd}")

    # --- Compose ---

    def action_open_compose(self) -> None:
        row_cache: dict[str, CatalogRowData] = {}
        for screen in self.app.screen_stack:
            if isinstance(screen, CatalogScreen):
                row_cache = screen._row_cache
                break
        if not row_cache:
            self._update_status("No catalog entries loaded")
            return
        self.app.push_screen(
            ComposeScreen(row_cache, initial_source=self._row_data),
            callback=self._on_compose_dismissed,
        )

    def _on_compose_dismissed(self, result: tuple[str, ...] | None) -> None:
        if result is None:
            return
        self._update_status("Composing\u2026")
        self.query_one("#dv-data-table", DataTable).loading = True
        self._do_compose(result)

    @work(thread=True, exit_on_error=False)
    def _do_compose(self, entry_names: tuple[str, ...]) -> None:
        from xorq.catalog.composer import ExprComposer  # noqa: PLC0415
        from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

        catalog = self.app._catalog
        if catalog is None:
            self.app.call_from_thread(self._render_error, "No catalog")
            return
        try:
            resolved = tuple(
                catalog.get_catalog_entry(name, maybe_alias=True)
                for name in entry_names
            )
            new_expr = ExprComposer(
                source=resolved[0],
                transforms=resolved[1:],
            ).expr

            # Build and catalog
            build_path = build_expr(new_expr)
            aliases = (self._tracking_alias,) if self._tracking_alias else ()
            catalog.add(build_path, aliases=aliases, exist_ok=True)

            # Refresh state
            _catalog_list_cached.cache_clear()
            entry_name = build_path.name
            new_entry = catalog.get_catalog_entry(entry_name)
            self._entry = new_entry
            self._current_expr = new_expr
            self._sort_column = None
            label = " \u2192 ".join(entry_names)
            self._current_label = label

            self._execute_and_render(new_expr)
            self.app.call_from_thread(self._update_panel_title, label)
        except Exception as e:
            self.app.call_from_thread(self._render_error, f"Compose failed: {e}")

    def _update_panel_title(self, label: str) -> None:
        self.query_one("#dv-data-panel").border_title = f"Data \u2014 {label}"

    # --- :code ---

    def _cmd_code(self, code: str) -> None:
        if not code:
            self._update_status("Usage: code <ibis_expression>")
            return
        self.query_one("#dv-data-table", DataTable).loading = True
        self._update_status("Evaluating\u2026")
        self._do_code(code)

    @work(thread=True, exit_on_error=False)
    def _do_code(self, code: str) -> None:
        try:
            from xorq.catalog.bind import _eval_code  # noqa: PLC0415

            if self._current_expr is None:
                self.app.call_from_thread(self._render_error, "No expression loaded")
                return
            new_expr = _eval_code(code, self._current_expr)
            self._current_expr = new_expr
            self._sort_column = None
            new_label = f"{self._current_label} | code"
            self._current_label = new_label
            self._execute_and_render(new_expr)
            self.app.call_from_thread(self._update_panel_title, new_label)
        except Exception as e:
            self.app.call_from_thread(self._render_error, f"Code error: {e}")

    # --- :head ---

    def _cmd_head(self, n_str: str) -> None:
        try:
            n = int(n_str) if n_str else 50
            if n < 1:
                raise ValueError
        except ValueError:
            self._update_status("Usage: head <positive_integer>")
            return
        self._head_limit = n
        self._reload_data()

    # --- :describe ---

    def _cmd_describe(self) -> None:
        self._stats_visible = True
        self.query_one("#dv-stats-panel").display = True
        self._load_describe()

    # --- /search ---

    def _do_search(self, query: str) -> None:
        if not query:
            return
        dt = self.query_one("#dv-data-table", DataTable)
        query_lower = query.lower()
        found = 0
        first_found = False
        for row_idx in range(dt.row_count):
            for col_idx in range(len(dt.columns)):
                cell_value = str(dt.get_cell_at((row_idx, col_idx)))
                if query_lower in cell_value.lower():
                    if not first_found:
                        dt.move_cursor(row=row_idx, column=col_idx)
                        first_found = True
                    found += 1
        suffix = "es" if found != 1 else ""
        self._update_status(f"/{query} \u2014 {found} match{suffix}")


class CatalogScreen(Screen):
    BINDINGS = (
        ("q", "quit_app", "Quit"),
        ("ctrl+c", "quit_app", "Quit"),
        ("h", "scroll_left", "Left"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "scroll_right", "Right"),
        ("tab", "focus_next_panel", "Next"),
        ("shift+tab", "focus_prev_panel", "Prev"),
        ("v", "toggle_revisions", "Revisions"),
        ("g", "toggle_git_log", "Git Log"),
        ("1", "show_view('sql')", "SQL"),
        ("2", "show_view('lineage')", "Lineage"),
        ("3", "show_view('data')", "Data"),
        ("4", "show_view('profiles')", "Profiles"),
        ("5", "show_view('tags')", "Tags"),
        ("6", "show_view('full_lineage')", "Full Lineage"),
        ("e", "open_data_view", "Data View"),
        ("slash", "start_search", "Search"),
    )

    VIEW_PANELS = {
        "lineage": "#lineage-panel",
        "sql": "#sql-panel",
        "data": "#data-preview-panel",
        "profiles": "#profiles-panel",
        "tags": "#tags-panel",
        "full_lineage": "#full-lineage-panel",
    }

    FOCUS_CYCLE = (
        "#catalog-tree",
        "#lineage-tree",
        "#schema-in-table",
        "#schema-preview-table",
        "#revisions-preview-table",
    )

    def __init__(self, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._refresh_interval = refresh_interval
        self._row_cache: dict[str, CatalogRowData] = {}
        self._kind_nodes: dict[str, object] = {}
        self._node_by_hash: dict[str, object] = {}
        self._highlighted_related: set[str] = set()
        self._last_highlighted_hash: str | None = None
        self._revisions_visible = False
        self._git_log_visible = False
        self._git_log_loaded = False
        self._refresh_lock = threading.Lock()
        self._active_view = "sql"
        self._data_preview = _TogglePanelState()
        self._profiles_state = _TogglePanelState()
        self._search_query = ""
        self._sql_cache: dict[str, object] = {}
        self._sql_rendering_hash: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-split"):
            with Vertical(id="left-column"):
                with Vertical(id="catalog-panel"):
                    yield Input(
                        placeholder="search...", id="search-input", disabled=True
                    )
                    yield Tree("Expressions", id="catalog-tree")
                with Vertical(id="revisions-panel"):
                    yield DataTable(id="revisions-preview-table")
                with Vertical(id="git-log-panel"):
                    yield DataTable(id="git-log-table")
            with Vertical(id="right-column"):
                with Vertical(id="detail-view"):
                    with Vertical(id="lineage-panel"):
                        yield Tree("Lineage", id="lineage-tree")
                    with Vertical(id="sql-panel"):
                        yield RichLog(id="sql-preview", wrap=True, auto_scroll=False)
                    with Vertical(id="data-preview-panel"):
                        yield Static("", id="data-preview-status")
                        yield DataTable(id="data-preview-table")
                    with Vertical(id="profiles-panel"):
                        yield DataTable(id="profiles-table")
                    with Vertical(id="tags-panel"):
                        yield Tree("Tags", id="tags-tree")
                    with Vertical(id="full-lineage-panel"):
                        yield Tree("Full Lineage", id="full-lineage-tree")
                with Vertical(id="schema-panel"):
                    with Horizontal(id="schema-split"):
                        with Vertical(id="schema-in-half"):
                            yield DataTable(id="schema-in-table")
                        with Vertical(id="schema-out-half"):
                            yield DataTable(id="schema-preview-table")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#catalog-tree", Tree)
        tree.show_root = False
        tree.guide_depth = 3

        schema_in_table = self.query_one("#schema-in-table", DataTable)
        schema_in_table.cursor_type = "row"
        schema_in_table.zebra_stripes = True
        for col in SCHEMA_PREVIEW_COLUMNS:
            schema_in_table.add_column(col, key=col)

        schema_table = self.query_one("#schema-preview-table", DataTable)
        schema_table.cursor_type = "row"
        schema_table.zebra_stripes = True
        for col in SCHEMA_PREVIEW_COLUMNS:
            schema_table.add_column(col, key=col)

        rev_table = self.query_one("#revisions-preview-table", DataTable)
        rev_table.cursor_type = "row"
        rev_table.zebra_stripes = True
        for col in REVISION_COLUMNS:
            rev_table.add_column(col, key=col)

        git_log_table = self.query_one("#git-log-table", DataTable)
        git_log_table.cursor_type = "row"
        git_log_table.zebra_stripes = True
        for col in GIT_LOG_COLUMNS:
            git_log_table.add_column(col, key=col)

        data_table = self.query_one("#data-preview-table", DataTable)
        data_table.cursor_type = "none"
        data_table.zebra_stripes = True
        data_table.loading = True

        info_table = self.query_one("#profiles-table", DataTable)
        info_table.cursor_type = "row"
        info_table.zebra_stripes = True
        info_table.add_column("NAME", key="name")
        info_table.add_column("BACKEND", key="backend")
        info_table.add_column("PARAMETERS", key="params")
        info_table.add_column("ENV VARS", key="env_vars")

        self.query_one("#catalog-panel").border_title = "Expressions"
        self.query_one("#schema-panel").border_title = "Schema"
        self.query_one("#schema-in-half").display = False
        self.query_one("#lineage-panel").border_title = "Lineage"
        self.query_one("#sql-panel").border_title = "SQL"
        data_panel = self.query_one("#data-preview-panel")
        data_panel.border_title = "Data Preview"
        self.query_one("#tags-panel").border_title = "Tags"
        self.query_one("#full-lineage-panel").border_title = "Full Lineage"
        self._apply_view("sql")
        rev_panel = self.query_one("#revisions-panel")
        rev_panel.border_title = "Revisions"
        rev_panel.display = False
        git_log_panel = self.query_one("#git-log-panel")
        git_log_panel.border_title = "Git Log"
        git_log_panel.display = False
        self.query_one("#profiles-panel").border_title = "Profiles"
        self.query_one("#status-bar", Static).update(" Loading catalog...")

        self.set_interval(self._refresh_interval, self._do_refresh)

    def _get_entry_hash_from_node(self, node) -> str | None:
        """Walk up from node to find an entry hash stored as node data."""
        current = node
        while current is not None:
            if isinstance(current.data, str):
                return current.data
            current = current.parent
        return None

    def _selected_entry_hash(self) -> str | None:
        """Return the entry hash of the currently highlighted catalog tree node."""
        tree = self.query_one("#catalog-tree", Tree)
        node = tree.cursor_node
        if node is None:
            return None
        return self._get_entry_hash_from_node(node)

    @staticmethod
    def _entry_label(row_data: CatalogRowData) -> str:
        return row_data.aliases_display or f"[dim]{row_data.hash[:12]}[/dim]"

    @staticmethod
    def _entry_label_related(row_data: CatalogRowData) -> str:
        name = row_data.aliases_display or row_data.hash[:12]
        return f"[#5abfb5]› {name}[/]"

    def _compute_related_hashes(self, entry_hash: str) -> set[str]:
        """Find entries related to entry_hash (both upstream and downstream)."""
        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            return set()
        # Upstream: entries this one is composed from
        related = set(row_data.composed_from_hashes & self._row_cache.keys())
        # Downstream: entries that are composed from this one
        for other_hash, other_data in self._row_cache.items():
            if (
                other_hash != entry_hash
                and entry_hash in other_data.composed_from_hashes
            ):
                related.add(other_hash)
        return related

    def _update_related_highlights(self, entry_hash: str) -> None:
        """Highlight tree nodes related to the selected entry."""
        # Clear previous highlights
        for h in self._highlighted_related:
            node = self._node_by_hash.get(h)
            rd = self._row_cache.get(h)
            if node is not None and rd is not None:
                node.set_label(self._entry_label(rd))
        # Compute and apply new highlights
        related = self._compute_related_hashes(entry_hash)
        self._highlighted_related = related
        for h in related:
            node = self._node_by_hash.get(h)
            rd = self._row_cache.get(h)
            if node is not None and rd is not None:
                node.set_label(self._entry_label_related(rd))

    def action_open_data_view(self) -> None:
        entry_hash = self._selected_entry_hash()
        if entry_hash is None:
            return
        row_data = self._row_cache.get(entry_hash)
        if row_data is not None:
            self.app.push_screen(DataViewScreen(row_data))

    @on(Tree.NodeHighlighted, "#catalog-tree")
    def _on_catalog_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        entry_hash = self._get_entry_hash_from_node(event.node)
        if entry_hash is None or entry_hash == self._last_highlighted_hash:
            return
        self._last_highlighted_hash = entry_hash
        self._update_related_highlights(entry_hash)

        schema_in_table = self.query_one("#schema-in-table", DataTable)
        schema_in_table.clear()
        schema_out_table = self.query_one("#schema-preview-table", DataTable)
        schema_out_table.clear()
        sql_preview = self.query_one("#sql-preview", RichLog)
        lineage_tree = self.query_one("#lineage-tree", Tree)

        rev_table = self.query_one("#revisions-preview-table", DataTable)
        rev_table.clear()

        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            sql_preview.clear()
            lineage_tree.clear()
            lineage_tree.root.set_label("Lineage")
            self.query_one("#schema-in-half").display = False
            self.query_one("#revisions-panel").border_title = "Revisions"
            self._reset_toggle_panels()
            return

        # Schema
        schema_panel = self.query_one("#schema-panel")
        match row_data.schema_in:
            case None:
                self.query_one("#schema-in-half").display = False
                schema_panel.border_title = "Schema"
                schema_panel.border_subtitle = f"{len(row_data.schema_out)} cols"
            case schema_in:
                schema_in_half = self.query_one("#schema-in-half")
                schema_in_half.display = True
                schema_in_half.border_title = "In"
                schema_in_half.border_subtitle = f"{len(schema_in)} cols"
                schema_panel.border_title = "Schemas"
                schema_panel.border_subtitle = ""
                for name, dtype in schema_in:
                    schema_in_table.add_row(name, dtype)
        schema_out_half = self.query_one("#schema-out-half")
        schema_out_half.border_title = "Out" if row_data.schema_in else ""
        schema_out_half.border_subtitle = (
            f"{len(row_data.schema_out)} cols" if row_data.schema_in else ""
        )
        for name, dtype in row_data.schema_out:
            schema_out_table.add_row(name, dtype)

        # Lineage panel (sync)
        lineage_panel = self.query_one("#lineage-panel")
        dag = row_data.lineage_dag
        if dag:
            _populate_lineage_tree(lineage_tree, dag)
        else:
            lineage_tree.clear()
            lineage_tree.root.set_label("(no lineage)")
        lineage_panel.border_subtitle = row_data.cache_info_text

        # Tags panel (sync)
        tags_tree = self.query_one("#tags-tree", Tree)
        if dag:
            _populate_tags_tree(tags_tree, dag)
        else:
            tags_tree.clear()
            tags_tree.root.set_label("(no tags)")

        # Full lineage panel (sync)
        full_lineage_tree = self.query_one("#full-lineage-tree", Tree)
        if dag:
            _populate_full_lineage_tree(full_lineage_tree, dag)
        else:
            full_lineage_tree.clear()
            full_lineage_tree.root.set_label("(no lineage)")

        # SQL preview — serve from cache or render async
        entry_hash = row_data.hash
        sql_panel = self.query_one("#sql-panel")
        sql_preview.clear()
        match row_data.sqls:
            case ():
                sql_preview.write("(SQL unavailable)")
                sql_panel.border_subtitle = ""
            case _ if entry_hash in self._sql_cache:
                syntax, truncated, total_lines = self._sql_cache[entry_hash]
                self._write_sql_to_widget(syntax, truncated, total_lines)
                match row_data.sqls:
                    case ((_, engine, _),):
                        sql_panel.border_subtitle = engine
                    case sqls:
                        engines = sorted({engine for _, engine, _ in sqls})
                        sql_panel.border_subtitle = (
                            f"{len(sqls)} queries \u00b7 {', '.join(engines)}"
                        )
            case ((_, engine, sql),):
                sql_preview.loading = True
                sql_panel.border_subtitle = engine
                self._render_sql_async(entry_hash, sql)
            case sqls:
                sql_preview.loading = True
                engines = sorted({engine for _, engine, _ in sqls})
                sql_panel.border_subtitle = (
                    f"{len(sqls)} queries \u00b7 {', '.join(engines)}"
                )
                self._render_sql_async(entry_hash, _render_sql_dag(sqls))

        # Revisions preview (async — only when panel visible)
        if self._revisions_visible:
            self._update_revisions(row_data)

        # Update toggle panels for new row (preserve visibility)
        self._update_toggle_panels(row_data, entry_hash)

    def _update_toggle_panels(self, row_data, entry_hash) -> None:
        """Update toggle panels for a new row, preserving visibility."""
        if self._active_view == "data":
            dt = self.query_one("#data-preview-table", DataTable)
            if row_data.cached:
                self._data_preview = _TogglePanelState(
                    visible=True,
                    loaded=True,
                    entry_hash=entry_hash,
                )
                dt.loading = True
                self.query_one("#data-preview-status", Static).update(
                    " Loading data preview..."
                )
                self._load_data_preview(row_data.entry)
            else:
                self._data_preview = _TogglePanelState(visible=True)
                self.query_one("#data-preview-status", Static).update(
                    " uncached — run to materialize"
                )
                dt.clear(columns=True)
                dt.loading = False
        else:
            self._data_preview = _TogglePanelState()
            dt = self.query_one("#data-preview-table", DataTable)
            dt.clear(columns=True)
            dt.loading = True

        if self._active_view == "profiles":
            self._profiles_state = _TogglePanelState()
            self._ensure_profiles_loaded()

    def _apply_view(self, name: str) -> None:
        """Show the named detail view, hiding the others."""
        self._active_view = name
        for view_name, selector in self.VIEW_PANELS.items():
            self.query_one(selector).display = view_name == name

    def _reset_toggle_panels(self) -> None:
        self._apply_view("sql")

        self._data_preview = _TogglePanelState()
        dt = self.query_one("#data-preview-table", DataTable)
        dt.clear(columns=True)
        dt.loading = True

        self._profiles_state = _TogglePanelState()
        self.query_one("#profiles-table", DataTable).clear()

    @work(thread=True, exit_on_error=False)
    def _do_refresh(self) -> None:
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._do_refresh_locked()
        finally:
            self._refresh_lock.release()

    @property
    def catalog_aliases(self) -> tuple:
        catalog = self.app._catalog
        if catalog is None:
            return ()
        return _get_catalog_aliases(catalog)

    def _do_refresh_locked(self) -> None:
        catalog = self.app._catalog
        if catalog is None:
            return
        repo_path = catalog.repo.working_dir
        expected_keys = frozenset(_get_catalog_list(catalog))
        alias_multimap = _build_alias_multimap(self.catalog_aliases)

        # preserve insertion order from _row_cache (dict is ordered in Python 3.7+)
        cached_rows = tuple(
            self._row_cache[k] for k in self._row_cache if k in expected_keys
        )
        new_keys = expected_keys - self._row_cache.keys()
        removed = self._row_cache.keys() - expected_keys

        # evict removed entries
        self._row_cache = {
            k: v for k, v in self._row_cache.items() if k in expected_keys
        }

        match (bool(removed), bool(cached_rows)):
            case (True, _) | (_, False):
                # re-render when rows were removed or on first refresh
                self.app.call_from_thread(self._render_refresh, repo_path, cached_rows)
            case _:
                pass

        # load new entries incrementally (expensive I/O, off the main thread)
        for entry_hash in new_keys:
            entry = catalog.get_catalog_entry(entry_hash)
            aliases = alias_multimap.get(entry_hash, ())
            row_data = _load_catalog_row(entry, aliases)
            self._row_cache[row_data.row_key] = row_data
            self.app.call_from_thread(self._render_catalog_row, row_data)

        if self._git_log_visible:
            git_rows = _build_git_log_rows(catalog.repo)
            self.app.call_from_thread(self._render_git_log, git_rows)

        stamp = datetime.now().strftime("%H:%M:%S")
        self.app.call_from_thread(self._render_status, stamp, repo_path)

    def _rebuild_catalog_tree(self, entries=None, query: str = "") -> None:
        """Rebuild the catalog tree from entries, grouped by kind."""
        tree = self.query_one("#catalog-tree", Tree)
        tree.clear()
        self._kind_nodes.clear()
        self._node_by_hash.clear()
        self._highlighted_related.clear()

        if entries is None:
            entries = self._row_cache.values()

        query_lower = query.lower()
        sorted_entries = sorted(entries, key=lambda r: r.sort_key)
        if query_lower:
            sorted_entries = [
                r for r in sorted_entries if self._matches_search(r, query_lower)
            ]

        for row_data in sorted_entries:
            self._add_entry_to_tree(tree, row_data)

    def _render_refresh(self, repo_path, cached_rows) -> None:
        with self.app.batch_update():
            catalog_name = Path(repo_path).name
            self.query_one(
                "#catalog-panel"
            ).border_title = f"Expressions — {catalog_name}"
            self._rebuild_catalog_tree(entries=cached_rows, query=self._search_query)

    def _add_entry_to_tree(self, tree, row_data) -> None:
        """Add a single entry node to the catalog tree under its kind group."""
        kind = row_data.kind
        if kind not in self._kind_nodes:
            self._kind_nodes[kind] = tree.root.add(f"[bold]{kind}[/bold]", expand=True)
        kind_node = self._kind_nodes[kind]
        entry_node = kind_node.add(self._entry_label(row_data), data=row_data.row_key)
        self._node_by_hash[row_data.row_key] = entry_node
        entry_node.add_leaf(f"[dim]{row_data.hash[:12]}[/dim]")
        for tag in row_data.tags:
            entry_node.add_leaf(f"[#5abfb5]{tag}[/]")

    def _render_catalog_row(self, row_data) -> None:
        if self._search_query and not self._matches_search(
            row_data, self._search_query.lower()
        ):
            return
        with self.app.batch_update():
            tree = self.query_one("#catalog-tree", Tree)
            self._add_entry_to_tree(tree, row_data)

    def _render_status(self, stamp, repo_path) -> None:
        count = len(self._row_cache)
        self.query_one("#status-bar", Static).update(
            f" {count} entries \u00b7 {repo_path} \u00b7 {stamp}"
        )

    # --- Search ---

    def action_start_search(self) -> None:
        search_input = self.query_one("#search-input", Input)
        search_input.disabled = False
        search_input.display = True
        search_input.value = self._search_query
        search_input.focus()

    def _finish_search(self, keep_filter: bool) -> None:
        search_input = self.query_one("#search-input", Input)
        if not keep_filter:
            self._search_query = ""
            search_input.value = ""
            self._apply_search_filter("")
        search_input.display = False
        search_input.disabled = True
        self.query_one("#catalog-tree", Tree).focus()

    @on(Input.Changed, "#search-input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._search_query = event.value
        self._apply_search_filter(event.value)

    @on(Input.Submitted, "#search-input")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        self._finish_search(keep_filter=True)

    def key_escape(self) -> None:
        search_input = self.query_one("#search-input", Input)
        if not search_input.disabled and search_input.has_focus:
            self._finish_search(keep_filter=False)

    def _apply_search_filter(self, query: str) -> None:
        self._rebuild_catalog_tree(query=query)

        query_lower = query.lower()
        total = len(self._row_cache)
        if query_lower:
            count = sum(
                1
                for r in self._row_cache.values()
                if self._matches_search(r, query_lower)
            )
            self.query_one("#catalog-panel").border_subtitle = f"{count}/{total}"
        else:
            self.query_one("#catalog-panel").border_subtitle = ""

    def _matches_search(self, row_data: CatalogRowData, query: str) -> bool:
        return (
            query in row_data.aliases_display.lower()
            or query in row_data.hash.lower()
            or query in row_data.kind.lower()
            or query in (row_data.entry.metadata.root_tag or "").lower()
        )

    # --- Toggle: Git Log ---

    def action_toggle_git_log(self) -> None:
        self._git_log_visible = not self._git_log_visible
        self.query_one("#git-log-panel").display = self._git_log_visible
        if self._git_log_visible and not self._git_log_loaded:
            self._load_git_log()

    @work(thread=True)
    def _load_git_log(self) -> None:
        catalog = self.app._catalog
        if catalog is None:
            return
        rows = _build_git_log_rows(catalog.repo)
        self._git_log_loaded = True
        self.app.call_from_thread(self._render_git_log, rows)

    def _render_git_log(self, rows) -> None:
        with self.app.batch_update():
            table = self.query_one("#git-log-table", DataTable)
            table.clear()
            for i, row_data in enumerate(rows):
                table.add_row(*row_data.row, key=str(i))

    # --- Toggle: Revisions ---

    def action_toggle_revisions(self) -> None:
        self._revisions_visible = not self._revisions_visible
        self.query_one("#revisions-panel").display = self._revisions_visible
        if self._revisions_visible:
            entry_hash = self._selected_entry_hash()
            if entry_hash is None:
                return
            row_data = self._row_cache.get(entry_hash)
            if row_data is not None:
                self._update_revisions(row_data)

    def _update_revisions(self, row_data) -> None:
        rev_table = self.query_one("#revisions-preview-table", DataTable)
        rev_table.clear()
        match row_data.aliases:
            case (first_alias, *_):
                catalog_alias = next(
                    (ca for ca in self.catalog_aliases if ca.alias == first_alias),
                    None,
                )
                match catalog_alias:
                    case None:
                        self.query_one(
                            "#revisions-panel"
                        ).border_title = "Revisions — (alias not found)"
                    case _:
                        self.query_one(
                            "#revisions-panel"
                        ).border_title = f"Revisions — {first_alias}"
                        self._load_revisions_preview(catalog_alias)
            case _:
                self.query_one(
                    "#revisions-panel"
                ).border_title = "Revisions — (no alias)"

    # --- View switching (1/2/3) ---

    def action_show_view(self, name: str) -> None:
        self._apply_view(name)
        if name == "data":
            self._ensure_data_loaded()
        elif name == "profiles":
            self._ensure_profiles_loaded()

    def _ensure_data_loaded(self) -> None:
        entry_hash = self._selected_entry_hash()
        if entry_hash is None:
            return
        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            return

        if self._data_preview.loaded and self._data_preview.entry_hash == entry_hash:
            return

        match row_data.cached:
            case True:
                self._data_preview = _TogglePanelState(
                    visible=True,
                    loaded=True,
                    entry_hash=entry_hash,
                )
                self.query_one("#data-preview-status", Static).update(
                    " Loading data preview..."
                )
                self._load_data_preview(row_data.entry)
            case _:
                self.query_one("#data-preview-status", Static).update(
                    " uncached — run to materialize"
                )
                self.query_one("#data-preview-table", DataTable).loading = False

    @work(thread=True, exit_on_error=False)
    def _load_data_preview(self, entry) -> None:
        try:
            df = entry.expr.head(50).execute()
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(str(round(v, 2)) if isinstance(v, float) else str(v) for v in row)
                for row in df.itertuples(index=False)
            )
            total_rows = len(df)
            self.app.call_from_thread(
                self._render_data_preview, columns, rows, total_rows
            )
        except Exception as e:
            self.app.call_from_thread(self._render_data_preview_error, str(e))

    def _render_data_preview(self, columns, rows, total_rows) -> None:
        with self.app.batch_update():
            self.query_one("#data-preview-status", Static).update(
                f" Data Preview — {total_rows} rows (max 50)"
            )
            data_table = self.query_one("#data-preview-table", DataTable)
            data_table.clear(columns=True)
            data_table.loading = False
            for col in columns:
                data_table.add_column(col, key=col)
            for i, row in enumerate(rows):
                data_table.add_row(*row, key=str(i))
            data_table.cursor_type = "row"

    def _render_data_preview_error(self, message) -> None:
        self.query_one("#data-preview-status", Static).update(f" Error: {message}")
        self.query_one("#data-preview-table", DataTable).loading = False

    # --- View: Profiles (4) ---

    def _ensure_profiles_loaded(self) -> None:
        entry_hash = self._selected_entry_hash()
        if entry_hash is None:
            return
        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            return
        if (
            self._profiles_state.loaded
            and self._profiles_state.entry_hash == entry_hash
        ):
            return
        self._profiles_state = _TogglePanelState(
            visible=True,
            loaded=True,
            entry_hash=entry_hash,
        )
        self._load_profiles(row_data.entry)

    @work(thread=True, exit_on_error=False)
    def _load_profiles(self, entry) -> None:
        if not entry.catalog_path.exists() or not zipfile.is_zipfile(
            entry.catalog_path
        ):
            return

        env_re = re.compile(r"^\$\{(.+)\}$|^\$(.+)$")

        def _extract_env_vars(kwargs):
            return tuple(
                m.group(1) or m.group(2)
                for v in kwargs.values()
                if isinstance(v, str) and (m := env_re.match(v))
            )

        with zipfile.ZipFile(entry.catalog_path, "r") as zf:
            member_path = f"{entry.name}/profiles.yaml"
            if member_path not in zf.namelist():
                return
            data = yaml12.parse_yaml(zf.read(member_path).decode())
        match data:
            case dict():
                pass
            case _:
                return
        rows = tuple(
            (
                pname,
                pdata.get("con_name", "?"),
                ", ".join(
                    f"{k}={v}"
                    for k, v in (pdata.get("kwargs_tuple") or {}).items()
                    if v is not None
                ),
                ", ".join(_extract_env_vars(pdata.get("kwargs_tuple") or {})),
            )
            for pname, pdata in data.items()
        )
        self.app.call_from_thread(self._render_profiles, rows)

    def _render_profiles(self, rows) -> None:
        with self.app.batch_update():
            table = self.query_one("#profiles-table", DataTable)
            table.clear()
            for i, (name, backend, params, env_vars) in enumerate(rows):
                table.add_row(name, backend, params, env_vars, key=str(i))

    # --- SQL Rendering (off main thread) ---

    _SQL_MAX_LINES = 1000

    @work(thread=True, exit_on_error=False)
    def _render_sql_async(self, entry_hash: str, sql_text: str) -> None:
        self._sql_rendering_hash = entry_hash
        lines = sql_text.split("\n")
        truncated = len(lines) > self._SQL_MAX_LINES
        if truncated:
            sql_text = "\n".join(lines[: self._SQL_MAX_LINES])
        syntax = Syntax(sql_text, "sql", theme=XorqSQLStyle, word_wrap=True)
        self.app.call_from_thread(
            self._render_sql_done, entry_hash, syntax, truncated, len(lines)
        )

    def _render_sql_done(
        self, entry_hash: str, syntax, truncated: bool, total_lines: int
    ) -> None:
        self._sql_cache[entry_hash] = (syntax, truncated, total_lines)
        if self._sql_rendering_hash == entry_hash:
            self._write_sql_to_widget(syntax, truncated, total_lines)

    def _write_sql_to_widget(self, syntax, truncated, total_lines) -> None:
        sql_preview = self.query_one("#sql-preview", RichLog)
        sql_preview.clear()
        sql_preview.write(syntax)
        if truncated:
            sql_preview.write(
                f"\n[dim]… truncated ({self._SQL_MAX_LINES}/{total_lines} lines)[/dim]"
            )
        sql_preview.loading = False

    # --- Revisions Preview ---

    @work(thread=True, exit_on_error=False)
    def _load_revisions_preview(self, catalog_alias) -> None:
        try:
            raw_revisions = catalog_alias.list_revisions()
        except (KeyError, ValueError, OSError):
            return
        revision_rows = tuple(
            row
            for i, (rev_entry, commit) in enumerate(raw_revisions)
            for row, _ in (_revision_pair(i, rev_entry, commit),)
        )
        self.app.call_from_thread(self._render_revisions_preview, revision_rows)

    def _render_revisions_preview(self, revision_rows) -> None:
        with self.app.batch_update():
            rev_table = self.query_one("#revisions-preview-table", DataTable)
            rev_table.clear()
            for i, row_data in enumerate(revision_rows):
                rev_table.add_row(*row_data.row, key=str(i))
            rev_panel = self.query_one("#revisions-panel")
            rev_panel.border_subtitle = f"{len(revision_rows)} revisions"

    # --- Navigation ---

    def _focused_widget(self) -> DataTable | VerticalScroll | Tree | None:
        focused = self.app.focused
        if isinstance(focused, Input):
            return None
        if isinstance(focused, (DataTable, VerticalScroll, Tree)):
            return focused
        return self.query_one("#catalog-tree", Tree)

    def action_scroll_left(self) -> None:
        match self._focused_widget():
            case DataTable() as w:
                w.action_scroll_left()
            case Tree() as w:
                w.action_scroll_left()

    def action_cursor_down(self) -> None:
        match self._focused_widget():
            case DataTable() as w:
                w.action_cursor_down()
            case Tree() as w:
                w.action_cursor_down()
            case VerticalScroll() as w:
                w.scroll_down()

    def action_cursor_up(self) -> None:
        match self._focused_widget():
            case DataTable() as w:
                w.action_cursor_up()
            case Tree() as w:
                w.action_cursor_up()
            case VerticalScroll() as w:
                w.scroll_up()

    def action_scroll_right(self) -> None:
        match self._focused_widget():
            case DataTable() as w:
                w.action_scroll_right()
            case Tree() as w:
                w.action_scroll_right()

    def action_focus_next_panel(self) -> None:
        self._cycle_focus(1)

    def action_focus_prev_panel(self) -> None:
        self._cycle_focus(-1)

    def _is_visible(self, widget) -> bool:
        """Check widget and all ancestors are visible."""
        node = widget
        while node is not None:
            if node.display is False:
                return False
            node = node.parent
        return True

    def _cycle_focus(self, direction: int) -> None:
        visible = tuple(
            sel for sel in self.FOCUS_CYCLE if self._is_visible(self.query_one(sel))
        )
        if not visible:
            return
        current = self.app.focused

        def _matches(widget, target) -> bool:
            """Check if the focused widget is, or is inside, the cycle target."""
            if widget is target:
                return True
            node = widget
            while node is not None:
                if node is target:
                    return True
                node = node.parent
            return False

        current_idx = next(
            (
                i
                for i, sel in enumerate(visible)
                if _matches(current, self.query_one(sel))
            ),
            0,
        )
        next_idx = (current_idx + direction) % len(visible)
        self.query_one(visible[next_idx]).focus()

    def action_quit_app(self) -> None:
        self.app.exit()


class CatalogTUI(App):
    TITLE = "xorq catalog"
    CSS = """
    * {
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    *:hover {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-color: #3b4261;
        scrollbar-color-hover: #565f89;
        scrollbar-color-active: #7aa2f7;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
    }
    #main-split {
        height: 1fr;
    }
    #left-column {
        width: 2fr;
        max-width: 60;
    }
    #right-column {
        width: 3fr;
    }
    #catalog-panel {
        height: 1fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        background: $surface;
    }
    #search-input {
        display: none;
        height: 1;
        dock: top;
        margin: 0;
        border: none;
        padding: 0 1;
        background: $surface;
    }
    #catalog-panel:focus-within {
        border: double #C1F0FF;
    }
    #catalog-tree {
        height: 1fr;
        padding: 0 1;
    }
    #revisions-panel {
        height: 1fr;
        border: solid #5abfb5;
        border-title-color: #5abfb5;
        border-subtitle-color: #5abfb5;
    }
    #revisions-preview-table {
        height: 1fr;
    }
    #git-log-panel {
        height: 1fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
    }
    #git-log-table {
        height: 1fr;
    }
    #detail-view {
        height: 2fr;
    }
    #lineage-panel {
        height: 1fr;
        border: solid #5abfb5;
        border-title-color: #5abfb5;
        border-subtitle-color: #5abfb5;
        padding: 0 1;
    }
    #lineage-panel:focus-within {
        border: double #5abfb5;
    }
    #lineage-tree {
        height: 1fr;
        padding: 0 1;
    }
    #sql-panel {
        height: 1fr;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
        border-subtitle-color: #2BBE75;
    }
    #sql-panel:focus-within {
        border: double #2BBE75;
    }
    #sql-preview {
        height: 1fr;
        padding: 0 1;
    }
    #data-preview-panel {
        height: 1fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
    }
    DataTable:focus {
        border: none;
    }
    #schema-panel {
        height: 1fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
        border-subtitle-color: #4AA8EC;
    }
    #schema-split {
        height: 1fr;
    }
    #schema-in-half {
        width: 1fr;
        border: solid #4AA8EC 50%;
        border-title-color: #4AA8EC;
        border-subtitle-color: #4AA8EC;
    }
    #schema-out-half {
        width: 1fr;
        border: solid #4AA8EC 50%;
        border-title-color: #4AA8EC;
        border-subtitle-color: #4AA8EC;
    }
    #schema-in-table {
        height: 1fr;
    }
    #schema-preview-table {
        height: 1fr;
    }
    #data-preview-status {
        height: 1;
        padding: 0 2;
    }
    #data-preview-table {
        height: 1fr;
    }
    #profiles-panel {
        height: 1fr;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
    }
    #profiles-table {
        height: 1fr;
    }
    #tags-panel {
        height: 1fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
        padding: 0 1;
    }
    #tags-tree {
        height: 1fr;
        padding: 0 1;
    }
    #full-lineage-panel {
        height: 1fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        padding: 0 1;
    }
    #full-lineage-panel:focus-within {
        border: double #C1F0FF;
    }
    #full-lineage-tree {
        height: 1fr;
        padding: 0 1;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    /* ComposeScreen (modal overlay) */
    ComposeScreen {
        align: center middle;
        background: rgba(5, 24, 26, 0.85);
    }
    #compose-dialog {
        width: 90%;
        height: 80%;
        max-width: 100;
        max-height: 34;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
        background: $surface;
        padding: 0 1;
    }
    #compose-title {
        height: 1;
        color: #2BBE75;
    }
    #compose-split {
        height: 1fr;
    }
    #compose-source-panel {
        width: 2fr;
        border: solid #C1F0FF 50%;
        border-title-color: #C1F0FF;
    }
    #compose-source-panel:focus-within {
        border: double #C1F0FF;
    }
    #compose-entries-table {
        height: 1fr;
    }
    #compose-transform-panel {
        width: 3fr;
        border: solid #5abfb5 50%;
        border-title-color: #5abfb5;
    }
    #compose-chain-table {
        height: auto;
        max-height: 10;
    }
    #compose-validation {
        height: auto;
        max-height: 4;
        padding: 0 1;
    }
    #compose-status {
        height: 1;
        padding: 0 1;
    }
    /* DataViewScreen */
    #dv-data-panel {
        height: 3fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        border-subtitle-color: #C1F0FF;
    }
    #dv-data-table {
        height: 1fr;
    }
    #dv-stats-panel {
        height: 1fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
        border-subtitle-color: #F5CA2C;
    }
    #dv-stats-table {
        height: 1fr;
    }
    #dv-command-input {
        display: none;
        dock: bottom;
        height: 3;
        margin: 0;
        border: tall #5abfb5;
        padding: 0 1;
        background: $surface;
        color: #C1F0FF;
    }
    #dv-status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    """

    def __init__(self, make_catalog, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._catalog = None
        self._make_catalog = make_catalog
        self._refresh_interval = refresh_interval
        self.register_theme(XORQ_DARK)
        self.theme = "xorq-dark"

    def on_mount(self) -> None:
        self.push_screen(CatalogScreen(refresh_interval=self._refresh_interval))
        self._load_catalog()

    @work(thread=True)
    def _load_catalog(self) -> None:
        catalog = self._make_catalog()
        self.app.call_from_thread(self._set_catalog, catalog)

    def _set_catalog(self, catalog) -> None:
        self._catalog = catalog
        self.screen._do_refresh()
