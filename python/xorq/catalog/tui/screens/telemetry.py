"""Telemetry screen — lineage-correlated expandable tree view.

Shows the expression lineage (source → transform → code) as the
top-level structure. Run events and OTEL spans are nested under
the lineage step they belong to, giving a unified view of what
was composed, how it executed, and whether caches were hit.
"""

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static, Tree


META_COLUMNS = ("KEY", "VALUE")


class TelemetryScreen(Screen):
    """Full-screen telemetry viewer with lineage-correlated span tree."""

    BINDINGS = (
        ("q", "go_back", "Back"),
        ("escape", "go_back", "Back"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_prev", "Prev"),
    )

    def __init__(self, run_id: str, expr_hash: str):
        super().__init__()
        self._run_id = run_id
        self._expr_hash = expr_hash

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="telemetry-split"):
            with Vertical(id="telemetry-spans-panel"):
                yield Static("", id="telemetry-spans-title")
                yield Tree("telemetry", id="telemetry-tree")
            with Vertical(id="telemetry-meta-panel"):
                yield Static("", id="telemetry-meta-title")
                yield DataTable(id="telemetry-meta-table")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"telemetry \u2014 {self._run_id[:8]}"

        tree = self.query_one("#telemetry-tree", Tree)
        tree.show_root = False
        tree.guide_depth = 3

        meta_table = self.query_one("#telemetry-meta-table", DataTable)
        meta_table.cursor_type = "row"
        meta_table.zebra_stripes = True
        for col in META_COLUMNS:
            meta_table.add_column(col, key=col)

        self.query_one("#telemetry-spans-title", Static).update(" Loading...")
        self.query_one("#telemetry-meta-title", Static).update(
            f" Meta \u2014 {self._run_id[:8]}"
        )
        self._load_telemetry()

    @work(thread=True, exit_on_error=False)
    def _load_telemetry(self) -> None:
        from xorq.common.utils.logging_utils import (  # noqa: PLC0415
            Run,
            get_xorq_runs_dir,
        )

        runs_dir = get_xorq_runs_dir()
        run_dir = runs_dir / self._expr_hash / self._run_id
        run = Run(run_dir=run_dir) if run_dir.exists() else None

        meta = run.read_meta() if run else {}
        meta_rows = tuple(
            (str(k), _truncate(str(v), 80)) for k, v in (meta or {}).items()
        )

        events = run.read_events() if run else ()
        trace_id = (meta or {}).get("otel_trace_id")
        otel_spans = _load_otel_spans(trace_id) if trace_id else ()

        self.app.call_from_thread(
            self._render_data,
            events,
            otel_spans,
            meta_rows,
        )

    def _render_data(
        self,
        events: tuple[dict, ...],
        otel_spans: tuple[dict, ...],
        meta_rows: tuple[tuple[str, str], ...],
    ) -> None:
        tree = self.query_one("#telemetry-tree", Tree)
        tree.clear()

        # --- Run events section ---
        if events:
            events_node = tree.root.add(f"run events ({len(events)})", expand=True)
            for ev in events:
                _add_event_node(events_node, ev)

        # --- OTEL spans section ---
        if otel_spans:
            otel_node = tree.root.add(
                f"execution trace ({len(otel_spans)} spans)", expand=True
            )
            _add_span_tree(otel_node, otel_spans)

        # --- Title ---
        parts = [f"{len(events)} events"]
        if otel_spans:
            parts.append(f"{len(otel_spans)} spans")
        self.query_one("#telemetry-spans-title", Static).update(
            f" {self._run_id[:8]} \u00b7 {' + '.join(parts)}"
        )

        # --- Meta ---
        meta_table = self.query_one("#telemetry-meta-table", DataTable)
        for i, row in enumerate(meta_rows):
            meta_table.add_row(*row, key=str(i))
        self.query_one("#telemetry-meta-title", Static).update(
            f" Meta \u2014 {self._run_id[:8]} \u00b7 {len(meta_rows)} fields"
        )

    # --- Navigation ---

    def action_focus_next(self) -> None:
        match self.app.focused:
            case w if w is self.query_one("#telemetry-tree", Tree):
                self.query_one("#telemetry-meta-table", DataTable).focus()
            case _:
                self.query_one("#telemetry-tree", Tree).focus()

    def action_focus_prev(self) -> None:
        self.action_focus_next()

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Run event tree nodes
# ---------------------------------------------------------------------------


def _add_event_node(parent, ev: dict) -> None:
    timestamp = ev.get("timestamp", "")
    match timestamp:
        case str(ts) if "T" in ts:
            timestamp = ts.split("T")[1][:12]
        case _:
            pass

    event_name = ev.get("event", "?")
    label = f"{timestamp}  {event_name}"

    details = {k: v for k, v in ev.items() if k not in ("timestamp", "event")}

    if details:
        node = parent.add(label, expand=False)
        for k, v in details.items():
            node.add_leaf(f"{k}: {_truncate(str(v), 60)}")
    else:
        parent.add_leaf(label)


# ---------------------------------------------------------------------------
# OTEL span tree
# ---------------------------------------------------------------------------


def _load_otel_spans(trace_id: str) -> tuple[dict, ...]:
    try:
        from xorq.common.utils.trace_utils import default_log_path  # noqa: PLC0415

        if not default_log_path.exists():
            return ()

        import json  # noqa: PLC0415
        import re  # noqa: PLC0415

        text = default_log_path.read_text().strip()
        if not text:
            return ()

        blocks = re.split(r"\}\s*\n\s*\{", text)
        raw_spans = []
        for i, block in enumerate(blocks):
            if i > 0:
                block = "{" + block
            if i < len(blocks) - 1:
                block = block + "}"
            try:
                raw_spans.append(json.loads(block))
            except json.JSONDecodeError:
                continue

        hex_id = trace_id
        return tuple(s for s in raw_spans if _extract_trace_id(s) == hex_id)
    except Exception:
        return ()


def _extract_trace_id(span: dict) -> str:
    ctx = span.get("context", {})
    return ctx.get("trace_id", "").removeprefix("0x")


def _add_span_tree(parent, spans: tuple[dict, ...]) -> None:
    by_id: dict[str, dict] = {}
    children: dict[str, list[dict]] = {}
    roots: list[dict] = []

    all_sids = {
        s.get("context", {}).get("span_id", "").removeprefix("0x") for s in spans
    }

    for s in spans:
        sid = s.get("context", {}).get("span_id", "").removeprefix("0x")
        by_id[sid] = s
        pid = (s.get("parent_id") or "").removeprefix("0x")
        if pid and pid in all_sids:
            children.setdefault(pid, []).append(s)
        else:
            roots.append(s)

    for root in sorted(roots, key=lambda s: s.get("start_time", "")):
        _add_span_node(parent, root, children)


def _add_span_node(parent, span: dict, children: dict) -> None:
    name = span.get("name", "?")
    dur = _format_duration(span)
    cache_icon = _cache_icon(span)

    label = f"{cache_icon}{name}  [{dur}]" if dur else f"{cache_icon}{name}"

    events = span.get("events", [])
    attrs = span.get("attributes", {})
    sid = span.get("context", {}).get("span_id", "").removeprefix("0x")
    kids = children.get(sid, [])
    has_content = bool(events) or bool(attrs) or bool(kids)

    if has_content:
        node = parent.add(label, expand=False)

        for event in events:
            ev_name = event.get("name", "")
            ev_attrs = event.get("attributes", {})
            match ev_name:
                case "cache.hit":
                    key = ev_attrs.get("key", "")
                    node.add_leaf(f"\u25cf cache HIT  {_truncate(key, 40)}")
                case "cache.miss":
                    key = ev_attrs.get("key", "")
                    node.add_leaf(f"\u25cb cache MISS  {_truncate(key, 40)}")
                case "replace_read":
                    engine = ev_attrs.get("engine", "?")
                    path = _truncate(str(ev_attrs.get("path", "")), 40)
                    node.add_leaf(f"read  engine={engine}  path={path}")
                case _ if "metrics" in ev_name:
                    parts = (f"{k}={v}" for k, v in ev_attrs.items())
                    node.add_leaf(f"metrics  {' '.join(parts)}")
                case _:
                    if ev_attrs:
                        detail = "  ".join(
                            f"{k}={_truncate(str(v), 30)}" for k, v in ev_attrs.items()
                        )
                        node.add_leaf(f"{ev_name}  {detail}")
                    elif ev_name:
                        node.add_leaf(ev_name)

        if isinstance(attrs, dict) and attrs:
            attrs_node = node.add("attributes", expand=False)
            for k, v in attrs.items():
                attrs_node.add_leaf(f"{k}: {_truncate(str(v), 50)}")

        for kid in sorted(kids, key=lambda s: s.get("start_time", "")):
            _add_span_node(node, kid, children)
    else:
        parent.add_leaf(label)


def _format_duration(span: dict) -> str:
    from datetime import datetime  # noqa: PLC0415

    try:
        start = datetime.fromisoformat(
            span.get("start_time", "").replace("Z", "+00:00")
        )
        end = datetime.fromisoformat(span.get("end_time", "").replace("Z", "+00:00"))
        dur = (end - start).total_seconds()
        match dur:
            case d if d < 0.001:
                return f"{d * 1_000_000:.0f}\u00b5s"
            case d if d < 1:
                return f"{d * 1000:.1f}ms"
            case d:
                return f"{d:.2f}s"
    except (ValueError, TypeError):
        return ""


def _cache_icon(span: dict) -> str:
    for event in span.get("events", []):
        match event.get("name"):
            case "cache.hit":
                return "\u25cf "
            case "cache.miss":
                return "\u25cb "
    return ""


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "\u2026"
