"""
CLI command for inspecting catalog entries using Typer and Rich.
"""
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, is_dataclass, fields
from datetime import datetime, timezone

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich import box

# Mapping of operation types to Rich styles
OP_TYPE_STYLES = {
    "read": "bold white",
    "project": "cyan",
    "filter": "magenta",
    "cache": "green",
    "predict": "yellow",
}

app = typer.Typer(help="Catalog inspection commands")
catalog_app = typer.Typer()
app.add_typer(catalog_app, name="catalog")

@dataclass
class Column:
    name: str
    type: str

@dataclass
class PlanNode:
    id: str
    op: str
    details: str
    engine: Optional[str] = None

@dataclass
class Profile:
    id: str
    engine: str
    params: Dict[str, Any]

@dataclass
class CacheRecord:
    id: str
    node: str
    backend: str
    source: str

@dataclass
class InspectResult:
    entry: str
    entry_id: str
    created_at: datetime
    expr_id: str
    meta: Optional[str]
    revision: str
    schema: List[Column] = None
    plan: List[PlanNode] = None
    profiles: List[Profile] = None
    caches: List[CacheRecord] = None

def shorten_id(id: str, style: str = "short") -> str:
    if style == "full" or not id:
        return id
    if len(id) <= 12:
        return id
    return f"{id[:8]}…{id[-4:]}"

def humanize_time(dt: datetime, tz: str = "utc") -> str:
    if tz == "local":
        dt = dt.astimezone()
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def dataclass_to_dict(obj):
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(val)
        return result
    if isinstance(obj, list):
        return [dataclass_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def render_summary(console: Console, result: InspectResult, show_ids: str, tz: str, wide: bool):
    table = Table(box=box.SIMPLE_HEAD, show_header=False, pad_edge=False, expand=False)
    table.add_column(justify="left", overflow="ellipsis", no_wrap=not wide)
    table.add_column(justify="left", overflow="ellipsis", no_wrap=not wide)
    table.add_column(justify="left", overflow="ellipsis", no_wrap=not wide)
    table.title = result.entry
    table.title_style = "bold"
    entry_id = shorten_id(result.entry_id, show_ids)
    created = humanize_time(result.created_at, tz)
    expr_id = shorten_id(result.expr_id, show_ids)
    table.add_row(f"Entry ▸ {entry_id}", f"Created ▸ {created}", f"Expr ▸ {expr_id}")
    meta = shorten_id(result.meta, show_ids) if result.meta else "-"
    count = len(result.schema) if result.schema else 0
    schema_info = f"{count} column" + ("s" if count != 1 else "")
    table.add_row(f"Meta  ▸ {meta}", f"Revision ▸ {result.revision}", f"Schema ▸ {schema_info}")
    console.print(table)

def render_plan(console: Console, result: InspectResult, show_ids: str, wide: bool):
    if not result.plan:
        console.print("No plan available.", style="dim")
        return
    tree = Tree("Plan", guide_style="bold")
    for node in result.plan:
        nid = shorten_id(node.id, show_ids)
        style = OP_TYPE_STYLES.get(node.op.lower(), "white")
        label = f"{nid}  [{style}]{node.op}[/] {node.details}"
        if node.engine:
            label += f" [{node.engine}]"
        tree.add(label)
    console.print(tree)

def render_schema(console: Console, result: InspectResult, show_ids: str, wide: bool):
    if not result.schema:
        console.print("No schema available.", style="dim")
        return
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold", row_styles=["none", "dim"])
    table.add_column("column", overflow="ellipsis", no_wrap=not wide)
    table.add_column("type", overflow="ellipsis", no_wrap=not wide)
    for col in result.schema:
        table.add_row(col.name, col.type)
    console.print("\nSchema")
    console.print(table)

def render_profiles(console: Console, result: InspectResult, show_ids: str, wide: bool):
    if not result.profiles:
        console.print("No profiles available.", style="dim")
        return
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold", row_styles=["none", "dim"])
    table.add_column("id", overflow="ellipsis", no_wrap=not wide)
    table.add_column("engine", overflow="ellipsis", no_wrap=not wide)
    table.add_column("params", overflow="ellipsis", no_wrap=not wide)
    for prof in result.profiles:
        pid = shorten_id(prof.id, show_ids)
        params = json.dumps(prof.params)
        table.add_row(pid, prof.engine, params)
    console.print("\nProfiles")
    console.print(table)

def render_caches(console: Console, result: InspectResult, show_ids: str, wide: bool):
    if not result.caches:
        console.print("No caches available.", style="dim")
        return
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold", row_styles=["none", "dim"])
    table.add_column("id", overflow="ellipsis", no_wrap=not wide)
    table.add_column("node", overflow="ellipsis", no_wrap=not wide)
    table.add_column("backend", overflow="ellipsis", no_wrap=not wide)
    table.add_column("source", overflow="ellipsis", no_wrap=not wide)
    for cache in result.caches:
        cid = shorten_id(cache.id, show_ids)
        table.add_row(cid, cache.node, cache.backend, cache.source)
    console.print("\nCaches")
    console.print(table)

@catalog_app.command("inspect")
def inspect_catalog(
    entry: str = typer.Argument(..., help="Entry ID, alias, or entry@revision to inspect"),
    revision: Optional[str] = typer.Option(None, "--revision", "-r", help="Revision ID"),
    schema: bool = typer.Option(False, "--schema", help="Show schema section"),
    plan: bool = typer.Option(False, "--plan", help="Show plan section"),
    profiles: bool = typer.Option(False, "--profiles", help="Show profiles section"),
    caches: bool = typer.Option(False, "--caches", help="Show caches section"),
    full: bool = typer.Option(False, "--full", help="Show all sections"),
    show_ids: str = typer.Option("short", "--show-ids", help="ID style: short or full"),
    tz: str = typer.Option("utc", "--tz", help="Timezone: utc or local"),
    wide: bool = typer.Option(False, "--wide", help="Disable truncation"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
    yaml_out: bool = typer.Option(False, "--yaml", help="Output as YAML"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colorized output"),
    no_pager: bool = typer.Option(False, "--no-pager", help="Disable pager"),
):
    """Inspect a catalog entry and display details."""
    # Load catalog and resolve target
    from xorq.catalog import load_catalog, resolve_target
    cat = load_catalog()
    tgt = resolve_target(entry, cat)
    if tgt is None:
        typer.echo(f"Entry {entry} not found in catalog")
        raise typer.Exit(code=1)
    entry_id = tgt.entry_id
    rev_id = revision or tgt.rev
    # Find entry and revision data
    entries = cat.get("entries", [])
    e = next((e for e in entries if e.get("entry_id") == entry_id), None)
    if e is None:
        typer.echo(f"Entry {entry_id} not found in catalog")
        raise typer.Exit(code=1)
    # Parse timestamps
    try:
        created = datetime.fromisoformat(e.get("created_at"))
    except Exception:
        created = datetime.now(timezone.utc)
    # Determine revision
    if not rev_id:
        rev_id = e.get("current_revision")
    history = e.get("history", [])
    r = next((r for r in history if r.get("revision_id") == rev_id), None)
    if r is None:
        typer.echo(f"Revision {rev_id} not found for entry {entry_id}")
        raise typer.Exit(code=1)
    try:
        rev_created = datetime.fromisoformat(r.get("created_at"))
    except Exception:
        rev_created = datetime.now(timezone.utc)
    # Expr ID: try expr_hashes, else build ID
    expr_hash = (r.get("expr_hashes") or {}).get("expr") or r.get("build", {}).get("build_id")
    meta_digest = r.get("meta_digest")
    # Build result object (sections loaded later)
    result = InspectResult(
        entry=entry,
        entry_id=entry_id,
        created_at=created,
        expr_id=expr_hash,
        meta=meta_digest,
        revision=rev_id,
        schema=None,
        plan=None,
        profiles=None,
        caches=None,
    )
    # Machine-readable output
    data_dict = dataclass_to_dict(result)
    if json_out:
        typer.echo(json.dumps(data_dict, indent=2))
        raise typer.Exit()
    if yaml_out:
        typer.echo(yaml.safe_dump(data_dict, sort_keys=False))
        raise typer.Exit()
    # Initialize console
    console = Console(no_color=no_color)
    def display():
        render_summary(console, result, show_ids, tz, wide)
        # Determine which sections to render
        need_plan = full or plan or not any([schema, plan, profiles, caches])
        if need_plan:
            console.print()
            render_plan(console, result, show_ids, wide)
        if full or schema:
            console.print()
            render_schema(console, result, show_ids, wide)
        if full or profiles:
            console.print()
            render_profiles(console, result, show_ids, wide)
        if full or caches:
            console.print()
            render_caches(console, result, show_ids, wide)
        if not (full or schema or plan or profiles or caches):
            console.print("\nHints: add --schema, --profiles, --caches, or --full", style="dim")
    # Use pager if not disabled
    if no_pager:
        display()
    else:
        with console.pager():
            display()