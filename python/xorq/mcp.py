"""MCP server exposing the xorq CLI as Claude-compatible tools.

Wraps CLI commands via subprocess so that:
- The heavy ``xorq.api`` import happens in the child process, not the server.
- Every tool maps 1:1 to a CLI invocation the user already knows.

Start with::

    xorq mcp serve            # stdio transport (default, for Claude integrations)
    xorq mcp serve --sse      # SSE transport (for debugging / remote)
"""

import asyncio

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "xorq",
    instructions=(
        "xorq is a framework for building, versioning, and running "
        "composable data expressions. Use the catalog tools to discover "
        "available entries and inspect their schemas before running them."
    ),
)

_XORQ_BIN = "xorq"


async def _run_cli(*args: str, timeout: float = 120.0) -> str:
    """Run ``xorq <args>`` and return combined stdout/stderr."""
    cmd = [_XORQ_BIN, *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return f"Error: command timed out after {timeout}s: {' '.join(cmd)}"

    out = stdout.decode() if stdout else ""
    err = stderr.decode() if stderr else ""

    if proc.returncode != 0:
        return f"Error (exit {proc.returncode}):\n{err or out}".strip()

    return out or err


def _catalog_args(
    catalog_name: str | None = None,
    catalog_path: str | None = None,
) -> list[str]:
    """Build the ``xorq catalog [--name N | --path P]`` prefix."""
    args = ["catalog"]
    if catalog_name:
        args += ["--name", catalog_name]
    elif catalog_path:
        args += ["--path", catalog_path]
    return args


# ---------------------------------------------------------------------------
# Catalog tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def catalog_list(
    catalog_name: str | None = None,
    catalog_path: str | None = None,
    show_kind: bool = False,
) -> str:
    """List all entries in a xorq catalog.

    Returns one entry per line. Pass show_kind=True to include the entry kind
    (source vs. partial/unbound).
    """
    args = _catalog_args(catalog_name, catalog_path)
    args.append("list")
    if show_kind:
        args.append("--kind")
    return await _run_cli(*args)


@mcp.tool()
async def catalog_list_aliases(
    catalog_name: str | None = None,
    catalog_path: str | None = None,
) -> str:
    """List all aliases in a xorq catalog."""
    args = _catalog_args(catalog_name, catalog_path)
    args.append("list-aliases")
    return await _run_cli(*args)


@mcp.tool()
async def catalog_schema(
    name: str,
    catalog_name: str | None = None,
    catalog_path: str | None = None,
    as_json: bool = False,
) -> str:
    """Show the schema (input/output columns and types) of a catalog entry.

    The ``name`` argument can be an entry name or an alias.
    Set as_json=True for machine-readable output.
    """
    args = _catalog_args(catalog_name, catalog_path)
    args += ["schema", name]
    if as_json:
        args.append("--json")
    return await _run_cli(*args)


@mcp.tool()
async def catalog_info(
    catalog_name: str | None = None,
    catalog_path: str | None = None,
) -> str:
    """Show catalog metadata: path, current commit, remotes, entry/alias counts."""
    args = _catalog_args(catalog_name, catalog_path)
    args.append("info")
    return await _run_cli(*args)


@mcp.tool()
async def catalog_log(
    catalog_name: str | None = None,
    catalog_path: str | None = None,
    as_json: bool = True,
) -> str:
    """Show catalog history as structured operations.

    Returns JSON by default for easier parsing.
    """
    args = _catalog_args(catalog_name, catalog_path)
    args.append("log")
    if as_json:
        args.append("--json")
    return await _run_cli(*args)


@mcp.tool()
async def catalog_check(
    catalog_name: str | None = None,
    catalog_path: str | None = None,
) -> str:
    """Validate catalog consistency. Returns 'OK' if healthy."""
    args = _catalog_args(catalog_name, catalog_path)
    args.append("check")
    return await _run_cli(*args)


# ---------------------------------------------------------------------------
# Build / run tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def build(
    script_path: str,
    expr_name: str = "expr",
    builds_dir: str = "builds",
) -> str:
    """Compile a xorq expression script into versioned build artifacts.

    Parameters
    ----------
    script_path
        Path to the Python script containing the expression.
    expr_name
        Name of the expression variable in the script.
    builds_dir
        Directory where build artifacts are written.
    """
    args = ["build", script_path, "-e", expr_name, "--builds-dir", builds_dir]
    return await _run_cli(*args)


@mcp.tool()
async def run(
    build_path: str,
    output_format: str = "json",
    limit: int | None = None,
    params: dict[str, str] | None = None,
) -> str:
    """Execute a built xorq expression and return results.

    Parameters
    ----------
    build_path
        Path to the build directory (output of ``build``).
    output_format
        One of: csv, json, parquet, arrow. Defaults to json for readability.
    limit
        Maximum number of rows to return.
    params
        Named parameters as key-value pairs (e.g. {"threshold": "0.5"}).
    """
    args = ["run", build_path, "-f", output_format, "-o", "/dev/stdout"]
    if limit is not None:
        args += ["--limit", str(limit)]
    if params:
        for k, v in params.items():
            args += ["-p", f"{k}={v}"]
    return await _run_cli(*args)


@mcp.tool()
async def run_cached(
    build_path: str,
    output_format: str = "json",
    limit: int | None = None,
    cache_type: str = "modification-time",
    ttl: int | None = None,
    params: dict[str, str] | None = None,
) -> str:
    """Execute a built expression with caching for efficient repeated runs.

    Parameters
    ----------
    build_path
        Path to the build directory.
    output_format
        One of: csv, json, parquet, arrow.
    limit
        Maximum number of rows to return.
    cache_type
        'modification-time' (default) or 'snapshot'.
    ttl
        TTL in seconds for snapshot cache.
    params
        Named parameters as key-value pairs.
    """
    args = [
        "run-cached",
        build_path,
        "-f",
        output_format,
        "-o",
        "/dev/stdout",
        "--cache-type",
        cache_type,
    ]
    if limit is not None:
        args += ["--limit", str(limit)]
    if ttl is not None:
        args += ["--ttl", str(ttl)]
    if params:
        for k, v in params.items():
            args += ["-p", f"{k}={v}"]
    return await _run_cli(*args)


# ---------------------------------------------------------------------------
# Catalog mutation tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def catalog_add(
    paths: list[str],
    catalog_name: str | None = None,
    catalog_path: str | None = None,
    aliases: list[str] | None = None,
    sync: bool = True,
) -> str:
    """Add build artifacts to a catalog.

    Parameters
    ----------
    paths
        Paths to archive files or build directories to add.
    aliases
        Optional aliases to assign to the added entries.
    sync
        Whether to push after adding (default True).
    """
    args = _catalog_args(catalog_name, catalog_path)
    args.append("add")
    if not sync:
        args.append("--no-sync")
    if aliases:
        for alias in aliases:
            args += ["--alias", alias]
    args += paths
    return await _run_cli(*args)


@mcp.tool()
async def catalog_remove(
    names: list[str],
    catalog_name: str | None = None,
    catalog_path: str | None = None,
    sync: bool = True,
) -> str:
    """Remove entries from a catalog by name.

    Parameters
    ----------
    names
        Entry names to remove.
    sync
        Whether to push after removing (default True).
    """
    args = _catalog_args(catalog_name, catalog_path)
    args.append("remove")
    if not sync:
        args.append("--no-sync")
    args += names
    return await _run_cli(*args)


@mcp.tool()
async def catalog_sync(
    catalog_name: str | None = None,
    catalog_path: str | None = None,
) -> str:
    """Pull then push a catalog to its remote(s)."""
    args = _catalog_args(catalog_name, catalog_path)
    args.append("sync")
    return await _run_cli(*args)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def serve(transport: str = "stdio"):
    """Run the MCP server."""
    mcp.run(transport=transport)
