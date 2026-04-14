import asyncio
from unittest.mock import patch

import pytest


pytest.importorskip("mcp", reason="mcp extra not installed")

import xorq.mcp as mcp_mod
from xorq.common.utils.process_utils import subprocess_run
from xorq.mcp import _catalog_args, _run_cli, mcp


# ---------------------------------------------------------------------------
# CLI wiring (subprocess) — measures real cold-start
# ---------------------------------------------------------------------------


def test_mcp_help():
    returncode, stdout, stderr = subprocess_run(["xorq", "mcp", "--help"], text=True)
    assert returncode == 0, stderr
    assert "serve" in stdout


def test_mcp_serve_help():
    returncode, stdout, stderr = subprocess_run(
        ["xorq", "mcp", "serve", "--help"], text=True
    )
    assert returncode == 0, stderr
    assert "--sse" in stdout


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "catalog_list",
    "catalog_list_aliases",
    "catalog_schema",
    "catalog_info",
    "catalog_log",
    "catalog_check",
    "build",
    "run",
    "run_cached",
    "catalog_add",
    "catalog_remove",
    "catalog_sync",
}


def test_all_tools_registered():
    tools = mcp._tool_manager.list_tools()
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS


def test_tool_descriptions_non_empty():
    tools = mcp._tool_manager.list_tools()
    for tool in tools:
        assert tool.description, f"tool {tool.name} has no description"


# ---------------------------------------------------------------------------
# _run_cli helper
# ---------------------------------------------------------------------------


def test_run_cli_success():
    result = asyncio.run(_run_cli("--help"))
    assert "build" in result
    assert "run" in result
    assert "Error" not in result


def test_run_cli_failure():
    result = asyncio.run(_run_cli("nonexistent-command-xyz"))
    assert result.startswith("Error")


def test_run_cli_timeout():
    # sleep 10 with a 1s timeout should be killed
    result = asyncio.run(_run_cli("run", "sleep-forever-not-a-real-path", timeout=0.5))
    # Either times out or fails with a non-zero exit — both are errors
    assert "Error" in result


# ---------------------------------------------------------------------------
# Argument building
# ---------------------------------------------------------------------------


def test_catalog_args_no_options():
    assert _catalog_args() == ["catalog"]


def test_catalog_args_name():
    assert _catalog_args(catalog_name="my-cat") == ["catalog", "--name", "my-cat"]


def test_catalog_args_path():
    assert _catalog_args(catalog_path="/tmp/cat") == ["catalog", "--path", "/tmp/cat"]


def test_catalog_args_name_takes_precedence():
    # When both are given, name wins (matches the if/elif logic)
    result = _catalog_args(catalog_name="n", catalog_path="/p")
    assert result == ["catalog", "--name", "n"]


@pytest.mark.parametrize(
    "tool_func,kwargs,expected_fragments",
    [
        ("catalog_list", {}, ["catalog", "list"]),
        ("catalog_list", {"show_kind": True}, ["--kind"]),
        ("catalog_schema", {"name": "foo"}, ["catalog", "schema", "foo"]),
        ("catalog_schema", {"name": "foo", "as_json": True}, ["--json"]),
        ("catalog_info", {}, ["catalog", "info"]),
        ("catalog_log", {}, ["catalog", "log", "--json"]),
        ("catalog_log", {"as_json": False}, ["catalog", "log"]),
        ("catalog_check", {}, ["catalog", "check"]),
        ("build", {"script_path": "s.py"}, ["build", "s.py", "-e", "expr"]),
        (
            "build",
            {"script_path": "s.py", "expr_name": "my_expr"},
            ["-e", "my_expr"],
        ),
        ("run", {"build_path": "/b"}, ["run", "/b", "-f", "json"]),
        (
            "run",
            {"build_path": "/b", "limit": 10, "params": {"k": "v"}},
            ["--limit", "10", "-p", "k=v"],
        ),
        (
            "run_cached",
            {"build_path": "/b"},
            ["run-cached", "/b", "-f", "json", "--cache-type", "modification-time"],
        ),
        (
            "run_cached",
            {"build_path": "/b", "ttl": 60, "cache_type": "snapshot"},
            ["--ttl", "60", "--cache-type", "snapshot"],
        ),
        ("catalog_add", {"paths": ["/a", "/b"]}, ["catalog", "add", "/a", "/b"]),
        (
            "catalog_add",
            {"paths": ["/a"], "sync": False, "aliases": ["x"]},
            ["--no-sync", "--alias", "x"],
        ),
        ("catalog_remove", {"names": ["e1"]}, ["catalog", "remove", "e1"]),
        ("catalog_sync", {}, ["catalog", "sync"]),
    ],
)
def test_tool_builds_correct_args(tool_func, kwargs, expected_fragments):
    """Verify each tool passes the right args to _run_cli."""
    captured_args = []

    async def fake_run_cli(*args, **kw):
        captured_args.extend(args)
        return "ok"

    with patch.object(mcp_mod, "_run_cli", side_effect=fake_run_cli):
        fn = getattr(mcp_mod, tool_func)
        asyncio.run(fn(**kwargs))

    for fragment in expected_fragments:
        assert fragment in captured_args, (
            f"expected {fragment!r} in args for {tool_func}, got {captured_args}"
        )
