"""Multi-root (`-e NAME=ENTRY`) join mode for catalog run/compose/run-cached.

These tests exercise the join surface added on top of the linear
source+transforms chain: binding parsing/validation, same-backend joins
(which stay native after same-profile rebinding), provenance capture, the
`into_backend` transport helper, the rebuild boundary, and uv-reinvoke
forwarding of `-e` flags.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd
import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.catalog import cli as cli_mod
from xorq.catalog.bind import _rebind_same_profile_backends, compose_join
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import (
    _parse_entry_bindings,
    _resolve_mode,
    cli,
)
from xorq.catalog.composer import ExprComposer
from xorq.expr.relations import into_backend
from xorq.ibis_yaml.enums import DumpFiles


@pytest.fixture
def two_source_catalog(tmp_path: Path) -> Catalog:
    """A plain-git catalog with three independent memtable source entries.

    Memtable-backed entries share the process let-backend, so a bare
    `left.join(right, ...)` resolves to a single backend (no transport).
    """
    cat = Catalog.from_repo_path(tmp_path / "repo", init=True, annex=False)
    cat.add(
        xo.memtable({"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}, name="sales"),
        aliases=("sales",),
    )
    cat.add(
        xo.memtable({"id": [1, 2, 3], "name": ["a", "b", "c"]}, name="customers"),
        aliases=("customers",),
    )
    cat.add(
        xo.memtable({"id": [1, 2, 3], "region": ["x", "y", "z"]}, name="regions"),
        aliases=("regions",),
    )
    return cat


# --- _parse_entry_bindings: pure validation ---


def test_parse_entry_bindings_empty() -> None:
    assert _parse_entry_bindings(()) == {}


def test_parse_entry_bindings_basic_ordered() -> None:
    bindings = _parse_entry_bindings(("left=sales", "right=customers"))
    assert bindings == {"left": "sales", "right": "customers"}
    # insertion order preserved (first binding = default target)
    assert list(bindings) == ["left", "right"]


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        pytest.param("no-equals", "NAME=ENTRY", id="no-equals"),
        pytest.param("=customers", "NAME=ENTRY", id="empty-name"),
        pytest.param("left=", "NAME=ENTRY", id="empty-entry"),
        pytest.param("1bad=sales", "identifier", id="non-identifier-digit"),
        pytest.param("with-dash=sales", "identifier", id="non-identifier-dash"),
        pytest.param("class=sales", "keyword", id="keyword"),
        pytest.param("__x__=sales", "dunder", id="dunder"),
        pytest.param("xo=sales", "reserved", id="reserved-xo"),
        pytest.param("ibis=sales", "reserved", id="reserved-ibis"),
        pytest.param("into_backend=sales", "reserved", id="reserved-into-backend"),
    ],
)
def test_parse_entry_bindings_rejects(raw: str, needle: str) -> None:
    with pytest.raises(click.BadParameter, match=needle):
        _parse_entry_bindings((raw,))


def test_parse_entry_bindings_rejects_duplicate() -> None:
    with pytest.raises(click.BadParameter, match="duplicate"):
        _parse_entry_bindings(("left=sales", "left=customers"))


# --- _resolve_mode ---


def test_resolve_mode_linear_returns_empty() -> None:
    assert _resolve_mode(("src", "trn"), (), None) == {}


def test_resolve_mode_rejects_mixing() -> None:
    with pytest.raises(click.UsageError, match="Cannot combine"):
        _resolve_mode(("customers",), ("left=sales",), "left")


def test_resolve_mode_requires_code() -> None:
    with pytest.raises(click.UsageError, match="requires inline code"):
        _resolve_mode((), ("left=sales",), None)


# --- compose_join: same-backend semantics + provenance ---


def test_compose_join_same_backend_executes(two_source_catalog: Catalog) -> None:
    expr = compose_join(
        two_source_catalog,
        {"left": "sales", "right": "customers"},
        "left.join(right, 'id')",
    )
    df = expr.ls.fused.execute().sort_values("id").reset_index(drop=True)
    assert list(df["id"]) == [1, 2, 3]
    assert set(df.columns) >= {"id", "amount", "name"}


def test_compose_join_provenance_captures_all_operands(
    two_source_catalog: Catalog,
) -> None:
    expr = compose_join(
        two_source_catalog,
        {"left": "sales", "right": "customers"},
        "left.join(right, 'id')",
    )
    composed = expr.ls.metadata.composed_from
    aliases = {c["alias"] for c in composed}
    assert aliases == {"left", "right"}
    # entry_name records the real catalog entry (alias resolved), not the binding
    entry_names = {c["entry_name"] for c in composed}
    sales_name = two_source_catalog.get_catalog_entry("sales", maybe_alias=True).name
    assert sales_name in entry_names


def test_compose_join_three_way(two_source_catalog: Catalog) -> None:
    expr = compose_join(
        two_source_catalog,
        {"a": "sales", "b": "customers", "c": "regions"},
        "a.join(b, 'id').join(c, 'id')",
    )
    composed = expr.ls.metadata.composed_from
    assert {c["alias"] for c in composed} == {"a", "b", "c"}
    df = expr.ls.fused.execute()
    assert set(df.columns) >= {"id", "amount", "name", "region"}


def test_compose_join_rebind_off_same_object(two_source_catalog: Catalog) -> None:
    # memtable entries share one backend object, so no-rebind still resolves.
    expr = compose_join(
        two_source_catalog,
        {"left": "sales", "right": "customers"},
        "left.join(right, 'id')",
        rebind_backends=False,
    )
    assert len(expr.ls.fused.execute()) == 3


def test_rebind_preserves_explicit_into_backend_transport() -> None:
    # Two distinct same-profile backends where each side is an in-connection
    # table that only lives on its own connection. An explicit into_backend
    # transport must survive automatic rebind: rebind only touches top-level
    # sources, not the operand moved behind the RemoteTable transport, so it
    # must NOT raise the transfer_tables=False error and must stay one backend.
    con_a = xo.duckdb.connect()
    con_b = xo.duckdb.connect()
    a = con_a.create_table("a", pd.DataFrame({"id": [1, 2, 3], "x": [4, 5, 6]}))
    b = con_b.create_table("b", pd.DataFrame({"id": [1, 2, 3], "y": [7, 8, 9]}))
    expr = a.join(into_backend(b, con_a), "id")
    resolved = {"a": (a, con_a), "b": (b, con_b)}

    out = _rebind_same_profile_backends(expr, resolved, rebind=True)
    assert len(out._find_backends()[0]) == 1


def test_compose_join_into_backend_helper_accepts_bound_entry(
    two_source_catalog: Catalog,
) -> None:
    # into_backend(right, left): resolves left's backend from the bound entry.
    expr = compose_join(
        two_source_catalog,
        {"left": "sales", "right": "customers"},
        "left.join(into_backend(right, left), 'id')",
    )
    assert len(expr.execute()) == 3


def test_compose_join_requires_bindings(two_source_catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="at least one"):
        compose_join(two_source_catalog, {}, "x")


def test_compose_join_requires_code(two_source_catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="requires inline"):
        compose_join(two_source_catalog, {"left": "sales"}, None)


# --- rebuild boundary: ExprComposer cannot recover a multi-source join ---


def test_exprcomposer_from_expr_rejects_multi_source(
    two_source_catalog: Catalog,
) -> None:
    expr = compose_join(
        two_source_catalog,
        {"left": "sales", "right": "customers"},
        "left.join(right, 'id')",
    )
    with pytest.raises(ValueError, match="multi-root join"):
        ExprComposer.from_expr(expr, two_source_catalog)


# --- CLI: run / compose / run-cached in multi-root mode ---


def test_cli_run_multi_root_join(
    runner: CliRunner, two_source_catalog: Catalog
) -> None:
    cp = str(two_source_catalog.repo_path)
    result = runner.invoke(
        cli,
        [
            "--path",
            cp,
            "run",
            "-e",
            "left=sales",
            "-e",
            "right=customers",
            "-c",
            "left.join(right, 'id')",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 4  # header + 3 rows


def test_cli_compose_multi_root_catalogs(
    runner: CliRunner, two_source_catalog: Catalog
) -> None:
    cp = str(two_source_catalog.repo_path)
    result = runner.invoke(
        cli,
        [
            "--path",
            cp,
            "compose",
            "-e",
            "left=sales",
            "-e",
            "right=customers",
            "-c",
            "left.join(right, 'id')",
            "-a",
            "joined",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as 'joined'" in result.output
    reopened = Catalog.from_kwargs(path=cp, init=False)
    assert reopened.catalog_yaml.contains_alias("joined")


def test_cli_compose_multi_root_dry_run(
    runner: CliRunner, two_source_catalog: Catalog
) -> None:
    cp = str(two_source_catalog.repo_path)
    result = runner.invoke(
        cli,
        [
            "--path",
            cp,
            "compose",
            "-e",
            "left=sales",
            "-e",
            "right=customers",
            "-c",
            "left.join(right, 'id')",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Bindings:" in result.output
    assert "left=sales" in result.output
    assert "Cataloged" not in result.output


def test_cli_run_cached_multi_root(
    runner: CliRunner, two_source_catalog: Catalog, tmp_path: Path
) -> None:
    cp = str(two_source_catalog.repo_path)
    result = runner.invoke(
        cli,
        [
            "--path",
            cp,
            "run-cached",
            "-e",
            "left=sales",
            "-e",
            "right=customers",
            "-c",
            "left.join(right, 'id')",
            "--cache-dir",
            str(tmp_path / "cache"),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cache").exists()


def test_cli_run_reject_mixing(runner: CliRunner, two_source_catalog: Catalog) -> None:
    cp = str(two_source_catalog.repo_path)
    result = runner.invoke(
        cli,
        ["--path", cp, "run", "-e", "left=sales", "customers", "-c", "left"],
    )
    assert result.exit_code != 0
    assert "Cannot combine" in result.output


def test_cli_run_reject_no_code(runner: CliRunner, two_source_catalog: Catalog) -> None:
    cp = str(two_source_catalog.repo_path)
    result = runner.invoke(cli, ["--path", cp, "run", "-e", "left=sales"])
    assert result.exit_code != 0
    assert "requires inline code" in result.output


def test_cli_run_no_rebind_backends_flag(
    runner: CliRunner, two_source_catalog: Catalog
) -> None:
    # memtable entries share one backend object, so --no-rebind-backends still
    # resolves (nothing distinct to collapse) and the join executes.
    cp = str(two_source_catalog.repo_path)
    result = runner.invoke(
        cli,
        [
            "--path",
            cp,
            "run",
            "--use-this-venv",
            "--no-rebind-backends",
            "-e",
            "left=sales",
            "-e",
            "right=customers",
            "-c",
            "left.join(right, 'id')",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(result.output.strip().splitlines()) == 4


# --- uv-reinvoke: -e flags forwarded, wheels harvested from bound entries ---


def _fake_uv_tool_run(captured: dict) -> Callable:
    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return fake


def test_cli_run_multi_root_reinvokes_with_entry_flags(
    two_source_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --use-this-venv, multi-root run must re-emit the -e bindings
    into the inner argv and harvest wheels from the bound entries."""
    captured: dict = {}
    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run", _fake_uv_tool_run(captured)
    )
    cp = str(two_source_catalog.repo_path)
    result = CliRunner().invoke(
        cli,
        [
            "--path",
            cp,
            "run",
            "-e",
            "left=sales",
            "-e",
            "right=customers",
            "-c",
            "left.join(right, 'id')",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert "run" in args
    assert "--entry" in args
    assert "left=sales" in args and "right=customers" in args
    assert "--use-this-venv" in args
    # wheels harvested from the bound entries
    assert any(str(w).endswith(".whl") for w in captured["kwargs"]["with_"])


def test_cli_compose_multi_root_reinvokes(
    two_source_catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        idx = args.index("--emit-build-path-to")
        pre_built = tmp_path / "fake-build"
        pre_built.mkdir(exist_ok=True)
        (pre_built / DumpFiles.requirements).write_text("")
        pathlib.Path(args[idx + 1]).write_text(str(pre_built))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake)
    monkeypatch.setattr(
        cli_mod, "_stage_bundle_into_build", lambda bundle, build_path: None
    )
    monkeypatch.setattr(Catalog, "add", lambda self, *a, **k: SimpleNamespace())

    cp = str(two_source_catalog.repo_path)
    result = CliRunner().invoke(
        cli,
        [
            "--path",
            cp,
            "compose",
            "-e",
            "left=sales",
            "-e",
            "right=customers",
            "-c",
            "left.join(right, 'id')",
        ],
    )
    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert "compose" in args
    assert "--entry" in args and "left=sales" in args
    assert "--use-this-venv" in args
