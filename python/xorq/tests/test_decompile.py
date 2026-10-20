from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
from xorq.expr.relations import flight_udxf
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.decompile import decompile


def run(code: str) -> ibis.Expr:
    namespace = {}
    exec(code, namespace)
    return namespace["result"]


def double_x(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(x2=df["x"] * 2)


@pytest.fixture
def table() -> ibis.Table:
    return ibis.table(name="t", schema={"x": "int64", "y": "string"})


@pytest.mark.parametrize(
    "name",
    (
        pytest.param(name, id=name)
        for name in ("row_number", "rank", "dense_rank", "percent_rank", "cume_dist")
    ),
)
def test_zero_arg_window_function(table: ibis.Table, name: str) -> None:
    expr = table.mutate(rn=getattr(ibis, name)())
    assert f"ibis.{name}()" in decompile(expr)


@pytest.mark.parametrize(
    "make_expr, call",
    (
        pytest.param(
            lambda t: t.into_backend(xo.duckdb.connect(), "moved"),
            ".into_backend(xo.duckdb.connect(), 'moved')",
            id="into_backend",
        ),
        pytest.param(
            lambda t: t.cache(xo.SourceCache.from_kwargs(source=xo.connect())),
            ".cache(xo.SourceCache.from_kwargs(source=xo.connect()))",
            id="cache",
        ),
    ),
)
def test_upstream_of_xorq_relation_is_kept(
    table: ibis.Table, make_expr: Callable[[ibis.Table], ibis.Table], call: str
) -> None:
    code = decompile(make_expr(table.filter(table.x > 1)))
    assert "t.x > 1" in code
    assert call in code
    assert decompile(run(code)) == code


def test_deferred_read_runs(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("x,y\n1,a\n2,b\n")
    expr = xo.deferred_read_csv(path, xo.duckdb.connect(), "t")
    code = decompile(expr)
    assert f"xo.deferred_read_csv({str(path)!r}, xo.duckdb.connect(), 't')" in code
    assert run(code).execute().equals(expr.execute())


@pytest.mark.parametrize(
    "built", (pytest.param(False, id="live"), pytest.param(True, id="built"))
)
def test_flight_udxf_emits_process_df(tmp_path: Path, built: bool) -> None:
    con = xo.connect()
    expr = flight_udxf(
        con.create_table("t", {"x": [1, 2]}),
        process_df=double_x,
        maybe_schema_in=xo.schema({"x": "int64"}),
        maybe_schema_out=xo.schema({"x": "int64", "x2": "int64"}),
        name="doubled",
    )
    if built:
        expr = xo.load_expr(xo.build_expr(expr, builds_dir=tmp_path))
    code = decompile(expr)
    assert "def double_x(df: pd.DataFrame) -> pd.DataFrame:" in code
    assert "make_udxf(double_x, " in code
    assert ".into_backend(xo.connect(), " in code
    compile(code, "<decompiled>", "exec")
