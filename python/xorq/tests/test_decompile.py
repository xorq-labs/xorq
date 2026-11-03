from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
from xorq.common.utils.file_utils import normalize_read_source_identity
from xorq.expr.relations import Read, flight_expr, flight_udxf
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.decompile import decompile


def run(code: str) -> ibis.Expr:
    namespace = {}
    exec(code, namespace)
    return namespace["result"]


def double_x(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(x2=df["x"] * 2)


def udxf_expr(process_df: Callable, **kwargs: object) -> ibis.Table:
    return flight_udxf(
        xo.connect().create_table("t", {"x": [1, 2]}),
        process_df=process_df,
        maybe_schema_in=xo.schema({"x": "int64"}),
        maybe_schema_out=xo.schema({"x": "int64", "x2": "int64"}),
        **kwargs,
    )


@pytest.fixture
def table() -> ibis.Table:
    return ibis.table(name="t", schema={"x": "int64", "y": "string"})


@pytest.mark.parametrize(
    "make_expr",
    (
        pytest.param(lambda t: t.mutate(rn=ibis.row_number()), id="row_number"),
        pytest.param(lambda t: t.mutate(rn=ibis.rank()), id="rank"),
        pytest.param(lambda t: t.mutate(rn=ibis.dense_rank()), id="dense_rank"),
        pytest.param(lambda t: t.mutate(rn=ibis.percent_rank()), id="percent_rank"),
        pytest.param(lambda t: t.mutate(rn=ibis.cume_dist()), id="cume_dist"),
        pytest.param(
            lambda t: t.mutate(s=t.x.sum().over(group_by=t.y, order_by=t.x)),
            id="group_order",
        ),
        pytest.param(
            lambda t: t.mutate(s=t.x.sum().over(rows=(-2, 0), order_by=t.x)),
            id="rows_frame",
        ),
    ),
)
def test_window_function_round_trips(
    table: ibis.Table, make_expr: Callable[[ibis.Table], ibis.Table]
) -> None:
    expr = make_expr(table)
    assert run(decompile(expr)).equals(expr)


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
    path.write_text("x,y\n1,001\n")
    schema = xo.schema({"x": "int64", "y": "string"})
    expr = xo.deferred_read_csv(path, xo.duckdb.connect(), "t", schema)
    code = decompile(expr)
    assert f"xo.deferred_read_csv({str(path)!r}, xo.duckdb.connect(), 't'" in code
    assert run(code).execute().equals(expr.execute())


@pytest.mark.parametrize(
    "make_expr",
    (
        pytest.param(
            lambda: Read(
                method_name="fetch_resource",
                name="t",
                schema=xo.schema({"x": "int64"}),
                source=xo.connect(),
                read_kwargs=(("resource", "things"),),
                normalize_method=normalize_read_source_identity,
            ).to_expr(),
            id="pathless_read",
        ),
        pytest.param(
            lambda: flight_expr(
                xo.connect().create_table("u", {"x": [1]}),
                ibis.table({"x": "int64"}, name="u").filter(ibis._.x > 0),
                inner_name="t",
            ),
            id="flight_expr",
        ),
    ),
)
def test_relation_without_handler_renders_as_table(
    make_expr: Callable[[], ibis.Table],
) -> None:
    assert "ibis.table(name='t'" in decompile(make_expr())


@pytest.mark.parametrize(
    "built", (pytest.param(False, id="live"), pytest.param(True, id="built"))
)
def test_flight_udxf_emits_process_df(tmp_path: Path, built: bool) -> None:
    expr = udxf_expr(double_x, name="doubled")
    if built:
        expr = xo.load_expr(xo.build_expr(expr, builds_dir=tmp_path))
    code = decompile(expr)
    assert "def double_x(df: pd.DataFrame) -> pd.DataFrame:" in code
    assert "make_udxf(double_x, " in code
    assert ".into_backend(xo.connect(), " in code
    compile(code, "<decompiled>", "exec")


@pytest.mark.parametrize(
    "process_df, make_udxf_kwargs, call",
    (
        pytest.param(
            double_x, (("name", "custom"),), "make_udxf(double_x, ", id="renamed"
        ),
        pytest.param(lambda df: df, (), "make_udxf(process_df, ", id="lambda"),
    ),
)
def test_flight_udxf_names_process_df(
    process_df: Callable, make_udxf_kwargs: tuple, call: str
) -> None:
    code = decompile(udxf_expr(process_df, make_udxf_kwargs=make_udxf_kwargs))
    assert call in code
    compile(code, "<decompiled>", "exec")
