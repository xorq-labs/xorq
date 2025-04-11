from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest
from pytest import param

import xorq as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.tests.conftest import TEST_TABLES


def test_list_tables(con):
    tables = con.list_tables()
    assert isinstance(tables, list)
    key = "functional_alltypes"
    assert key in tables or key.upper() in tables
    assert all(isinstance(table, str) for table in tables)


def test_tables_accessor_mapping(con):
    if con.name == "snowflake":
        pytest.skip("snowflake sometimes counts more tables than are around")

    name = "functional_alltypes"

    assert isinstance(con.tables[name], ir.Table)

    with pytest.raises(KeyError, match="doesnt_exist"):
        con.tables["doesnt_exist"]

    # temporary might pop into existence in parallel test runs, in between the
    # first `list_tables` call and the second, so we check that there's a
    # non-empty intersection
    assert TEST_TABLES.keys() & set(map(str.lower, con.list_tables()))
    assert TEST_TABLES.keys() & set(map(str.lower, con.tables))


def test_tables_accessor_getattr(con):
    name = "functional_alltypes"
    assert isinstance(getattr(con.tables, name), ir.Table)

    with pytest.raises(AttributeError, match="doesnt_exist"):
        con.tables.doesnt_exist  # noqa: B018

    # Underscore/double-underscore attributes are never available, since many
    # python apis expect that checking for the absence of these to be cheap.
    with pytest.raises(AttributeError, match="_private_attr"):
        con.tables._private_attr  # noqa: B018


def test_tables_accessor_tab_completion(con):
    name = "functional_alltypes"
    attrs = dir(con.tables)
    assert name in attrs
    assert "keys" in attrs  # type methods also present

    keys = con.tables._ipython_key_completions_()
    assert name in keys


def test_tables_accessor_repr(con):
    name = "functional_alltypes"
    result = repr(con.tables)
    assert f"- {name}" in result


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t.limit(5).limit(10), id="small_big"),
        param(lambda t: t.limit(10).limit(5), id="big_small"),
    ],
)
def test_limit_chain(alltypes, expr_fn):
    expr = expr_fn(alltypes)
    result = expr.execute()
    assert len(result) == 5


@pytest.mark.xfail
@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t, id="alltypes table"),
        param(lambda t: t.join(t.view(), [("id", "int_col")]), id="self join"),
    ],
)
def test_unbind(alltypes, expr_fn: Callable):
    xo.options.interactive = False

    expr = expr_fn(alltypes)
    assert expr.unbind() != expr
    assert expr.unbind().schema() == expr.schema()

    assert "Unbound" not in repr(expr)
    assert "Unbound" in repr(expr.unbind())


@pytest.mark.parametrize(
    ("extension", "method"),
    [("parquet", xo.read_parquet), ("csv", xo.read_csv)],
)
def test_read(data_dir, extension, method):
    table = method(
        data_dir / extension / f"batting.{extension}", table_name=f"batting-{extension}"
    )
    assert table.execute() is not None


@pytest.mark.parametrize(
    ("extension", "write", "read"),
    [
        (
            "parquet",
            xo.to_parquet,
            lambda path, schema: xo.read_parquet(
                path, schema=schema.to_pyarrow()
            ).execute(),
        ),
        (
            "csv",
            xo.to_csv,
            lambda path, schema: xo.read_csv(
                path, schema=schema.to_pyarrow()
            ).execute(),
        ),
        (
            "json",
            xo.to_json,
            lambda path, schema: pd.read_json(
                path, lines=True, dtype=dict(schema.to_pandas())
            ),
        ),
    ],
)
def test_write(alltypes, df, tmp_path, extension, write, read):
    output_path = tmp_path / f"alltypes.{extension}"
    write(alltypes, output_path)
    actual = read(output_path, alltypes.schema())

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert isinstance(actual, pd.DataFrame)


@pytest.mark.parametrize(
    "method",
    [
        "deferred_read_csv",
        "deferred_read_parquet",
    ],
)
def test_deferred_read(method):
    assert hasattr(xo, method)
