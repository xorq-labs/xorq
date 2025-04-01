from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import rich.console
from pytest import param

import xorq as xo
from xorq.tests.util import assert_frame_equal
from xorq.vendor import ibis


if TYPE_CHECKING:
    pass


@pytest.fixture
def new_schema():
    return xo.schema([("a", "string"), ("b", "bool"), ("c", "int32")])


def _create_temp_table_with_schema(con, temp_table_name, schema, data=None):
    temporary = con.create_table(temp_table_name, schema=schema)
    assert temporary.to_pandas().empty

    if data is not None and isinstance(data, pd.DataFrame):
        assert not data.empty
        tmp = con.create_table(temp_table_name, data, overwrite=True)
        result = tmp.to_pandas()
        assert len(result) == len(data.index)
        tm.assert_frame_equal(
            result.sort_values(result.columns[0]).reset_index(drop=True),
            data.sort_values(result.columns[0]).reset_index(drop=True),
        )
        return tmp

    return temporary


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(
            xo.memtable([(1, 2.0, "3")], columns=list("abc")),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")),
            id="simple",
        ),
        param(
            xo.memtable([(1, 2.0, "3")]),
            pd.DataFrame([(1, 2.0, "3")], columns=["col0", "col1", "col2"]),
            id="simple_auto_named",
        ),
        param(
            xo.memtable(
                [(1, 2.0, "3")],
                schema=xo.schema(dict(a="int8", b="float32", c="string")),
            ),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")).astype(
                {"a": "int8", "b": "float32"}
            ),
            id="simple_schema",
        ),
        param(
            xo.memtable(
                pd.DataFrame({"a": [1], "b": [2.0], "c": ["3"]}).astype(
                    {"a": "int8", "b": "float32"}
                )
            ),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")).astype(
                {"a": "int8", "b": "float32"}
            ),
            id="dataframe",
        ),
        param(
            xo.memtable([dict(a=1), dict(a=2)]),
            pd.DataFrame({"a": [1, 2]}),
            id="list_of_dicts",
        ),
    ],
)
def test_in_memory_table(con, expr, expected):
    result = con.execute(expr)
    assert_frame_equal(result, expected)


def test_filter_memory_table(con):
    t = xo.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.filter(t.x > 1)
    expected = pd.DataFrame({"x": [3, 5], "y": [4, 6]})
    result = con.execute(expr)
    assert_frame_equal(result, expected)


def test_agg_memory_table(con):
    t = xo.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.x.count()
    result = con.execute(expr)
    assert result == 3


def test_self_join_memory_table(con):
    t = xo.memtable({"x": [1, 2], "y": [2, 1], "z": ["a", "b"]})
    t_view = t.view()
    expr = t.join(t_view, t.x == t_view.y).select("x", "y", "z", "z_right")
    result = con.execute(expr).sort_values("x").reset_index(drop=True)
    expected = pd.DataFrame(
        {"x": [1, 2], "y": [2, 1], "z": ["a", "b"], "z_right": ["b", "a"]}
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, "f8"])
def test_dunder_array_table(alltypes, dtype):
    expr = alltypes.group_by("string_col").int_col.sum().order_by("string_col")
    result = np.asarray(expr, dtype=dtype)
    expected = np.asarray(expr.execute(), dtype=dtype)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, "f8"])
def test_dunder_array_column(alltypes, dtype):
    from xorq.vendor.ibis import _

    expr = alltypes.group_by("string_col").agg(int_col=_.int_col.sum()).int_col
    result = np.sort(np.asarray(expr, dtype=dtype))
    expected = np.sort(np.asarray(xo.execute(expr), dtype=dtype))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.xfail
@pytest.mark.parametrize("interactive", [True, False])
def test_repr(alltypes, interactive, monkeypatch):
    monkeypatch.setattr(xo.options, "interactive", interactive)

    expr = alltypes.select("date_string_col")

    s = repr(expr)
    # no control characters
    assert all(c.isprintable() or c in "\n\r\t" for c in s)
    if interactive:
        assert "/" in s
    else:
        assert "/" not in s


@pytest.mark.xfail
@pytest.mark.parametrize("show_types", [True, False])
def test_interactive_repr_show_types(alltypes, show_types, monkeypatch):
    monkeypatch.setattr(xo.options, "interactive", True)
    monkeypatch.setattr(xo.options.repr.interactive, "show_types", show_types)

    expr = alltypes.select("id")
    s = repr(expr)
    if show_types:
        assert "int" in s
    else:
        assert "int" not in s


@pytest.mark.xfail
@pytest.mark.parametrize("is_jupyter", [True, False])
def test_interactive_repr_max_columns(alltypes, is_jupyter, monkeypatch):
    monkeypatch.setattr(xo.options, "interactive", True)

    cols = {f"c_{i}": ibis._.id + i for i in range(50)}
    expr = alltypes.mutate(**cols).select(*cols)

    console = rich.console.Console(force_jupyter=is_jupyter, width=80)
    options = console.options.copy()

    # max_columns = 0
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_0 " in text
    if is_jupyter:
        # All columns are written
        assert " c_49 " in text
    else:
        # width calculations truncate well before 20 columns
        assert " c_19 " not in text

    # max_columns = 3
    monkeypatch.setattr(xo.options.repr.interactive, "max_columns", 3)
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_2 " in text
    assert " c_3 " not in text

    # max_columns = None
    monkeypatch.setattr(xo.options.repr.interactive, "max_columns", None)
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_0 " in text
    if is_jupyter:
        # All columns written
        assert " c_49 " in text
    else:
        # width calculations still truncate
        assert " c_19 " not in text


@pytest.mark.xfail
@pytest.mark.parametrize("expr_type", ["table", "column"])
@pytest.mark.parametrize("interactive", [True, False])
def test_repr_mimebundle(alltypes, interactive, expr_type, monkeypatch):
    monkeypatch.setattr(xo.options, "interactive", interactive)

    if expr_type == "column":
        expr = alltypes.date_string_col
    else:
        expr = alltypes.select("date_string_col")

    reprs = expr._repr_mimebundle_(include=["text/plain", "text/html"], exclude=[])
    for fmt in ["text/plain", "text/html"]:
        if interactive:
            assert "r0.date_string_col" not in reprs[fmt]
        else:
            assert "r0.date_string_col" in reprs[fmt]


@pytest.mark.parametrize(
    "option", ["max_rows", "max_length", "max_string", "max_depth"]
)
def test_ibis_config_wrapper(option, monkeypatch):
    from xorq.vendor import ibis

    xorq_option_value = getattr(xo.options.repr.interactive, option)
    assert xorq_option_value == getattr(ibis.options.repr.interactive, option)

    monkeypatch.setattr(xo.options.repr.interactive, option, xorq_option_value + 1)
    assert getattr(ibis.options.repr.interactive, option) == xorq_option_value + 1
