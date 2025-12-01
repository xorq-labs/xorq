import pytest
from pytest import param

import xorq.api as xo
from xorq.tests.util import assert_frame_equal


@pytest.fixture(scope="function")
def con():
    return xo.connect()


def test_fill_null_with_dict(compiler, t):
    expr = t.fill_null({"a": 0, "b": "unknown"})
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "FillNull"
    assert "a" in expression["replacements"]
    assert "b" in expression["replacements"]

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(xo.null().fill_null(5), 5, id="na_fill_null"),
        param(xo.literal(5).fill_null(10), 5, id="non_na_fill_null"),
    ],
)
def test_scalar_fill_null_nullif(con, expr, expected, compiler):
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert con.execute(roundtrip_expr) == expected


@pytest.mark.parametrize(
    "replacements",
    [
        param({"int_col": 20}, id="int"),
        param({"double_col": -1, "string_col": "missing"}, id="double-int-str"),
        param({"double_col": -1.5, "string_col": "missing"}, id="double-str"),
        param({}, id="empty"),
    ],
)
def test_table_fill_null_mapping(replacements, con, compiler, parquet_dir):
    alltypes = xo.deferred_read_parquet(
        parquet_dir / "functional_alltypes.parquet", con=con
    )

    table = alltypes.mutate(
        int_col=alltypes.int_col.nullif(1),
        double_col=alltypes.double_col.nullif(3.0),
        string_col=alltypes.string_col.nullif("2"),
    ).select("id", "int_col", "double_col", "string_col")

    profiles = {
        con._profile.hash_name: con,
    }

    expr = table.fill_null(replacements)
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute().reset_index(drop=True)
    pd_table = table.execute()
    expected = pd_table.fillna(replacements).reset_index(drop=True)

    assert_frame_equal(result, expected, check_dtype=False)


def test_table_unnest(con, compiler):
    t = con.create_table("test", {"x": [[1, 2, 3], [4, 5], [6]]})

    expr = t.unnest("x")

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "TableUnnest"
    assert "parent" in expression
    assert "column" in expression
    assert "column_name" in expression
    assert "keep_empty" in expression

    profiles = {
        con._profile.hash_name: con,
    }
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute().reset_index(drop=True)
    expected = expr.execute().reset_index(drop=True)

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("offset", "keep_empty"),
    [
        param(None, False, id="no_offset_no_keep_empty"),
        param("idx", False, id="with_offset_no_keep_empty"),
        param(None, True, id="no_offset_with_keep_empty"),
    ],
)
def test_table_unnest_options(con, compiler, offset, keep_empty):
    t = con.create_table("test", {"x": [[1, 2], [], [3]]})

    expr = t.unnest("x", offset=offset, keep_empty=keep_empty)

    yaml_dict = compiler.to_yaml(expr)

    expression = yaml_dict["expression"]
    assert expression["op"] == "TableUnnest"

    roundtrip_expr = compiler.from_yaml(
        yaml_dict,
        profiles={
            con._profile.hash_name: con,
        },
    )

    assert_frame_equal(
        con.execute(roundtrip_expr).reset_index(drop=True),
        expr.execute().reset_index(drop=True),
    )
