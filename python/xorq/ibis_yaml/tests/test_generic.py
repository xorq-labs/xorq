import pytest
from pytest import param

import xorq.api as xo
from xorq.tests.util import assert_frame_equal


@pytest.fixture(scope="session")
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
