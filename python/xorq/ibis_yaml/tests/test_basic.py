import datetime
import decimal

import pytest

import xorq.vendor.ibis as ibis
from xorq.ibis_yaml.tests.conftest import get_dtype_yaml


def test_unbound_table(t, compiler):
    yaml_dict = compiler.to_yaml(t)
    node_ref = yaml_dict["expression"]["node_ref"]
    expression = yaml_dict["definitions"]["nodes"][node_ref]
    assert expression["op"] == "UnboundTable"
    assert expression["name"] == "test_table"
    assert expression["schema_ref"]

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == t.schema()
    assert roundtrip_expr.op().name == t.op().name


def test_field(t, compiler):
    expr = t.a
    yaml_dict = compiler.to_yaml(expr)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Field"
    assert expression["name"] == "a"
    assert get_dtype_yaml(yaml_dict, expression) == {
        "op": "DataType",
        "type": "Int64",
        "nullable": {"op": "bool", "value": True},
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
    assert roundtrip_expr.get_name() == expr.get_name()


def test_literal(compiler):
    lit = ibis.literal(42)
    yaml_dict = compiler.to_yaml(lit)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    assert expression["value"] == 42
    dtype_yaml = yaml_dict["definitions"]["dtypes"][expression["type"]["dtype_ref"]]
    assert dtype_yaml == {
        "op": "DataType",
        "type": "Int8",
        "nullable": {"op": "bool", "value": True},
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)


def test_binary_op(t, compiler):
    expr = t.a + 1
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Add"
    assert expression["right"]["op"] == "Literal"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


@pytest.mark.parametrize(
    "primitive",
    [
        (ibis.literal(True), "Boolean"),
        (ibis.literal(1), "Int8"),
        (ibis.literal(1000), "Int16"),
        (ibis.literal(1.0), "Float64"),
        (ibis.literal("hello"), "String"),
        (ibis.literal(None), "Null"),
    ],
)
def test_primitive_types(compiler, primitive):
    lit, expected_type = primitive
    yaml_dict = compiler.to_yaml(lit)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    assert get_dtype_yaml(yaml_dict, expression)["type"] == expected_type

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().name == lit.type().name


def test_temporal_types(compiler):
    now = datetime.datetime.now()
    today = datetime.date.today()
    time = datetime.time(12, 0)
    temporals = [
        (ibis.literal(now), "Timestamp"),
        (ibis.literal(today), "Date"),
        (ibis.literal(time), "Time"),
    ]
    for lit, expected_type in temporals:
        yaml_dict = compiler.to_yaml(lit)
        expression = yaml_dict["expression"]
        assert expression["op"] == "Literal"
        assert get_dtype_yaml(yaml_dict, expression)["type"] == expected_type

        roundtrip_expr = compiler.from_yaml(yaml_dict)
        assert roundtrip_expr.equals(lit)
        assert roundtrip_expr.type().name == lit.type().name


def test_decimal_type(compiler):
    dec = decimal.Decimal("123.45")
    lit = ibis.literal(dec)
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml["type"] == "Decimal"
    assert dtype_yaml["nullable"]

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().name == lit.type().name


def test_array_type(compiler):
    lit = ibis.literal([1, 2, 3])
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    inner = get_dtype_yaml(yaml_dict, dtype_yaml, key="value_type")
    assert expression["op"] == "Literal"
    assert expression["value"] == (1, 2, 3)
    assert dtype_yaml["type"] == "Array"
    assert inner["type"] == "Int8"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().value_type == lit.type().value_type


def test_map_type(compiler):
    lit = ibis.literal({"a": 1, "b": 2})
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    key_dtype_yaml = get_dtype_yaml(yaml_dict, dtype_yaml, key="key_type")
    value_dtype_yaml = get_dtype_yaml(yaml_dict, dtype_yaml, key="value_type")
    assert expression["op"] == "Literal"
    assert dtype_yaml["type"] == "Map"
    assert key_dtype_yaml["type"] == "String"
    assert value_dtype_yaml["type"] == "Int8"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().key_type == lit.type().key_type
    assert roundtrip_expr.type().value_type == lit.type().value_type


def test_complex_expression_roundtrip(t, compiler):
    expr = (t.a + 1).abs() * 2
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_window_function_roundtrip(t, compiler):
    expr = t.a.sum().over(ibis.window(group_by=t.a))
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_join_roundtrip(t, compiler):
    t2 = ibis.table({"b": "int64"}, name="test_table_2")
    expr = t.join(t2, t.a == t2.b)
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == expr.schema()


def test_aggregation_roundtrip(t, compiler):
    expr = t.group_by(t.a).aggregate(count=t.a.count())
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == expr.schema()


def test_aggregation_with_alias_roundtrip(t, compiler):
    expr = t.group_by(t.a.name("cust_id")).aggregate(count=t.a.count())
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == expr.schema()


def test_table_perserves_namespace(compiler):
    expr = ibis.table({"b": "int64"}, name="t2", database="db", catalog="catalog")
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.op().namespace == expr.op().namespace
