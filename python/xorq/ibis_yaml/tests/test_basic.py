import datetime
import decimal

import pytest

import xorq.vendor.ibis as ibis
from xorq.ibis_yaml.enums import RefEnum, RegistryEnum


def test_unbound_table(t, compiler):
    yaml_dict = compiler.to_yaml(t)
    node_ref = yaml_dict["expression"][RefEnum.node_ref]
    expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]
    assert expression["op"] == "UnboundTable"
    assert expression["name"] == "test_table"
    assert expression[RefEnum.schema_ref]

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == t.schema()
    assert roundtrip_expr.op().name == t.op().name


def test_field(t, compiler):
    expr = t.a
    yaml_dict = compiler.to_yaml(expr)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Field"
    assert expression["name"] == "a"
    # Field no longer serializes its (computed) dtype; it is re-inferred from the
    # relation schema on load.
    assert "type" not in expression
    assert expr.type().name == "Int64"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
    assert roundtrip_expr.get_name() == expr.get_name()


def test_literal(compiler):
    lit = ibis.literal(42)
    yaml_dict = compiler.to_yaml(lit)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    assert expression["value"] == 42
    # dtypes are inlined as their canonical ibis string
    assert expression["type"] == "int8"
    assert expression["type"] == str(lit.type())

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
    assert expression["type"] == str(lit.type())
    assert lit.type().name == expected_type

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
        assert expression["type"] == str(lit.type())
        assert lit.type().name == expected_type

        roundtrip_expr = compiler.from_yaml(yaml_dict)
        assert roundtrip_expr.equals(lit)
        assert roundtrip_expr.type().name == lit.type().name


def test_decimal_type(compiler):
    dec = decimal.Decimal("123.45")
    lit = ibis.literal(dec)
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    # parametric dtypes inline as their full ibis string, e.g. "decimal(5, 2)"
    assert expression["type"] == str(lit.type())
    assert expression["type"].startswith("decimal")
    assert not expression["type"].startswith("!")  # nullable

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().name == lit.type().name


def test_array_type(compiler):
    lit = ibis.literal([1, 2, 3])
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    assert expression["value"] == (1, 2, 3)
    assert expression["type"] == str(lit.type())
    assert expression["type"] == "array<int8>"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().value_type == lit.type().value_type


def test_map_type(compiler):
    lit = ibis.literal({"a": 1, "b": 2})
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    assert expression["type"] == str(lit.type())
    assert expression["type"] == "map<string, int8>"

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


def test_join_roundtrip(t, compiler, con):
    t2 = con.create_table("test_table_2", schema={"b": "int64"}, overwrite=True)
    expr = t.join(t2, t.a == t2.b)
    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)
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
