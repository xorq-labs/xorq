import datetime
import decimal

import ibis


def test_unbound_table(t, compiler):
    yaml_dict = compiler.compile_to_yaml(t)
    assert yaml_dict["op"] == "UnboundTable"
    assert yaml_dict["name"] == "test_table"
    assert yaml_dict["schema"]["a"] == {"name": "Int64", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == t.schema()
    assert roundtrip_expr.op().name == t.op().name


def test_field(t, compiler):
    expr = t.a
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Field"
    assert yaml_dict["name"] == "a"
    assert yaml_dict["type"] == {"name": "Int64", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
    assert roundtrip_expr.get_name() == expr.get_name()


def test_literal(compiler):
    lit = ibis.literal(42)
    yaml_dict = compiler.compile_to_yaml(lit)
    assert yaml_dict["op"] == "Literal"
    assert yaml_dict["value"] == 42
    assert yaml_dict["type"] == {"name": "Int8", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)


def test_binary_op(t, compiler):
    expr = t.a + 1
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Add"
    assert yaml_dict["args"][0]["op"] == "Field"
    assert yaml_dict["args"][1]["op"] == "Literal"

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_primitive_types(compiler):
    primitives = [
        (ibis.literal(True), "Boolean"),
        (ibis.literal(1), "Int8"),
        (ibis.literal(1000), "Int16"),
        (ibis.literal(1.0), "Float64"),
        (ibis.literal("hello"), "String"),
        (ibis.literal(None), "Null"),
    ]
    for lit, expected_type in primitives:
        yaml_dict = compiler.compile_to_yaml(lit)
        assert yaml_dict["op"] == "Literal"
        assert yaml_dict["type"]["name"] == expected_type

        roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
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
        yaml_dict = compiler.compile_to_yaml(lit)
        assert yaml_dict["op"] == "Literal"
        assert yaml_dict["type"]["name"] == expected_type

        roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
        assert roundtrip_expr.equals(lit)
        assert roundtrip_expr.type().name == lit.type().name


def test_decimal_type(compiler):
    dec = decimal.Decimal("123.45")
    lit = ibis.literal(dec)
    yaml_dict = compiler.compile_to_yaml(lit)
    assert yaml_dict["op"] == "Literal"
    assert yaml_dict["type"]["name"] == "Decimal"
    assert yaml_dict["type"]["nullable"]

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().name == lit.type().name


def test_array_type(compiler):
    lit = ibis.literal([1, 2, 3])
    yaml_dict = compiler.compile_to_yaml(lit)
    assert yaml_dict["op"] == "Literal"
    assert yaml_dict["type"]["name"] == "Array"
    assert yaml_dict["type"]["value_type"]["name"] == "Int8"
    assert yaml_dict["value"] == (1, 2, 3)

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().value_type == lit.type().value_type


def test_map_type(compiler):
    lit = ibis.literal({"a": 1, "b": 2})
    yaml_dict = compiler.compile_to_yaml(lit)
    assert yaml_dict["op"] == "Literal"
    assert yaml_dict["type"]["name"] == "Map"
    assert yaml_dict["type"]["key_type"]["name"] == "String"
    assert yaml_dict["type"]["value_type"]["name"] == "Int8"

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().key_type == lit.type().key_type
    assert roundtrip_expr.type().value_type == lit.type().value_type


def test_complex_expression_roundtrip(t, compiler):
    expr = (t.a + 1).abs() * 2
    yaml_dict = compiler.compile_to_yaml(expr)
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_window_function_roundtrip(t, compiler):
    expr = t.a.sum().over(ibis.window(group_by=t.a))
    yaml_dict = compiler.compile_to_yaml(expr)
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_join_roundtrip(t, compiler):
    t2 = ibis.table({"b": "int64"}, name="test_table_2")
    expr = t.join(t2, t.a == t2.b)
    yaml_dict = compiler.compile_to_yaml(expr)
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == expr.schema()


def test_aggregation_roundtrip(t, compiler):
    expr = t.group_by(t.a).aggregate(count=t.a.count())
    yaml_dict = compiler.compile_to_yaml(expr)
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == expr.schema()
