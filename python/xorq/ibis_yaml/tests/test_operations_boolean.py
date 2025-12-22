import xorq.vendor.ibis as ibis
from xorq.ibis_yaml.tests.conftest import get_dtype_yaml


def test_equals(compiler):
    a = ibis.literal(5)
    b = ibis.literal(5)
    expr = a == b
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Equals"
    assert expression["left"]["value"] == 5
    assert expression["right"]["value"] == 5
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml == {
        "op": "DataType",
        "type": "Boolean",
        "nullable": {"op": "bool", "value": True},
    }
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_not_equals(compiler):
    a = ibis.literal(5)
    b = ibis.literal(3)
    expr = a != b
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "NotEquals"
    assert expression["left"]["value"] == 5
    assert expression["right"]["value"] == 3
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_greater_than(compiler):
    a = ibis.literal(5)
    b = ibis.literal(3)
    expr = a > b
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Greater"
    assert expression["left"]["value"] == 5
    assert expression["right"]["value"] == 3
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_less_than(compiler):
    a = ibis.literal(3)
    b = ibis.literal(5)
    expr = a < b
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Less"
    assert expression["left"]["value"] == 3
    assert expression["right"]["value"] == 5
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_and_or(compiler):
    a = ibis.literal(5)
    b = ibis.literal(3)
    c = ibis.literal(10)

    expr_and = (a > b) & (a < c)
    yaml_dict = compiler.to_yaml(expr_and)
    expression = yaml_dict["expression"]
    assert expression["op"] == "And"
    assert expression["left"]["op"] == "Greater"
    assert expression["right"]["op"] == "Less"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr_and)

    expr_or = (a > b) | (a < c)
    yaml_dict = compiler.to_yaml(expr_or)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Or"
    assert expression["left"]["op"] == "Greater"
    assert expression["right"]["op"] == "Less"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr_or)


def test_not(compiler):
    a = ibis.literal(True)
    expr = ~a
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Not"
    assert expression["arg"]["value"]
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_is_null(compiler):
    a = ibis.literal(None)
    expr = a.isnull()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "IsNull"
    assert expression["arg"]["value"] is None
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_is_inf(compiler):
    a = ibis.literal(float("inf"))
    expr = a.isinf()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "IsInf"
    assert expression["arg"]["value"] == float("inf")
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_is_nan(compiler):
    from math import isnan

    a = ibis.literal(float("nan"))
    expr = a.isnan()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "IsNan"
    isnan(expression["arg"]["value"])
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_between(compiler):
    a = ibis.literal(5)
    expr = a.between(3, 7)
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Between"
    assert expression["arg"]["value"] == 5
    assert expression["lower_bound"]["value"] == 3
    assert expression["upper_bound"]["value"] == 7
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
