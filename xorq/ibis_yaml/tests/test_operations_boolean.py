import xorq.vendor.ibis as ibis


def test_equals(compiler):
    a = ibis.literal(5)
    b = ibis.literal(5)
    expr = a == b
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Equals"
    assert expression["args"][0]["value"] == 5
    assert expression["args"][1]["value"] == 5
    assert expression["type"] == {
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
    assert expression["args"][0]["value"] == 5
    assert expression["args"][1]["value"] == 3
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_greater_than(compiler):
    a = ibis.literal(5)
    b = ibis.literal(3)
    expr = a > b
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Greater"
    assert expression["args"][0]["value"] == 5
    assert expression["args"][1]["value"] == 3
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_less_than(compiler):
    a = ibis.literal(3)
    b = ibis.literal(5)
    expr = a < b
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Less"
    assert expression["args"][0]["value"] == 3
    assert expression["args"][1]["value"] == 5
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
    assert expression["args"][0]["op"] == "Greater"
    assert expression["args"][1]["op"] == "Less"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr_and)

    expr_or = (a > b) | (a < c)
    yaml_dict = compiler.to_yaml(expr_or)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Or"
    assert expression["args"][0]["op"] == "Greater"
    assert expression["args"][1]["op"] == "Less"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr_or)


def test_not(compiler):
    a = ibis.literal(True)
    expr = ~a
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Not"
    assert expression["args"][0]["value"]
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_is_null(compiler):
    a = ibis.literal(None)
    expr = a.isnull()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "IsNull"
    assert expression["args"][0]["value"] is None
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_is_inf(compiler):
    a = ibis.literal(float("inf"))
    expr = a.isinf()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "IsInf"
    assert expression["args"][0]["value"] == float("inf")
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_is_nan(compiler):
    from math import isnan

    a = ibis.literal(float("nan"))
    expr = a.isnan()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "IsNan"
    isnan(expression["args"][0]["value"])
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_between(compiler):
    a = ibis.literal(5)
    expr = a.between(3, 7)
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Between"
    assert expression["args"][0]["value"] == 5
    assert expression["args"][1]["value"] == 3
    assert expression["args"][2]["value"] == 7
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
