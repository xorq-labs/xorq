import xorq.vendor.ibis as ibis


def test_add(compiler):
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 + lit2

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Add"
    assert expression["args"][0]["op"] == "Literal"
    assert expression["args"][0]["value"] == 5
    assert expression["args"][1]["op"] == "Literal"
    assert expression["args"][1]["value"] == 3
    assert expression["type"] == {
        "op": "DataType",
        "type": "Int8",
        "nullable": {"op": "bool", "value": True},
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_subtract(compiler):
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 - lit2

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Subtract"
    assert expression["args"][0]["op"] == "Literal"
    assert expression["args"][0]["value"] == 5
    assert expression["args"][1]["op"] == "Literal"
    assert expression["args"][1]["value"] == 3
    assert expression["type"] == {
        "op": "DataType",
        "type": "Int8",
        "nullable": {"op": "bool", "value": True},
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_multiply(compiler):
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 * lit2

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Multiply"
    assert expression["args"][0]["op"] == "Literal"
    assert expression["args"][0]["value"] == 5
    assert expression["args"][1]["op"] == "Literal"
    assert expression["args"][1]["value"] == 3
    assert expression["type"] == {
        "op": "DataType",
        "type": "Int8",
        "nullable": {"op": "bool", "value": True},
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_divide(compiler):
    lit1 = ibis.literal(6.0)
    lit2 = ibis.literal(2.0)
    expr = lit1 / lit2

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Divide"
    assert expression["args"][0]["op"] == "Literal"
    assert expression["args"][0]["value"] == 6.0
    assert expression["args"][1]["op"] == "Literal"
    assert expression["args"][1]["value"] == 2.0
    assert expression["type"] == {
        "op": "DataType",
        "type": "Float64",
        "nullable": {"op": "bool", "value": True},
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_mixed_arithmetic(compiler):
    i = ibis.literal(5)
    f = ibis.literal(2.5)
    expr = i * f

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Multiply"
    assert expression["type"] == {
        "op": "DataType",
        "type": "Float64",
        "nullable": {"op": "bool", "value": True},
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_complex_arithmetic(compiler):
    a = ibis.literal(10)
    b = ibis.literal(5)
    c = ibis.literal(2.0)
    expr = (a + b) * c

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Multiply"
    assert expression["args"][0]["op"] == "Add"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
