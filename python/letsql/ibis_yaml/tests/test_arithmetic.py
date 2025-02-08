import ibis


def test_add(compiler):
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 + lit2

    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Add"
    assert yaml_dict["args"][0]["op"] == "Literal"
    assert yaml_dict["args"][0]["value"] == 5
    assert yaml_dict["args"][1]["op"] == "Literal"
    assert yaml_dict["args"][1]["value"] == 3
    assert yaml_dict["type"] == {"name": "Int8", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_subtract(compiler):
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 - lit2

    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Subtract"
    assert yaml_dict["args"][0]["op"] == "Literal"
    assert yaml_dict["args"][0]["value"] == 5
    assert yaml_dict["args"][1]["op"] == "Literal"
    assert yaml_dict["args"][1]["value"] == 3
    assert yaml_dict["type"] == {"name": "Int8", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_multiply(compiler):
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 * lit2

    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Multiply"
    assert yaml_dict["args"][0]["op"] == "Literal"
    assert yaml_dict["args"][0]["value"] == 5
    assert yaml_dict["args"][1]["op"] == "Literal"
    assert yaml_dict["args"][1]["value"] == 3
    assert yaml_dict["type"] == {"name": "Int8", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_divide(compiler):
    lit1 = ibis.literal(6.0)
    lit2 = ibis.literal(2.0)
    expr = lit1 / lit2

    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Divide"
    assert yaml_dict["args"][0]["op"] == "Literal"
    assert yaml_dict["args"][0]["value"] == 6.0
    assert yaml_dict["args"][1]["op"] == "Literal"
    assert yaml_dict["args"][1]["value"] == 2.0
    assert yaml_dict["type"] == {"name": "Float64", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_mixed_arithmetic(compiler):
    i = ibis.literal(5)
    f = ibis.literal(2.5)
    expr = i * f

    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Multiply"
    assert yaml_dict["type"] == {"name": "Float64", "nullable": True}

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_complex_arithmetic(compiler):
    a = ibis.literal(10)
    b = ibis.literal(5)
    c = ibis.literal(2.0)
    expr = (a + b) * c

    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Multiply"
    assert yaml_dict["args"][0]["op"] == "Add"

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
