import ibis


def test_equals(compiler):
    a = ibis.literal(5)
    b = ibis.literal(5)
    expr = a == b
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Equals"
    assert yaml_dict["args"][0]["value"] == 5
    assert yaml_dict["args"][1]["value"] == 5
    assert yaml_dict["type"] == {"name": "Boolean", "nullable": True}
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_not_equals(compiler):
    a = ibis.literal(5)
    b = ibis.literal(3)
    expr = a != b
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "NotEquals"
    assert yaml_dict["args"][0]["value"] == 5
    assert yaml_dict["args"][1]["value"] == 3
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_greater_than(compiler):
    a = ibis.literal(5)
    b = ibis.literal(3)
    expr = a > b
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Greater"
    assert yaml_dict["args"][0]["value"] == 5
    assert yaml_dict["args"][1]["value"] == 3
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_less_than(compiler):
    a = ibis.literal(3)
    b = ibis.literal(5)
    expr = a < b
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Less"
    assert yaml_dict["args"][0]["value"] == 3
    assert yaml_dict["args"][1]["value"] == 5
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_and_or(compiler):
    a = ibis.literal(5)
    b = ibis.literal(3)
    c = ibis.literal(10)

    expr_and = (a > b) & (a < c)
    yaml_dict = compiler.compile_to_yaml(expr_and)
    assert yaml_dict["op"] == "And"
    assert yaml_dict["args"][0]["op"] == "Greater"
    assert yaml_dict["args"][1]["op"] == "Less"
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr_and)

    expr_or = (a > b) | (a < c)
    yaml_dict = compiler.compile_to_yaml(expr_or)
    assert yaml_dict["op"] == "Or"
    assert yaml_dict["args"][0]["op"] == "Greater"
    assert yaml_dict["args"][1]["op"] == "Less"
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr_or)


def test_not(compiler):
    a = ibis.literal(True)
    expr = ~a
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Not"
    assert yaml_dict["args"][0]["value"]
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_is_null(compiler):
    a = ibis.literal(None)
    expr = a.isnull()
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "IsNull"
    assert yaml_dict["args"][0]["value"] is None
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_between(compiler):
    a = ibis.literal(5)
    expr = a.between(3, 7)
    yaml_dict = compiler.compile_to_yaml(expr)
    assert yaml_dict["op"] == "Between"
    assert yaml_dict["args"][0]["value"] == 5
    assert yaml_dict["args"][1]["value"] == 3
    assert yaml_dict["args"][2]["value"] == 7
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
