import ibis


def test_explicit_cast(compiler):
    expr = ibis.literal(42).cast("float64")
    yaml_dict = compiler.compile_to_yaml(expr)

    assert yaml_dict["op"] == "Cast"
    assert yaml_dict["args"][0]["op"] == "Literal"
    assert yaml_dict["args"][0]["value"] == 42
    assert yaml_dict["type"]["name"] == "Float64"

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_implicit_cast(compiler):
    i = ibis.literal(1)
    f = ibis.literal(2.5)
    expr = i + f
    yaml_dict = compiler.compile_to_yaml(expr)

    assert yaml_dict["op"] == "Add"
    assert yaml_dict["args"][0]["type"]["name"] == "Int8"
    assert yaml_dict["args"][1]["type"]["name"] == "Float64"
    assert yaml_dict["type"]["name"] == "Float64"

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_string_cast(compiler):
    expr = ibis.literal("42").cast("int64")
    yaml_dict = compiler.compile_to_yaml(expr)

    assert yaml_dict["op"] == "Cast"
    assert yaml_dict["args"][0]["value"] == "42"
    assert yaml_dict["type"]["name"] == "Int64"

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_timestamp_cast(compiler):
    expr = ibis.literal("2024-01-01").cast("timestamp")
    yaml_dict = compiler.compile_to_yaml(expr)

    assert yaml_dict["op"] == "Cast"
    assert yaml_dict["args"][0]["value"] == "2024-01-01"
    assert yaml_dict["type"]["name"] == "Timestamp"

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
