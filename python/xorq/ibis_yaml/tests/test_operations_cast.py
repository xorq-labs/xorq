import xorq.vendor.ibis as ibis


# NB: Cast serializes its target dtype under the `to` constructor arg as an
# inline ibis string (the write-only result `type:` field was dropped). Implicit
# casts (Add) have no `to`; their result dtype is re-inferred on load.


def test_explicit_cast(compiler):
    expr = ibis.literal(42).cast("float64")
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Cast"
    assert expression["arg"]["op"] == "Literal"
    assert expression["arg"]["value"] == 42
    assert expression["to"] == "float64"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_implicit_cast(compiler):
    i = ibis.literal(1)
    f = ibis.literal(2.5)
    expr = i + f
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Add"
    assert expression["left"]["type"] == "int8"
    assert expression["right"]["type"] == "float64"
    assert expr.type().name == "Float64"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_string_cast(compiler):
    expr = ibis.literal("42").cast("int64")
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Cast"
    assert expression["arg"]["value"] == "42"
    assert expression["to"] == "int64"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_timestamp_cast(compiler):
    expr = ibis.literal("2024-01-01").cast("timestamp")
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Cast"
    assert expression["arg"]["value"] == "2024-01-01"
    assert expression["to"] == "timestamp"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
