import xorq.vendor.ibis as ibis
from xorq.ibis_yaml.tests.conftest import get_dtype_yaml


def test_explicit_cast(compiler):
    expr = ibis.literal(42).cast("float64")
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)

    assert expression["op"] == "Cast"
    assert expression["arg"]["op"] == "Literal"
    assert expression["arg"]["value"] == 42
    assert dtype_yaml["type"] == "Float64"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_implicit_cast(compiler):
    i = ibis.literal(1)
    f = ibis.literal(2.5)
    expr = i + f
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)

    assert expression["op"] == "Add"
    assert get_dtype_yaml(yaml_dict, expression["left"])["type"] == "Int8"
    assert get_dtype_yaml(yaml_dict, expression["right"])["type"] == "Float64"
    assert dtype_yaml["type"] == "Float64"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_string_cast(compiler):
    expr = ibis.literal("42").cast("int64")
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)

    assert expression["op"] == "Cast"
    assert expression["arg"]["value"] == "42"
    assert dtype_yaml["type"] == "Int64"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_timestamp_cast(compiler):
    expr = ibis.literal("2024-01-01").cast("timestamp")
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)

    assert expression["op"] == "Cast"
    assert expression["arg"]["value"] == "2024-01-01"
    assert dtype_yaml["type"] == "Timestamp"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
