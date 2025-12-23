import xorq.vendor.ibis as ibis
from xorq.ibis_yaml.tests.conftest import get_dtype_yaml


def test_add(compiler):
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 + lit2

    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Add"
    assert expression["left"]["op"] == "Literal"
    assert expression["left"]["value"] == 5
    assert expression["right"]["op"] == "Literal"
    assert expression["right"]["value"] == 3
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml == {
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
    assert expression["left"]["op"] == "Literal"
    assert expression["left"]["value"] == 5
    assert expression["right"]["op"] == "Literal"
    assert expression["right"]["value"] == 3
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml == {
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
    assert expression["left"]["op"] == "Literal"
    assert expression["left"]["value"] == 5
    assert expression["right"]["op"] == "Literal"
    assert expression["right"]["value"] == 3
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml == {
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
    assert expression["left"]["op"] == "Literal"
    assert expression["left"]["value"] == 6.0
    assert expression["right"]["op"] == "Literal"
    assert expression["right"]["value"] == 2.0
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml == {
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
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml == {
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
    assert expression["left"]["op"] == "Add"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
