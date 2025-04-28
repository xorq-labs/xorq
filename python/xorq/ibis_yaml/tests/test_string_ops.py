import xorq.vendor.ibis as ibis


def test_string_concat(compiler):
    s1 = ibis.literal("hello")
    s2 = ibis.literal("world")
    expr = s1 + s2
    yaml_dict = compiler.to_yaml(expr)["expression"]

    assert yaml_dict["op"] == "StringConcat"
    assert yaml_dict["args"][0]["value"] == "hello"
    assert yaml_dict["args"][1]["value"] == "world"
    assert yaml_dict["type"] == {
        "op": "DataType",
        "type": "String",
        "nullable": {"op": "bool", "value": True},
    }


def test_string_upper_lower(compiler):
    s = ibis.literal("Hello")
    upper_expr = s.upper()
    lower_expr = s.lower()

    upper_yaml = compiler.to_yaml(upper_expr)["expression"]
    assert upper_yaml["op"] == "Uppercase"
    assert upper_yaml["args"][0]["value"] == "Hello"

    lower_yaml = compiler.to_yaml(lower_expr)["expression"]
    assert lower_yaml["op"] == "Lowercase"
    assert lower_yaml["args"][0]["value"] == "Hello"


def test_string_length(compiler):
    s = ibis.literal("hello")
    expr = s.length()
    yaml_dict = compiler.to_yaml(expr)["expression"]

    assert yaml_dict["op"] == "StringLength"
    assert yaml_dict["args"][0]["value"] == "hello"
    assert yaml_dict["type"] == {
        "op": "DataType",
        "type": "Int32",
        "nullable": {"op": "bool", "value": True},
    }


def test_string_substring(compiler):
    s = ibis.literal("hello world")
    expr = s.substr(0, 5)
    yaml_dict = compiler.to_yaml(expr)["expression"]

    assert yaml_dict["op"] == "Substring"
    assert yaml_dict["args"][0]["value"] == "hello world"
    assert yaml_dict["args"][1]["value"] == 0
    assert yaml_dict["args"][2]["value"] == 5
