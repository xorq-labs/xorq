import xorq.vendor.ibis as ibis


def test_string_concat(compiler):
    s1 = ibis.literal("hello")
    s2 = ibis.literal("world")
    expr = s1 + s2
    yaml_dict = compiler.to_yaml(expr)["expression"]

    assert yaml_dict["op"] == "StringConcat"
    assert yaml_dict["args"][0]["value"] == "hello"
    assert yaml_dict["args"][1]["value"] == "world"
    assert yaml_dict["type"] == {"name": "String", "nullable": True}


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


def test_string_to_date(compiler):
    value = "20170206"
    format_str = "%Y%m%d"
    s = ibis.literal(value)
    expr = s.as_date(format_str)

    expr_yaml = compiler.to_yaml(expr)
    yaml_dict = expr_yaml["expression"]
    assert yaml_dict["op"] == "StringToDate"
    assert yaml_dict["arg"]["value"] == value
    assert yaml_dict["format_str"]["value"] == format_str

    roundtrip = compiler.from_yaml(expr_yaml)
    assert roundtrip.equals(expr)


def test_string_length(compiler):
    s = ibis.literal("hello")
    expr = s.length()
    yaml_dict = compiler.to_yaml(expr)["expression"]

    assert yaml_dict["op"] == "StringLength"
    assert yaml_dict["args"][0]["value"] == "hello"
    assert yaml_dict["type"] == {"name": "Int32", "nullable": True}


def test_string_substring(compiler):
    s = ibis.literal("hello world")
    expr = s.substr(0, 5)
    yaml_dict = compiler.to_yaml(expr)["expression"]

    assert yaml_dict["op"] == "Substring"
    assert yaml_dict["args"][0]["value"] == "hello world"
    assert yaml_dict["args"][1]["value"] == 0
    assert yaml_dict["args"][2]["value"] == 5
