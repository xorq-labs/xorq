import xorq.vendor.ibis as ibis
from xorq.ibis_yaml.common import (
    RefEnum,
    RegistryEnum,
)


def test_filter(compiler, t):
    expr = t.filter(t.a > 0)
    yaml_dict = compiler.to_yaml(expr)
    node_ref = yaml_dict["expression"][RefEnum.node_ref]

    expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]

    assert expression["op"] == "Filter"
    assert expression["predicates"][0]["op"] == "Greater"
    parent_ref = expression["parent"][RefEnum.node_ref]
    parent = yaml_dict["definitions"][RegistryEnum.nodes][parent_ref]
    assert parent["op"] == "UnboundTable"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_projection(compiler, t):
    expr = t.select(["a", "b"])
    yaml_dict = compiler.to_yaml(expr)
    node_ref = yaml_dict["expression"][RefEnum.node_ref]

    expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]

    assert expression["op"] == "Project"
    parent_ref = expression["parent"][RefEnum.node_ref]
    parent = yaml_dict["definitions"][RegistryEnum.nodes][parent_ref]
    assert parent["op"] == "UnboundTable"
    assert set(expression["values"]) == {"a", "b"}

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_aggregation(compiler, t):
    expr = t.group_by("a").aggregate(avg_c=t.c.mean())
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Aggregate"
    assert expression["by"]["a"]["op"] == "Field"
    assert expression["by"]["a"]["name"] == "a"
    assert expression["metrics"]["avg_c"]["op"] == "Mean"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_join(compiler):
    t1 = ibis.table(dict(a="int", b="string"), name="t1")
    t2 = ibis.table(dict(b="string", c="float"), name="t2")
    expr = t1.join(t2, t1.b == t2.b)
    yaml_dict = compiler.to_yaml(expr)
    node_ref = yaml_dict["expression"][RefEnum.node_ref]
    expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]

    assert expression["op"] == "JoinChain"
    assert expression["rest"][0]["predicates"][0]["op"] == "Equals"
    assert expression["rest"][0]["how"] == "inner"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_order_by(compiler, t):
    expr = t.order_by(["a", "b"])
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Sort"
    assert len(expression["keys"]) == 2

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_limit(compiler, t):
    expr = t.limit(10)
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Limit"
    assert expression["n"] == 10
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
