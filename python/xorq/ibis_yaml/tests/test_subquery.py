import xorq.vendor.ibis.expr.operations as ops
from xorq.ibis_yaml.common import (
    RefEnum,
    RegistryEnum,
)
from xorq.ibis_yaml.tests.conftest import get_dtype_yaml


def test_scalar_subquery(compiler, t):
    expr = ops.ScalarSubquery(t.c.mean().as_table()).to_expr()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "ScalarSubquery"
    node_ref = expression["rel"][RefEnum.node_ref]
    agg_expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]
    assert agg_expression["op"] == "Aggregate"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_exists_subquery(compiler, con):
    t1 = con.create_table("t1", schema={"a": "int32", "b": "string"}, overwrite=True)
    t2 = con.create_table("t2", schema={"a": "int32", "c": "float32"}, overwrite=True)

    filtered = t2.filter(t2.a == t1.a)
    expr = ops.ExistsSubquery(filtered).to_expr()
    yaml_dict = compiler.to_yaml(expr)

    expression = yaml_dict["expression"]

    assert expression["op"] == "ExistsSubquery"
    node_ref = expression["rel"][RefEnum.node_ref]
    assert yaml_dict["definitions"][RegistryEnum.nodes][node_ref]["op"] == "Filter"

    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)
    assert roundtrip_expr.equals(expr)


def test_in_subquery(compiler, con):
    t1 = con.create_table("t1", schema={"a": "int32", "b": "string"}, overwrite=True)
    t2 = con.create_table("t2", schema={"a": "int32", "c": "float32"}, overwrite=True)

    expr = ops.InSubquery(t1.select("a"), t2.a).to_expr()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)

    assert expression["op"] == "InSubquery"
    assert dtype_yaml["type"] == "Boolean"

    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)
    assert roundtrip_expr.equals(expr)
