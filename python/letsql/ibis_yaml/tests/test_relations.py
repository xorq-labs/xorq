import ibis


def test_filter(compiler, t):
    expr = t.filter(t.a > 0)
    yaml_dict = compiler.compile_to_yaml(expr)

    # Original assertions
    assert yaml_dict["op"] == "Filter"
    assert yaml_dict["predicates"][0]["op"] == "Greater"
    assert yaml_dict["parent"]["op"] == "UnboundTable"

    # Roundtrip test: compile from YAML and verify equality
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_projection(compiler, t):
    expr = t.select(["a", "b"])
    yaml_dict = compiler.compile_to_yaml(expr)

    # Original assertions
    assert yaml_dict["op"] == "Project"
    assert yaml_dict["parent"]["op"] == "UnboundTable"
    assert set(yaml_dict["values"]) == {"a", "b"}

    # Roundtrip test
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_aggregation(compiler, t):
    expr = t.group_by("a").aggregate(avg_c=t.c.mean())
    yaml_dict = compiler.compile_to_yaml(expr)

    # Original assertions
    assert yaml_dict["op"] == "Aggregate"
    assert yaml_dict["by"][0]["name"] == "a"
    assert yaml_dict["metrics"]["avg_c"]["op"] == "Mean"

    # Roundtrip test
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_join(compiler):
    t1 = ibis.table(dict(a="int", b="string"), name="t1")
    t2 = ibis.table(dict(b="string", c="float"), name="t2")
    expr = t1.join(t2, t1.b == t2.b)
    yaml_dict = compiler.compile_to_yaml(expr)

    # Original assertions
    assert yaml_dict["op"] == "JoinChain"
    # The first join link's predicates
    assert yaml_dict["rest"][0]["predicates"][0]["op"] == "Equals"
    assert yaml_dict["rest"][0]["how"] == "inner"

    # Roundtrip test
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_order_by(compiler, t):
    expr = t.order_by(["a", "b"])
    yaml_dict = compiler.compile_to_yaml(expr)

    # Original assertions
    assert yaml_dict["op"] == "Sort"
    assert len(yaml_dict["keys"]) == 2

    # Roundtrip test
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_limit(compiler, t):
    expr = t.limit(10)
    yaml_dict = compiler.compile_to_yaml(expr)

    # Original assertions
    assert yaml_dict["op"] == "Limit"
    assert yaml_dict["n"] == 10

    # Roundtrip test
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
