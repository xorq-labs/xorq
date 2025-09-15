import xorq.vendor.ibis as ibis


def test_selection_on_view(compiler):
    T = ibis.table({"id": "int32", "name": "string"}, name="T")
    T_view = T.view()
    q = T.join(T_view, T.id == T_view.id)
    q = q.select({"alias_name": T_view.name})
    q = q.filter(q.alias_name == "X")

    yaml_dict = compiler.to_yaml(q)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(q)
