import pandas as pd
import pytest


# Minimal test mimicking h07 join chain with self-reference in projection
def test_minimal_joinchain_self_reference(
    compiler, con, orders, supplier, lineitem, customer, nation
):
    q = supplier.join(lineitem, supplier.s_suppkey == lineitem.l_suppkey)
    q = q.join(orders, orders.o_orderkey == lineitem.l_orderkey)
    q = q.join(customer, customer.c_custkey == orders.o_custkey)
    n1 = nation
    n2 = nation.view()
    q = q.join(n1, supplier.s_nationkey == n1.n_nationkey)
    q = q.join(n2, customer.c_nationkey == n2.n_nationkey)
    q = q.projection(
        {
            "supp_nation": n1.n_name,
            "cust_nation": n2.n_name,
            "l_shipdate": lineitem.l_shipdate,
            "l_extendedprice": lineitem.l_extendedprice,
            "l_discount": lineitem.l_discount,
        }
    )
    q = q.filter(
        (
            ((q.cust_nation == "FRANCE") & (q.supp_nation == "GERMANY"))
            | ((q.cust_nation == "GERMANY") & (q.supp_nation == "FRANCE"))
        )
    )

    yaml_dict = compiler.to_yaml(q)
    profiles = {con._profile.hash_name: con}
    q_roundtrip = compiler.from_yaml(yaml_dict, profiles)

    try:
        _ = q_roundtrip["cust_nation"]
    except Exception as e:
        pytest.fail(f"Accessing 'cust_nation' on the roundtrip query failed: {e}")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"on": "t"},
        {"on": "t", "predicates": "k"},
        {"on": "t", "predicates": "k", "tolerance": 1},
    ],
)
def test_asof_join_roundtrip(compiler, con, kwargs):
    left = con.create_table(
        "asof_left", pd.DataFrame({"k": [1, 1], "t": [1, 5]}), overwrite=True
    )
    right = con.create_table(
        "asof_right",
        pd.DataFrame({"k": [1, 1], "t": [0, 3], "v": [10, 30]}),
        overwrite=True,
    )
    expr = left.asof_join(right, **kwargs)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip = compiler.from_yaml(yaml_dict, {con._profile.hash_name: con})

    pd.testing.assert_frame_equal(
        roundtrip.execute().sort_values("t").reset_index(drop=True),
        expr.execute().sort_values("t").reset_index(drop=True),
    )
