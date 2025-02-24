import pytest

import xorq.vendor.ibis as ibis


@pytest.fixture
def orders():
    return ibis.table(
        {
            "o_orderkey": "int32",
            "o_orderpriority": "string",
            "o_custkey": "int32",
            "o_orderdate": "date",
        },
        name="orders",
    )


@pytest.fixture
def supplier():
    return ibis.table(
        {
            "s_suppkey": "int32",
            "s_nationkey": "int32",
        },
        name="supplier",
    )


@pytest.fixture
def lineitem():
    return ibis.table(
        {
            "l_orderkey": "int64",
            "l_suppkey": "int32",
            "l_shipdate": "date",
            "l_extendedprice": "decimal(15,2)",
            "l_discount": "decimal(15,2)",
        },
        name="lineitem",
    )


@pytest.fixture
def customer():
    return ibis.table(
        {
            "c_custkey": "int32",
            "c_nationkey": "int32",
        },
        name="customer",
    )


@pytest.fixture
def nation():
    return ibis.table(
        {
            "n_nationkey": "int32",
            "n_name": "string",
        },
        name="nation",
    )


# Minimal test mimicking h07 join chain with self-reference in projection
def test_minimal_joinchain_self_reference(
    compiler, orders, supplier, lineitem, customer, nation
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
    q_roundtrip = compiler.from_yaml(yaml_dict)

    try:
        _ = q_roundtrip["cust_nation"]
    except Exception as e:
        pytest.fail(f"Accessing 'cust_nation' on the roundtrip query failed: {e}")
