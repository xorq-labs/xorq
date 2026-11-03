import sqlglot as sg

from xorq.expr.remote_table_exec import tables_under_rescans


def test_tables_under_rescans_resolves_each_cte_once():
    # a chain of forty CTEs, each a self-join of the one before it; resolving
    # a CTE once per reference would visit 2**40 nodes and never return
    depth = 40
    ctes = ["c0 AS (SELECT id, k, v FROM s)"] + [
        f"c{i} AS (SELECT a.id, a.k, a.v FROM c{i - 1} AS a JOIN c{i - 1} AS b ON a.id = b.id)"
        for i in range(1, depth)
    ]
    query = sg.parse_one(
        f"WITH {', '.join(ctes)} "
        f"SELECT SUM(v) OVER (PARTITION BY k) FROM c{depth - 1} JOIN r ON TRUE",
        dialect="duckdb",
    )
    assert tables_under_rescans(query) == {"s", "r"}
