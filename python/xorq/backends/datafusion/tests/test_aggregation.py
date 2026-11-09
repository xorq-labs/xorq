import xorq.api as xo


def test_group_by_key_shadowing_a_column_reads_the_column():
    con = xo.datafusion.connect()
    t = con.create_table("t", {"a": [1, 3, 2], "b": ["x", "x", "y"]})
    expr = t.group_by(a=t.a % 2).agg(s=t.a.sum(), n=t.a.nunique())
    result = expr.execute().sort_values("a").to_dict("records")
    assert result == [
        {"a": 0, "s": 2, "n": 1},
        {"a": 1, "s": 4, "n": 2},
    ]
