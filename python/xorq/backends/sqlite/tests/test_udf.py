def test_hash(sqlite_con, df):
    t = sqlite_con.create_table("test", df)
    expr = t.mutate(my_hash=t.c.hash())

    assert not expr.execute().empty
