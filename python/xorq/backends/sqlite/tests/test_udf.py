def test_hash(sqlite_con, df):
    t = sqlite_con.create_table("test", df)
    expr = t.mutate(my_hash=t.c.hash())

    assert not expr.execute().empty


def test_hash_pinned_values(sqlite_con, snapshot):
    # Pin the sqlite Hash op (ibis_hash_32 udf, a 32-bit blake2b digest) so an
    # unintended algorithm/encoding change is caught. Regenerate deliberately
    # with --snapshot-update; a diff here is a reviewable behavioral break.
    import json  # noqa: PLC0415

    import xorq.api as xo  # noqa: PLC0415

    inputs = ["", "a", "hello", "0", "123"]
    t = xo.memtable({"c": inputs}).into_backend(sqlite_con)
    df = t.mutate(h=t.c.hash()).execute()
    actual = json.dumps(
        {c: int(h) for c, h in zip(df["c"], df["h"])},
        indent=2,
        sort_keys=True,
    )
    snapshot.assert_match(actual, "ibis_hash_32.json")
