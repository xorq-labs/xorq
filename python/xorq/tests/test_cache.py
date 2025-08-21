import xorq as xo
from xorq.caching import ParquetStorage, SourceStorage
from xorq.tests.util import assert_frame_equal


def test_cross_source_storage(pg):
    name = "astronauts"
    expr = (
        xo.duckdb.connect()
        .create_table(name, pg.table(name).to_pyarrow())[lambda t: t.number > 22]
        .cache(storage=SourceStorage(source=pg))
    )
    expr.execute()


def test_caching_of_registered_arbitrary_expression(con, pg, tmp_path):
    table_name = "batting"
    t = pg.table(table_name)

    expr = t.filter(t.playerID == "allisar01")[
        ["playerID", "yearID", "stint", "teamID", "lgID"]
    ]
    expected = expr.execute()

    result = expr.cache(
        storage=ParquetStorage(source=con, relative_path=tmp_path)
    ).execute()

    assert result is not None
    assert_frame_equal(result, expected, check_like=True)


def test_cache_record_batch_provider_exec(pg):
    batches = pg.table("batting").to_pyarrow_batches()
    t = (ls_con := xo.connect()).register(batches, table_name="batting_batches")
    storage = SourceStorage(source=ls_con)

    assert storage.get_key(t) is not None
