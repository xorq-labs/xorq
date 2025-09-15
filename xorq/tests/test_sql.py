import pytest

import xorq.api as xo


@pytest.mark.parametrize("file_format", ["parquet", "csv"])
def test_sql_on_deferred_read(file_format, request):
    ddb = xo.duckdb.connect()
    diamonds_path = (
        request.getfixturevalue(f"{file_format}_dir") / f"diamonds.{file_format}"
    )

    deferred_read = getattr(xo, f"deferred_read_{file_format}")

    expr = (
        deferred_read(diamonds_path, con=ddb, table_name="diamonds_ddb")
        .limit(2)
        .select(xo._.y, xo._.z, xo._.cut)
        .sql("SELECT cut, avg(y + z) as c FROM diamonds_ddb group by cut")
    )

    assert not expr.execute().empty
