import pandas as pd
import pytest
from pytest import param

import xorq.api as xo
import xorq.vendor.ibis as ibis
from xorq.backends.snowflake.tests.conftest import inside_temp_schema
from xorq.tests.util import assert_series_equal


@pytest.mark.parametrize(
    ("start", "end", "unit", "expected"),
    [
        param(
            ibis.time("01:58:00"),
            ibis.time("23:59:59"),
            "hour",
            22,
            id="time",
        ),
        param(ibis.date("1992-09-30"), ibis.date("1992-10-01"), "day", 1, id="date"),
        param(
            ibis.timestamp("1992-09-30 23:59:59"),
            ibis.timestamp("1992-10-01 01:58:00"),
            "hour",
            2,
            id="timestamp",
        ),
    ],
)
def test_delta(sf_con, start, end, unit, expected):
    expr = end.delta(start, part=unit)
    assert sf_con.execute(expr) == expected


date_value = pd.Timestamp("2017-12-31")
timestamp_value = pd.Timestamp("2018-01-01 18:18:18")


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        # param(
        #     lambda t: t.timestamp_col - ibis.interval(days=17),
        #     lambda t: t.timestamp_col - pd.Timedelta(days=17),
        #     id="timestamp-subtract-interval",
        # ),
        # param(
        #     lambda t: t.timestamp_col.date() + ibis.interval(days=4),
        #     lambda t: t.timestamp_col.dt.floor("d").add(pd.Timedelta(days=4)),
        #     id="date-add-interval",
        # ),
        # param(
        #     lambda t: t.timestamp_col.date() - ibis.interval(days=14),
        #     lambda t: t.timestamp_col.dt.floor("d").sub(pd.Timedelta(days=14)),
        #     id="date-subtract-interval",
        # ),
        # param(
        #     lambda t: t.timestamp_col - ibis.timestamp(timestamp_value),
        #     lambda t: pd.Series(
        #         t.timestamp_col.sub(timestamp_value).values.astype("timedelta64[s]")
        #     ).dt.floor("s"),
        #     id="timestamp-subtract-timestamp",
        # ),
        param(
            lambda t: t.timestamp_col.date() - ibis.date(date_value),
            lambda t: pd.Series(
                (t.timestamp_col.dt.floor("d") - date_value).values.astype(
                    "timedelta64[D]"
                )
            ),
            id="date-subtract-date",
        ),
    ],
)
def test_temporal_binop(
    sf_con, temp_catalog, temp_db, parquet_dir, expr_fn, expected_fn
):
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        name = "functional_alltypes"
        alltypes = (
            xo.deferred_read_parquet(
                parquet_dir.joinpath(f"{name}.parquet"), table_name=name
            )
            .limit(1000)
            .into_backend(sf_con)
        )

        expr = expr_fn(alltypes).name("tmp")
        expected = expected_fn(alltypes.execute())

        result = expr.execute()
        expected = expected.rename("tmp")

        assert_series_equal(result, expected.astype(result.dtype), check_dtype=False)
