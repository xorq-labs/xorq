import pandas as pd
import pyarrow as pa
from datafusion import udwf
from datafusion.user_defined import WindowEvaluator

import xorq as xo
from xorq.expr.relations import into_backend


class SmoothTwoColumn(WindowEvaluator):
    """Smooth once column based on a condition of another column.

    If the second column is above a threshold, then smooth over the first column from
    the previous and next rows.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def evaluate_all(self, values: list[pa.Array], num_rows: int) -> pa.Array:
        results = []
        values_a = values[0]
        values_b = values[1]
        for idx in range(num_rows):
            if not values_b[idx].is_valid:
                if idx == 0:
                    results.append(values_a[1].cast(pa.float64()))
                elif idx == num_rows - 1:
                    results.append(values_a[num_rows - 2].cast(pa.float64()))
                else:
                    results.append(
                        pa.scalar(
                            values_a[idx - 1].as_py() * self.alpha
                            + values_a[idx + 1].as_py() * (1.0 - self.alpha)
                        )
                    )
            else:
                results.append(values_a[idx].cast(pa.float64()))

        return pa.array(results)


def test_cross_engine_analytics(con, ddb_con):
    # register udwf in bare datafusion
    context = con.con
    smooth_two_col = udwf(
        lambda: SmoothTwoColumn(0.9),
        [pa.float64(), pa.int64()],
        pa.float64(),
        volatility="immutable",
        name="smooth_two_col",
    )
    context.register_udwf(smooth_two_col)

    # create new table with the new computed column
    query = "select trades.*, smooth_two_col(trades.price, trades.volume) over (partition by trades.symbol) as vwap from trades"
    vwap_df = con.raw_sql(query).to_pandas()
    trades = con.create_table("vwap", vwap_df, temp=False).pipe(into_backend, ddb_con)

    # Perform asof join in DuckDB
    asof_result = (
        ddb_con.table("quotes")
        .asof_join(trades, predicates=["symbol"], on="timestamp")
        .pipe(xo.execute)
    )

    assert isinstance(asof_result, pd.DataFrame)
    assert len(asof_result) > 0
