from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from xorq.common.utils.ibis_utils import from_ibis


if TYPE_CHECKING:
    from ibis.backends import BaseBackend


STRING_COLUMNS = ("origin", "destination", "flight_num", "tail_num")
TIMESTAMP_COLUMNS = ("dep_time", "arr_time")
INT_COLUMNS = ("flight_time", "dep_delay", "arr_delay", "taxi_out", "taxi_in", "id2")
BOOL_COLUMNS = ("cancelled", "diverted")


@pytest.mark.slow(level=3)
def test_complex_expr_from_boring_semantic_layer(ibis_con: BaseBackend) -> None:
    flights = ibis_con.create_table(
        "complex_flights",
        pa.table(
            {
                "carrier": ["AA", "AA", "UA"],
                "distance": [100, 300, 600],
                **{name: ["a", "b", "c"] for name in STRING_COLUMNS},
                **{
                    name: pa.array([0, 1, 2], type=pa.timestamp("us"))
                    for name in TIMESTAMP_COLUMNS
                },
                **{name: [0, 1, 2] for name in INT_COLUMNS},
                **{name: [False, True, False] for name in BOOL_COLUMNS},
            }
        ),
        overwrite=True,
    )
    carriers = ibis_con.create_table(
        "complex_carriers",
        pa.table(
            {
                "code": ["AA", "UA"],
                "name": ["American Airlines", "United Air Lines"],
                "nickname": ["American", "United"],
            }
        ),
        overwrite=True,
    )

    # Join flights with carrier information
    flights_with_carriers = flights.inner_join(
        carriers,
        (flights.carrier == carriers.code),
    ).select(
        flights.carrier,
        flights.origin,
        flights.destination,
        flights.flight_num,
        flights.flight_time,
        flights.tail_num,
        flights.dep_time,
        flights.arr_time,
        flights.dep_delay,
        flights.arr_delay,
        flights.taxi_out,
        flights.taxi_in,
        flights.distance,
        flights.cancelled,
        flights.diverted,
        flights.id2,
        carriers.code,
        carriers.name,
        carriers.nickname,
    )

    # Calculate statistics by carrier
    distance_sum_expr = flights_with_carriers.distance.sum()
    carrier_stats = flights_with_carriers.aggregate(
        [
            flights_with_carriers.count().name("flight_count"),
            distance_sum_expr.name("total_distance"),
        ],
        by=[flights_with_carriers.nickname],
    )

    # Calculate overall total distance
    overall_totals = flights_with_carriers.aggregate(
        [distance_sum_expr.name("total_distance")]
    )

    # Join carrier stats with overall totals
    carrier_stats_with_totals = carrier_stats.cross_join(overall_totals).select(
        carrier_stats.nickname,
        carrier_stats.flight_count,
        carrier_stats.total_distance,
        overall_totals.total_distance.name("overall_total_distance"),
    )

    # Calculate distance share percentage for each carrier
    carrier_distance_shares = carrier_stats_with_totals.select(
        carrier_stats_with_totals.nickname,
        carrier_stats_with_totals.flight_count,
        (
            (
                carrier_stats_with_totals.total_distance.cast("float64")
                / carrier_stats_with_totals.overall_total_distance.cast("float64")
            )
            * 100
        ).name("distance_share_percentage"),
    )

    # Get top 10 carriers by distance share
    top_carriers_by_distance_share = carrier_distance_shares.order_by(
        carrier_distance_shares.distance_share_percentage.desc()
    ).limit(10)

    xorq_expr = from_ibis(top_carriers_by_distance_share)

    result = xorq_expr.execute()
    assert result.to_dict("list") == {
        "nickname": ["United", "American"],
        "flight_count": [1, 2],
        "distance_share_percentage": [60.0, 40.0],
    }
