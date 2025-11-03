import pytest

from xorq.common.utils.ibis_utils import from_ibis


BASE_URL = "https://pub-a45a6a332b4646f2a6f44775695c64df.r2.dev"


@pytest.mark.slow(level=3)
def test_complex_expr_from_boring_semantic_layer(ibis_con):
    flights = ibis_con.read_parquet(f"{BASE_URL}/flights.parquet")
    carriers = ibis_con.read_parquet(f"{BASE_URL}/carriers.parquet")

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

    assert xorq_expr is not None
    assert not xorq_expr.execute().empty
