"""YAML serialization and deserialization of expressions across multiple backends.

Traditional approach: There is no standard way to serialize a query plan across
engines. You would typically save raw SQL strings or pickle Python objects, both
of which lose portability across different database backends. Reproducing a pipeline
on another system means manually reconstructing the query logic.

With xorq: Expressions serialize to YAML and deserialize back, preserving the full
computation graph including cross-backend joins. Pipelines can be version-controlled,
shared as config files, and rebuilt on any system with access to the same backends.
"""
import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.relations import into_backend
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
)


pg = xo.postgres.connect_examples()
db = xo.duckdb.connect()

batting = pg.table("batting")

backend = xo.duckdb.connect()
awards_players = deferred_read_parquet(
    xo.config.options.pins.get_path("awards_players"),
    backend,
    table_name="award_players",
)
left = batting.filter(batting.yearID == 2015)
right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
expr = left.join(
    into_backend(right, pg, "pg-filtered-table"), ["playerID"], how="semi"
)[["yearID", "stint"]]


if __name__ == "__pytest_main__":
    build_path = build_expr(expr, builds_dir="builds")
    roundtrip_expr = load_expr(build_path)
    pytest_examples_passed = True
