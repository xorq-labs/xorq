"""
GizmoSQL Backend Demo — Multi-Engine Execution with XORQ
=========================================================

GizmoSQL exposes DuckDB over Arrow Flight SQL, enabling client-server
analytics with the full DuckDB SQL dialect.  This demo shows how to use
the GizmoSQL backend with XORQ's `into_backend` method for composing
queries across engines.

Prerequisites
-------------
1. Install the GizmoSQL extra:
       uv sync --extra gizmosql --extra duckdb --group dev
2. Start the GizmoSQL server:
       docker compose up gizmosql --wait
"""

import xorq.api as xo
from xorq.caching import SourceCache
from xorq.vendor import ibis
from xorq.vendor.ibis import _


# ---------------------------------------------------------------------------
# 1. Connect to GizmoSQL (DuckDB over Arrow Flight SQL)
# ---------------------------------------------------------------------------

gizmosql_con = xo.gizmosql.connect(
    host="localhost",
    user="ibis",
    password="ibis_password",
    port=31337,
    use_encryption=True,
    disable_certificate_verification=True,
)

print("Connected to GizmoSQL")
print(f"  Backend name : {gizmosql_con.name}")
print(f"  Catalog      : {gizmosql_con.current_catalog}")
print(f"  Database     : {gizmosql_con.current_database}")
print(f"  Version      : {gizmosql_con.version}")
print()

# ---------------------------------------------------------------------------
# 2. Load data into GizmoSQL via read_parquet
#    (reads locally with DuckDB, streams Arrow batches to the server)
# ---------------------------------------------------------------------------

batting = gizmosql_con.read_parquet(
    "ci/ibis-testing-data/parquet/batting.parquet",
    table_name="batting",
)

awards = gizmosql_con.read_parquet(
    "ci/ibis-testing-data/parquet/awards_players.parquet",
    table_name="awards_players",
)

print(f"Tables on GizmoSQL server: {gizmosql_con.list_tables()}")
print(f"  batting       : {batting.count().execute()} rows")
print(f"  awards_players: {awards.count().execute()} rows")
print()

# ---------------------------------------------------------------------------
# 3. Basic query — runs entirely on GizmoSQL
# ---------------------------------------------------------------------------

top_batters = (
    batting.filter(_.yearID >= 2010)
    .group_by("playerID")
    .agg(
        total_hits=_.H.sum(),
        seasons=_.yearID.nunique(),
    )
    .order_by(ibis.desc("total_hits"))
    .limit(10)
)

print("Top 10 batters (2010+) — executed on GizmoSQL:")
print(top_batters.execute().to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 4. into_backend: GizmoSQL → DuckDB
#    Move data from the GizmoSQL server into a local DuckDB engine.
#    XORQ handles the Arrow Flight transfer automatically.
# ---------------------------------------------------------------------------

duckdb_con = xo.duckdb.connect()

# Transfer batting data from GizmoSQL → DuckDB
batting_in_ddb = batting.into_backend(duckdb_con, "batting_local")

# Verify: the expression now lives on two backends
print("into_backend: GizmoSQL → DuckDB")
print(f"  Expression backends: {batting_in_ddb.ls.backends}")
print()

# Query runs on DuckDB, data was pulled from GizmoSQL via Arrow
recent_batting = (
    batting_in_ddb.filter(_.yearID == 2015)
    .select("playerID", "yearID", "teamID", "H", "AB")
    .order_by(ibis.desc("H"))
    .limit(5)
)
print("Recent batting (2015, top 5 by hits) — executed on DuckDB:")
print(recent_batting.execute().to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 5. into_backend: GizmoSQL → DataFusion (XORQ's embedded engine)
#    DataFusion is the default XORQ engine, accessed via xo.connect().
# ---------------------------------------------------------------------------

xorq_con = xo.connect()  # embedded DataFusion

awards_in_xorq = awards.into_backend(xorq_con, "awards_local")

print("into_backend: GizmoSQL → DataFusion")
print(f"  Expression backends: {awards_in_xorq.ls.backends}")
print()

nl_awards = (
    awards_in_xorq.filter(_.lgID == "NL")
    .group_by("playerID")
    .agg(num_awards=_.playerID.count())
    .order_by(ibis.desc("num_awards"))
    .limit(5)
)
print("NL award winners (top 5) — executed on DataFusion:")
print(nl_awards.execute().to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 6. Multi-engine join: GizmoSQL data on DuckDB joined with local data
#    This shows the real power — compose across engines seamlessly.
# ---------------------------------------------------------------------------

# Pull both tables into DuckDB for a join
batting_ddb = batting.into_backend(duckdb_con, "batting_join")
awards_ddb = awards.into_backend(duckdb_con, "awards_join")

# Join batting with awards — runs on DuckDB, data sourced from GizmoSQL
enriched = (
    batting_ddb.filter(_.yearID >= 2010)
    .join(
        awards_ddb.select("playerID", "awardID", yearID="yearID"),
        predicates=["playerID", "yearID"],
    )
    .group_by(["playerID", "awardID"])
    .agg(total_hits=_.H.sum())
    .order_by(ibis.desc("total_hits"))
    .limit(10)
)

print("Award-winning batters (2010+) — GizmoSQL → DuckDB join:")
print(enriched.execute().to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 7. Multi-hop: GizmoSQL → DuckDB → DataFusion
#    Chain into_backend calls to move data through a pipeline.
# ---------------------------------------------------------------------------

pipeline = (
    batting.filter(_.yearID >= 2014)
    .select("playerID", "yearID", "teamID", "H", "AB")
    .into_backend(duckdb_con, "pipeline_step1")  # GizmoSQL → DuckDB
    .filter(_.H > 100)
    .into_backend(xorq_con, "pipeline_step2")  # DuckDB → DataFusion
    .group_by("teamID")
    .agg(
        avg_hits=_.H.mean(),
        num_players=_.playerID.nunique(),
    )
    .order_by(ibis.desc("avg_hits"))
    .limit(10)
)

print("Multi-hop pipeline (GizmoSQL → DuckDB → DataFusion):")
print(pipeline.execute().to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 8. Caching with into_backend
#    Cache intermediate results to avoid re-computation.
# ---------------------------------------------------------------------------

cached_pipeline = (
    batting.filter(_.yearID >= 2010)
    .into_backend(duckdb_con, "cached_batting")
    .group_by(["playerID", "yearID"])
    .agg(total_hits=_.H.sum())
    .cache(SourceCache.from_kwargs(source=duckdb_con))  # cache on DuckDB
    .into_backend(xorq_con)  # move to DataFusion
    .filter(_.total_hits > 150)
    .order_by(ibis.desc("total_hits"))
    .limit(10)
)

print("Cached pipeline (GizmoSQL → DuckDB [cached] → DataFusion):")
print(cached_pipeline.execute().to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 9. Arrow RecordBatch streaming
#    Expressions are tools, Arrow is the pipe.
# ---------------------------------------------------------------------------

batches = (
    batting.filter(_.yearID == 2015)
    .select("playerID", "teamID", "H")
    .to_pyarrow_batches()
)

print("Arrow RecordBatch streaming from GizmoSQL:")
print(f"  Reader type: {type(batches)}")
batch = next(batches)
print(f"  First batch: {batch.num_rows} rows, {batch.num_columns} columns")
print(f"  Schema: {batch.schema}")
print()

# ---------------------------------------------------------------------------
# 10. In-memory tables (memtable) on GizmoSQL
#     Create a table from a Python dict, push it to the server.
# ---------------------------------------------------------------------------

players = xo.memtable(
    {
        "playerID": ["troutmi01", "harpebr03", "bikiebo01"],
        "name": ["Mike Trout", "Bryce Harper", "Bo Bichette"],
    }
)

# Execute the memtable on GizmoSQL — data is sent via ADBC
result = gizmosql_con.execute(players)
print("Memtable executed on GizmoSQL:")
print(result.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("GizmoSQL backend demo complete!")
print()
print("Key takeaways:")
print("  - GizmoSQL = DuckDB over Arrow Flight SQL (client-server)")
print("  - into_backend() moves expressions between engines via Arrow")
print("  - Chain into_backend() calls for multi-engine pipelines")
print("  - Cache intermediate results with .cache() at any node")
print("  - .to_pyarrow_batches() streams results as Arrow batches")
print("=" * 60)

# Clean up: close the ADBC connection explicitly to avoid
# spurious errors during interpreter shutdown.
gizmosql_con.con.close()
