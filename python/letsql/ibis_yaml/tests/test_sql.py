import pytest

import letsql as ls
import letsql.vendor.ibis.expr.operations as ops
from letsql.expr.relations import RemoteTable, into_backend
from letsql.ibis_yaml.sql import find_remote_tables, generate_sql_plans


def test_find_remote_tables_simple():
    db = ls.duckdb.connect()
    db.profile_name = "duckdb"
    table = ls.memtable([(1, "a"), (2, "b")], columns=["id", "val"])
    backend = table._find_backend()
    backend.profile_name = "duckdb"
    remote_expr = into_backend(table, db)

    remote_tables = find_remote_tables(remote_expr.op())

    assert len(remote_tables) == 1
    table_name = next(iter(remote_tables))
    assert table_name.startswith("ibis_remote")
    assert remote_tables[table_name]["engine"] == "duckdb"


def test_find_remote_tables_raises():
    db = ls.connect()

    awards_players = db.read_parquet(
        ls.config.options.pins.get_path("awards_players"),
        table_name="awards_players",
    )

    db2 = ls.datafusion.connect()

    remote_expr = into_backend(awards_players, db2)
    with pytest.raises(
        AttributeError, match="Backend does not have a valid 'profile_name' attribute."
    ):
        find_remote_tables(remote_expr.op())


def test_find_remote_tables_nested():
    db1 = ls.duckdb.connect()
    db1.profile_name = "duckdb"
    db2 = ls.datafusion.connect()
    db2.profile_name = "datafusion"

    table1 = ls.memtable([(1, "a"), (2, "b")], columns=["id", "val1"])
    table2 = ls.memtable([(1, "x"), (2, "y")], columns=["id", "val2"])

    remote1 = into_backend(table1, db1)
    remote2 = into_backend(table2, db2)
    expr = remote1.join(remote2, "id")

    remote_tables = find_remote_tables(expr.op())

    assert len(remote_tables) == 2
    assert all(name.startswith("ibis_remote") for name in remote_tables)
    assert all("engine" in info and "sql" in info for info in remote_tables.values())


def test_find_remote_tables():
    pg = ls.postgres.connect_examples()
    pg.profile_name = "postgres"
    db = ls.duckdb.connect()
    db.profile_name = "duckdb"

    batting = pg.table("batting")
    awards_players = db.read_parquet(
        ls.config.options.pins.get_path("awards_players"),
        table_name="awards_players",
    )

    left = batting.filter(batting.yearID == 2015)
    right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
    expr = left.join(into_backend(right, pg), ["playerID"], how="semi")[
        ["yearID", "stint"]
    ]

    def print_tree(node, level=0):
        indent = "  " * level
        print(f"{indent}{type(node).__name__}")
        if hasattr(node, "args"):
            for arg in node.args:
                if isinstance(arg, (ops.Node, RemoteTable)):
                    print_tree(arg, level + 1)

    print_tree(expr.op())

    remote_tables = find_remote_tables(expr.op())

    assert len(remote_tables) == 1, (
        f"Expected 1 remote table, found {len(remote_tables)}"
    )

    first_table = next(iter(remote_tables.values()))
    assert "sql" in first_table, "SQL query missing from remote table info"
    assert "engine" in first_table, "Engine info missing from remote table info"


def test_generate_sql_plans_simple():
    db = ls.duckdb.connect()
    db.profile_name = "duckdb"
    table = ls.memtable([(1, "a"), (2, "b")], columns=["id", "val"])
    expr = into_backend(table, db).filter(ls._.id > 1)

    plans = generate_sql_plans(expr)

    assert "queries" in plans
    assert "main" in plans["queries"]
    assert len(plans["queries"]) == 2
    assert all("sql" in q and "engine" in q for q in plans["queries"].values())


def test_generate_sql_plans_raises():
    db = ls.duckdb.connect()
    table = ls.memtable([(1, "a"), (2, "b")], columns=["id", "val"])
    expr = into_backend(table, db).filter(ls._.id > 1)
    with pytest.raises(
        AttributeError, match="Backend does not have a valid 'profile_name' attribute."
    ):
        generate_sql_plans(expr)


def test_generate_sql_plans_complex_example():
    pg = ls.postgres.connect_examples()
    pg.profile_name = "postgres"

    db = ls.duckdb.connect()
    db.profile_name = "duckdb"

    batting = pg.table("batting")
    awards_players = db.read_parquet(
        ls.config.options.pins.get_path("awards_players"),
        table_name="awards_players",
    )

    left = batting.filter(batting.yearID == 2015)
    right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
    expr = left.join(into_backend(right, pg), ["playerID"], how="semi")[
        ["yearID", "stint"]
    ]

    plans = generate_sql_plans(expr)

    assert "queries" in plans
    assert len(plans["queries"]) == 2
    assert "main" in plans["queries"]

    remote_table_names = [k for k in plans["queries"].keys() if k != "main"]
    assert len(remote_table_names) == 1
    remote_table_name = remote_table_names[0]
    assert remote_table_name.startswith("ibis_remote")

    expected_main_sql = f'''SELECT
  "t4"."yearID",
  "t4"."stint"
FROM (
  SELECT
    *
  FROM "batting" AS "t0"
  WHERE
    "t0"."yearID" = 2015
) AS "t4"
WHERE
  EXISTS(
    SELECT
      1
    FROM "{remote_table_name}" AS "t2"
    WHERE
      "t4"."playerID" = "t2"."playerID"
  )'''

    expected_remote_sql = """SELECT
  "t0"."playerID",
  "t0"."awardID",
  "t0"."tie",
  "t0"."notes"
FROM "awards_players" AS "t0"
WHERE
  "t0"."lgID" = 'NL\'"""

    main_query = plans["queries"]["main"]
    assert main_query["engine"] == "postgres", (
        f"Expected 'postgres', got '{main_query['engine']}'"
    )
    assert main_query["sql"].strip() == expected_main_sql.strip()

    remote_query = plans["queries"][remote_table_name]
    assert remote_query["engine"] == "duckdb", (
        f"Expected 'duckdb', got '{remote_query['engine']}'"
    )
    assert remote_query["sql"].strip() == expected_remote_sql.strip()
