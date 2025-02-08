import pytest

import letsql as ls
from letsql.expr.relations import into_backend
from letsql.ibis_yaml.compiler import IbisYamlCompiler


@pytest.fixture(scope="session")
def duckdb_path(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("duckdb") / "test.db"
    return str(db_path)


@pytest.fixture(scope="session")
def prepare_duckdb_con(duckdb_path):
    con = ls.duckdb.connect(duckdb_path)
    con.profile_name = "my_duckdb"  # patch

    con.raw_sql(
        """
        CREATE TABLE IF NOT EXISTS mytable (
            id INT,
            val VARCHAR
        )
        """
    )
    con.raw_sql(
        """
        INSERT INTO mytable
        SELECT i, 'val' || i::VARCHAR
        FROM range(1, 6) t(i)
        """
    )
    return con


def test_duckdb_database_table_roundtrip(prepare_duckdb_con):
    con = prepare_duckdb_con

    profiles = {"my_duckdb": con}

    table_expr = con.table("mytable")  # DatabaseTable op

    expr1 = table_expr.mutate(new_val=(table_expr.val + "_extra"))
    compiler = IbisYamlCompiler()
    compiler.profiles = profiles

    yaml_dict = compiler.compile_to_yaml(expr1)

    print("Serialized YAML:\n", yaml_dict)

    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)

    df_original = expr1.execute()
    df_roundtrip = roundtrip_expr.execute()

    assert df_original.equals(df_roundtrip), "Roundtrip expression data differs!"


def test_memtable(prepare_duckdb_con, tmp_path_factory):
    table = ls.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    backend.profile_name = "default-duckdb"
    expr = table.mutate(new_val=2 * ls._.val)

    profiles = {"default-duckdb": backend}

    compiler = IbisYamlCompiler()
    compiler.tmp_path = tmp_path_factory.mktemp("duckdb")
    compiler.profiles = profiles

    yaml_dict = compiler.compile_to_yaml(expr)
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)

    expr.equals(roundtrip_expr)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_into_backend(prepare_duckdb_con, tmp_path_factory):
    table = ls.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    backend.profile_name = "default-duckdb"
    expr = table.mutate(new_val=2 * ls._.val)

    con2 = ls.connect()
    con2.profile_name = "default-let"
    con3 = ls.connect()
    con3.profile_name = "default-datafusion"

    expr = into_backend(expr, con2, "ls_mem").mutate(x=4 * ls._.val)
    expr = into_backend(expr, con3, "df_mem")

    profiles = {
        "default-duckdb": backend,
        "default-let": con2,
        "default-datafusion": con3,
    }

    compiler = IbisYamlCompiler()
    compiler.tmp_path = tmp_path_factory.mktemp("duckdb")
    compiler.profiles = profiles

    yaml_dict = compiler.compile_to_yaml(expr)
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)

    assert ls.execute(expr).equals(ls.execute(roundtrip_expr))


def test_memtable_cache(prepare_duckdb_con, tmp_path_factory):
    table = ls.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    backend.profile_name = "default-duckdb"
    expr = table.mutate(new_val=2 * ls._.val).cache()

    profiles = {"default-duckdb": backend}

    compiler = IbisYamlCompiler()
    compiler.tmp_path = tmp_path_factory.mktemp("duckdb")
    compiler.profiles = profiles

    yaml_dict = compiler.compile_to_yaml(expr)
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)

    expr.equals(roundtrip_expr)

    assert expr.execute().equals(roundtrip_expr.execute())
