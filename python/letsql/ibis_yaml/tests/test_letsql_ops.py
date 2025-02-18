import pytest

import letsql as ls
from letsql import _
from letsql.common.utils.defer_utils import (
    deferred_read_csv,
)
from letsql.expr.relations import into_backend
from letsql.ibis_yaml.compiler import YamlExpressionTranslator


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


def test_duckdb_database_table_roundtrip(prepare_duckdb_con, build_dir):
    con = prepare_duckdb_con

    profiles = {con._profile.hash_name: con}

    table_expr = con.table("mytable")

    expr1 = table_expr.mutate(new_val=(table_expr.val + "_extra"))
    compiler = YamlExpressionTranslator(current_path=build_dir, profiles=profiles)

    yaml_dict = compiler.to_yaml(expr1)

    print("Serialized YAML:\n", yaml_dict)

    roundtrip_expr = compiler.from_yaml(yaml_dict)

    df_original = expr1.execute()
    df_roundtrip = roundtrip_expr.execute()

    assert df_original.equals(df_roundtrip), "Roundtrip expression data differs!"


def test_memtable(build_dir):
    table = ls.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    expr = table.mutate(new_val=2 * ls._.val)

    profiles = {backend._profile.hash_name: backend}

    compiler = YamlExpressionTranslator(current_path=build_dir, profiles=profiles)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    expr.equals(roundtrip_expr)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_into_backend(build_dir):
    table = ls.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    expr = table.mutate(new_val=2 * ls._.val)

    con2 = ls.connect()
    con3 = ls.connect()

    expr = into_backend(expr, con2, "ls_mem").mutate(x=4 * ls._.val)
    expr = into_backend(expr, con3, "df_mem")

    profiles = {
        backend._profile.hash_name: backend,
        con2._profile.hash_name: con2,
        con3._profile.hash_name: con3,
    }

    compiler = YamlExpressionTranslator(current_path=build_dir, profiles=profiles)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert ls.execute(expr).equals(ls.execute(roundtrip_expr))


def test_memtable_cache(build_dir):
    table = ls.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    expr = table.mutate(new_val=2 * ls._.val).cache()
    backend1 = expr._find_backend()

    profiles = {
        backend._profile.hash_name: backend,
        backend1._profile.hash_name: backend1,
    }

    compiler = YamlExpressionTranslator(profiles=profiles, current_path=build_dir)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert ls.execute(expr).equals(ls.execute(roundtrip_expr))


def test_deferred_read_csv(build_dir):
    csv_name = "iris"
    csv_path = ls.options.pins.get_path(csv_name)
    pd_con = ls.pandas.connect()
    expr = deferred_read_csv(con=pd_con, path=csv_path, table_name=csv_name).filter(
        _.sepal_length > 6
    )

    profiles = {pd_con._profile.hash_name: pd_con}
    compiler = YamlExpressionTranslator(profiles=profiles, current_path=build_dir)
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert ls.execute(expr).equals(ls.execute(roundtrip_expr))
