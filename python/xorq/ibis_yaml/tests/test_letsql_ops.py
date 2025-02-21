import pytest

import xorq as xo
from xorq import _
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
)
from xorq.expr.relations import into_backend
from xorq.ibis_yaml.compiler import YamlExpressionTranslator


@pytest.fixture(scope="session")
def duckdb_path(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("duckdb") / "test.db"
    return str(db_path)


@pytest.fixture(scope="session")
def prepare_duckdb_con(duckdb_path):
    con = xo.duckdb.connect(duckdb_path)
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
    compiler = YamlExpressionTranslator()

    yaml_dict = compiler.to_yaml(expr1, profiles)
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    df_original = expr1.execute()
    df_roundtrip = roundtrip_expr.execute()

    assert df_original.equals(df_roundtrip), "Roundtrip expression data differs!"


@pytest.mark.xfail(reason="MemTable is not serializable")
def test_memtable(build_dir):
    table = xo.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    expr = table.mutate(new_val=2 * xo._.val)

    profiles = {backend._profile.hash_name: backend}

    compiler = YamlExpressionTranslator()

    yaml_dict = compiler.to_yaml(expr, profiles, build_dir)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    expr.equals(roundtrip_expr)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_into_backend(build_dir):
    parquet_path = xo.config.options.pins.get_path("awards_players")
    backend = xo.duckdb.connect()
    table = deferred_read_parquet(backend, parquet_path, table_name="award_players")
    expr = table.mutate(new_id=2 * xo._.playerID)

    con2 = xo.connect()
    con3 = xo.connect()

    expr = into_backend(expr, con2, "ls_mem").mutate(x=4 * xo._.new_id)
    expr = into_backend(expr, con3, "df_mem")

    profiles = {
        backend._profile.hash_name: backend,
        con2._profile.hash_name: con2,
        con3._profile.hash_name: con3,
    }

    compiler = YamlExpressionTranslator()

    yaml_dict = compiler.to_yaml(expr, profiles)
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    assert xo.execute(expr).equals(xo.execute(roundtrip_expr))


@pytest.mark.xfail(reason="MemTable is not serializable")
def test_memtable_cache(build_dir):
    table = xo.memtable([(i, "val") for i in range(10)], columns=["key1", "val"])
    backend = table._find_backend()
    expr = table.mutate(new_val=2 * xo._.val).cache()
    backend1 = expr._find_backend()

    profiles = {
        backend._profile.hash_name: backend,
        backend1._profile.hash_name: backend1,
    }

    compiler = YamlExpressionTranslator(profiles=profiles)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert xo.execute(expr).equals(xo.execute(roundtrip_expr))


def test_deferred_read_csv(build_dir):
    csv_name = "iris"
    csv_path = xo.options.pins.get_path(csv_name)
    pd_con = xo.pandas.connect()
    expr = deferred_read_csv(con=pd_con, path=csv_path, table_name=csv_name).filter(
        _.sepal_length > 6
    )

    profiles = {pd_con._profile.hash_name: pd_con}
    compiler = YamlExpressionTranslator()
    yaml_dict = compiler.to_yaml(expr, profiles)
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    assert xo.execute(expr).equals(xo.execute(roundtrip_expr))
