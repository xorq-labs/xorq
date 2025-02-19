import os
import pathlib

import dask
import pytest
import yaml

import letsql as ls
from letsql.common.utils.defer_utils import deferred_read_parquet
from letsql.ibis_yaml.compiler import ArtifactStore, BuildManager
from letsql.vendor.ibis.common.collections import FrozenOrderedDict


def test_build_manager_expr_hash(t, build_dir):
    expected = "c6527994ad9a"
    build_manager = ArtifactStore(build_dir)
    result = build_manager.get_expr_hash(t)
    assert expected == result


def test_build_manager_roundtrip(t, build_dir):
    build_manager = ArtifactStore(build_dir)
    expr_hash = "c6527994ad9a"
    yaml_dict = {"a": "string"}
    build_manager.save_yaml(yaml_dict, expr_hash, "expr.yaml")

    with open(build_dir / expr_hash / "expr.yaml") as f:
        out = f.read()
    assert out == "a: string\n"
    result = build_manager.load_yaml(expr_hash, "expr.yaml")
    assert result == yaml_dict


def test_build_manager_paths(t, build_dir):
    new_path = build_dir / "new_path"

    assert not os.path.exists(new_path)
    build_manager = ArtifactStore(new_path)
    assert os.path.exists(new_path)

    build_manager.get_build_path("hash")
    assert os.path.exists(new_path / "hash")


def test_clean_frozen_dict_yaml(build_dir):
    build_manager = ArtifactStore(build_dir)
    data = FrozenOrderedDict(
        {"string": "text", "integer": 42, "float": 3.14, "boolean": True, "none": None}
    )

    expected_yaml = """string: text
integer: 42
float: 3.14
boolean: true
none: null
"""
    out_path = build_manager.save_yaml(data, "hash", "expr.yaml")
    result = out_path.read_text()

    assert expected_yaml == result


@pytest.mark.xfail(reason="MemTable is not serializable")
def test_ibis_compiler(t, build_dir):
    t = ls.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    expr_hash = dask.base.tokenize(expr)[:12]

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_ibis_compiler_parquet_reader(build_dir):
    backend = ls.duckdb.connect()
    parquet_path = ls.config.options.pins.get_path("awards_players")
    awards_players = deferred_read_parquet(
        backend, parquet_path, table_name="award_players"
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    print(dask.base.tokenize(expr)[:12])
    expr_hash = "9a7d0b20d41a"
    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_compiler_sql(build_dir):
    backend = ls.datafusion.connect()
    awards_players = deferred_read_parquet(
        backend,
        ls.config.options.pins.get_path("awards_players"),
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    expr_hash = "79d83e9c89ad"
    _roundtrip_expr = compiler.load_expr(expr_hash)

    assert os.path.exists(build_dir / expr_hash / "sql.yaml")

    sql_text = pathlib.Path(build_dir / expr_hash / "sql.yaml").read_text()
    expected_result = (
        "queries:\n"
        "  main:\n"
        "    engine: let\n"
        f"    profile_name: {expr._find_backend()._profile.hash_name}\n"
        "    relations:\n"
        "    - awards_players-eaf5fdf4554ae9098af6c7e7dfea1a9f\n"
        '    sql: "SELECT\\n  \\"t0\\".\\"playerID\\",\\n  \\"t0\\".\\"awardID\\",\\n  \\"t0\\".\\"tie\\"'
        '\\\n      ,\\n  \\"t0\\".\\"notes\\"\\nFROM \\"awards_players-eaf5fdf4554ae9098af6c7e7dfea1a9f\\"'
        '\\\n      \\ AS \\"t0\\"\\nWHERE\\n  \\"t0\\".\\"lgID\\" = \'NL\'"\n'
        "  awards_players-eaf5fdf4554ae9098af6c7e7dfea1a9f:\n"
        "    engine: datafusion\n"
        "    relations: awards_players-eaf5fdf4554ae9098af6c7e7dfea1a9f\n"
        "    profile_name: a506210f56203e8f9b4a84ef73d95eaa\n"
        '    sql: "SELECT\\n  *\\nFROM \\"awards_players-eaf5fdf4554ae9098af6c7e7dfea1a9f\\""\n'
    )
    assert sql_text == expected_result


def test_ibis_compiler_expr_schema_ref(t, build_dir):
    t = ls.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    expr_hash = dask.base.tokenize(expr)[:12]

    with open(build_dir / expr_hash / "expr.yaml") as f:
        yaml_dict = yaml.safe_load(f)

    assert yaml_dict["expression"]["schema_ref"]
