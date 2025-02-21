import os
import pathlib

import dask
import pytest
import yaml

import xorq as xo
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.ibis_yaml.compiler import ArtifactStore, BuildManager
from xorq.vendor.ibis.common.collections import FrozenOrderedDict


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
    t = xo.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    expr_hash = dask.base.tokenize(expr)[:12]

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_ibis_compiler_parquet_reader(build_dir):
    backend = xo.duckdb.connect()
    parquet_path = xo.config.options.pins.get_path("awards_players")
    awards_players = deferred_read_parquet(
        backend, parquet_path, table_name="award_players"
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    expr_hash = "9a7d0b20d41a"
    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_compiler_sql(build_dir):
    backend = xo.datafusion.connect()
    awards_players = deferred_read_parquet(
        backend,
        xo.config.options.pins.get_path("awards_players"),
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
        "    options: {}\n"
        "    sql_file: df34d95d62bc.sql\n"
    )
    assert sql_text == expected_result


def test_deferred_reads_yaml(build_dir):
    backend = xo.datafusion.connect()
    awards_players = deferred_read_parquet(
        backend,
        xo.config.options.pins.get_path("awards_players"),
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    expr_hash = "79d83e9c89ad"
    _roundtrip_expr = compiler.load_expr(expr_hash)
    assert os.path.exists(build_dir / expr_hash / "deferred_reads.yaml")

    sql_text = pathlib.Path(build_dir / expr_hash / "deferred_reads.yaml").read_text()

    expected_result = (
        "reads:\n"
        "  awards_players-eaf5fdf4554ae9098af6c7e7dfea1a9f:\n"
        "    engine: datafusion\n"
        "    profile_name: a506210f56203e8f9b4a84ef73d95eaa\n"
        "    relations:\n"
        "    - awards_players-eaf5fdf4554ae9098af6c7e7dfea1a9f\n"
        "    options:\n"
        "      method_name: read_parquet\n"
        "      name: awards_players\n"
        "      read_kwargs:\n"
        "      - path: /home/hussainsultan/.cache/pins-py/gs_d3037fb8920d01eb3b262ab08d52335c89ba62aa41299e5236f01807aa8b726d/awards_players/20240711T171119Z-886c4/awards_players.parquet\n"
        "      - table_name: awards_players\n"
        "    sql_file: c0907dab80b0.sql\n"
    )
    assert sql_text == expected_result


def test_ibis_compiler_expr_schema_ref(t, build_dir):
    t = xo.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    compiler.compile_expr(expr)
    expr_hash = dask.base.tokenize(expr)[:12]

    with open(build_dir / expr_hash / "expr.yaml") as f:
        yaml_dict = yaml.safe_load(f)

    assert yaml_dict["expression"]["schema_ref"]


def test_multi_engine_deferred_reads(build_dir):
    con0 = xo.connect()
    con1 = xo.connect()
    con2 = xo.duckdb.connect()
    con3 = xo.connect()

    awards_players = xo.examples.awards_players.fetch(con0).into_backend(con1)
    batting = xo.examples.batting.fetch(con2).into_backend(con1)
    expr = awards_players.join(
        batting, predicates=["playerID", "yearID", "lgID"]
    ).into_backend(con3)[lambda t: t.G == 1]
    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())
