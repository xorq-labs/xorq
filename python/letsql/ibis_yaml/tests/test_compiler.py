import os
import pathlib

import dask
import yaml

import letsql as ls
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


def test_ibis_compiler(t, build_dir):
    t = ls.memtable({"a": [0, 1], "b": [0, 1]})
    backend = t._find_backend()
    backend.profile_name = "default"
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    compiler.profiles = {"default": backend}
    compiler.compile_expr(expr)
    expr_hash = dask.base.tokenize(expr)[:12]

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_ibis_compiler_parquet_reader(t, build_dir):
    backend = ls.datafusion.connect()
    backend.profile_name = "default"
    awards_players = backend.read_parquet(
        ls.config.options.pins.get_path("awards_players"),
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    compiler = BuildManager(build_dir)
    compiler.profiles = {"default": backend}
    compiler.compile_expr(expr)
    expr_hash = "5ebaf6a7a02d"
    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_compiler_sql(build_dir):
    backend = ls.datafusion.connect()
    backend.profile_name = "default"
    awards_players = backend.read_parquet(
        ls.config.options.pins.get_path("awards_players"),
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    compiler = BuildManager(build_dir)
    compiler.profiles = {"default": backend}
    compiler.compile_expr(expr)
    expr_hash = "5ebaf6a7a02d"
    _roundtrip_expr = compiler.load_expr(expr_hash)

    assert os.path.exists(build_dir / expr_hash / "sql.yaml")

    sql_text = pathlib.Path(build_dir / expr_hash / "sql.yaml").read_text()
    expected_result = (
        "queries:\n"
        "  main:\n"
        "    engine: datafusion\n"
        "    profile_name: default\n"
        '    sql: "SELECT\\n  \\"t0\\".\\"playerID\\",\\n  '
        '\\"t0\\".\\"awardID\\",\\n  \\"t0\\".\\"tie\\"\\\n'
        '      ,\\n  \\"t0\\".\\"notes\\"\\nFROM \\"awards_players\\" AS '
        '\\"t0\\"\\nWHERE\\n  \\"t0\\".\\"\\\n'
        "      lgID\\\" = 'NL'\"\n"
    )

    assert sql_text == expected_result


def test_ibis_compiler_expr_schema_ref(t, build_dir):
    t = ls.memtable({"a": [0, 1], "b": [0, 1]})
    backend = t._find_backend()
    backend.profile_name = "default"
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    compiler.profiles = {"default": backend}
    compiler.compile_expr(expr)
    expr_hash = dask.base.tokenize(expr)[:12]

    with open(build_dir / expr_hash / "expr.yaml") as f:
        yaml_dict = yaml.safe_load(f)

    assert yaml_dict["expression"]["schema_ref"]
