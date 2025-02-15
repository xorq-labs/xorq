import os

from ibis.common.collections import FrozenOrderedDict

from letsql.ibis_yaml.compiler import BuildManager, IbisYamlCompiler


def test_build_manager_expr_hash(t, build_dir):
    expected = "c6527994ad9a"
    build_manager = BuildManager(build_dir)
    result = build_manager.get_expr_hash(t)
    assert expected == result


def test_build_manager_roundtrip(t, build_dir):
    build_manager = BuildManager(build_dir)
    expr_hash = "c6527994ad9a"
    yaml_dict = {"a": "string"}
    build_manager.save_yaml(yaml_dict, expr_hash)

    with open(build_dir / expr_hash / "expr.yaml") as f:
        out = f.read()
    assert out == "a: string\n"
    result = build_manager.load_yaml(expr_hash)
    assert result == yaml_dict


def test_build_manager_paths(t, build_dir):
    new_path = build_dir / "new_path"

    assert not os.path.exists(new_path)
    build_manager = BuildManager(new_path)
    assert os.path.exists(new_path)

    build_manager.get_build_path("hash")
    assert os.path.exists(new_path / "hash")


def test_clean_frozen_dict_yaml(build_dir):
    build_manager = BuildManager(build_dir)
    data = FrozenOrderedDict(
        {"string": "text", "integer": 42, "float": 3.14, "boolean": True, "none": None}
    )

    expected_yaml = """string: text
integer: 42
float: 3.14
boolean: true
none: null
"""
    out_path = build_manager.save_yaml(data, "hash")
    result = out_path.read_text()

    assert expected_yaml == result


def test_ibis_compiler(t, build_dir):
    compiler = IbisYamlCompiler(build_dir)
    compiler.compile(t)
    expr_hash = "c6527994ad9a"

    roundtrip_expr = compiler.from_hash(expr_hash)

    assert t.equals(roundtrip_expr)
