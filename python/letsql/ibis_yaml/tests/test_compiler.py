import os

import pytest

from letsql.ibis_yaml.compiler import BuildManager


@pytest.fixture
def build_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("builds")


def test_build_manager_expr_hash(t, build_dir):
    expected = "c6527994ad9a"
    build_manager = BuildManager(build_dir)
    result = build_manager.get_expr_hash(t)
    assert expected == result


def test_build_manager_roundtrip(t, build_dir):
    build_manager = BuildManager(build_dir)
    expr_hash = "c6527994ad9a"
    yaml_dict = {"a": "string"}
    build_manager.save_yaml(yaml_dict, t)

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
