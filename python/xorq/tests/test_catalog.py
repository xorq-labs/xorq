import pytest

from xorq.catalog import (
    CATALOG_YAML_FILENAME,
    XorqCatalog,
    load_catalog,
    resolve_build_dir,
)
from xorq.catalog_api import CatalogAPI


def test_load_and_save_catalog(tmp_path):
    cat = load_catalog(path=str(tmp_path / CATALOG_YAML_FILENAME))
    assert hasattr(cat, "entries") and isinstance(cat.entries, tuple)
    cat.save(path=str(tmp_path / CATALOG_YAML_FILENAME))
    cat2 = load_catalog(path=str(tmp_path / CATALOG_YAML_FILENAME))
    assert cat2.entries == cat.entries


def test_resolve_target_alias_and_entry():
    catalog = XorqCatalog.from_dict(
        {
            "aliases": {"foo": {"entry_id": "e1", "revision_id": "r1"}},
            "entries": [
                {
                    "entry_id": "e1",
                    "current_revision": "r1",
                    "history": [{"revision_id": "r1"}],
                }
            ],
        }
    )
    t = catalog.resolve_target("foo")
    assert t.entry_id == "e1"
    assert t.rev == "r1"
    assert t.alias is True
    t2 = catalog.resolve_target("e1@r1")
    assert t2.entry_id == "e1"
    assert t2.rev == "r1"
    assert t2.alias is False


def test_resolve_build_dir_by_path(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    cat = {"aliases": {}, "entries": []}
    p = resolve_build_dir(str(d), cat)
    assert p == d


def test_resolve_build_dir_by_build_id(tmp_path):
    d = tmp_path / "build1"
    d.mkdir()
    entry = {
        "entry_id": "e1",
        "current_revision": "r1",
        "history": [{"revision_id": "r1", "build": {"build_id": "b1", "path": str(d)}}],
    }
    cat = XorqCatalog.from_dict({"aliases": {}, "entries": [entry]})
    p = resolve_build_dir("e1", cat)
    assert p == d


def test_load_expr_non_existent_path(tmp_path):
    """Test load_expr with non-existent path raises ValueError"""
    api = CatalogAPI(catalog_path=tmp_path / CATALOG_YAML_FILENAME)
    non_existent = tmp_path / "does_not_exist"

    with pytest.raises(ValueError, match="Build directory not found"):
        api.load_expr(non_existent)


def test_load_expr_path_is_file(tmp_path):
    """Test load_expr with file instead of directory raises ValueError"""
    api = CatalogAPI(catalog_path=tmp_path / CATALOG_YAML_FILENAME)
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError, match="not a directory"):
        api.load_expr(file_path)


def test_load_expr_missing_expr_yaml(tmp_path):
    """Test load_expr with directory missing expr.yaml raises ValueError"""
    api = CatalogAPI(catalog_path=tmp_path / CATALOG_YAML_FILENAME)
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    with pytest.raises(ValueError, match="No expr.yaml found"):
        api.load_expr(build_dir)


def test_get_placeholder_non_existent_alias(tmp_path):
    """Test get_placeholder with non-existent alias raises ValueError"""
    # Create empty catalog
    catalog_path = tmp_path / CATALOG_YAML_FILENAME
    load_catalog(path=str(catalog_path)).save(path=str(catalog_path))

    api = CatalogAPI(catalog_path=catalog_path)

    with pytest.raises(ValueError, match="Build directory not found"):
        api.get_placeholder("non-existent-alias")
