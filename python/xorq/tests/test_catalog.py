from xorq.catalog import (
    XorqCatalog,
    load_catalog,
    resolve_build_dir,
)


def test_load_and_save_catalog(tmp_path):
    cat = load_catalog(path=str(tmp_path / "catalog.yaml"))
    assert hasattr(cat, "entries") and isinstance(cat.entries, tuple)
    cat.save(path=str(tmp_path / "catalog.yaml"))
    cat2 = load_catalog(path=str(tmp_path / "catalog.yaml"))
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
