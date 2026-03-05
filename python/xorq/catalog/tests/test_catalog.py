import json
import shutil
from pathlib import Path

import pytest
from attr import evolve

import xorq.api as xo
from xorq.catalog.catalog import (
    BuildTgz,
    Catalog,
    CatalogAddition,
    CatalogAlias,
    CatalogEntry,
    with_pure_suffix,
)
from xorq.catalog.constants import CatalogInfix
from xorq.catalog.expr_utils import (
    build_expr_context_tgz,
)
from xorq.catalog.tar_utils import (
    write_tgz,
)
from xorq.catalog.tests.conftest import (
    compare_repo_and_catalog,
)
from xorq.ibis_yaml.compiler import REQUIRED_TGZ_NAMES, ExprKind, build_expr


def test_catalog_add(catalog, data_dict):
    catalog_entries = tuple(catalog.add(path) for path in data_dict.values())
    assert all(catalog_entry.exists() for catalog_entry in catalog_entries)
    for catalog_entry in catalog_entries:
        catalog_entry.assert_consistency()
    catalog.assert_consistency()
    assert set(catalog.list()) == {
        with_pure_suffix(path).name for path in data_dict.values()
    }

    # test not exists condition
    path = next(iter(data_dict.values()))
    with pytest.raises(AssertionError):
        catalog.add(path)


def test_catalog_addition_from_expr(catalog):
    expr = xo.memtable({"from-expr": ["from-expr"]})
    catalog_addition = CatalogAddition.from_expr(expr, catalog)
    assert catalog_addition.build_tgz.path.exists()
    assert catalog_addition._maybe_tmpfile is not None
    catalog_entry = catalog_addition.add()
    assert catalog_entry.exists()
    catalog.assert_consistency()
    assert catalog_entry.name in catalog.list()


def test_catalog_addition_with_aliases(catalog):
    expr = xo.memtable({"with-aliases": ["with-aliases"]})
    catalog_addition = CatalogAddition.from_expr(expr, catalog)
    aliases = ("alias-x", "alias-y")
    catalog_addition = evolve(catalog_addition, aliases=aliases)
    catalog_entry = catalog_addition.add()

    commit_message = catalog.repo.head.commit.message.strip()
    assert all(alias in commit_message for alias in aliases)

    assert catalog_entry.exists()
    assert {ca.alias for ca in catalog_entry.aliases} == set(aliases)
    catalog.assert_consistency()


def test_catalog_addition_from_expr_tmpfile_lifecycle(catalog):
    expr = xo.memtable({"lifecycle": ["lifecycle"]})
    catalog_addition = CatalogAddition.from_expr(expr, catalog)
    tgz_path = catalog_addition.build_tgz.path
    assert tgz_path.exists()
    del catalog_addition
    assert not tgz_path.exists()


def test_catalog_rm(catalog, data_dict):
    catalog_entries = tuple(catalog.add(path) for path in data_dict.values())
    for catalog_entry in catalog_entries:
        catalog.remove(catalog_entry.name)
    assert not any(catalog_entry.exists() for catalog_entry in catalog_entries)
    for catalog_entry in catalog_entries:
        catalog_entry.assert_consistency()
    catalog.assert_consistency()
    assert not catalog.list()

    # test exists condition
    name = next(iter(data_dict.keys()))
    with pytest.raises(AssertionError):
        catalog.remove(name)


def test_catalog_rm_removes_aliases(catalog_populated):
    name = catalog_populated.list()[0]
    alias_a = "alias-one"
    alias_b = "alias-two"
    catalog_populated.add_alias(name, alias_a)
    catalog_populated.add_alias(name, alias_b)

    catalog_entry = catalog_populated.get_catalog_entry(name)
    assert len(catalog_entry.aliases) == 2

    catalog_populated.remove(name)

    commit_message = catalog_populated.repo.head.commit.message.strip()
    assert alias_a in commit_message
    assert alias_b in commit_message

    assert not catalog_entry.exists()
    assert not any(
        ca.alias_path.exists()
        for ca in (catalog_populated.catalog_aliases or [])
        if ca.alias in (alias_a, alias_b)
    )
    catalog_populated.assert_consistency()


def test_catalog_clone_from_push(repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("cloned")
    )
    before = cloned.list()
    compare_repo_and_catalog(repo_cloned_bare, cloned)

    with build_expr_context_tgz(xo.memtable({"to-push": ["to-push"]})) as tgz_path:
        cloned.add(tgz_path)
        cloned.push()

    after = cloned.list()
    compare_repo_and_catalog(repo_cloned_bare, cloned)

    assert before != after


@pytest.mark.parametrize("elide", REQUIRED_TGZ_NAMES)
def test_test_tgz(elide, catalog, tmpdir):
    tgz_path = write_tgz(
        Path(tmpdir).joinpath("build.tgz"),
        {name: b"" for name in REQUIRED_TGZ_NAMES if name != elide},
    )
    with pytest.raises(AssertionError, match=elide):
        BuildTgz(tgz_path)


def test_assert_consistency(catalog, tmpdir):
    tgz_path = write_tgz(
        Path(tmpdir).joinpath("build.tgz"),
        dict.fromkeys(REQUIRED_TGZ_NAMES, b""),
    )
    catalog_addition = CatalogAddition(BuildTgz(tgz_path), catalog)
    catalog_addition.ensure_dirs()
    catalog_path = catalog_addition.catalog_entry.catalog_path
    with catalog.commit_context("bad commit"):
        shutil.copy(
            tgz_path,
            catalog_path,
        )
        catalog.repo.index.add((catalog_path,))
    with pytest.raises(AssertionError):
        catalog.assert_consistency()
    with pytest.raises(AssertionError):
        CatalogEntry(catalog_addition.name, catalog, False)


def test_add_alias(catalog_populated):
    name = catalog_populated.list()[0]
    alias = "my-alias"
    catalog_alias = catalog_populated.add_alias(name, alias)

    assert isinstance(catalog_alias, CatalogAlias)
    assert catalog_alias.alias_path.is_symlink()
    assert catalog_alias.alias_path.exists()
    assert catalog_alias.alias_path.parent.name == CatalogInfix.ALIAS
    assert catalog_alias.target == Path("..") / CatalogInfix.ENTRY / (name + ".tgz")
    catalog_populated.assert_consistency()


def test_add_alias_unknown_name_raises(catalog_populated):
    with pytest.raises(AssertionError):
        catalog_populated.add_alias("nonexistent", "my-alias")


def test_add_alias_overwrite(catalog_populated):
    names = catalog_populated.list()
    name_a, name_b = names[0], names[1]
    alias = "shared-alias"

    catalog_populated.add_alias(name_a, alias)
    catalog_alias = catalog_populated.add_alias(name_b, alias)

    assert catalog_alias.catalog_entry.name == name_b
    assert catalog_alias.alias_path.is_symlink()
    assert (
        catalog_alias.alias_path.resolve()
        == (
            catalog_populated.repo_path / CatalogInfix.ENTRY / (name_b + ".tgz")
        ).resolve()
    )
    catalog_populated.assert_consistency()


def test_add_alias_multiple(catalog_populated):
    names = catalog_populated.list()
    aliases = [f"alias-{i}" for i in range(len(names))]

    for name, alias in zip(names, aliases):
        catalog_populated.add_alias(name, alias)

    catalog_aliases = catalog_populated.catalog_aliases
    assert len(catalog_aliases) == len(names)
    assert {ca.alias for ca in catalog_aliases} == set(aliases)
    catalog_populated.assert_consistency()


def test_add_alias_symlink_is_relative(catalog_populated):
    name = catalog_populated.list()[0]
    catalog_alias = catalog_populated.add_alias(name, "rel-alias")

    raw_target = Path(catalog_alias.alias_path.parent).joinpath(
        catalog_alias.alias_path.readlink()
    )
    assert not catalog_alias.alias_path.readlink().is_absolute()
    assert raw_target.resolve() == catalog_alias.alias_path.resolve()


def test_list_revisions_single(catalog_populated):
    name = catalog_populated.list()[0]
    catalog_alias = catalog_populated.add_alias(name, "rev-alias")

    revisions = catalog_alias.list_revisions()

    assert len(revisions) == 1
    entry, commit = revisions[0]
    assert isinstance(entry, CatalogEntry)
    assert entry.name == name
    assert commit.message.strip() == f"add alias: rev-alias -> {name}"


def test_list_revisions_overwrite(catalog_populated):
    names = catalog_populated.list()
    name_a, name_b = names[0], names[1]
    alias = "rev-alias"

    catalog_populated.add_alias(name_a, alias)
    catalog_alias = catalog_populated.add_alias(name_b, alias)

    revisions = catalog_alias.list_revisions()

    assert len(revisions) == 2
    # most recent first
    assert revisions[0][0].name == name_b
    assert revisions[1][0].name == name_a


def test_list_revisions_entries_require_exists_false(catalog_populated):
    names = catalog_populated.list()
    name_a, name_b = names[0], names[1]
    alias = "rev-alias"

    catalog_populated.add_alias(name_a, alias)
    catalog_alias = catalog_populated.add_alias(name_b, alias)
    catalog_populated.remove(name_a)

    revisions = catalog_alias.list_revisions()

    assert len(revisions) == 2
    entry_b, entry_a = revisions[0][0], revisions[1][0]
    assert entry_b.exists()
    assert not entry_a.exists()


def test_list_revisions_commit_objects(catalog_populated):
    name = catalog_populated.list()[0]
    catalog_alias = catalog_populated.add_alias(name, "rev-alias")

    revisions = catalog_alias.list_revisions()
    _, commit = revisions[0]

    assert hasattr(commit, "hexsha")
    assert hasattr(commit, "authored_datetime")
    assert hasattr(commit, "author")


def test_catalog_alias_from_name(catalog_populated):
    name = catalog_populated.list()[0]
    alias = "from-name-alias"
    catalog_populated.add_alias(name, alias)

    catalog_alias = CatalogAlias.from_name(alias, catalog_populated)

    assert isinstance(catalog_alias, CatalogAlias)
    assert catalog_alias.alias == alias
    assert catalog_alias.catalog_entry.name == name
    assert catalog_alias.alias_path.is_symlink()
    assert catalog_alias.catalog_entry.exists()


def test_catalog_alias_from_name_nonexistent_raises(catalog_populated):
    with pytest.raises(ValueError, match="no such alias"):
        CatalogAlias.from_name("does-not-exist", catalog_populated)


def test_catalog_alias_from_name_entry_consistency(catalog_populated):
    name = catalog_populated.list()[0]
    alias = "consistency-alias"
    catalog_populated.add_alias(name, alias)

    catalog_alias = CatalogAlias.from_name(alias, catalog_populated)

    catalog_alias.catalog_entry.assert_consistency()
    assert catalog_alias.catalog_entry.metadata_path.exists()
    assert catalog_alias.catalog_entry.catalog_path.exists()


def test_catalog_alias_from_name_matches_catalog_aliases(catalog_populated):
    name = catalog_populated.list()[0]
    alias = "match-alias"
    catalog_populated.add_alias(name, alias)

    from_name = CatalogAlias.from_name(alias, catalog_populated)
    from_catalog = next(
        ca for ca in catalog_populated.catalog_aliases if ca.alias == alias
    )

    assert from_name.alias == from_catalog.alias
    assert from_name.catalog_entry.name == from_catalog.catalog_entry.name


def test_catalog_entry_relocatable(repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("cloned")
    )
    catalog_entries = cloned.catalog_entries
    exprs = tuple(catalog_entry.expr for catalog_entry in catalog_entries)
    assert exprs


def test_build_expr_kind_bound(tmp_path):
    expr = xo.memtable({"a": [1, 2, 3]})
    build_dir = build_expr(expr, builds_dir=tmp_path)
    meta = json.loads((build_dir / "metadata.json").read_text())
    assert meta["kind"] == ExprKind.Expr


def test_build_expr_kind_partial(tmp_path):
    t = xo.table(schema={"a": "int64"})
    expr = t.filter(t.a > 0)
    build_dir = build_expr(expr, builds_dir=tmp_path)
    meta = json.loads((build_dir / "metadata.json").read_text())
    assert meta["kind"] == ExprKind.UnboundExpr


def test_extract_kind_bound(catalog):
    expr = xo.memtable({"a": [1, 2, 3]})
    entry = catalog.add(expr)
    assert entry.kind == ExprKind.Expr


def test_extract_kind_partial(catalog):
    t = xo.table(schema={"a": "int64"})
    expr = t.filter(t.a > 0)
    entry = catalog.add(expr)
    assert entry.kind == ExprKind.UnboundExpr
