import shutil
from pathlib import Path

import pytest

import xorq.api as xo
from xorq.catalog.catalog import (
    BuildTgz,
    Catalog,
    CatalogAddition,
    CatalogEntry,
    with_pure_suffix,
)
from xorq.catalog.constants import (
    REQUIRED_TGZ_NAMES,
)
from xorq.catalog.expr_utils import (
    build_expr_context_tgz,
)
from xorq.catalog.tar_utils import (
    write_tgz,
)
from xorq.catalog.tests.conftest import (
    compare_repo_and_catalog,
)


def test_catalog_add(catalog, data_dict):
    catalog_entries = tuple(catalog.add(path) for path in data_dict.values())
    assert all(catalog_entry.exists() for catalog_entry in catalog_entries)
    for catalog_entry in catalog_entries:
        catalog_entry.assert_consistency()
    catalog.assert_consistency()
    assert set(catalog.list()) == set(
        with_pure_suffix(path).name for path in data_dict.values()
    )

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
        {name: b"" for name in REQUIRED_TGZ_NAMES},
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


def test_catalog_entry_relocatable(repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("cloned")
    )
    catalog_entries = cloned.catalog_entries
    exprs = tuple(catalog_entry.expr for catalog_entry in catalog_entries)
    assert exprs
