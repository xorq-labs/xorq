"""Tests for the plain-git (annex=None) Catalog backend."""

from pathlib import Path

import pytest
from git import Repo

import xorq.api as xo
from xorq.catalog.backend import GitBackend
from xorq.catalog.catalog import Catalog, CatalogAddition
from xorq.catalog.tests.conftest import compare_repo_and_catalog, make_build_zip


@pytest.fixture
def git_repo(tmpdir):
    repo_path = Path(tmpdir) / "repo"
    Catalog.init_repo_path(repo_path, annex=None)
    return Repo(repo_path)


@pytest.fixture
def catalog(git_repo):
    backend = GitBackend(repo=git_repo)
    return Catalog(backend=backend)


@pytest.fixture
def data_dict(tmpdir):
    data_dict = {}
    for name in ("a", "b", "c"):
        target = make_build_zip(tmpdir, name)
        data_dict[target.name] = target
    return data_dict


@pytest.fixture
def catalog_populated(catalog, data_dict):
    tuple(catalog.add(path) for path in data_dict.values())
    return catalog


def test_init_repo_path_no_annex(tmpdir):
    repo_path = Path(tmpdir) / "new-catalog"
    repo = Catalog.init_repo_path(repo_path, annex=None)
    assert isinstance(repo, Repo)
    assert not (repo_path / ".git" / "annex").exists()


def test_from_repo_path_init_no_annex(tmpdir):
    repo_path = Path(tmpdir) / "catalog"
    cat = Catalog.from_repo_path(repo_path, init=True, annex=None)
    assert isinstance(cat.backend, GitBackend)


def test_from_repo_path_reopen_no_annex(tmpdir):
    repo_path = Path(tmpdir) / "catalog"
    Catalog.from_repo_path(repo_path, init=True, annex=None)
    cat = Catalog.from_repo_path(repo_path, init=False, annex=None)
    assert isinstance(cat.backend, GitBackend)


def test_from_name_no_annex(tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    cat = Catalog.from_name("test-plain", annex=None)
    assert isinstance(cat.backend, GitBackend)
    assert cat.repo_path == Path(tmpdir) / "test-plain"


def test_from_default_no_annex(tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    cat = Catalog.from_default(annex=None)
    assert isinstance(cat.backend, GitBackend)


def test_from_kwargs_no_annex(tmpdir):
    repo_path = str(Path(tmpdir) / "kw-catalog")
    cat = Catalog.from_kwargs(path=repo_path, init=True, annex=None)
    assert isinstance(cat.backend, GitBackend)


def test_add(catalog, data_dict):
    entries = tuple(catalog.add(path) for path in data_dict.values())
    assert all(entry.exists() for entry in entries)
    for entry in entries:
        entry.assert_consistency()
    catalog.assert_consistency()


def test_add_from_expr(catalog):
    expr = xo.memtable({"plain-git": ["value"]})
    addition = CatalogAddition.from_expr(expr, catalog)
    entry = addition.add()
    assert entry.exists()
    catalog.assert_consistency()


def test_remove(catalog, data_dict):
    entries = tuple(catalog.add(path) for path in data_dict.values())
    for entry in entries:
        catalog.remove(entry.name)
    assert not any(entry.exists() for entry in entries)
    catalog.assert_consistency()
    assert not catalog.list()


def test_add_alias(catalog_populated):
    name = catalog_populated.list()[0]
    alias = catalog_populated.add_alias(name, "my-alias")
    assert alias.alias_path.is_symlink()
    catalog_populated.assert_consistency()


def test_remove_entry_removes_aliases(catalog_populated):
    name = catalog_populated.list()[0]
    catalog_populated.add_alias(name, "rm-alias")
    assert "rm-alias" in catalog_populated.list_aliases()
    catalog_populated.remove(name)
    assert "rm-alias" not in catalog_populated.list_aliases()
    catalog_populated.assert_consistency()


def test_clone_from_no_annex(catalog_populated, tmpdir):
    bare_path = Path(tmpdir) / "bare"
    bare = Repo.clone_from(catalog_populated.repo_path, bare_path, bare=True)
    cloned = Catalog.clone_from(bare.working_dir, Path(tmpdir) / "cloned", annex=None)
    assert isinstance(cloned.backend, GitBackend)
    compare_repo_and_catalog(bare, cloned)
