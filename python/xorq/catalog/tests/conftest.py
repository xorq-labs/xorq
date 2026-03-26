import shutil
from pathlib import Path

import pytest
import toolz
from git import (
    Blob,
    Repo,
)

import xorq.api as xo
from xorq.catalog.annex import LOCAL_ANNEX, Annex, _do_inside
from xorq.catalog.backend import GitAnnexBackend
from xorq.catalog.catalog import (
    CATALOG_YAML_NAME,
    METADATA_APPEND,
    PREFERRED_SUFFIX,
    Catalog,
)
from xorq.catalog.constants import CatalogInfix
from xorq.catalog.expr_utils import build_expr_context_zip
from xorq.catalog.zip_utils import with_pure_suffix


def get_split_tree(repo):
    dct = toolz.groupby(
        Path.parent.fget,
        (
            Path(blob.path)
            for blob in repo.head.commit.tree.list_traverse()
            if isinstance(blob, Blob)
        ),
    )
    return {
        str(parent): tuple(
            zip(
                *(
                    (str(with_pure_suffix(p, "")), "".join(p.suffixes))
                    for p in (p.relative_to(parent) for p in ps)
                )
            )
        )
        for parent, ps in dct.items()
    }


def compare_repo_and_catalog(repo, catalog):
    tree = get_split_tree(repo)

    toplevel_names, toplevel_suffixes = tree["."]
    assert toplevel_names == (Path(CATALOG_YAML_NAME).stem,)
    assert toplevel_suffixes == (Path(CATALOG_YAML_NAME).suffix,)

    entry_names, entry_suffixes = tree[CatalogInfix.ENTRY]
    (entry_suffix, *rest) = set(entry_suffixes)
    assert entry_suffix == PREFERRED_SUFFIX and not rest, (entry_suffix, *rest)

    metadata_names, metadata_suffixes = tree[CatalogInfix.METADATA]
    (metadata_suffix, *rest) = set(metadata_suffixes)
    assert metadata_suffix == PREFERRED_SUFFIX + METADATA_APPEND and not rest, (
        metadata_suffix,
        *rest,
    )

    actual = tuple(sorted(catalog.list()))
    expecteds = tuple(tuple(sorted(el)) for el in (entry_names, metadata_names))
    assert all(actual == expected for expected in expecteds), (actual, expecteds)

    alias_names, alias_suffixes = tree.get(CatalogInfix.ALIAS, ((), ()))
    if alias_suffixes:
        (alias_suffix, *rest) = set(alias_suffixes)
        assert alias_suffix == PREFERRED_SUFFIX and not rest, (alias_suffix, *rest)
    assert tuple(sorted(alias_names)) == tuple(sorted(catalog.list_aliases()))


@pytest.fixture
def repo(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("repo"), annex=LOCAL_ANNEX)
    yield repo


@pytest.fixture
def catalog(repo):
    repo_path = Path(repo.working_dir)
    backend = GitAnnexBackend(repo=repo, annex=Annex(repo_path=repo_path))
    yield Catalog(backend=backend)


@pytest.fixture
def catalog_path(catalog):
    yield str(catalog.repo_path)


def make_build_zip(tmpdir, name):
    expr = xo.memtable({name: [name]})
    with build_expr_context_zip(expr) as zip_path:
        target = Path(tmpdir).joinpath(zip_path.name)
        shutil.copy(zip_path, target)
        return target


@pytest.fixture
def data_dict(tmpdir):
    data_dict = {}
    for name in map(
        chr,
        (
            *range(ord("a"), ord("c")),
            *range(ord("A"), ord("C")),
        ),
    ):
        target = make_build_zip(tmpdir, name)
        data_dict[target.name] = target
    yield data_dict


@pytest.fixture
def catalog_populated(catalog, data_dict):
    tuple(catalog.add(path) for path in data_dict.values())
    yield catalog


@pytest.fixture
def root_repo(tmpdir):
    root_path = Path(tmpdir).joinpath("root")
    root_path.mkdir()
    repo = Repo.init(root_path)
    (root_path / "README.md").write_text("root repo")
    repo.index.add(["README.md"])
    repo.index.commit("initial commit")
    yield repo


@pytest.fixture
def repo_cloned_bare(catalog_populated, tmpdir):
    bare_path = Path(tmpdir).joinpath("catalog-populated-bare")
    repo_cloned_bare = Repo.clone_from(
        catalog_populated.repo_path,
        bare_path,
        bare=True,
    )
    # init annex in bare repo so it can serve content to clones
    _do_inside(bare_path, "init")
    # sync content from origin (the populated catalog)
    _do_inside(bare_path, "sync", "--content")
    yield repo_cloned_bare
