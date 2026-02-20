import shutil
from pathlib import Path

import pytest
import toolz
from git import (
    Blob,
    Repo,
)

import xorq.api as xo
from xorq.catalog.catalog import (
    CATALOG_YAML_NAME,
    ENTRY_INFIX,
    METADATA_APPEND,
    METADATA_INFIX,
    PREFERRED_SUFFIX,
    VALID_SUFFIXES,
    Catalog,
    with_pure_suffix,
)
from xorq.catalog.expr_utils import build_expr_context_tgz


VALID_SUFFIX0, VALID_SUFFIX1, *_ = VALID_SUFFIXES


def get_split_tree(repo):
    dct = toolz.groupby(
        Path.parent.fget,
        (
            Path(blob.path)
            for blob in repo.head.commit.tree.list_traverse()
            if isinstance(blob, Blob)
        ),
    )
    (parents, (toplevel, entries, metadata)) = zip(
        *(
            (
                str(parent),
                tuple(
                    zip(
                        *(
                            (str(with_pure_suffix(p, "")), "".join(p.suffixes))
                            for p in (p.relative_to(parent) for p in ps)
                        )
                    )
                ),
            )
            for parent, ps in dct.items()
        )
    )
    return (parents, (toplevel, entries, metadata))


def compare_repo_and_catalog(repo, catalog):
    (
        parents,
        (
            (toplevel_names, toplevel_suffixes),
            (entry_names, entry_suffixes),
            (metadata_names, metadata_suffixes),
        ),
    ) = get_split_tree(repo)
    assert parents == (".", ENTRY_INFIX, METADATA_INFIX)
    assert toplevel_names == (Path(CATALOG_YAML_NAME).stem,)
    assert toplevel_suffixes == (Path(CATALOG_YAML_NAME).suffix,)
    #
    (entry_suffix, *rest) = set(entry_suffixes)
    assert entry_suffix == PREFERRED_SUFFIX and not rest, (entry_suffix, *rest)
    #
    (metadata_suffix, *rest) = set(metadata_suffixes)
    assert metadata_suffix == PREFERRED_SUFFIX + METADATA_APPEND and not rest, (
        metadata_suffix,
        *rest,
    )

    actual = tuple(sorted(catalog.list()))
    expecteds = tuple(tuple(sorted(el)) for el in (entry_names, metadata_names))
    assert all(actual == expected for expected in expecteds), (actual, expecteds)


@pytest.fixture
def repo(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("repo"))
    yield repo


@pytest.fixture
def catalog(repo):
    yield Catalog(repo=repo)


@pytest.fixture
def catalog_path(catalog):
    yield str(catalog.repo_path)


def make_build_tgz(tmpdir, name):
    expr = xo.memtable({name: [name]})
    with build_expr_context_tgz(expr) as tgz_path:
        target = Path(tmpdir).joinpath(tgz_path.name)
        shutil.copy(tgz_path, target)
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
        target = make_build_tgz(tmpdir, name)
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
    repo_cloned_bare = Repo.clone_from(
        catalog_populated.repo_path,
        Path(tmpdir).joinpath("catalog-populated-bare"),
        bare=True,
    )
    yield repo_cloned_bare
