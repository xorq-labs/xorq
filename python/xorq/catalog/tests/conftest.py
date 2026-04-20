import shutil
from pathlib import Path

import pytest
import toolz
from git import (
    Blob,
    Repo,
)

import xorq.api as xo
import xorq.catalog.catalog as catalog_mod
from xorq.catalog.annex import LOCAL_ANNEX, Annex, _do_inside
from xorq.catalog.backend import GitAnnexBackend, GitBackend
from xorq.catalog.catalog import (
    CATALOG_YAML_NAME,
    METADATA_APPEND,
    PREFERRED_SUFFIX,
    Catalog,
)
from xorq.catalog.constants import MAIN_BRANCH, CatalogInfix
from xorq.catalog.expr_utils import build_expr_context_zip
from xorq.catalog.zip_utils import with_pure_suffix
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import (
    PYPROJECT_NAME,
    WheelPackager,
    find_file_upwards,
)


# Fake wheel name for tests that construct synthetic archives.
TEST_WHEEL_NAME = "pkg-0.0.0-py3-none-any.whl"


@pytest.fixture(scope="session", autouse=True)
def _cached_wheel_artifacts():
    """Build the wheel once per test session and patch _ensure_wheel_artifacts to reuse it."""
    pyproject_path = find_file_upwards(Path.cwd(), PYPROJECT_NAME)
    packager = WheelPackager(pyproject_path.parent)
    bundle = packager.build()

    original = catalog_mod._ensure_wheel_artifacts

    def _fast_ensure(build_dir, project_path=None):
        build_dir = Path(build_dir)
        reqs_path = build_dir / DumpFiles.requirements
        if not list(build_dir.glob("*.whl")):
            shutil.copy2(bundle.wheel_path, build_dir / bundle.wheel_path.name)
        if not reqs_path.exists():
            shutil.copy2(bundle.requirements_path, reqs_path)

    catalog_mod._ensure_wheel_artifacts = _fast_ensure
    yield
    catalog_mod._ensure_wheel_artifacts = original


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


@pytest.fixture(scope="session", params=["git", "annex"])
def backend_type(request):
    return request.param


def _make_catalog_from_repo(repo, backend_type):
    repo_path = Path(repo.working_dir)
    if backend_type == "annex":
        backend = GitAnnexBackend(repo=repo, annex=Annex(repo_path=repo_path))
    else:
        backend = GitBackend(repo=repo)
    return Catalog(backend=backend)


def _init_catalog_repo(repo_path, backend_type):
    annex = LOCAL_ANNEX if backend_type == "annex" else None
    return Catalog.init_repo_path(repo_path, annex=annex)


def make_build_zip(tmpdir, name):
    expr = xo.memtable({name: [name]})
    with build_expr_context_zip(expr) as zip_path:
        target = Path(tmpdir).joinpath(zip_path.name)
        shutil.copy(zip_path, target)
        return target


_DATA_DICT_NAMES = tuple(
    map(
        chr,
        (
            *range(ord("a"), ord("c")),
            *range(ord("A"), ord("C")),
        ),
    )
)


@pytest.fixture(scope="session")
def _data_dict_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("data-dict-template")
    return {
        (target := make_build_zip(root, name)).name: target for name in _DATA_DICT_NAMES
    }


@pytest.fixture
def data_dict(tmpdir, _data_dict_template):
    dst_root = Path(tmpdir)
    return {
        name: Path(shutil.copy(src, dst_root.joinpath(name)))
        for name, src in _data_dict_template.items()
    }


@pytest.fixture(scope="session")
def _catalog_populated_template(tmp_path_factory, backend_type, _data_dict_template):
    root = tmp_path_factory.mktemp(f"catalog-populated-template-{backend_type}")
    repo_path = root / "repo"
    repo = _init_catalog_repo(repo_path, backend_type)
    catalog = _make_catalog_from_repo(repo, backend_type)
    for path in _data_dict_template.values():
        catalog.add(path)
    return repo_path


@pytest.fixture
def repo(tmpdir, backend_type):
    repo = _init_catalog_repo(Path(tmpdir).joinpath("repo"), backend_type)
    yield repo


@pytest.fixture
def catalog(repo, backend_type):
    yield _make_catalog_from_repo(repo, backend_type)


@pytest.fixture
def catalog_path(catalog):
    yield str(catalog.repo_path)


@pytest.fixture
def catalog_populated(tmpdir, backend_type, _catalog_populated_template):
    dst = Path(tmpdir).joinpath("repo")
    shutil.copytree(_catalog_populated_template, dst, symlinks=True)
    repo = Repo(dst)
    yield _make_catalog_from_repo(repo, backend_type)


@pytest.fixture
def root_repo(tmpdir):
    root_path = Path(tmpdir).joinpath("root")
    root_path.mkdir()
    repo = Repo.init(root_path, initial_branch=MAIN_BRANCH)
    (root_path / "README.md").write_text("root repo")
    repo.index.add(["README.md"])
    repo.index.commit("initial commit")
    yield repo


@pytest.fixture
def repo_cloned_bare(catalog_populated, backend_type, tmpdir):
    if backend_type != "annex":
        pytest.skip("repo_cloned_bare requires annex backend")
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
