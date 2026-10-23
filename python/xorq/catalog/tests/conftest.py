import shutil
import uuid
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
import toolz
from click.testing import CliRunner
from git import (
    Blob,
    Repo,
)

import xorq.api as xo
import xorq.catalog.catalog as catalog_mod
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog.annex import LOCAL_ANNEX, Annex, _do_inside
from xorq.catalog.backend import GitAnnexBackend, GitBackend, GitPointerBackend
from xorq.catalog.catalog import (
    CATALOG_YAML_NAME,
    METADATA_APPEND,
    PREFERRED_SUFFIX,
    Catalog,
    CatalogAlias,
)
from xorq.catalog.constants import CONTENT_STORE_YAML, MAIN_BRANCH, POINTER_SUFFIX
from xorq.catalog.content_store import (
    ContentCache,
    DirectoryContentStoreConfig,
)
from xorq.catalog.enums import CatalogInfix
from xorq.catalog.expr_utils import build_expr_context_zip
from xorq.catalog.replay import Replayer
from xorq.catalog.zip_utils import with_pure_suffix
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import (
    PYPROJECT_NAME,
    WheelPackager,
    find_file_upwards,
)
from xorq.vendor.ibis.expr import operations as ops


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


def _replay_rebuild(source_catalog_obj, target_path, on_unrebuilt_builder="raise"):
    target = Catalog.from_repo_path(target_path, init=True)
    Replayer(
        from_catalog=source_catalog_obj,
        rebuild=True,
        on_unrebuilt_builder=on_unrebuilt_builder,
    ).replay(target)
    return target


# Fake wheel name for tests that construct synthetic archives.
TEST_WHEEL_NAME = "pkg-0.0.0-py3-none-any.whl"


def dist_info_only_wheel(wheel_path: Path, out_dir: Path) -> Path:
    """A copy of ``wheel_path`` holding only its ``.dist-info`` members.

    Same file name and metadata, a few KB instead of the 2 MB package. Catalog
    tests carry an entry's wheel but, unless marked ``installs_entry_wheel``,
    never install it: ``run`` and ``compose`` get ``--use-this-venv`` and
    ``uv tool run`` is stubbed.
    """
    out_path = out_dir / wheel_path.name
    with (
        zipfile.ZipFile(wheel_path) as src,
        zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            if ".dist-info/" in info.filename:
                dst.writestr(info, src.read(info))
    return out_path


def wheel_artifacts_copier(wheel_path: Path, requirements_path: Path) -> Callable:
    """An ``_ensure_wheel_artifacts`` that copies these files instead of building."""

    def ensure(build_dir, project_path=None):
        build_dir = Path(build_dir)
        reqs_path = build_dir / DumpFiles.requirements
        if not list(build_dir.glob("*.whl")):
            shutil.copy2(wheel_path, build_dir / wheel_path.name)
        if not reqs_path.exists():
            shutil.copy2(requirements_path, reqs_path)

    return ensure


@pytest.fixture(scope="session", autouse=True)
def _cached_wheel_artifacts(tmp_path_factory):
    """Build the wheel once per test session; entries get its metadata-only
    copy, including those the session-scoped templates add."""
    pyproject_path = find_file_upwards(Path.cwd(), PYPROJECT_NAME)
    bundle = WheelPackager(pyproject_path.parent).build()
    metadata_only = dist_info_only_wheel(
        bundle.wheel_path, tmp_path_factory.mktemp("wheel")
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            catalog_mod,
            "_ensure_wheel_artifacts",
            wheel_artifacts_copier(metadata_only, bundle.requirements_path),
        )
        yield bundle


@pytest.fixture(autouse=True)
def _installable_entry_wheel(request, monkeypatch, _cached_wheel_artifacts):
    # autouse, so it is in place before the test's catalog fixtures add entries
    if request.node.get_closest_marker("installs_entry_wheel"):
        bundle = _cached_wheel_artifacts
        monkeypatch.setattr(
            catalog_mod,
            "_ensure_wheel_artifacts",
            wheel_artifacts_copier(bundle.wheel_path, bundle.requirements_path),
        )


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
    is_pointer = isinstance(catalog.backend, GitPointerBackend)
    tree = get_split_tree(repo)

    toplevel_names, toplevel_suffixes = tree["."]
    if is_pointer:
        expected_top_names = tuple(
            sorted(
                (
                    Path(CATALOG_YAML_NAME).stem,
                    ".gitignore",  # dotfile: no suffix to strip
                    "content_store",  # content_store.yaml stem
                )
            )
        )
        assert tuple(sorted(toplevel_names)) == expected_top_names
        expected_top_suffixes = tuple(
            sorted(
                ("", Path(CATALOG_YAML_NAME).suffix, Path(CONTENT_STORE_YAML).suffix)
            )
        )
        assert tuple(sorted(toplevel_suffixes)) == expected_top_suffixes
    else:
        assert toplevel_names == (Path(CATALOG_YAML_NAME).stem,)
        assert toplevel_suffixes == (Path(CATALOG_YAML_NAME).suffix,)

    expected_entry_suffix = POINTER_SUFFIX if is_pointer else PREFERRED_SUFFIX
    entry_names, entry_suffixes = tree[CatalogInfix.ENTRY]
    (entry_suffix, *rest) = set(entry_suffixes)
    assert entry_suffix == expected_entry_suffix and not rest, (entry_suffix, *rest)

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


_annex_available = shutil.which("git-annex") is not None

requires_annex = pytest.mark.skipif(
    not _annex_available, reason="git-annex not installed"
)


@pytest.fixture(
    scope="session",
    params=[
        "git",
        pytest.param(
            "annex",
            marks=pytest.mark.skipif(
                not _annex_available, reason="git-annex not installed"
            ),
        ),
        "pointer",
    ],
)
def backend_type(request: pytest.FixtureRequest) -> str:
    return request.param


def _make_pointer_backend(repo: Repo, store_dir: str | Path) -> GitPointerBackend:
    repo_path = Path(repo.working_dir)
    config_path = repo_path / CONTENT_STORE_YAML
    if not config_path.exists():
        config = DirectoryContentStoreConfig(
            catalog_id=str(uuid.uuid4()),
            directory=store_dir,
        )
        config.write_yaml(config_path)
        repo.index.add([CONTENT_STORE_YAML])
        repo.index.commit("add content store config")
    cache = ContentCache(
        cache_dir=repo_path / ".xorq-cache",
        max_bytes=1024 * 1024 * 1024,
    )
    return GitPointerBackend.from_repo(repo, cache=cache)


def _make_catalog_from_repo(
    repo: Repo, backend_type: str, store_dir: str | Path | None = None
) -> Catalog:
    repo_path = Path(repo.working_dir)
    if backend_type == "annex":
        backend = GitAnnexBackend(repo=repo, annex=Annex(repo_path=repo_path))
    elif backend_type == "pointer":
        backend = _make_pointer_backend(repo, store_dir or repo_path / ".store")
    else:
        backend = GitBackend(repo=repo)
    return Catalog(backend=backend)


def directory_store_config(
    directory: str | Path, catalog_id: str | None = None
) -> DirectoryContentStoreConfig:
    kwargs = {"directory": str(directory)}
    if catalog_id is not None:
        kwargs["catalog_id"] = catalog_id
    return DirectoryContentStoreConfig(**kwargs)


def _init_catalog_repo(
    repo_path: str | Path, backend_type: str, store_dir: Path | None = None
) -> Repo:
    if backend_type == "pointer":
        store_dir = store_dir or Path(repo_path).parent / "content-store"
        store_dir.mkdir(parents=True, exist_ok=True)
        config = directory_store_config(store_dir)
        return Catalog.init_repo_path(repo_path, content_store_config=config)
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
    store_dir = root / "content-store"
    repo = _init_catalog_repo(repo_path, backend_type, store_dir=store_dir)
    catalog = _make_catalog_from_repo(repo, backend_type, store_dir=store_dir)
    for path in _data_dict_template.values():
        catalog.add(path)
    return repo_path, store_dir


@pytest.fixture(autouse=True)
def _cwd_in_tmp_path(tmp_path, monkeypatch):
    # compose, join and union build into ./builds; keep that out of the repo
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def repo(tmpdir, backend_type):
    repo = _init_catalog_repo(Path(tmpdir).joinpath("repo"), backend_type)
    yield repo


@pytest.fixture
def catalog(repo, backend_type):
    store_dir = Path(repo.working_dir).parent / "content-store"
    yield _make_catalog_from_repo(repo, backend_type, store_dir=store_dir)


@pytest.fixture
def catalog_path(catalog):
    yield str(catalog.repo_path)


@pytest.fixture
def world_aliases() -> tuple[str, ...]:
    """Aliases the shared `world` entry is added under; override per module."""
    return ("live",)


@pytest.fixture
def world(
    tmp_path: Path, catalog_path: str, world_aliases: tuple[str, ...]
) -> SimpleNamespace:
    """A sqlite table `t` holding RECORDED, and a catalog entry built over it."""
    db_path = tmp_path / "live.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", RECORDED.to_pandas())
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    t = con.table("t")
    entry = catalog.add(t.filter(t.a > 1), aliases=world_aliases)
    return SimpleNamespace(
        db_path=db_path,
        con=con,
        catalog=catalog,
        catalog_path=catalog_path,
        name=entry.name,
        tmp_path=tmp_path,
    )


@pytest.fixture
def catalog_populated(tmpdir, backend_type, _catalog_populated_template):
    template_repo_path, template_store_dir = _catalog_populated_template
    dst = Path(tmpdir).joinpath("repo")
    shutil.copytree(template_repo_path, dst, symlinks=True)
    store_dir = Path(tmpdir).joinpath("content-store")
    if template_store_dir.exists():
        shutil.copytree(template_store_dir, store_dir)
    repo = Repo(dst)
    yield _make_catalog_from_repo(repo, backend_type, store_dir=store_dir)


@pytest.fixture
def root_repo(tmpdir):
    root_path = Path(tmpdir).joinpath("root")
    root_path.mkdir()
    repo = Repo.init(root_path, initial_branch=MAIN_BRANCH)
    (root_path / "README.md").write_text("root repo")
    repo.index.add(["README.md"])
    repo.index.commit("initial commit")
    yield repo


class _AutoVenvCliRunner(CliRunner):
    """CliRunner that auto-injects ``--use-this-venv`` on ``run`` /
    ``run-cached`` / ``compose`` invocations.

    These commands default to spawning ``uv tool run`` for isolated execution.
    Under CliRunner the subprocess writes to OS fd 1, which bypasses
    CliRunner's ``sys.stdout`` capture and yields an empty ``result.output``.
    The ``--use-this-venv`` flag opts into the in-process path so tests stay
    fast and output-capturable.
    """

    def invoke(self, cli, args=(), **kwargs):
        args = list(args)
        catalog_opts_with_value = {
            "-n",
            "--name",
            "-p",
            "--path",
            "-u",
            "--url",
            "-r",
            "--root-repo",
        }
        i = 0
        while i < len(args):
            a = args[i]
            if a in catalog_opts_with_value:
                i += 2
                continue
            if any(a.startswith(f"{opt}=") for opt in catalog_opts_with_value):
                i += 1
                continue
            if a.startswith("-"):
                i += 1
                continue
            if a in ("run", "run-cached", "compose"):
                args = args[: i + 1] + ["--use-this-venv"] + args[i + 1 :]
            break
        return super().invoke(cli, args, **kwargs)


@pytest.fixture
def runner():
    yield _AutoVenvCliRunner()


@pytest.fixture
def catalog_with_source_and_transform(catalog_path):
    """Populate a catalog with a source entry and an unbound transform entry."""
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)

    source = xo.memtable(
        {"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0], "name": ["a", "b", "c"]}
    )
    source_entry = catalog.add(source, aliases=("src",))

    schema = source.schema()
    unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
    transform = unbound.filter(unbound.amount > 0).select("user_id", "amount")
    transform_entry = catalog.add(transform, aliases=("trn",))

    return catalog_path, source_entry.name, transform_entry.name


@pytest.fixture
def catalog_with_two_sources(catalog_path: str) -> tuple[str, str, str]:
    """Populate a catalog with two independent, joinable/unionable source entries."""
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)

    left = xo.memtable(
        {"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}, name="left_source"
    )
    left_entry = catalog.add(left, aliases=("left-src",))

    right = xo.memtable(
        {"user_id": [1, 2, 4], "name": ["a", "b", "d"]}, name="right_source"
    )
    right_entry = catalog.add(right, aliases=("right-src",))

    return catalog_path, left_entry.name, right_entry.name


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


# ---------------------------------------------------------------------------
# Shared helpers for catalog pull tests
# ---------------------------------------------------------------------------


def add_expr_entry(catalog, table_name, value, aliases=()):
    """Add a tiny memtable expression as a catalog entry; return its hash name."""
    expr = xo.memtable({"x": [value]}, name=table_name)
    with build_expr_context_zip(expr) as zip_path:
        entry = catalog.add(zip_path, sync=False, aliases=aliases)
    return entry.name


def remove_alias(catalog, alias):
    CatalogAlias.from_name(alias, catalog).remove()


def alias_target_hash(catalog: Catalog, alias: str) -> str:
    return CatalogAlias.from_name(alias, catalog).catalog_entry.name


def fail_nth_alias_move(
    monkeypatch: pytest.MonkeyPatch, n: int, after_write: bool = False
) -> None:
    """Fail the ``n``th alias move; ``after_write`` fails it once the symlink is
    written, as a failed commit would."""
    add = CatalogAlias.add
    calls = []

    def failing_add(self: CatalogAlias) -> None:
        calls.append(self.alias)
        if len(calls) == n:
            if after_write:
                self._add()
            raise RuntimeError("alias move failed")
        return add(self)

    monkeypatch.setattr(CatalogAlias, "add", failing_add)


def pull_then(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[Catalog], object]
) -> None:
    """Stand in for a sync whose pull applies ``change`` to the catalog."""

    @contextmanager
    def pulling(self: Catalog, sync: bool):
        if sync:
            change(self)
        yield

    monkeypatch.setattr(Catalog, "maybe_synchronizing", pulling)


@pytest.fixture
def two_clones(tmpdir):
    """Two plain-git catalogs sharing a bare origin, both at the boot commit.

    Initial state on origin/main:
        entries: ['boot']
        aliases: ['boot-alias' -> boot]
    """
    workdir = Path(tmpdir)
    origin_path = workdir / "origin.git"
    user_a_path = workdir / "user-a"
    user_b_path = workdir / "user-b"

    Repo.init(origin_path, bare=True, initial_branch=MAIN_BRANCH)

    cat_a = Catalog.from_repo_path(user_a_path, init=True, annex=False)
    add_expr_entry(cat_a, "boot", value=0, aliases=("boot-alias",))
    cat_a.repo.create_remote(name="origin", url=str(origin_path))
    cat_a.repo.git.push("-u", "origin", MAIN_BRANCH)

    cat_b = Catalog.clone_from(url=str(origin_path), repo_path=user_b_path, annex=False)
    return cat_a, cat_b
