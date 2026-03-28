import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from attr import evolve
from git import Repo as GitRepo

import xorq.api as xo
from xorq.catalog.annex import LOCAL_ANNEX, Annex, DirectoryRemoteConfig, S3RemoteConfig
from xorq.catalog.backend import GitAnnexBackend, GitBackend
from xorq.catalog.catalog import (
    Catalog,
    CatalogAddition,
    CatalogAlias,
    CatalogEntry,
)
from xorq.catalog.constants import CatalogInfix
from xorq.catalog.expr_utils import (
    build_expr_context_zip,
)
from xorq.catalog.tests.conftest import (
    compare_repo_and_catalog,
)
from xorq.catalog.zip_utils import (
    BuildZip,
    with_pure_suffix,
    write_zip,
)
from xorq.ibis_yaml.enums import REQUIRED_ARCHIVE_NAMES, ExprKind


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
    with pytest.raises(ValueError, match="already exists"):
        catalog.add(path)


def test_catalog_addition_from_expr(catalog):
    expr = xo.memtable({"from-expr": ["from-expr"]})
    catalog_addition = CatalogAddition.from_expr(expr, catalog)
    assert catalog_addition.build_zip.path.exists()
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
    zip_path = catalog_addition.build_zip.path
    assert zip_path.exists()
    del catalog_addition
    assert not zip_path.exists()


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

    with build_expr_context_zip(xo.memtable({"to-push": ["to-push"]})) as zip_path:
        cloned.add(zip_path)
        cloned.push()

    after = cloned.list()
    compare_repo_and_catalog(repo_cloned_bare, cloned)

    assert before != after


# ---------------------------------------------------------------------------
# clone_from auto-detection
# ---------------------------------------------------------------------------


def test_clone_from_auto_detects_annex(repo_cloned_bare, tmpdir):
    """clone_from with annex=None auto-detects git-annex branch."""
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("auto-detect")
    )
    assert isinstance(cloned.backend, GitAnnexBackend)
    assert cloned.list()


def test_clone_from_no_annex_branch(tmpdir):
    """clone_from with annex=None on a plain-git repo returns GitBackend."""

    origin_path = Path(tmpdir).joinpath("origin")
    catalog = Catalog.from_repo_path(origin_path, init=True)
    with build_expr_context_zip(xo.memtable({"plain": ["plain"]})) as zip_path:
        catalog.add(zip_path)

    cloned = Catalog.clone_from(str(origin_path), Path(tmpdir).joinpath("clone-plain"))
    assert isinstance(cloned.backend, GitBackend)
    assert cloned.list()


def test_clone_from_false_forces_plain_git(repo_cloned_bare, tmpdir):
    """annex=False forces GitBackend even when git-annex branch exists."""

    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir,
        Path(tmpdir).joinpath("force-plain"),
        annex=False,
    )
    assert isinstance(cloned.backend, GitBackend)


# ---------------------------------------------------------------------------
# from_repo_path auto-detection
# ---------------------------------------------------------------------------


def test_from_repo_path_auto_detects_annex(tmpdir):
    """from_repo_path with annex=None auto-detects .git/annex."""
    repo_path = Path(tmpdir).joinpath("annex-repo")
    Catalog.from_repo_path(repo_path, init=True, annex=LOCAL_ANNEX)

    reopened = Catalog.from_repo_path(repo_path)
    assert isinstance(reopened.backend, GitAnnexBackend)


def test_remote_log_available_after_init(tmpdir):
    """remote.log is readable immediately after initremote (journal flushed)."""
    remote_dir = Path(tmpdir).joinpath("remote-store")
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = Path(tmpdir).joinpath("repo")
    Catalog.from_repo_path(repo_path, init=True, annex=remote_config)

    annex = Annex(repo_path=repo_path)
    remote_log = annex.remote_log
    assert remote_log
    config = next(iter(remote_log.values()))
    assert config["name"] == "mydir"
    assert config["type"] == "directory"


def test_from_repo_path_no_annex(tmpdir):
    """from_repo_path with annex=None on a plain-git repo returns GitBackend."""
    repo_path = Path(tmpdir).joinpath("plain-repo")
    Catalog.from_repo_path(repo_path, init=True)

    reopened = Catalog.from_repo_path(repo_path)
    assert isinstance(reopened.backend, GitBackend)


def test_from_repo_path_false_forces_plain_git(tmpdir):
    """annex=False forces GitBackend even when .git/annex exists."""
    repo_path = Path(tmpdir).joinpath("annex-repo")
    Catalog.from_repo_path(repo_path, init=True, annex=LOCAL_ANNEX)

    reopened = Catalog.from_repo_path(repo_path, annex=False)
    assert isinstance(reopened.backend, GitBackend)


@pytest.mark.parametrize("elide", REQUIRED_ARCHIVE_NAMES)
def test_test_zip(elide, catalog, tmpdir):
    zip_path = write_zip(
        Path(tmpdir).joinpath("build.zip"),
        {name: b"" for name in REQUIRED_ARCHIVE_NAMES if name != elide},
    )
    with pytest.raises(AssertionError, match=elide):
        BuildZip(zip_path)


def test_assert_consistency(catalog, tmpdir):
    zip_path = write_zip(
        Path(tmpdir).joinpath("build.zip"),
        dict.fromkeys(REQUIRED_ARCHIVE_NAMES, b""),
    )
    catalog_addition = CatalogAddition(BuildZip(zip_path), catalog)
    catalog_addition.ensure_dirs()
    catalog_path = catalog_addition.catalog_entry.catalog_path
    with catalog.commit_context("bad commit"):
        shutil.copy(
            zip_path,
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
    assert catalog_alias.target == Path("..") / CatalogInfix.ENTRY / (name + ".zip")
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
            catalog_populated.repo_path / CatalogInfix.ENTRY / (name_b + ".zip")
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
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("cloned"), annex=LOCAL_ANNEX
    )
    catalog_entries = cloned.catalog_entries
    exprs = tuple(catalog_entry.expr for catalog_entry in catalog_entries)
    assert exprs


def test_extract_kind_source(catalog):
    expr = xo.memtable({"a": [1, 2, 3]})
    entry = catalog.add(expr)
    assert entry.kind == ExprKind.Source


def test_extract_kind_bound(catalog):
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    entry = catalog.add(expr)
    assert entry.kind == ExprKind.Expr


def test_extract_kind_partial(catalog):
    t = xo.table(schema={"a": "int64"})
    expr = t.filter(t.a > 0)
    entry = catalog.add(expr)
    assert entry.kind == ExprKind.UnboundExpr


def test_schema_out_bound(catalog):
    expr = xo.memtable({"col_a": [1, 2], "col_b": ["x", "y"]})
    entry = catalog.add(expr)
    assert entry.metadata.schema_out == xo.Schema({"col_a": "int64", "col_b": "string"})


def test_schema_in_none_for_bound(catalog):
    expr = xo.memtable({"a": [1, 2, 3]})
    entry = catalog.add(expr)
    assert entry.metadata.schema_in is None


def test_schema_out_unbound(catalog):
    t = xo.table(schema={"amount": "float64", "currency": "string"})
    expr = t.mutate(amount_usd=t.amount * 1.2)
    entry = catalog.add(expr)
    assert entry.metadata.schema_out == xo.Schema(
        {
            "amount": "float64",
            "currency": "string",
            "amount_usd": "float64",
        }
    )


def test_schema_in_unbound(catalog):
    t = xo.table(schema={"amount": "float64", "currency": "string"})
    expr = t.filter(t.amount > 0)
    entry = catalog.add(expr)
    assert entry.metadata.schema_in == xo.Schema(
        {"amount": "float64", "currency": "string"}
    )


def test_get_entry_by_alias(catalog):
    expr = xo.memtable({"x": [1]})
    entry = catalog.add(expr, aliases=("my-alias",))
    resolved = catalog.get_catalog_entry("my-alias", maybe_alias=True)
    assert resolved.name == entry.name


def test_get_entry_unknown_raises(catalog):
    with pytest.raises(ValueError, match="no-such"):
        catalog.get_catalog_entry("no-such")


def test_directory_remote(tmpdir):
    """Add entries, copy to a directory remote, drop local, get back."""
    remote_dir = Path(tmpdir).joinpath("remote-store")
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = Path(tmpdir).joinpath("repo")
    Catalog.init_repo_path(repo_path, annex=remote_config)
    annex = Annex(repo_path=repo_path)
    backend = GitAnnexBackend(repo=GitRepo(repo_path), annex=annex)
    catalog = Catalog(backend=backend)

    # add two entries
    with build_expr_context_zip(xo.memtable({"x": [1]})) as zp:
        entry = catalog.add(zp, sync=False)
    catalog.assert_consistency()
    assert entry.catalog_path.is_symlink()

    # copy content to directory remote
    annex.copy(to="mydir")
    # content is in the remote directory
    assert any(remote_dir.rglob("*"))

    # drop local content — symlink stays, but target is gone
    annex.drop()
    assert entry.catalog_path.is_symlink()
    assert not entry.catalog_path.exists()

    # get content back from directory remote
    annex.get()
    assert entry.catalog_path.exists()


def test_annex_is_content_local_after_drop(tmpdir):
    """is_content_local is False when annex content has been dropped."""
    remote_dir = Path(tmpdir).joinpath("remote-store")
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = Path(tmpdir).joinpath("repo")
    Catalog.init_repo_path(repo_path, annex=remote_config)
    annex = Annex(repo_path=repo_path)
    backend = GitAnnexBackend(repo=GitRepo(repo_path), annex=annex)
    catalog = Catalog(backend=backend)

    with build_expr_context_zip(xo.memtable({"x": [1]})) as zp:
        entry = catalog.add(zp, sync=False)

    assert entry.is_content_local
    assert entry.is_available

    annex.copy(to="mydir")
    annex.drop()

    assert entry.exists(), "entry should still be registered"
    assert not entry.is_content_local
    assert not entry.is_available


def test_annex_read_after_drop_raises_content_not_available(tmpdir):
    """Accessing content after drop raises ContentNotAvailableError."""
    remote_dir = Path(tmpdir).joinpath("remote-store")
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = Path(tmpdir).joinpath("repo")
    Catalog.init_repo_path(repo_path, annex=remote_config)
    annex = Annex(repo_path=repo_path)
    backend = GitAnnexBackend(repo=GitRepo(repo_path), annex=annex)
    catalog = Catalog(backend=backend)

    with build_expr_context_zip(xo.memtable({"x": [1]})) as zp:
        entry = catalog.add(zp, sync=False)

    annex.copy(to="mydir")
    annex.drop()

    # metadata reads from sidecar — works without content
    assert entry.metadata is not None
    assert entry.kind is not None

    # expr auto-fetches from the remote
    assert entry.expr is not None

    # drop again and verify get also auto-fetches
    annex.drop()
    out_dir = str(tmpdir.join("out"))
    Path(out_dir).mkdir()
    result = entry.get(dir_path=out_dir)
    assert result.exists()


def test_annex_fetch_restores_content(tmpdir):
    """entry.fetch() retrieves annex content for a single entry."""
    remote_dir = Path(tmpdir).joinpath("remote-store")
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = Path(tmpdir).joinpath("repo")
    Catalog.init_repo_path(repo_path, annex=remote_config)
    annex = Annex(repo_path=repo_path)
    backend = GitAnnexBackend(repo=GitRepo(repo_path), annex=annex)
    catalog = Catalog(backend=backend)

    with build_expr_context_zip(xo.memtable({"x": [1]})) as zp:
        entry = catalog.add(zp, sync=False)

    annex.copy(to="mydir")
    annex.drop()
    assert not entry.is_content_local

    entry.fetch()
    assert entry.is_content_local
    assert entry.is_available
    # content is actually readable after fetch
    assert entry.metadata is not None


def test_plain_git_is_content_local(catalog):
    """Plain-git entries always have content local."""
    expr = xo.memtable({"x": [1]})
    entry = catalog.add(expr)
    assert entry.is_content_local
    assert entry.is_available


def test_sidecar_contains_promoted_fields(catalog):
    """Sidecar metadata file contains expr_metadata and backends."""
    expr = xo.memtable({"x": [1], "y": ["a"]})
    entry = catalog.add(expr)
    sidecar = entry.sidecar_metadata
    assert "md5sum" in sidecar
    assert "expr_metadata" in sidecar
    em = sidecar["expr_metadata"]
    assert "kind" in em
    assert "schema_out" in em
    assert set(em["schema_out"]) == {"x", "y"}
    assert "backends" in sidecar
    assert isinstance(sidecar["backends"], list)


def test_metadata_from_sidecar_after_drop(tmpdir):
    """All metadata properties work from sidecar after annex content is dropped."""
    remote_dir = Path(tmpdir).joinpath("remote-store")
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = Path(tmpdir).joinpath("repo")
    Catalog.init_repo_path(repo_path, annex=remote_config)
    annex = Annex(repo_path=repo_path)
    backend = GitAnnexBackend(repo=GitRepo(repo_path), annex=annex)
    catalog = Catalog(backend=backend)

    with build_expr_context_zip(xo.memtable({"x": [1]})) as zp:
        entry = catalog.add(zp, sync=False)

    annex.copy(to="mydir")
    annex.drop()
    assert not entry.is_content_local

    # all sidecar-backed properties work without content
    assert entry.kind == ExprKind.Source
    assert entry.columns == ("x",)
    assert entry.root_tag is not None or entry.root_tag == ""
    assert isinstance(entry.backends, tuple)
    assert entry.metadata.to_dict() is not None

    # expr auto-fetches content from the remote
    assert entry.expr is not None


@pytest.mark.s3
def test_s3_remote_minio(tmpdir):
    """Add entries, copy to S3 (minio), drop local, get back."""
    remote_config = S3RemoteConfig.from_env(
        name="mys3",
        bucket=f"test-annex-{uuid.uuid4().hex[:12]}",
        host="minio",
        port="9000",
        aws_access_key_id="accesskey",
        aws_secret_access_key="secretkey",
        protocol="http",
        requeststyle="path",
        signature="v2",
    )
    # check minio is reachable
    result = subprocess.run(
        [
            "curl",
            "-sf",
            f"http://{remote_config.host}:{remote_config.port}/minio/health/live",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("minio not reachable")
    repo_path = Path(tmpdir).joinpath("repo")
    repo_path.mkdir(parents=True)
    repo = GitRepo.init(repo_path)
    repo.index.commit("initial commit")
    # allow private IPs for minio
    subprocess.run(
        ["git", "config", "annex.security.allowed-ip-addresses", "all"],
        cwd=repo_path,
        check=True,
    )
    Annex.init_repo_path(repo_path, remote_config=remote_config)

    annex = Annex(repo_path=repo_path, env=remote_config.env)
    backend = GitAnnexBackend(repo=repo, annex=annex)
    catalog = Catalog(backend=backend)

    # add an entry
    with build_expr_context_zip(xo.memtable({"s3col": [42]})) as zp:
        entry = catalog.add(zp, sync=False)
    catalog.assert_consistency()
    assert entry.catalog_path.is_symlink()

    # copy to S3
    annex.copy(to="mys3")

    # drop local content
    annex.drop()
    assert entry.catalog_path.is_symlink()
    assert not entry.catalog_path.exists()

    # get content back from S3
    annex.get()
    assert entry.catalog_path.exists()
