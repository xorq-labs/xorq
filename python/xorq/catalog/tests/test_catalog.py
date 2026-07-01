import gc
import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pandas as pd
import pytest
from attr import evolve
from git import PushInfo
from git import Repo as GitRepo

import xorq.api as xo
import xorq.catalog.catalog as catalog_mod
from xorq.caching import ParquetSnapshotCache
from xorq.catalog.annex import (
    LOCAL_ANNEX,
    Annex,
    AnnexError,
    DirectoryRemoteConfig,
    S3RemoteConfig,
    _do_inside,
)
from xorq.catalog.backend import (
    GitAnnexBackend,
    GitBackend,
    GitPointerBackend,
)
from xorq.catalog.catalog import (
    Catalog,
    CatalogAddition,
    CatalogAlias,
    CatalogEntry,
    _format_push_failures,
)
from xorq.catalog.constants import CONTENT_STORE_YAML, MAIN_BRANCH
from xorq.catalog.content_store import (
    POINTER_VERSION,
    ContentCache,
    ContentIntegrityError,
    ContentStoreConfig,
    DirectoryContentStore,
    DirectoryContentStoreConfig,
    S3ContentStoreConfig,
    _coerce_port,
    compute_content_key,
    compute_sha256,
    parse_pointer,
    write_pointer,
)
from xorq.catalog.enums import CatalogInfix
from xorq.catalog.exceptions import (
    CatalogConfigurationError,
    CatalogPushError,
)
from xorq.catalog.expr_utils import (
    _live_extract_dirs,
    build_expr_context_zip,
)
from xorq.catalog.tests.conftest import (
    TEST_WHEEL_NAME,
    compare_repo_and_catalog,
    directory_store_config,
    requires_annex,
)
from xorq.catalog.tui import get_cache_key_path
from xorq.catalog.zip_utils import (
    BuildZip,
    with_pure_suffix,
    write_zip,
)
from xorq.common.utils.caching_utils import CacheKey
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


def test_catalog_add_expr_threads_project_path(catalog, tmp_path, monkeypatch):
    # Verify the project_path kwarg on Catalog.add is plumbed all the way
    # through to _ensure_wheel_artifacts, so callers outside the project cwd
    # (e.g. Jupyter kernels) can opt out of the upward-walk.
    explicit_project = tmp_path / "explicit-project"
    explicit_project.mkdir()
    captured = {}
    original = catalog_mod._ensure_wheel_artifacts

    def spy(build_dir, project_path=None):
        captured["project_path"] = project_path
        return original(build_dir, project_path=project_path)

    monkeypatch.setattr(catalog_mod, "_ensure_wheel_artifacts", spy)

    expr = xo.memtable({"threaded": ["threaded"]})
    catalog_entry = catalog.add(expr, project_path=explicit_project)

    assert captured["project_path"] == explicit_project
    assert catalog_entry.exists()
    catalog.assert_consistency()


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
    with pytest.raises(ValueError):
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


def test_push_surfaces_remote_rejection(repo_cloned_bare, tmpdir):
    """catalog.push() must surface a remote rejection rather than returning silently.

    Scenario: two clones from the same bare remote, both add a different
    entry. User A pushes first (clean fast-forward). User B's local main
    has now diverged from origin/main, so the next push is rejected by
    the remote with "fetch first". xorq#1898 — today catalog.push()
    returns normally despite GitPython reporting REJECTED|ERROR on the
    main ref.
    """
    user_a = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("a")
    )
    user_b = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("b")
    )

    with build_expr_context_zip(xo.memtable({"a-only": ["a"]})) as zp:
        user_a.add(zp, sync=False)
    user_a.push()

    with build_expr_context_zip(xo.memtable({"b-only": ["b"]})) as zp:
        user_b.add(zp, sync=False)

    with pytest.raises(CatalogPushError, match="(?i)reject"):
        user_b.push()


def test_push_annex_branch_rejection(repo_cloned_bare, tmpdir, monkeypatch):
    """push() surfaces git-annex branch rejections, not just main rejections.

    Main pushes cleanly but the git-annex branch is rejected. push() must
    raise CatalogPushError with a message that names the git-annex ref,
    proving the annex failure path through _format_push_failures is wired up.
    """
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("c")
    )
    with build_expr_context_zip(xo.memtable({"annex-rej": ["v"]})) as zp:
        cloned.add(zp, sync=False)
    assert "git-annex" in cloned.repo.heads

    class FakeInfo:
        def __init__(self, flags, local_ref, summary):
            self.flags = flags
            self.local_ref = local_ref
            self.summary = summary

    call_count = 0
    original_push = type(cloned.repo.remotes.origin).push

    def patched_push(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_push(self, *args, **kwargs)
        return [
            FakeInfo(
                PushInfo.REJECTED, "refs/heads/git-annex", "[rejected] (fetch first)"
            )
        ]

    monkeypatch.setattr(type(cloned.repo.remotes.origin), "push", patched_push)

    with pytest.raises(CatalogPushError, match="git-annex") as exc_info:
        cloned.push()
    assert "rejected" in str(exc_info.value).lower()


class TestFormatPushFailures:
    """Unit tests for _format_push_failures — the string the user sees on error."""

    def _make_info(self, flags, local_ref="refs/heads/main", summary="rejected"):
        class FakePushInfo:
            pass

        info = FakePushInfo()
        info.flags = flags
        info.local_ref = local_ref
        info.summary = summary
        return info

    def test_filters_only_failures(self):
        ok = self._make_info(PushInfo.FAST_FORWARD)
        bad = self._make_info(PushInfo.REJECTED, summary="fetch first")
        result = _format_push_failures([ok, bad], "origin")
        assert len(result) == 1
        assert "fetch first" in result[0]
        assert "origin/" in result[0]

    def test_all_failure_flag_variants(self):
        for flag in (
            PushInfo.REJECTED,
            PushInfo.REMOTE_REJECTED,
            PushInfo.REMOTE_FAILURE,
            PushInfo.ERROR,
        ):
            result = _format_push_failures(
                [self._make_info(flag, summary="oops")], "origin"
            )
            assert len(result) == 1

    def test_local_ref_none_fallback(self):
        info = self._make_info(PushInfo.REJECTED, local_ref=None)
        result = _format_push_failures([info], "origin")
        assert "origin/?" in result[0]

    def test_summary_none_fallback(self):
        info = self._make_info(PushInfo.REJECTED, summary=None)
        result = _format_push_failures([info], "origin")
        assert "origin/" in result[0]
        assert result[0].endswith(": ")

    def test_summary_stripped(self):
        info = self._make_info(PushInfo.REJECTED, summary="  spaces  ")
        result = _format_push_failures([info], "origin")
        assert result[0].endswith(": spaces")

    def test_empty_list(self):
        assert _format_push_failures([], "origin") == ()


@pytest.mark.parametrize("op", ["push", "pull", "fetch", "sync"])
def test_multi_remote_raises_configuration_error(tmpdir, op):
    """ADR-0011: catalog refuses to operate on configurations with 2+ git remotes.

    push/pull/fetch/sync each must raise rather than attempt best-effort
    multi-remote semantics. The error message names both remotes so the
    user can see what the catalog discovered.
    """
    bare_1 = Path(tmpdir).joinpath("bare_1")
    bare_2 = Path(tmpdir).joinpath("bare_2")
    GitRepo.init(bare_1, bare=True, initial_branch=MAIN_BRANCH)
    GitRepo.init(bare_2, bare=True, initial_branch=MAIN_BRANCH)

    catalog = Catalog.from_repo_path(Path(tmpdir).joinpath("local"), init=True)
    catalog.repo.create_remote("r1", str(bare_1))
    catalog.repo.create_remote("r2", str(bare_2))

    with pytest.raises(
        CatalogConfigurationError, match="(?i)single git remote"
    ) as exc_info:
        getattr(catalog, op)()
    msg = str(exc_info.value)
    assert "r1" in msg
    assert "r2" in msg


def test_set_remote_refuses_when_remote_exists(tmpdir):
    """ADR-0011: Catalog.set_remote refuses to silently replace an existing remote.

    A second call without ``force=True`` raises CatalogConfigurationError so
    the user has to explicitly opt in to overwriting the previously
    configured remote (it would otherwise vanish without trace).
    """
    bare_1 = Path(tmpdir).joinpath("bare_1")
    bare_2 = Path(tmpdir).joinpath("bare_2")
    GitRepo.init(bare_1, bare=True, initial_branch=MAIN_BRANCH)
    GitRepo.init(bare_2, bare=True, initial_branch=MAIN_BRANCH)

    catalog = Catalog.from_repo_path(Path(tmpdir).joinpath("local"), init=True)
    catalog.set_remote("origin", str(bare_1))

    with pytest.raises(CatalogConfigurationError, match="(?i)remote.*already") as exc:
        catalog.set_remote("origin", str(bare_2))
    assert "force" in str(exc.value).lower()
    # the existing remote is preserved
    assert len(catalog._git_remotes) == 1
    assert catalog._git_remotes[0].url == str(bare_1)


def test_set_remote_force_replaces_existing(tmpdir):
    """ADR-0011: Catalog.set_remote(..., force=True) replaces any existing git remote.

    Successive calls with ``force=True`` must leave the catalog with exactly
    one git remote, so users can reconfigure without ever creating the
    multi-remote configuration that triggers CatalogConfigurationError.
    """
    bare_1 = Path(tmpdir).joinpath("bare_1")
    bare_2 = Path(tmpdir).joinpath("bare_2")
    GitRepo.init(bare_1, bare=True, initial_branch=MAIN_BRANCH)
    GitRepo.init(bare_2, bare=True, initial_branch=MAIN_BRANCH)

    catalog = Catalog.from_repo_path(Path(tmpdir).joinpath("local"), init=True)
    catalog.set_remote("origin", str(bare_1))
    assert len(catalog._git_remotes) == 1

    catalog.set_remote("origin", str(bare_2), force=True)
    assert len(catalog._git_remotes) == 1
    assert catalog._git_remotes[0].url == str(bare_2)


def test_set_remote_force_recovers_from_multi_remote_state(tmpdir):
    """set_remote(force=True) is the recovery path from an ADR-0011 violation.

    A user (or another tool) can put the catalog into a 2+ remote state via
    raw ``git remote add``, after which ``push``/``pull``/``fetch``/``sync``
    refuse to operate. ``set_remote(force=True)`` must collapse the state to
    exactly one remote so the catalog is operable again.
    """
    bare_1 = Path(tmpdir).joinpath("bare_1")
    bare_2 = Path(tmpdir).joinpath("bare_2")
    bare_3 = Path(tmpdir).joinpath("bare_3")
    for bare in (bare_1, bare_2, bare_3):
        GitRepo.init(bare, bare=True, initial_branch=MAIN_BRANCH)

    catalog = Catalog.from_repo_path(Path(tmpdir).joinpath("local"), init=True)
    catalog.repo.create_remote("r1", str(bare_1))
    catalog.repo.create_remote("r2", str(bare_2))
    assert len(catalog._git_remotes) == 2

    catalog.set_remote("origin", str(bare_3), force=True)

    assert len(catalog._git_remotes) == 1
    assert catalog._git_remotes[0].name == "origin"
    assert catalog._git_remotes[0].url == str(bare_3)


def test_set_remote_preserves_annex_special_remote(tmpdir):
    """set_remote replaces only git remotes; annex special remotes survive.

    `_git_remotes` filters by fetch refspec, so a remote configured with no
    fetch line (the shape of a git-annex special remote — credentials live
    in remote.log on the git-annex branch, not in .git/config) is not a
    git remote for ADR-0011 purposes. set_remote must not delete it.
    """
    bare = Path(tmpdir).joinpath("bare")
    GitRepo.init(bare, bare=True, initial_branch=MAIN_BRANCH)

    catalog = Catalog.from_repo_path(Path(tmpdir).joinpath("local"), init=True)

    config_section = 'remote "s3-bucket"'
    with catalog.repo.config_writer() as writer:
        writer.add_section(config_section)
        writer.set(config_section, "annex-uuid", "fake-uuid")

    assert "s3-bucket" in {r.name for r in catalog.repo.remotes}
    assert "s3-bucket" not in {r.name for r in catalog._git_remotes}

    catalog.set_remote("origin", str(bare))

    remote_names = {r.name for r in catalog.repo.remotes}
    assert "s3-bucket" in remote_names
    assert "origin" in remote_names
    assert {r.name for r in catalog._git_remotes} == {"origin"}


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


def test_push_no_remote_skips_consistency_check(tmpdir, monkeypatch):
    """``Catalog.push()`` is a true no-op for zero-remote catalogs.

    The docstring says push is a no-op when no git remote is configured.
    A no-op cannot run ``assert_consistency`` — that check could surface
    unrelated repo errors even though the user did not ask the catalog
    to talk to anything. Asserts the zero-remote path returns ``()``
    without calling consistency.
    """
    catalog = Catalog.from_repo_path(Path(tmpdir).joinpath("local-only"), init=True)
    assert catalog._git_remotes == ()

    def boom(self, *args, **kwargs):
        raise AssertionError("assert_consistency should not run for zero-remote push")

    monkeypatch.setattr(Catalog, "assert_consistency", boom)
    assert catalog.push() == ()


def test_push_skips_missing_annex_branch(tmpdir):
    """Catalog.push() succeeds when the local repo has no git-annex branch."""

    bare_path = Path(tmpdir).joinpath("bare")
    GitRepo.init(bare_path, bare=True, initial_branch=MAIN_BRANCH)

    origin_path = Path(tmpdir).joinpath("origin")
    catalog = Catalog.from_repo_path(origin_path, init=True)
    remote = catalog.repo.create_remote("origin", str(bare_path))
    remote.push(MAIN_BRANCH, set_upstream=True)

    with build_expr_context_zip(xo.memtable({"plain": ["plain"]})) as zip_path:
        catalog.add(zip_path, sync=False)

    assert "git-annex" not in catalog.repo.heads
    result = catalog.push()
    assert len(result) == 1


def test_push_returns_two_element_tuple_with_annex(repo_cloned_bare, tmpdir):
    """push() returns (main_result, annex_result) when a local git-annex branch exists."""
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("cloned")
    )
    with build_expr_context_zip(xo.memtable({"shape-check": ["v"]})) as zp:
        cloned.add(zp, sync=False)

    assert "git-annex" in cloned.repo.heads
    result = cloned.push()
    assert len(result) == 2


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


@requires_annex
def test_from_repo_path_auto_detects_annex(tmp_path: Path) -> None:
    """from_repo_path with annex=None auto-detects .git/annex."""
    repo_path = tmp_path / "annex-repo"
    Catalog.from_repo_path(repo_path, init=True, annex=LOCAL_ANNEX)

    reopened = Catalog.from_repo_path(repo_path)
    assert isinstance(reopened.backend, GitAnnexBackend)


@requires_annex
def test_remote_log_available_after_init(tmp_path: Path) -> None:
    """remote.log is readable immediately after initremote (journal flushed)."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
    Catalog.from_repo_path(repo_path, init=True, annex=remote_config)

    annex = Annex(repo_path=repo_path)
    remote_log = annex.remote_log
    assert remote_log
    config = next(iter(remote_log.values()))
    assert config["name"] == "mydir"
    assert config["type"] == "directory"


@requires_annex
def test_enableremote_falls_back_to_initremote_on_empty_remote_log(
    tmp_path: Path,
) -> None:
    """enableremote on a fresh annex repo (no remote.log) creates the remote."""
    repo_path = tmp_path / "repo"
    Catalog.from_repo_path(repo_path, init=True, annex=LOCAL_ANNEX)
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    rc = DirectoryRemoteConfig(name="newdir", directory=str(remote_dir))

    Annex.from_repo_path(repo_path).enableremote(rc)

    remote_log = Annex.from_repo_path(repo_path).remote_log
    assert remote_log
    assert next(iter(remote_log.values()))["name"] == "newdir"


@requires_annex
def test_enableremote_raises_when_name_missing_from_nonempty_remote_log(
    tmp_path: Path,
) -> None:
    """enableremote refuses to silently create a second remote next to an existing one."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    existing = DirectoryRemoteConfig(name="existing", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
    Catalog.from_repo_path(repo_path, init=True, annex=existing)

    other_dir = tmp_path / "other-store"
    other_dir.mkdir()
    other = DirectoryRemoteConfig(name="other", directory=str(other_dir))

    with pytest.raises(AnnexError, match="not registered"):
        Annex.from_repo_path(repo_path).enableremote(other)


def test_from_repo_path_no_annex(tmpdir):
    """from_repo_path with annex=None on a plain-git repo returns GitBackend."""
    repo_path = Path(tmpdir).joinpath("plain-repo")
    Catalog.from_repo_path(repo_path, init=True)

    reopened = Catalog.from_repo_path(repo_path)
    assert isinstance(reopened.backend, GitBackend)


@requires_annex
def test_from_repo_path_false_forces_plain_git(tmp_path: Path) -> None:
    """annex=False forces GitBackend even when .git/annex exists."""
    repo_path = tmp_path / "annex-repo"
    Catalog.from_repo_path(repo_path, init=True, annex=LOCAL_ANNEX)

    reopened = Catalog.from_repo_path(repo_path, annex=False)
    assert isinstance(reopened.backend, GitBackend)


# ---------------------------------------------------------------------------
# Pointer backend / content store
# ---------------------------------------------------------------------------


def test_content_cache_fetch_from_protects_fetched_file(tmp_path: Path) -> None:
    """A fetched file larger than max_bytes is not evicted by its own insertion."""
    root = tmp_path
    store = DirectoryContentStore(directory=root / "store")
    cache = ContentCache(cache_dir=root / "cache", max_bytes=10)
    key = "cat/aa/bb/deadbeef.zip"
    src = root / "big.bin"
    src.write_bytes(b"x" * 100)  # 100 bytes > max_bytes
    store.put(key, src)

    dest = cache.fetch_from(store, key)
    assert dest.exists()  # protected from its own eviction
    assert dest.read_bytes() == b"x" * 100
    assert cache.get_path(key) is not None


def test_content_cache_put_protects_and_refreshes_atime(tmp_path: Path) -> None:
    """put() survives a full cache and resets the stale source atime for LRU."""
    root = tmp_path
    cache = ContentCache(cache_dir=root / "cache", max_bytes=10)
    src = root / "big.bin"
    src.write_bytes(b"x" * 100)  # 100 bytes > max_bytes
    os.utime(src, (1, 1))  # stale source atime that copy2 would otherwise preserve

    cache.put("cat/aa/bb/big.zip", src)

    dest = cache._path("cat/aa/bb/big.zip")
    assert dest.exists()  # not evicted by its own insertion (protect=key)
    assert dest.stat().st_atime > 1  # atime refreshed, not the stale source value


def test_content_cache_evicts_older_entries(tmp_path: Path) -> None:
    """Older entries are evicted once the cache exceeds max_bytes."""
    root = tmp_path
    store = DirectoryContentStore(directory=root / "store")
    cache = ContentCache(cache_dir=root / "cache", max_bytes=150)
    keys = []
    for i in range(3):
        src = root / f"f{i}.bin"
        src.write_bytes(bytes([i]) * 100)
        key = f"cat/{i:02d}/x/h{i}.zip"
        store.put(key, src)
        cache.fetch_from(store, key)
        keys.append(key)

    total = sum(p.stat().st_size for p in (root / "cache").rglob("*") if p.is_file())
    assert total <= 150
    assert cache.get_path(keys[-1]) is not None  # most recent survives
    assert cache.get_path(keys[0]) is None  # oldest evicted


def test_content_cache_rejects_unwritable_dir(tmp_path: Path) -> None:
    """Construction fails early when the cache directory is not writable."""
    unwritable = tmp_path / "sealed"
    unwritable.mkdir()
    os.chmod(unwritable, 0o444)
    with pytest.raises(OSError, match="not writable"):
        ContentCache(cache_dir=unwritable / "cache", max_bytes=1024)
    os.chmod(unwritable, 0o755)  # restore so tmp_path cleanup works


def test_content_cache_disabled_skips_put(tmp_path: Path) -> None:
    """max_bytes=0 disables caching: put() is a no-op."""
    root = tmp_path
    cache = ContentCache(cache_dir=root / "cache", max_bytes=0)
    src = root / "data.bin"
    src.write_bytes(b"hello")
    cache.put("cat/aa/bb/data.zip", src)
    assert not cache._path("cat/aa/bb/data.zip").exists()
    assert not cache.contains("cat/aa/bb/data.zip")
    assert cache.get_path("cat/aa/bb/data.zip") is None


def test_fetch_content_disabled_cache_bypasses_cache(tmp_path: Path) -> None:
    """max_bytes=0: fetch_content downloads directly to the archive path,
    bypassing the cache entirely — no files left in cache_dir."""
    repo = GitRepo.init(tmp_path / "repo")
    store_dir = tmp_path / "store"
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    config.write_yaml(Path(repo.working_dir) / CONTENT_STORE_YAML)
    store = DirectoryContentStore(directory=store_dir)
    cache_dir = tmp_path / "cache"
    cache = ContentCache(cache_dir=cache_dir, max_bytes=0)
    backend = GitPointerBackend.from_repo(repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"real content")
    sha = compute_sha256(archive)
    key = compute_content_key("testcat", sha)
    store.put(key, archive)

    target = tmp_path / "entry.zip"
    write_pointer(backend._pointer_path(target), sha, archive.stat().st_size)

    backend.fetch_content(target)
    assert target.exists()
    assert target.read_bytes() == b"real content"

    cached_files = [p for p in cache_dir.rglob("*") if p.is_file()]
    assert cached_files == [], "disabled cache should leave no files behind"


def test_content_cache_unlimited_never_evicts(tmp_path: Path) -> None:
    """max_bytes=-1: unlimited cache, nothing is ever evicted."""
    root = tmp_path
    store = DirectoryContentStore(directory=root / "store")
    cache = ContentCache(cache_dir=root / "cache", max_bytes=-1)
    keys = []
    for i in range(5):
        src = root / f"f{i}.bin"
        src.write_bytes(bytes([i]) * 1000)
        key = f"cat/{i:02d}/x/h{i}.zip"
        store.put(key, src)
        cache.fetch_from(store, key)
        keys.append(key)

    for key in keys:
        assert cache.get_path(key) is not None


def test_content_cache_default_respects_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ContentCache.default() reads XORQ_CONTENT_CACHE_DIR and _MAX_BYTES."""
    custom_dir = tmp_path / "custom-cache"
    monkeypatch.setenv("XORQ_CONTENT_CACHE_DIR", str(custom_dir))
    monkeypatch.setenv("XORQ_CONTENT_CACHE_MAX_BYTES", "42")

    cache = ContentCache.default()
    assert cache.cache_dir == custom_dir
    assert cache.max_bytes == 42


def test_directory_content_store_put_is_atomic(tmp_path: Path) -> None:
    """Interrupted put must not leave a partial file at the destination."""
    store = DirectoryContentStore(directory=tmp_path / "store")
    src = tmp_path / "data.bin"
    src.write_bytes(b"good data")

    store.put("cat/aa/bb/test.zip", src, sha256=compute_sha256(src))
    dest = store._key_path("cat/aa/bb/test.zip")
    assert dest.read_bytes() == b"good data"

    bad_sha = "0" * 64
    with pytest.raises(ContentIntegrityError, match="SHA256 mismatch"):
        store.put("cat/aa/bb/other.zip", src, sha256=bad_sha)
    assert not store._key_path("cat/aa/bb/other.zip").exists()
    tmp_leftover = store._key_path("cat/aa/bb/other.zip.tmp")
    assert not tmp_leftover.exists()


def test_directory_content_store_config_resolves_paths() -> None:
    """DirectoryContentStoreConfig resolves paths so equality is stable."""
    a = DirectoryContentStoreConfig(catalog_id="cat", directory="/tmp/./store")
    b = DirectoryContentStoreConfig(catalog_id="cat", directory="/tmp/store")
    assert a == b
    assert a.directory == b.directory


def test_content_store_config_round_trips() -> None:
    """Typed configs round-trip through to_dict/from_dict unchanged."""
    cfg = DirectoryContentStoreConfig(catalog_id="cat", directory="/tmp/store")
    assert ContentStoreConfig.from_dict(cfg.to_dict()) == cfg


def test_content_store_config_yaml_round_trips_relative(tmp_path: Path) -> None:
    """write_yaml stores relative directory paths; from_yaml resolves them back."""
    store_dir = tmp_path / "content-store"
    store_dir.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    yaml_path = repo_dir / "content_store.yaml"

    cfg = DirectoryContentStoreConfig(catalog_id="cat", directory=str(store_dir))
    cfg.write_yaml(yaml_path)

    raw = yaml_path.read_text()
    assert str(store_dir) not in raw, "absolute path should not appear in YAML"
    assert "../content-store" in raw

    loaded = ContentStoreConfig.from_yaml(yaml_path)
    assert loaded == cfg


def test_content_store_config_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown content store type"):
        ContentStoreConfig.from_dict({"type": "nope", "catalog_id": "cat"})


def test_compute_content_key_rejects_unsafe_inputs() -> None:
    """compute_content_key validates catalog_id and sha256 to block path traversal."""
    good_sha = "a" * 64
    assert compute_content_key("cat-1", good_sha).startswith("cat-1/")
    for bad_id in ("../evil", "a/b", "..", ""):
        with pytest.raises(ValueError, match="Unsafe catalog_id"):
            compute_content_key(bad_id, good_sha)
    for bad_sha in ("../../etc/passwd", "XYZ", "a" * 63):
        with pytest.raises(ValueError, match="Invalid sha256"):
            compute_content_key("cat", bad_sha)


def test_parse_pointer_rejects_malformed(tmp_path: Path) -> None:
    """A pointer line missing its value raises ValueError, not IndexError."""
    p = tmp_path / "bad.xorq-pointer"
    p.write_text(f"{POINTER_VERSION}\nsha256\nsize 10\n")  # sha256 line has no value
    with pytest.raises(ValueError, match="Invalid pointer file"):
        parse_pointer(p)


def test_fetch_content_drops_corrupt_cache_entry(tmp_path: Path) -> None:
    """A SHA mismatch removes the cached entry so the next fetch self-heals."""
    repo = GitRepo.init(tmp_path / "repo")
    store_dir = tmp_path / "store"
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    config.write_yaml(Path(repo.working_dir) / CONTENT_STORE_YAML)
    store = DirectoryContentStore(directory=store_dir)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"real content")
    sha = compute_sha256(archive)
    key = compute_content_key("testcat", sha)
    store.put(key, archive)

    target = tmp_path / "entry.zip"
    write_pointer(backend._pointer_path(target), sha, archive.stat().st_size)

    # prime the cache, then corrupt the cached copy
    cache.fetch_from(store, key)
    cached_path = cache._path(key)
    cached_path.write_bytes(b"corrupt")

    with pytest.raises(ContentIntegrityError):
        backend.fetch_content(target)
    assert not cached_path.exists()  # corrupt entry dropped

    # second attempt re-pulls the good copy from the store and succeeds
    backend.fetch_content(target)
    assert target.read_bytes() == b"real content"


def test_fetch_content_drops_cache_entry_on_size_mismatch(tmp_path: Path) -> None:
    """A size mismatch removes the cached entry so the next fetch self-heals."""
    repo = GitRepo.init(tmp_path / "repo")
    store_dir = tmp_path / "store"
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    config.write_yaml(Path(repo.working_dir) / CONTENT_STORE_YAML)
    store = DirectoryContentStore(directory=store_dir)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"real content")
    sha = compute_sha256(archive)
    key = compute_content_key("testcat", sha)
    store.put(key, archive)

    target = tmp_path / "entry.zip"
    # write a pointer with the correct sha but wrong size
    write_pointer(backend._pointer_path(target), sha, archive.stat().st_size + 999)

    cache.fetch_from(store, key)
    cached_path = cache._path(key)
    assert cached_path.exists()

    with pytest.raises(ContentIntegrityError, match="Size mismatch"):
        backend.fetch_content(target)
    assert not cached_path.exists()  # corrupt entry dropped

    # fix the pointer and verify self-healing
    write_pointer(backend._pointer_path(target), sha, archive.stat().st_size)
    backend.fetch_content(target)
    assert target.read_bytes() == b"real content"


def test_fetch_content_no_partial_file_on_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted archive copy leaves no file at the target; retry succeeds."""
    repo = GitRepo.init(tmp_path / "repo")
    store_dir = tmp_path / "store"
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    config.write_yaml(Path(repo.working_dir) / CONTENT_STORE_YAML)
    store = DirectoryContentStore(directory=store_dir)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"real content")
    sha = compute_sha256(archive)
    key = compute_content_key("testcat", sha)
    store.put(key, archive)
    cache.fetch_from(store, key)  # prime cache so the failure hits the archive copy

    target = tmp_path / "entry.zip"
    write_pointer(backend._pointer_path(target), sha, archive.stat().st_size)

    def boom(src, dst):
        Path(dst).write_bytes(b"partial")  # write a partial file, then fail
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", boom)
    with pytest.raises(OSError, match="disk full"):
        backend.fetch_content(target)
    assert not target.exists()  # atomic: no partial file at the final path

    monkeypatch.undo()
    backend.fetch_content(target)  # retry succeeds (cache intact)
    assert target.read_bytes() == b"real content"


def test_from_repo_path_conflicting_content_store_raises(tmp_path: Path) -> None:
    """Opening an existing pointer repo with a conflicting content_store_config raises."""
    repo_path = tmp_path / "pointer-repo"
    Catalog.from_repo_path(
        repo_path,
        init=True,
        content_store_config=directory_store_config(tmp_path / "store-a"),
    )

    conflicting = directory_store_config(tmp_path / "store-b", catalog_id="other")
    with pytest.raises(ValueError, match="conflicts with committed"):
        Catalog.from_repo_path(repo_path, init=False, content_store_config=conflicting)


def test_from_repo_path_matching_content_store_ok(tmp_path: Path) -> None:
    """Reopening with a content_store_config equal to the committed config is accepted."""
    repo_path = tmp_path / "pointer-repo"
    Catalog.from_repo_path(
        repo_path,
        init=True,
        content_store_config=directory_store_config(tmp_path / "store-a"),
    )

    committed = ContentStoreConfig.from_yaml(repo_path / CONTENT_STORE_YAML)
    reopened = Catalog.from_repo_path(
        repo_path, init=False, content_store_config=committed
    )
    assert isinstance(reopened.backend, GitPointerBackend)


def test_from_repo_path_content_store_without_yaml_raises(tmp_path: Path) -> None:
    """init=False with a content_store_config but no committed yaml fails loudly."""
    repo_path = tmp_path / "plain-repo"
    Catalog.from_repo_path(
        repo_path, init=True
    )  # plain-git repo, no content_store.yaml

    with pytest.raises(ValueError, match="is absent"):
        Catalog.from_repo_path(
            repo_path,
            init=False,
            content_store_config=directory_store_config(tmp_path / "store"),
        )


def test_compute_sha256_manual_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compute_sha256 works without hashlib.file_digest (the Python 3.10 path)."""
    f = tmp_path / "data.bin"
    payload = b"hello pointer content" * 1000
    f.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert compute_sha256(f) == expected  # native path (file_digest present)

    monkeypatch.delattr(hashlib, "file_digest", raising=False)  # simulate 3.10
    assert compute_sha256(f) == expected


def test_assert_consistency_strict_for_plain_git(tmp_path: Path) -> None:
    """A stray .gitignore is not silently whitelisted for non-pointer catalogs."""
    repo_path = tmp_path / "plain-repo"
    catalog = Catalog.from_repo_path(repo_path, init=True)
    catalog.assert_consistency()  # consistent to start

    (repo_path / ".gitignore").write_text("*.tmp\n")
    catalog.repo.index.add([".gitignore"])
    catalog.repo.index.commit("add stray .gitignore")

    with pytest.raises(AssertionError):
        catalog.assert_consistency()


def test_pointer_shared_content_ref_counting(tmp_path: Path) -> None:
    """Removing one entry does not delete content still referenced by another."""

    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir)
    git_repo = Catalog.init_repo_path(repo_path, content_store_config=config)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(git_repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"shared content blob")
    sha = compute_sha256(archive)
    key = compute_content_key(backend.catalog_id, sha)

    entries_dir = repo_path / CatalogInfix.ENTRY
    entries_dir.mkdir(exist_ok=True)

    path_a = entries_dir / "entry_a.zip"
    path_b = entries_dir / "entry_b.zip"

    backend.stage_content(archive, path_a)
    backend.stage_content(archive, path_b)
    git_repo.index.commit("add two entries sharing content")

    assert backend.content_store.exists(key)

    backend.stage_unlink(path_a)
    git_repo.index.commit("remove entry_a")
    assert backend.content_store.exists(key), (
        "content should survive: entry_b still references it"
    )

    backend.stage_unlink(path_b)
    git_repo.index.commit("remove entry_b")
    assert not backend.content_store.exists(key), (
        "content should be deleted: no remaining references"
    )


def test_clone_from_conflicting_content_store_raises(tmp_path: Path) -> None:
    """clone_from with a content_store_config that conflicts with the cloned repo raises."""
    repo_path = tmp_path / "origin"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir, catalog_id="original")
    Catalog.init_repo_path(repo_path, content_store_config=config)

    conflicting = directory_store_config(tmp_path / "other", catalog_id="other")
    clone_path = tmp_path / "cloned"
    with pytest.raises(ValueError, match="conflicts with committed"):
        Catalog.clone_from(
            url=str(repo_path),
            repo_path=clone_path,
            content_store_config=conflicting,
        )
    assert not clone_path.exists(), "failed clone_from should clean up repo_path"


def test_annex_false_on_pointer_repo_raises(tmp_path: Path) -> None:
    """annex=False on a repo with content_store.yaml raises ValueError."""
    repo_path = tmp_path / "origin"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir)
    Catalog.init_repo_path(repo_path, content_store_config=config)

    clone_path = tmp_path / "cloned"
    with pytest.raises(ValueError, match="pointer backend"):
        Catalog.clone_from(
            url=str(repo_path),
            repo_path=clone_path,
            annex=False,
        )
    assert not clone_path.exists(), "failed clone_from should clean up repo_path"

    with pytest.raises(ValueError, match="pointer backend"):
        Catalog.from_repo_path(repo_path, init=False, annex=False)


def test_pointer_entry_expr_roundtrip(tmp_path: Path) -> None:
    """entry.expr on the pointer backend round-trips through fetch_content."""
    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir)
    repo = Catalog.init_repo_path(repo_path, content_store_config=config)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(repo, cache=cache)
    catalog = Catalog(backend=backend)

    df = pd.DataFrame({"a": [1, 2, 3]})
    entry = catalog.add(xo.memtable(df))

    archive = entry.catalog_path
    if archive.exists():
        archive.unlink()

    assert not archive.exists()
    result = entry.expr.execute()
    assert archive.exists()
    pd.testing.assert_frame_equal(result, df)


def test_parse_pointer_rejects_negative_size(tmp_path: Path) -> None:
    """A pointer file with a negative size is rejected."""
    p = tmp_path / "neg.xorq-pointer"
    p.write_text(f"{POINTER_VERSION}\nsha256 {'a' * 64}\nsize -1\n")
    with pytest.raises(ValueError, match="Invalid pointer file"):
        parse_pointer(p)


def test_from_kwargs_forwards_content_store_config(tmp_path: Path) -> None:
    """from_kwargs passes content_store_config through to from_repo_path."""
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir)
    repo_path = tmp_path / "repo"
    catalog = Catalog.from_kwargs(
        path=repo_path, init=True, content_store_config=config
    )
    assert isinstance(catalog.backend, GitPointerBackend)


def test_init_repo_path_existing_raises(tmp_path: Path) -> None:
    """init_repo_path raises FileExistsError (not AssertionError) for existing paths."""
    repo_path = tmp_path / "repo"
    Catalog.init_repo_path(repo_path)
    with pytest.raises(FileExistsError, match="already exists"):
        Catalog.init_repo_path(repo_path)


def test_directory_content_store_config_from_env_missing_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XORQ_CONTENT_STORE_DIRECTORY_CATALOG_ID", raising=False)
    monkeypatch.delenv("XORQ_CONTENT_STORE_DIRECTORY_DIRECTORY", raising=False)

    with pytest.raises(ValueError, match="requires 'directory'"):
        DirectoryContentStoreConfig.from_env()


def test_stage_content_cleans_store_on_pointer_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the pointer write fails after upload, the content store blob is cleaned up."""
    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir)
    git_repo = Catalog.init_repo_path(repo_path, content_store_config=config)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(git_repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"content to upload")
    sha = compute_sha256(archive)
    key = compute_content_key(backend.catalog_id, sha)

    entries_dir = repo_path / CatalogInfix.ENTRY
    entries_dir.mkdir(exist_ok=True)
    target = entries_dir / "entry.zip"

    monkeypatch.setattr(
        "xorq.catalog.backend.write_pointer",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        backend.stage_content(archive, target)

    assert not backend.content_store.exists(key), (
        "content store blob should be cleaned up after pointer write failure"
    )


def test_directory_content_store_list_keys(tmp_path: Path) -> None:
    """list_keys returns all keys under a prefix."""
    store = DirectoryContentStore(directory=tmp_path / "store")
    src = tmp_path / "data.bin"
    src.write_bytes(b"content")

    store.put("cat/aa/bb/file1.zip", src)
    store.put("cat/aa/cc/file2.zip", src)
    store.put("other/aa/bb/file3.zip", src)

    all_keys = sorted(store.list_keys())
    assert len(all_keys) == 3

    cat_keys = sorted(store.list_keys(prefix="cat"))
    assert cat_keys == ["cat/aa/bb/file1.zip", "cat/aa/cc/file2.zip"]

    other_keys = sorted(store.list_keys(prefix="other"))
    assert other_keys == ["other/aa/bb/file3.zip"]

    empty_keys = list(store.list_keys(prefix="nonexistent"))
    assert empty_keys == []


def test_gc_content_store_finds_orphans(tmp_path: Path) -> None:
    """gc_content_store identifies and removes unreferenced content store keys."""
    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir)
    git_repo = Catalog.init_repo_path(repo_path, content_store_config=config)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(git_repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"real content")
    sha = compute_sha256(archive)
    key = compute_content_key(backend.catalog_id, sha)

    entries_dir = repo_path / CatalogInfix.ENTRY
    entries_dir.mkdir(exist_ok=True)
    target = entries_dir / "entry.zip"
    backend.stage_content(archive, target)
    git_repo.index.commit("add entry")

    # no orphans yet
    assert backend.gc_content_store(dry_run=True) == []

    # plant an orphan directly in the store
    orphan_data = tmp_path / "orphan.zip"
    orphan_data.write_bytes(b"orphaned blob")
    orphan_key = compute_content_key(backend.catalog_id, compute_sha256(orphan_data))
    backend.content_store.put(orphan_key, orphan_data)

    # dry run finds it but doesn't delete
    orphans = backend.gc_content_store(dry_run=True)
    assert orphan_key in orphans
    assert key not in orphans
    assert backend.content_store.exists(orphan_key)

    # actual run deletes it
    orphans = backend.gc_content_store(dry_run=False)
    assert orphan_key in orphans
    assert not backend.content_store.exists(orphan_key)
    assert backend.content_store.exists(key)


def test_gc_content_store_empty_catalog(tmp_path: Path) -> None:
    """gc on an empty pointer catalog with no entries finds nothing."""
    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = directory_store_config(store_dir)
    git_repo = Catalog.init_repo_path(repo_path, content_store_config=config)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(git_repo, cache=cache)

    assert backend.gc_content_store(dry_run=True) == []


# ---------------------------------------------------------------------------
# Tier 1: content_store.py error paths
# ---------------------------------------------------------------------------


def test_directory_content_store_get_missing_raises(tmp_path: Path) -> None:
    """get() on a nonexistent key raises FileNotFoundError."""
    store = DirectoryContentStore(directory=tmp_path / "store")
    with pytest.raises(FileNotFoundError, match="Content not found"):
        store.get("cat/aa/bb/missing.zip", tmp_path / "out.zip")


def test_content_cache_get_path_miss(tmp_path: Path) -> None:
    """get_path() returns None for a key that was never cached."""
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=1024)
    assert cache.get_path("cat/aa/bb/missing.zip") is None


def test_content_cache_contains_disabled(tmp_path: Path) -> None:
    """contains() always returns False when cache is disabled."""
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=0)
    assert cache.disabled
    assert not cache.contains("any/key")


def test_coerce_port_valid() -> None:
    assert _coerce_port(8080) == 8080
    assert _coerce_port("443") == 443
    assert _coerce_port(None) is None


def test_coerce_port_out_of_range() -> None:
    with pytest.raises(ValueError, match="port must be 1-65535"):
        _coerce_port(0)
    with pytest.raises(ValueError, match="port must be 1-65535"):
        _coerce_port(70000)


def test_non_empty_str_validator() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        S3ContentStoreConfig(bucket="", catalog_id="cat")


def test_s3_content_store_config_resolve_secrets_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_secrets returns empty dict when no secrets are set."""
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_AWS_SECRET_ACCESS_KEY", raising=False)
    config = S3ContentStoreConfig(catalog_id="cat", bucket="b")
    assert config._resolve_secrets() == {}


def test_content_store_config_from_dict_missing_type() -> None:
    with pytest.raises(ValueError, match="missing required 'type' field"):
        ContentStoreConfig.from_dict({"catalog_id": "cat"})


def test_content_store_config_from_dict_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        ContentStoreConfig.from_dict(
            {"type": "directory", "directory": "/tmp", "bogus": "x"}
        )


def test_content_store_config_from_dict_ignore_unknown() -> None:
    cfg = ContentStoreConfig.from_dict(
        {"type": "directory", "directory": "/tmp", "bogus": "x"},
        ignore_unknown=True,
    )
    assert isinstance(cfg, DirectoryContentStoreConfig)


# ---------------------------------------------------------------------------
# Tier 2: backend error paths
# ---------------------------------------------------------------------------


def test_fetch_content_missing_pointer_raises(tmp_path: Path) -> None:
    """fetch_content raises FileNotFoundError when pointer file is missing."""
    repo = GitRepo.init(tmp_path / "repo")
    store_dir = tmp_path / "store"
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    config.write_yaml(Path(repo.working_dir) / CONTENT_STORE_YAML)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(repo, cache=cache)

    target = tmp_path / "entry.zip"
    with pytest.raises(FileNotFoundError, match="Pointer file missing"):
        backend.fetch_content(target)


def test_stage_content_cleans_up_on_store_put_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If store.put() fails, the local archive copy is cleaned up."""
    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    git_repo = Catalog.init_repo_path(repo_path, content_store_config=config)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(git_repo, cache=cache)

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"content")

    entries_dir = repo_path / CatalogInfix.ENTRY
    entries_dir.mkdir(exist_ok=True)
    target = entries_dir / "entry.zip"

    monkeypatch.setattr(
        DirectoryContentStore,
        "put",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        backend.stage_content(archive, target)

    assert not target.exists(), "archive copy should be cleaned up after store failure"


def test_has_references_no_entries_dir(tmp_path: Path) -> None:
    """_has_references returns False when entries dir doesn't exist."""
    repo = GitRepo.init(tmp_path / "repo")
    store_dir = tmp_path / "store"
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    config.write_yaml(Path(repo.working_dir) / CONTENT_STORE_YAML)
    cache = ContentCache(cache_dir=tmp_path / "cache", max_bytes=10**9)
    backend = GitPointerBackend.from_repo(repo, cache=cache)

    assert not backend._has_references("a" * 64)


# ---------------------------------------------------------------------------
# Tier 2: catalog validation
# ---------------------------------------------------------------------------


def test_validate_content_store_config_annex_on_pointer_repo(tmp_path: Path) -> None:
    """Passing an AnnexConfig to a pointer-backend repo raises ValueError."""
    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    Catalog.init_repo_path(repo_path, content_store_config=config)

    with pytest.raises(ValueError, match="cannot use git-annex"):
        Catalog._validate_content_store_config(
            repo_path, content_store_config=None, annex=LOCAL_ANNEX
        )


def test_validate_content_store_config_annex_false_on_pointer_repo(
    tmp_path: Path,
) -> None:
    """Passing annex=False to a pointer-backend repo raises ValueError."""
    repo_path = tmp_path / "repo"
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    config = DirectoryContentStoreConfig(catalog_id="testcat", directory=str(store_dir))
    Catalog.init_repo_path(repo_path, content_store_config=config)

    with pytest.raises(ValueError, match="cannot force plain git"):
        Catalog._validate_content_store_config(
            repo_path, content_store_config=None, annex=False
        )


def test_validate_content_store_config_no_yaml_is_noop(tmp_path: Path) -> None:
    """When content_store.yaml doesn't exist, validation is a no-op."""
    repo_path = tmp_path / "repo"
    Catalog.init_repo_path(repo_path)
    Catalog._validate_content_store_config(
        repo_path, content_store_config=None, annex=None
    )


_VALID_ARCHIVE_NAMES = (*REQUIRED_ARCHIVE_NAMES, TEST_WHEEL_NAME)


@pytest.mark.parametrize("elide", REQUIRED_ARCHIVE_NAMES)
def test_test_zip(elide, catalog, tmpdir):
    zip_path = write_zip(
        Path(tmpdir).joinpath("build.zip"),
        {name: b"" for name in _VALID_ARCHIVE_NAMES if name != elide},
    )
    with pytest.raises(AssertionError, match=elide):
        BuildZip(zip_path)


def test_test_zip_missing_wheel(catalog, tmpdir):
    zip_path = write_zip(
        Path(tmpdir).joinpath("build.zip"),
        dict.fromkeys(REQUIRED_ARCHIVE_NAMES, b""),
    )
    with pytest.raises(AssertionError, match=r"\.whl"):
        BuildZip(zip_path)


def test_test_zip_multi_wheel(catalog, tmpdir):
    names = {**dict.fromkeys(REQUIRED_ARCHIVE_NAMES, b""), TEST_WHEEL_NAME: b""}
    names["extra-0.0.0-py3-none-any.whl"] = b""
    zip_path = write_zip(Path(tmpdir).joinpath("build.zip"), names)
    bz = BuildZip(zip_path)
    assert bz.path == zip_path


def test_assert_consistency(catalog, tmpdir):
    zip_path = write_zip(
        Path(tmpdir).joinpath("build.zip"),
        dict.fromkeys(_VALID_ARCHIVE_NAMES, b""),
    )
    catalog_addition = CatalogAddition(BuildZip(zip_path), catalog)
    catalog_addition.ensure_dirs()
    catalog_path = catalog_addition.catalog_entry.catalog_path
    tracked_path = catalog.backend.entry_tracked_path(catalog_path)
    with catalog.commit_context("bad commit"):
        if tracked_path != catalog_path:
            tracked_path.write_text("xorq-pointer v1\nsha256 abc\nsize 0\n")
            catalog.repo.index.add((tracked_path,))
        else:
            shutil.copy(zip_path, catalog_path)
            catalog.repo.index.add((catalog_path,))
    with pytest.raises(AssertionError):
        catalog.assert_consistency()
    with pytest.raises(AssertionError):
        CatalogEntry(catalog_addition.name, catalog)


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


def _commit_count(catalog: Catalog) -> int:
    return len(list(catalog.repo.iter_commits()))


def test_add_alias_noop_leaves_no_commit(catalog_populated: Catalog) -> None:
    # Re-adding an alias that already resolves to its target stages nothing
    # (CatalogAlias._add returns early), so commit_context must NOT leave an
    # empty commit behind. Runs across git/annex/pointer backends via the
    # catalog_populated fixture.
    name = catalog_populated.list()[0]
    catalog_populated.add_alias(name, "my-alias")

    before = _commit_count(catalog_populated)
    catalog_populated.add_alias(name, "my-alias")  # identical -> no-op
    after = _commit_count(catalog_populated)

    assert after == before
    catalog_populated.assert_consistency()


def test_add_alias_real_change_still_commits(catalog_populated: Catalog) -> None:
    # Guard sanity: a genuine alias add/retarget must still produce exactly one
    # commit (the empty-commit guard only skips no-ops).
    names = catalog_populated.list()
    name_a, name_b = names[0], names[1]

    catalog_populated.add_alias(name_a, "moves")
    before = _commit_count(catalog_populated)
    # a brand-new alias, then a retarget of it -- each a real staged change
    catalog_populated.add_alias(name_a, "fresh")
    catalog_populated.add_alias(name_b, "moves")
    after = _commit_count(catalog_populated)

    assert after == before + 2
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


@requires_annex
def test_directory_remote(tmp_path: Path) -> None:
    """Add entries, copy to a directory remote, drop local, get back."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
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


@requires_annex
def test_annex_is_content_local_after_drop(tmp_path: Path) -> None:
    """is_content_local is False when annex content has been dropped."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
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


@requires_annex
def test_annex_auto_fetch_after_drop(tmp_path: Path) -> None:
    """Sidecar metadata works after drop; expr and get auto-fetch from remote."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
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
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = entry.get(dir_path=str(out_dir))
    assert result.exists()


@requires_annex
def test_annex_fetch_restores_content(tmp_path: Path) -> None:
    """entry.fetch() retrieves annex content for a single entry."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
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


@requires_annex
def test_fetch_entries_bulk(tmp_path: Path) -> None:
    """catalog.fetch_entries() batch-fetches multiple entries in one operation."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
    Catalog.init_repo_path(repo_path, annex=remote_config)
    annex = Annex(repo_path=repo_path)
    backend = GitAnnexBackend(repo=GitRepo(repo_path), annex=annex)
    catalog = Catalog(backend=backend)

    with build_expr_context_zip(xo.memtable({"a": [1]})) as zp:
        entry_a = catalog.add(zp, sync=False)
    with build_expr_context_zip(xo.memtable({"b": [2]})) as zp:
        entry_b = catalog.add(zp, sync=False)

    annex.copy(to="mydir")
    annex.drop()
    assert not entry_a.is_content_local
    assert not entry_b.is_content_local

    catalog.fetch_entries(entry_a, entry_b)
    assert entry_a.is_content_local
    assert entry_b.is_content_local

    # also test string-based lookup
    annex.drop()
    catalog.fetch_entries(entry_a.name, entry_b.name)
    assert entry_a.is_content_local
    assert entry_b.is_content_local


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


@requires_annex
def test_metadata_from_sidecar_after_drop(tmp_path: Path) -> None:
    """All metadata properties work from sidecar after annex content is dropped."""
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    repo_path = tmp_path / "repo"
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
    repo = GitRepo.init(repo_path, initial_branch=MAIN_BRANCH)
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


@pytest.mark.s3
def test_s3_fileprefix_namespaced_in_remote_log(tmpdir):
    """initremote bakes ``{base}{name}/{remote_uuid}/`` into remote.log,
    enableremote of the same config is a no-op, and a re-config with a
    mismatched base raises AnnexError (ADR-0011)."""
    base_prefix = "annex-only/"
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
        fileprefix=base_prefix,
    )
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
    repo = GitRepo.init(repo_path, initial_branch=MAIN_BRANCH)
    repo.index.commit("initial commit")
    subprocess.run(
        ["git", "config", "annex.security.allowed-ip-addresses", "all"],
        cwd=repo_path,
        check=True,
    )
    Annex.init_repo_path(repo_path, remote_config=remote_config)

    annex = Annex(repo_path=repo_path, env=remote_config.env)
    [(remote_uuid, cfg)] = annex.remote_log.items()
    expected = f"{base_prefix}{remote_config.name}/{remote_uuid}/"
    assert cfg["fileprefix"] == expected

    # re-enabling with the same config is a no-op (verify passes)
    annex.enableremote(remote_config)
    assert annex.remote_log[remote_uuid]["fileprefix"] == expected

    # changing the base prefix must fail rather than silently rewriting
    mismatched = evolve(remote_config, fileprefix="other-base/")
    with pytest.raises(AnnexError, match="different base prefix"):
        annex.enableremote(mismatched)


@requires_annex
def test_from_repo_path_enables_special_remote_on_clone(tmp_path: Path) -> None:
    """from_repo_path enables the special remote on a fresh clone.

    Simulates the scenario where a catalog is cloned via git submodule add
    from a host like GitHub (annex-ignore on origin). Without enableremote,
    git-annex cannot locate the special remote and content fetch fails.
    """
    # create origin catalog with a directory remote
    remote_dir = tmp_path / "remote-store"
    remote_dir.mkdir()
    remote_config = DirectoryRemoteConfig(name="mydir", directory=str(remote_dir))
    origin_path = tmp_path / "origin"
    Catalog.init_repo_path(origin_path, annex=remote_config)
    annex = Annex(repo_path=origin_path)
    backend = GitAnnexBackend(repo=GitRepo(origin_path), annex=annex)
    origin_catalog = Catalog(backend=backend)

    with build_expr_context_zip(xo.memtable({"x": [1]})) as zp:
        entry = origin_catalog.add(zp, sync=False)
    entry_name = entry.name

    # copy content to directory remote
    annex.copy(to="mydir")

    # create a bare clone (intermediary, like GitHub)
    bare_path = tmp_path / "bare"
    GitRepo.clone_from(origin_path, bare_path, bare=True)
    _do_inside(bare_path, "init")
    _do_inside(bare_path, "sync", "--content")

    # clone from bare (simulates git submodule add)
    clone_path = tmp_path / "cloned"
    GitRepo.clone_from(bare_path, clone_path)
    _do_inside(clone_path, "init")

    # simulate GitHub: origin has annex-ignore set
    subprocess.run(
        ["git", "config", "remote.origin.annex-ignore", "true"],
        cwd=clone_path,
        check=True,
    )

    # open with from_repo_path — this should enableremote automatically
    cloned_catalog = Catalog.from_repo_path(clone_path, init=False, annex=remote_config)
    assert isinstance(cloned_catalog.backend, GitAnnexBackend)

    # verify content is fetchable from the directory remote
    cloned_entry = cloned_catalog.get_catalog_entry(entry_name)
    assert not cloned_entry.is_content_local
    cloned_entry.fetch()
    assert cloned_entry.is_content_local


def test_cache_keys_stores_key_and_relative_path(catalog, tmp_path):
    """CacheKey in the sidecar carries both the hash key and the relative_path
    so paths can be reconstructed without loading the expression."""
    relative = "my_cache"
    cache = ParquetSnapshotCache.from_kwargs(relative_path=relative)
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)
    entry = catalog.add(expr)

    ck = entry.projected_cache_key
    assert isinstance(ck, CacheKey)
    assert ck.relative_path == relative
    assert ck.key  # non-empty hash string


def test_cache_keys_paths_relocatable(catalog, tmp_path, monkeypatch):
    cache_dir_A = tmp_path / "cache_A"
    cache_dir_B = tmp_path / "cache_B"
    relative = "my_cache"

    monkeypatch.setattr(
        "xorq.common.utils.caching_utils.get_xorq_cache_dir", lambda: cache_dir_A
    )
    cache = ParquetSnapshotCache.from_kwargs(relative_path=relative)
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)
    entry = catalog.add(expr)

    ck = entry.projected_cache_key
    expected_name = ck.key + ".parquet"

    path_at_A = get_cache_key_path(entry.projected_cache_key)
    assert path_at_A == str(cache_dir_A / relative / expected_name)

    monkeypatch.setattr(
        "xorq.common.utils.caching_utils.get_xorq_cache_dir", lambda: cache_dir_B
    )
    path_at_B = get_cache_key_path(entry.projected_cache_key)
    assert path_at_B == str(cache_dir_B / relative / expected_name)


def test_base_path_is_silently_dropped_through_catalog_round_trip(catalog, tmp_path):
    cache = ParquetSnapshotCache.from_kwargs(
        relative_path="my_cache", base_path=tmp_path / "explicit_base"
    )
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)
    entry = catalog.add(expr)

    ck = entry.projected_cache_key
    assert ck.relative_path == "my_cache"
    assert ck.key


@requires_annex
def test_annex_fetch_content_no_remote_raises(tmp_path: Path) -> None:
    """fetch_content raises instead of hanging when no remote is configured."""
    repo_path = tmp_path / "no-remote"
    repo = Catalog.init_repo_path(repo_path, annex=LOCAL_ANNEX)
    backend = GitAnnexBackend(repo=repo, annex=Annex(repo_path=repo_path))
    catalog = Catalog(backend=backend)

    expr = xo.memtable({"x": [1, 2, 3]})
    entry = catalog.add(expr, sync=False)

    # Drop the annex content so is_content_local returns False
    backend.annex.drop(backend.get_relpath(entry.catalog_path))

    assert not backend.is_content_local(entry.catalog_path)
    assert not backend._has_any_remote()

    with pytest.raises(AnnexError, match="no remote configured"):
        backend.fetch_content(entry.catalog_path)


def test_catalog_entry_roundtrip_execute(catalog):
    """Loading an expression from a catalog zip and executing it returns the original data."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    expr = xo.memtable(df)
    entry = catalog.add(expr)
    result = entry.expr.execute()
    pd.testing.assert_frame_equal(result, df)


def test_database_table_roundtrip_execute(catalog):
    """A registered database_table survives zip roundtrip and can be executed."""
    df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})
    con = xo.connect()
    t = con.register(df, table_name="test_dt")
    entry = catalog.add(t)
    result = entry.expr.execute()
    pd.testing.assert_frame_equal(result, df)


def test_extract_dir_cleaned_up_on_expr_gc(catalog):
    """Temp extract directory is removed when the loaded expression is garbage-collected."""
    entry = catalog.add(xo.memtable(pd.DataFrame({"a": [1]})))

    before = frozenset(_live_extract_dirs)
    expr = entry.expr
    created = frozenset(_live_extract_dirs) - before
    assert len(created) == 1, f"expected exactly one new extract dir, got {created}"
    (td,) = created
    assert Path(td).is_dir()

    del expr
    for _ in range(3):
        gc.collect()

    assert td not in _live_extract_dirs
    assert not Path(td).exists()
