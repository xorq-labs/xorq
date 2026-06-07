"""Tests that backend classes reject repos with conflicting backend artifacts."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from git import Repo

from xorq.catalog.annex import LOCAL_ANNEX, Annex
from xorq.catalog.backend import GitAnnexBackend, GitPointerBackend
from xorq.catalog.catalog import Catalog
from xorq.catalog.constants import CONTENT_STORE_YAML, MAIN_BRANCH
from xorq.catalog.content_store import ContentCache, DirectoryContentStoreConfig


def _init_annex_repo(path: Path) -> Repo:
    return Catalog.init_repo_path(path, annex=LOCAL_ANNEX)


def _init_pointer_repo(path: Path, store_dir: Path) -> Repo:
    store_dir.mkdir(parents=True, exist_ok=True)
    config = DirectoryContentStoreConfig(
        catalog_id=str(uuid.uuid4()),
        directory=str(store_dir),
    )
    return Catalog.init_repo_path(path, content_store_config=config)


def _make_cache(repo_path: Path) -> ContentCache:
    return ContentCache(
        cache_dir=repo_path / ".xorq-cache",
        max_bytes=1024 * 1024 * 1024,
    )


def test_pointer_rejects_repo_with_annex_branch(tmp_path: Path) -> None:
    repo = _init_annex_repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="git-annex artifacts"):
        GitPointerBackend.from_repo(repo, cache=_make_cache(tmp_path))


def test_pointer_rejects_repo_with_annex_directory(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo = Repo.init(str(repo_path), mkdir=True, initial_branch=MAIN_BRANCH)
    (repo_path / ".git" / "annex").mkdir()
    with pytest.raises(ValueError, match="git-annex artifacts"):
        GitPointerBackend.from_repo(repo, cache=_make_cache(tmp_path))


def test_annex_rejects_repo_with_content_store_yaml(tmp_path: Path) -> None:
    repo = _init_annex_repo(tmp_path / "repo")
    repo_path = Path(repo.working_dir)
    (repo_path / CONTENT_STORE_YAML).write_text("fake: true\n")
    with pytest.raises(ValueError, match="pointer-backend artifacts"):
        GitAnnexBackend(repo=repo, annex=Annex(repo_path=repo_path))


def test_annex_rejects_repo_with_pointer_files(tmp_path: Path) -> None:
    repo = _init_annex_repo(tmp_path / "repo")
    repo_path = Path(repo.working_dir)
    entries_dir = repo_path / "entries"
    entries_dir.mkdir(exist_ok=True)
    (entries_dir / "foo.zip.pointer").write_text(
        "xorq-pointer v1\nsha256 abc\nsize 0\n"
    )
    with pytest.raises(ValueError, match="pointer-backend artifacts"):
        GitAnnexBackend(repo=repo, annex=Annex(repo_path=repo_path))


def test_normal_pointer_backend_without_annex(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    repo = _init_pointer_repo(tmp_path / "repo", store_dir)
    GitPointerBackend.from_repo(repo, cache=_make_cache(tmp_path))


def test_normal_annex_backend_without_pointer(tmp_path: Path) -> None:
    repo = _init_annex_repo(tmp_path / "repo")
    repo_path = Path(repo.working_dir)
    GitAnnexBackend(repo=repo, annex=Annex(repo_path=repo_path))
