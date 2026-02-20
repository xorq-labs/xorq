from pathlib import Path

import pytest
from git import (
    NoSuchPathError,
    Repo,
)

from xorq.catalog.catalog import (
    Catalog,
)
from xorq.catalog.tests.conftest import compare_repo_and_catalog


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_bare_clone(source_catalog, tmpdir_path):
    """Return a bare Repo cloned from *source_catalog* at *tmpdir_path*."""
    return Repo.clone_from(
        source_catalog.repo_path,
        tmpdir_path,
        bare=True,
    )


def test_catalog_ctor_fails(tmpdir):
    uninited_repo = Repo.init(Path(tmpdir), mkdir=True)
    with pytest.raises(
        ValueError, match="Reference at 'refs/heads/master' does not exist"
    ):
        Catalog(uninited_repo)


def test_catalog_ctor(repo):
    Catalog(repo)


def test_catalog_ctor_from_repo_path_init(tmpdir):
    repo_path = Path(tmpdir).joinpath("repo")
    with pytest.raises(NoSuchPathError):
        Catalog.from_repo_path(repo_path, init=False)
    Catalog.from_repo_path(repo_path, init=True)
    Catalog.from_repo_path(repo_path, init=False)
    with pytest.raises(AssertionError):
        Catalog.from_repo_path(repo_path, init=True)


def test_catalog_ctor_from_repo_path(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("repo"))
    assert isinstance(repo, Repo)
    Catalog.from_repo_path(repo.working_dir)


def test_catalog_clone_from(repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("cloned")
    )
    compare_repo_and_catalog(repo_cloned_bare, cloned)


def test_from_kwargs_from_path(catalog_path):
    catalog = Catalog.from_kwargs(path=catalog_path)
    assert isinstance(catalog, Catalog)
    assert str(catalog.repo_path) == catalog_path


def test_from_kwargs_name_and_path_mutually_exclusive(catalog_path):
    with pytest.raises(Exception, match="mutually exclusive"):
        Catalog.from_kwargs(name="foo", path=catalog_path)


@pytest.mark.parametrize("init", (None, True))
def test_from_kwargs_with_init(init, tmpdir):
    new_path = str(Path(tmpdir).joinpath("new-catalog"))
    catalog = Catalog.from_kwargs(path=new_path, init=init)
    assert isinstance(catalog, Catalog)
    assert Path(new_path).exists()


# ---------------------------------------------------------------------------
# from_kwargs – no root_repo, no url (name / path / default branches)
# ---------------------------------------------------------------------------


def test_from_kwargs_default(tmpdir, monkeypatch):
    """from_kwargs() with no args opens (or creates) the 'default' catalog."""
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    catalog = Catalog.from_kwargs()
    assert isinstance(catalog, Catalog)
    assert catalog.repo_path == Path(tmpdir) / "default"


def test_from_kwargs_name(tmpdir, monkeypatch):
    """from_kwargs(name=...) opens/creates a named catalog."""
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    catalog = Catalog.from_kwargs(name="my-catalog")
    assert isinstance(catalog, Catalog)
    assert catalog.repo_path == Path(tmpdir) / "my-catalog"


def test_from_kwargs_name_init_false_existing(tmpdir, monkeypatch):
    """from_kwargs(name=..., init=False) opens an existing named catalog."""
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    Catalog.from_kwargs(name="existing", init=True)
    catalog = Catalog.from_kwargs(name="existing", init=False)
    assert isinstance(catalog, Catalog)


def test_from_kwargs_name_init_false_missing_raises(tmpdir, monkeypatch):
    """from_kwargs(name=..., init=False) on a missing path raises."""
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    with pytest.raises(NoSuchPathError):
        Catalog.from_kwargs(name="nonexistent", init=False)


def test_from_kwargs_name_and_path_mutually_exclusive_message(catalog_path):
    """Error message names the two mutually-exclusive arguments."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Catalog.from_kwargs(name="foo", path=catalog_path)


# ---------------------------------------------------------------------------
# from_kwargs – url branch (no root_repo)
# ---------------------------------------------------------------------------


def test_from_kwargs_url_with_explicit_path(catalog, tmpdir):
    """from_kwargs(url=..., path=...) clones to the given path."""
    bare = _make_bare_clone(catalog, Path(tmpdir) / "bare")
    target = str(Path(tmpdir) / "cloned")
    result = Catalog.from_kwargs(url=str(bare.working_dir), path=target)
    assert isinstance(result, Catalog)
    assert result.repo_path == Path(target)


def test_from_kwargs_url_without_path(catalog, tmpdir, monkeypatch):
    """from_kwargs(url=...) with no path derives a name from the URL stem."""
    # Put the bare source and the by_name_base_path in separate subdirs
    # so the auto-derived clone path doesn't collide with the source.
    source_dir = Path(tmpdir) / "source"
    clones_dir = Path(tmpdir) / "clones"
    clones_dir.mkdir()
    monkeypatch.setattr(Catalog, "by_name_base_path", clones_dir)
    bare = _make_bare_clone(catalog, source_dir)
    result = Catalog.from_kwargs(url=str(bare.working_dir))
    assert isinstance(result, Catalog)
    expected_name = Path(bare.working_dir).stem
    assert result.repo_path == clones_dir / expected_name


def test_from_kwargs_url_and_name_raises(catalog, tmpdir):
    """`url` and `name` together must raise ValueError with a useful message."""
    bare = _make_bare_clone(catalog, Path(tmpdir) / "bare")
    with pytest.raises(ValueError, match="mutually exclusive"):
        Catalog.from_kwargs(url=str(bare.working_dir), name="something")


def test_from_kwargs_url_and_name_message_contains_values(catalog, tmpdir):
    """Error message for url+name conflict should echo back the bad values."""
    bare = _make_bare_clone(catalog, Path(tmpdir) / "bare")
    url = str(bare.working_dir)
    with pytest.raises(ValueError, match="something"):
        Catalog.from_kwargs(url=url, name="something")


# ---------------------------------------------------------------------------
# from_kwargs – root_repo branch
# ---------------------------------------------------------------------------


def test_from_kwargs_root_repo_with_name(root_repo, tmpdir):
    """from_kwargs(root_repo=..., name=...) creates a submodule catalog by name."""
    catalog = Catalog.from_kwargs(root_repo=root_repo, name="sub-catalog")
    assert isinstance(catalog, Catalog)
    expected = Path(root_repo.working_dir) / Catalog.submodule_rel_path / "sub-catalog"
    assert catalog.repo_path == expected
    submodule_paths = [sm.path for sm in root_repo.submodules]
    assert str(Catalog.submodule_rel_path / "sub-catalog") in submodule_paths


def test_from_kwargs_root_repo_with_url(root_repo, catalog, tmpdir):
    """from_kwargs(root_repo=..., url=...) clones a remote catalog as a submodule."""
    bare = _make_bare_clone(catalog, Path(tmpdir) / "bare")
    result = Catalog.from_kwargs(root_repo=root_repo, url=str(bare.working_dir))
    assert isinstance(result, Catalog)
    name = Path(bare.working_dir).stem
    expected = Path(root_repo.working_dir) / Catalog.submodule_rel_path / name
    assert result.repo_path == expected


def test_from_kwargs_root_repo_path_only_raises(root_repo, catalog_path):
    """Providing only path alongside root_repo is not a valid combination."""
    with pytest.raises(ValueError, match="root_repo"):
        Catalog.from_kwargs(root_repo=root_repo, path=catalog_path)


def test_from_kwargs_root_repo_no_name_or_url_raises(root_repo):
    """root_repo with neither name nor url must raise ValueError."""
    with pytest.raises(ValueError, match="root_repo"):
        Catalog.from_kwargs(root_repo=root_repo)


def test_from_kwargs_root_repo_name_and_url_raises(root_repo, catalog, tmpdir):
    """root_repo with both name and url must raise ValueError."""
    bare = _make_bare_clone(catalog, Path(tmpdir) / "bare")
    with pytest.raises(ValueError, match="root_repo"):
        Catalog.from_kwargs(root_repo=root_repo, name="x", url=str(bare.working_dir))


def test_from_kwargs_root_repo_name_and_path_raises(root_repo, catalog_path):
    """root_repo with both name and path must raise ValueError."""
    with pytest.raises(ValueError, match="root_repo"):
        Catalog.from_kwargs(root_repo=root_repo, name="x", path=catalog_path)


def test_from_kwargs_root_repo_error_message_echoes_args(root_repo, catalog_path):
    """Error message should include the bad argument values for easier debugging."""
    with pytest.raises(ValueError, match="path=") as exc_info:
        Catalog.from_kwargs(root_repo=root_repo, path=catalog_path)
    assert catalog_path in str(exc_info.value)
