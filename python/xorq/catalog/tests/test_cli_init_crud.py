# Tests for: init, add, remove, add-alias, remove-alias CLI commands
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from click.testing import CliRunner

from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.tests.conftest import (
    compare_repo_and_catalog,
    make_build_zip,
)
from xorq.catalog.zip_utils import extract_build_zip_context


@pytest.fixture
def runner():
    yield CliRunner()


# --- init command ---


def test_init_command(runner, tmpdir):
    new_path = str(Path(tmpdir).joinpath("init-catalog"))
    result = runner.invoke(cli, ["--path", new_path, "init"])
    assert result.exit_code == 0, result.output
    assert "Initialized catalog at" in result.output
    assert Path(new_path).exists()


def test_init_command_already_exists(runner, catalog_path):
    assert Path(catalog_path).exists()
    result = runner.invoke(cli, ["--path", catalog_path, "init"])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert catalog_path in result.output


def test_init_by_name(runner, tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    result = runner.invoke(cli, ["--name", "my-catalog", "init"])
    assert result.exit_code == 0, result.output
    assert "Initialized catalog at" in result.output
    assert Path(tmpdir).joinpath("my-catalog").exists()


def test_init_by_name_already_exists(runner, tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    runner.invoke(cli, ["--name", "my-catalog", "init"])
    result = runner.invoke(cli, ["--name", "my-catalog", "init"])
    assert result.exit_code != 0


def test_init_default(runner, tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0, result.output
    assert "Initialized catalog at" in result.output
    assert Path(tmpdir).joinpath("default").exists()


def test_init_default_already_exists(runner, tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    runner.invoke(cli, ["init"])
    result = runner.invoke(cli, ["init"])
    assert result.exit_code != 0


def test_init_with_root_repo_and_name(runner, root_repo):
    result = runner.invoke(
        cli,
        ["--root-repo", root_repo.working_dir, "--name", "sub-catalog", "init"],
    )
    assert result.exit_code == 0, result.output
    assert "Initialized catalog at" in result.output


# --- add command ---


def test_add_command(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    result = runner.invoke(cli, ["--path", catalog_path, "add", path])
    assert result.exit_code == 0, result.output
    assert "Added" in result.output


def test_add_multiple(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    result = runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    assert result.exit_code == 0, result.output
    assert result.output.count("Added") == len(data_dict)


def test_add_with_aliases(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "add",
            path,
            "--alias",
            "alias-x",
            "--alias",
            "alias-y",
        ],
    )
    assert result.exit_code == 0, result.output
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    assert {ca.alias for ca in catalog.catalog_aliases} == {"alias-x", "alias-y"}


def test_add_duplicate(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    result = runner.invoke(cli, ["--path", catalog_path, "add", path])
    assert result.exit_code != 0


def test_add_from_directory(runner, catalog_path, tmpdir):
    archive = make_build_zip(tmpdir, "build-dir-test")
    with extract_build_zip_context(archive) as build_dir:
        result = runner.invoke(cli, ["--path", catalog_path, "add", str(build_dir)])
    assert result.exit_code == 0, result.output
    assert "Added" in result.output


def test_add_nonexistent_path(runner, catalog_path):
    result = runner.invoke(
        cli, ["--path", catalog_path, "add", "/nonexistent/file.zip"]
    )
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "sync,expectation",
    (
        (True, does_not_raise()),
        (False, pytest.raises(AssertionError)),
    ),
)
def test_add_sync(sync, expectation, runner, repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("add-sync-test")
    )

    path = make_build_zip(tmpdir, "to-add")
    result = runner.invoke(
        cli,
        [
            "--path",
            cloned.repo_path,
            "add",
            str(path),
            "--sync" if sync else "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Added" in result.output

    # check sync condition
    with expectation:
        compare_repo_and_catalog(repo_cloned_bare, cloned)


# --- remove command ---


def test_remove_command(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    result = runner.invoke(cli, ["--path", catalog_path, "remove", name])
    assert result.exit_code == 0, result.output
    assert "Removed" in result.output


def test_remove_nonexistent(runner, catalog_path):
    result = runner.invoke(cli, ["--path", catalog_path, "remove", "nonexistent"])
    assert result.exit_code != 0


def test_remove_multiple(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    names = [
        Path(p).name.removesuffix("".join(Path(p).suffixes)) for p in data_dict.values()
    ]
    result = runner.invoke(cli, ["--path", catalog_path, "remove", *names])
    assert result.exit_code == 0, result.output
    assert result.output.count("Removed") == len(data_dict)


@pytest.mark.parametrize(
    "sync,expectation",
    (
        (True, does_not_raise()),
        (False, pytest.raises(AssertionError)),
    ),
)
def test_remove_sync(sync, expectation, runner, repo_cloned_bare, tmpdir, data_dict):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("remove-sync-test")
    )

    path = str(next(iter(data_dict.values())))
    result = runner.invoke(
        cli,
        [
            "--path",
            cloned.repo_path,
            "remove",
            Path(path).stem,
            "--sync" if sync else "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Removed" in result.output

    # check that sync condition
    with expectation:
        compare_repo_and_catalog(repo_cloned_bare, cloned)


# --- add-alias command ---


def test_add_alias_command(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    result = runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "my-alias"])
    assert result.exit_code == 0, result.output
    assert "my-alias" in result.output


def test_add_with_aliases_commit_message(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "add", path, "--alias", "v1", "--alias", "latest"],
    )
    assert result.exit_code == 0, result.output
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    commit_message = catalog.repo.head.commit.message.strip()
    assert "v1" in commit_message
    assert "latest" in commit_message


def test_add_alias_unknown_entry(runner, catalog_path):
    result = runner.invoke(
        cli, ["--path", catalog_path, "add-alias", "nonexistent", "my-alias"]
    )
    assert result.exit_code != 0


def test_add_alias_overwrite(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    names = [
        Path(p).name.removesuffix("".join(Path(p).suffixes)) for p in data_dict.values()
    ]
    runner.invoke(cli, ["--path", catalog_path, "add-alias", names[0], "shared"])
    result = runner.invoke(
        cli, ["--path", catalog_path, "add-alias", names[1], "shared"]
    )
    assert result.exit_code == 0, result.output
    assert "shared" in result.output


# --- remove-alias command ---


def test_remove_alias_command(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "to-remove"])
    result = runner.invoke(cli, ["--path", catalog_path, "remove-alias", "to-remove"])
    assert result.exit_code == 0, result.output
    assert "to-remove" in result.output


def test_remove_alias_multiple(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-a"])
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-b"])
    result = runner.invoke(
        cli, ["--path", catalog_path, "remove-alias", "alias-a", "alias-b"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("Removed alias") == 2


def test_remove_alias_nonexistent(runner, catalog_path):
    result = runner.invoke(
        cli, ["--path", catalog_path, "remove-alias", "no-such-alias"]
    )
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "sync,expectation",
    (
        (True, does_not_raise()),
        (False, pytest.raises(AssertionError)),
    ),
)
def test_add_alias_sync(sync, expectation, runner, repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("add-alias-sync-test")
    )
    path = make_build_zip(tmpdir, "to-alias")
    runner.invoke(cli, ["--path", str(cloned.repo_path), "add", str(path), "--sync"])
    name = path.stem
    result = runner.invoke(
        cli,
        [
            "--path",
            str(cloned.repo_path),
            "add-alias",
            name,
            "my-alias",
            "--sync" if sync else "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "my-alias" in result.output

    with expectation:
        compare_repo_and_catalog(repo_cloned_bare, cloned)


@pytest.mark.parametrize(
    "sync,expectation",
    (
        (True, does_not_raise()),
        (False, pytest.raises(AssertionError)),
    ),
)
def test_remove_alias_sync(sync, expectation, runner, repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("remove-alias-sync-test")
    )
    path = make_build_zip(tmpdir, "to-alias")
    runner.invoke(cli, ["--path", str(cloned.repo_path), "add", str(path), "--sync"])
    name = path.stem
    runner.invoke(
        cli,
        ["--path", str(cloned.repo_path), "add-alias", name, "my-alias", "--sync"],
    )
    result = runner.invoke(
        cli,
        [
            "--path",
            str(cloned.repo_path),
            "remove-alias",
            "my-alias",
            "--sync" if sync else "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "my-alias" in result.output

    with expectation:
        compare_repo_and_catalog(repo_cloned_bare, cloned)


def test_remove_entry_cascades_aliases(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-p"])
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-q"])

    result = runner.invoke(cli, ["--path", catalog_path, "remove", name])
    assert result.exit_code == 0, result.output

    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    assert catalog.list_aliases() == []
    catalog.assert_consistency()
