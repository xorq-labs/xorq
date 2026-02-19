# https://docs.pytest.org/en/7.1.x/example/parametrize.html#parametrizing-conditional-raising
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from click.testing import CliRunner

from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.tests.conftest import (
    compare_repo_and_catalog,
    make_build_tgz,
)


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


def test_add_duplicate(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    result = runner.invoke(cli, ["--path", catalog_path, "add", path])
    assert result.exit_code != 0


def test_add_nonexistent_path(runner, catalog_path):
    result = runner.invoke(
        cli, ["--path", catalog_path, "add", "/nonexistent/file.tgz"]
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

    path = make_build_tgz(tmpdir, "to-add")
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


# --- list command ---


def test_list_empty(runner, catalog_path):
    result = runner.invoke(cli, ["--path", catalog_path, "list"])
    assert result.exit_code == 0, result.output
    assert "No entries." in result.output


def test_list_populated(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    result = runner.invoke(cli, ["--path", catalog_path, "list"])
    assert result.exit_code == 0, result.output
    assert "No entries." not in result.output
    for p in data_dict.values():
        name = Path(p).name.removesuffix("".join(Path(p).suffixes))
        assert name in result.output


# --- get command ---


def test_get_command(runner, catalog_path, data_dict, tmpdir):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    output_dir = str(Path(tmpdir).joinpath("export"))
    Path(output_dir).mkdir()
    result = runner.invoke(cli, ["--path", catalog_path, "get", name, "-o", output_dir])
    assert result.exit_code == 0, result.output
    assert "Exported to" in result.output


def test_get_nonexistent_entry(runner, catalog_path, tmpdir):
    output_dir = str(Path(tmpdir).joinpath("export"))
    Path(output_dir).mkdir()
    result = runner.invoke(
        cli, ["--path", catalog_path, "get", "nonexistent", "-o", output_dir]
    )
    assert result.exit_code != 0


# --- check command ---


def test_check_command(runner, catalog_path):
    result = runner.invoke(cli, ["--path", catalog_path, "check"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_check_populated(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    result = runner.invoke(cli, ["--path", catalog_path, "check"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


# --- clone command ---


def test_clone_command(runner, catalog_populated, tmpdir):
    dest = str(Path(tmpdir).joinpath("cloned"))
    result = runner.invoke(
        cli, ["clone", str(catalog_populated.repo_path), "--path", dest]
    )
    assert result.exit_code == 0, result.output
    assert "Cloned to" in result.output
    assert Path(dest).exists()


def test_clone_name_and_path_mutually_exclusive(runner, catalog_populated):
    result = runner.invoke(
        cli,
        [
            "clone",
            str(catalog_populated.repo_path),
            "--name",
            "foo",
            "--path",
            "/tmp/bar",
        ],
    )
    assert result.exit_code != 0


def test_clone_derives_name_from_url(runner, catalog_populated, tmpdir, monkeypatch):
    clone_base = Path(tmpdir).joinpath("clone-base")
    clone_base.mkdir()
    monkeypatch.setattr(Catalog, "by_name_base_path", clone_base)
    result = runner.invoke(cli, ["clone", str(catalog_populated.repo_path)])
    assert result.exit_code == 0, result.output
    assert "Cloned to" in result.output


# --- push / pull / sync ---


def test_push_with_remote(runner, repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("push-test")
    )
    compare_repo_and_catalog(repo_cloned_bare, cloned)
    before = repo_cloned_bare.head.commit.hexsha

    result = runner.invoke(cli, ["--path", str(cloned.repo_path), "push"])
    assert result.exit_code == 0, result.output
    assert "Pushed." in result.output
    middle = repo_cloned_bare.head.commit.hexsha
    assert before == middle

    # check commit
    path = make_build_tgz(tmpdir, "to-add")
    result = runner.invoke(
        cli, ["--path", cloned.repo_path, "add", str(path), "--no-sync"]
    )
    assert result.exit_code == 0, result.output
    assert "Added" in result.output
    middle = repo_cloned_bare.head.commit.hexsha
    assert before == middle
    with pytest.raises(AssertionError):
        compare_repo_and_catalog(repo_cloned_bare, cloned)

    result = runner.invoke(cli, ["--path", str(cloned.repo_path), "push"])
    assert result.exit_code == 0, result.output
    assert "Pushed." in result.output
    after = repo_cloned_bare.head.commit.hexsha
    assert before != after
    compare_repo_and_catalog(repo_cloned_bare, cloned)


def test_pull_with_remote(runner, repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("pull-test")
    )
    pusher = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("pusher")
    )
    before = cloned.repo.head.commit.hexsha

    result = runner.invoke(cli, ["--path", str(cloned.repo_path), "pull"])
    assert result.exit_code == 0, result.output
    assert "Pulled." in result.output
    #
    middle = cloned.repo.head.commit.hexsha
    assert before == middle
    assert middle == pusher.repo.head.commit.hexsha

    path = make_build_tgz(tmpdir, "to-add")
    pusher.add(path, sync=True)
    middle = cloned.repo.head.commit.hexsha
    assert before == middle
    assert middle != pusher.repo.head.commit.hexsha

    result = runner.invoke(cli, ["--path", str(cloned.repo_path), "pull"])
    assert result.exit_code == 0, result.output
    assert "Pulled." in result.output
    #
    after = cloned.repo.head.commit.hexsha
    assert before != after
    assert after == pusher.repo.head.commit.hexsha


def test_sync_with_remote(runner, repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("sync-test")
    )
    result = runner.invoke(cli, ["--path", str(cloned.repo_path), "sync"])
    assert result.exit_code == 0, result.output
    assert "Synced." in result.output


# --- CLI option validation ---


def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Manage xorq build-artifact catalogs." in result.output


def test_subcommand_help(runner):
    for cmd in (
        "init",
        "add",
        "remove",
        "list",
        "get",
        "push",
        "pull",
        "sync",
        "clone",
        "check",
    ):
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed"


def test_list_invalid_path(runner):
    result = runner.invoke(cli, ["--path", "/nonexistent/path", "list"])
    assert result.exit_code != 0
