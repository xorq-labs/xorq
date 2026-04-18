# https://docs.pytest.org/en/7.1.x/example/parametrize.html#parametrizing-conditional-raising
import json
import shutil
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from click.testing import CliRunner

import xorq.api as xo
import xorq.expr.builders as _builders_mod
from xorq.catalog.catalog import (
    Catalog,
    CatalogAddition,
)
from xorq.catalog.cli import cli
from xorq.catalog.tests.conftest import (
    compare_repo_and_catalog,
    make_build_zip,
)
from xorq.catalog.zip_utils import (
    BuildZip,
    extract_build_zip_context,
    write_zip,
)
from xorq.cli import cli as top_cli
from xorq.expr.builders import (
    _FROM_TAG_NODE_REGISTRY,
    TagHandler,
    _reset_registry,
    register_tag_handler,
)
from xorq.ibis_yaml.enums import REQUIRED_ARCHIVE_NAMES
from xorq.vendor.ibis.expr import operations as ops


@pytest.fixture
def runner():
    yield CliRunner()


@pytest.fixture
def saved_registry():
    """Save and restore the handler registry around a test."""
    saved = dict(_FROM_TAG_NODE_REGISTRY)
    saved_keys = _builders_mod._BUILTIN_KEYS
    saved_init = _builders_mod._initialized
    yield
    _FROM_TAG_NODE_REGISTRY.clear()
    _FROM_TAG_NODE_REGISTRY.update(saved)
    _builders_mod._BUILTIN_KEYS = saved_keys
    _builders_mod._initialized = saved_init


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


# --- info command ---


def test_info_empty(runner, catalog_path):
    result = runner.invoke(cli, ["--path", catalog_path, "info"])
    assert result.exit_code == 0, result.output
    assert "path:" in result.output
    assert "commit:" in result.output
    assert "entries: 0" in result.output
    assert "aliases: 0" in result.output
    assert "(none)" in result.output


def test_info_populated(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    result = runner.invoke(cli, ["--path", catalog_path, "info"])
    assert result.exit_code == 0, result.output
    assert f"entries: {len(data_dict)}" in result.output
    assert "aliases: 0" in result.output


def test_info_alias_count(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-p"])
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-q"])
    result = runner.invoke(cli, ["--path", catalog_path, "info"])
    assert result.exit_code == 0, result.output
    assert "aliases: 2" in result.output


def test_info_shows_remotes(runner, repo_cloned_bare, tmpdir):
    cloned = Catalog.clone_from(
        repo_cloned_bare.working_dir, Path(tmpdir).joinpath("info-remote-test")
    )
    result = runner.invoke(cli, ["--path", str(cloned.repo_path), "info"])
    assert result.exit_code == 0, result.output
    assert "origin" in result.output


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


def test_list_with_kind(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    result = runner.invoke(cli, ["--path", catalog_path, "list", "--kind"])
    assert result.exit_code == 0, result.output
    assert "No entries." not in result.output
    for p in data_dict.values():
        name = Path(p).name.removesuffix("".join(Path(p).suffixes))
        assert name in result.output
    assert (
        "\tunbound_expr" in result.output
        or "\tsource" in result.output
        or "\texpr" in result.output
    )


# --- list-aliases command ---


def test_list_aliases_invalid_path(runner):
    result = runner.invoke(cli, ["--path", "/nonexistent/path", "list-aliases"])
    assert result.exit_code != 0
    assert "init" in result.output


def test_list_aliases_empty(runner, catalog_path):
    result = runner.invoke(cli, ["--path", catalog_path, "list-aliases"])
    assert result.exit_code == 0, result.output
    assert "No aliases." in result.output


def test_list_aliases_populated(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-a"])
    runner.invoke(cli, ["--path", catalog_path, "add-alias", name, "alias-b"])
    result = runner.invoke(cli, ["--path", catalog_path, "list-aliases"])
    assert result.exit_code == 0, result.output
    assert "alias-a" in result.output
    assert "alias-b" in result.output


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


def test_get_default_output_dir(runner, catalog_path, data_dict):
    path = str(next(iter(data_dict.values())))
    runner.invoke(cli, ["--path", catalog_path, "add", path])
    name = Path(path).name.removesuffix("".join(Path(path).suffixes))
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--path", catalog_path, "get", name])
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


def test_check_catches_inconsistency(runner, catalog_path, tmpdir):
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    zip_path = write_zip(
        Path(tmpdir).joinpath("build.zip"),
        dict.fromkeys(REQUIRED_ARCHIVE_NAMES, b""),
    )
    catalog_addition = CatalogAddition(BuildZip(zip_path), catalog)
    catalog_addition.ensure_dirs()
    entry_path = catalog_addition.catalog_entry.catalog_path
    with catalog.commit_context("bad commit"):
        shutil.copy(zip_path, entry_path)
        catalog.repo.index.add((entry_path,))

    result = runner.invoke(cli, ["--path", catalog_path, "check"])
    assert result.exit_code != 0


def test_check_populated(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    result = runner.invoke(cli, ["--path", catalog_path, "check"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


# --- log command ---


def test_log_command(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    result = runner.invoke(cli, ["--path", catalog_path, "log"])
    assert result.exit_code == 0, result.output
    assert "[init]" in result.output
    assert "[add]" in result.output
    assert "--- summary ---" in result.output


def test_log_json(runner, catalog_path, data_dict):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    result = runner.invoke(cli, ["--path", catalog_path, "log", "--json"])
    assert result.exit_code == 0, result.output
    ops = json.loads(result.output)
    assert isinstance(ops, list)
    assert any(op["type"] == "AddEntry" for op in ops)


def test_log_empty_catalog(runner, catalog_path):
    result = runner.invoke(cli, ["--path", catalog_path, "log"])
    assert result.exit_code == 0, result.output
    assert "[init]" in result.output


# --- replay command ---


def test_replay_command(runner, catalog_path, data_dict, tmpdir):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    target = str(Path(tmpdir).joinpath("replayed"))
    result = runner.invoke(cli, ["--path", catalog_path, "replay", target])
    assert result.exit_code == 0, result.output
    assert "Replayed" in result.output
    target_catalog = Catalog.from_repo_path(target, init=False)
    assert sorted(target_catalog.list()) == sorted(
        Catalog.from_repo_path(catalog_path, init=False).list()
    )


def test_replay_dry_run(runner, catalog_path, data_dict, tmpdir):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    target = str(Path(tmpdir).joinpath("replayed"))
    result = runner.invoke(cli, ["--path", catalog_path, "replay", target, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Replayed" not in result.output


def test_replay_no_preserve_commits(runner, catalog_path, data_dict, tmpdir):
    paths = [str(p) for p in data_dict.values()]
    runner.invoke(cli, ["--path", catalog_path, "add", *paths])
    target = str(Path(tmpdir).joinpath("replayed"))
    result = runner.invoke(
        cli, ["--path", catalog_path, "replay", target, "--no-preserve-commits"]
    )
    assert result.exit_code == 0, result.output
    assert "Replayed" in result.output


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
    path = make_build_zip(tmpdir, "to-add")
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

    path = make_build_zip(tmpdir, "to-add")
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


def test_no_subcommand_prints_help(runner):
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "Manage xorq build-artifact catalogs." in result.output


def test_subcommand_help(runner):
    for cmd in (
        "init",
        "add",
        "add-alias",
        "remove",
        "remove-alias",
        "info",
        "list",
        "list-aliases",
        "get",
        "push",
        "pull",
        "sync",
        "clone",
        "check",
        "run",
        "compose",
    ):
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed"


def test_name_and_path_mutually_exclusive(runner, catalog_path, tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    runner.invoke(cli, ["--name", "my-catalog", "init"])
    result = runner.invoke(
        cli, ["--name", "my-catalog", "--path", catalog_path, "list"]
    )
    assert result.exit_code != 0


def test_list_invalid_path(runner):
    result = runner.invoke(cli, ["--path", "/nonexistent/path", "list"])
    assert result.exit_code != 0
    assert "init" in result.output


def test_missing_catalog_hint_includes_path(runner, tmpdir):
    missing = str(Path(tmpdir).joinpath("no-such-catalog"))
    result = runner.invoke(cli, ["--path", missing, "list"])
    assert result.exit_code != 0
    assert f"--path {missing}" in result.output
    assert "init" in result.output


def test_missing_catalog_hint_includes_name(runner, tmpdir, monkeypatch):
    monkeypatch.setattr(Catalog, "by_name_base_path", Path(tmpdir))
    result = runner.invoke(cli, ["--name", "my-catalog", "list"])
    assert result.exit_code != 0
    assert "--name my-catalog" in result.output
    assert "init" in result.output


# --- TUI validation ---


def test_tui_missing_catalog_shows_hint(runner, tmpdir):
    missing = str(Path(tmpdir).joinpath("no-such-catalog"))
    result = runner.invoke(cli, ["--path", missing, "tui"])
    assert result.exit_code != 0
    assert "init" in result.output


def test_tui_missing_catalog_with_name(runner, tmpdir):
    result = runner.invoke(cli, ["--name", "no-such-catalog-xyz", "tui"])
    assert result.exit_code != 0
    assert "init" in result.output


# --- schema command ---


def test_schema_command(runner, catalog_path, tmpdir):
    archive = make_build_zip(tmpdir, "schema-entry")
    runner.invoke(cli, ["--path", catalog_path, "add", str(archive)])
    result = runner.invoke(cli, ["--path", catalog_path, "schema", archive.stem])
    assert result.exit_code == 0, result.output
    assert "Source (bound)" in result.output
    assert "Schema Out:" in result.output


def test_schema_json(runner, catalog_path, tmpdir):
    archive = make_build_zip(tmpdir, "schema-json")
    runner.invoke(cli, ["--path", catalog_path, "add", str(archive)])
    result = runner.invoke(
        cli, ["--path", catalog_path, "schema", archive.stem, "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["kind"] == "source"
    assert "schema_out" in data


def test_schema_nonexistent(runner, catalog_path):
    result = runner.invoke(cli, ["--path", catalog_path, "schema", "no-such-entry"])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert "list-aliases" in result.output


# --- run command ---


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


def test_run_single_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_default_output(runner, catalog_with_source_and_transform):
    """Default output (no -o) writes to /dev/null, exits 0."""
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src"],
    )
    assert result.exit_code == 0, result.output


def test_run_with_limit(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "-o", "-", "-f", "csv", "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 2  # header + 1 data row


def test_run_with_code(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "src",
            "-c",
            "source.filter(source.amount > 15)",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output
    # amount <= 15 should be filtered out (only 20.0 and 30.0 remain)
    lines = result.output.strip().splitlines()
    assert len(lines) == 3  # header + 2 data rows


def test_run_rejects_unbound_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "trn"],
    )
    assert result.exit_code != 0
    assert "unbound expression" in result.output


def test_run_two_entries(runner, catalog_with_source_and_transform):
    """run src trn  — compose + execute in one shot."""
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "trn", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_two_entries_with_code(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "src",
            "-c",
            "source.filter(source.amount > 15)",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 3  # header + 2 data rows


def test_run_two_entries_json(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "trn", "-o", "-", "-f", "json"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_two_entries_with_limit(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "src",
            "trn",
            "-o",
            "-",
            "-f",
            "csv",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 2  # header + 1 data row


def test_run_no_entries(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run"],
    )
    assert result.exit_code != 0


def test_run_piped_arrow_into_unbound(runner, catalog_with_source_and_transform):
    """run bound -o - -f arrow | run unbound  — simulated via CliRunner input."""

    catalog_path, _, _ = catalog_with_source_and_transform

    # Step 1: run the source entry, capture arrow bytes
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "-o", "-", "-f", "arrow"],
    )
    assert result.exit_code == 0, result.output
    arrow_bytes = result.output_bytes

    # Step 2: pipe arrow bytes into the unbound entry
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "trn", "-o", "-", "-f", "csv"],
        input=arrow_bytes,
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output
    assert "amount" in result.output


def test_run_nonexistent_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "no-such-entry"],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


# --- compose command ---


def test_compose_two_entries(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn"],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output


def test_compose_with_alias(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "composed-result"],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output
    assert "composed-result" in result.output


def test_compose_with_code(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "compose",
            "src",
            "-c",
            "source.filter(source.amount > 15)",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output


def test_compose_code_no_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "-c", "source.limit(1)"],
    )
    assert result.exit_code != 0


def test_compose_no_entries(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose"],
    )
    assert result.exit_code != 0


def test_compose_dry_run(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Schema:" in result.output
    assert "user_id" in result.output
    # dry-run should NOT catalog
    assert "Cataloged" not in result.output


def test_compose_without_alias_catalogs_by_hash(
    runner, catalog_with_source_and_transform
):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn"],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output


# --- compose + run roundtrip tests ---


def test_compose_then_run_roundtrip(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    # compose with alias
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "my-composed"],
    )
    assert result.exit_code == 0, result.output
    # run the composed entry
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "my-composed", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_compose_then_run_json(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "json-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "json-test", "-f", "json", "-o", "-"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_compose_then_run_parquet(runner, catalog_with_source_and_transform, tmpdir):
    catalog_path, _, _ = catalog_with_source_and_transform
    out = str(Path(tmpdir).joinpath("out.parquet"))
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "pq-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "pq-test", "-f", "parquet", "-o", out],
    )
    assert result.exit_code == 0, result.output
    assert Path(out).exists()


def test_compose_then_run_csv_file(runner, catalog_with_source_and_transform, tmpdir):
    catalog_path, _, _ = catalog_with_source_and_transform
    out = str(Path(tmpdir).joinpath("out.csv"))
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "csv-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "csv-test", "-f", "csv", "-o", out],
    )
    assert result.exit_code == 0, result.output
    assert Path(out).exists()
    assert "user_id" in Path(out).read_text()


def test_compose_then_run_arrow_file(runner, catalog_with_source_and_transform, tmpdir):
    catalog_path, _, _ = catalog_with_source_and_transform
    out = str(Path(tmpdir).joinpath("out.arrow"))
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "arrow-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "arrow-test", "-f", "arrow", "-o", out],
    )
    assert result.exit_code == 0, result.output
    assert Path(out).exists()
    with open(out, "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    assert len(table) > 0


def test_compose_then_run_arrow_stdout(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "arrow-stdout-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "arrow-stdout-test", "-f", "arrow", "-o", "-"],
    )
    assert result.exit_code == 0, result.output
    assert len(result.output) > 0


# --- --pdb flag ---


def test_pdb_flag_invokes_post_mortem(tmp_path, monkeypatch):
    """--pdb should let exceptions propagate to PdbGroup, which calls pdb.post_mortem."""
    mock_pm = MagicMock()
    monkeypatch.setattr("xorq.cli.pdb.post_mortem", mock_pm)

    # "catalog list" on a non-catalog directory fails inside the command body
    # (not at Click arg-parsing time), so it exercises click_context_catalog.
    runner = CliRunner()
    result = runner.invoke(
        top_cli,
        ["--pdb", "catalog", "--path", str(tmp_path), "list"],
    )
    assert result.exit_code != 0
    assert mock_pm.called, "pdb.post_mortem was not called with --pdb"


def test_no_pdb_flag_wraps_exception(tmp_path):
    """Without --pdb, errors should be wrapped as clean 'Error: ...' messages."""
    runner = CliRunner()
    result = runner.invoke(
        top_cli,
        ["catalog", "--path", str(tmp_path), "list"],
    )
    assert result.exit_code != 0
    assert "Error:" in result.output


# --- --rename-params tests ---


@pytest.fixture
def catalog_with_parameterized_entries(catalog_path):
    """Populate a catalog with source and transform entries containing NamedScalarParameter nodes."""
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)

    # Source: a memtable with a parameterized filter
    threshold = xo.param("threshold", "float64", default=5.0)
    source = xo.memtable(
        {"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0], "name": ["a", "b", "c"]}
    )
    source_filtered = source.filter(source.amount > threshold)
    source_entry = catalog.add(source_filtered, aliases=("psrc",))

    # Transform: an unbound expr with its own NamedScalarParameter
    limit_param = xo.param("threshold", "float64", default=15.0)
    schema = source_filtered.schema()
    unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
    transform = unbound.filter(unbound.amount > limit_param).select("user_id", "amount")
    transform_entry = catalog.add(transform, aliases=("ptrn",))

    return catalog_path, source_entry.name, transform_entry.name


def test_run_with_rename_params(runner, catalog_with_parameterized_entries):
    """run with --rename-params renames a parameter in a transform entry."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "ptrn,threshold,trn_threshold",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_compose_with_rename_params(runner, catalog_with_parameterized_entries):
    """compose with --rename-params renames params before composition."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "compose",
            "psrc",
            "ptrn",
            "--rename-params",
            "ptrn,threshold,trn_threshold",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "user_id" in result.output


def test_rename_params_bad_format(runner, catalog_with_parameterized_entries):
    """--rename-params with wrong format should show an error."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "bad_format",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code != 0
    assert "Expected" in result.output


def test_rename_params_unknown_entry(runner, catalog_with_parameterized_entries):
    """--rename-params with unknown entry name should show an error."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "nonexistent,threshold,new_threshold",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown entry" in result.output


# --- --params tests ---


def test_run_with_params_single_entry(runner, catalog_with_parameterized_entries):
    """run with -p binds a NamedScalarParameter value before execution."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "-p",
            "threshold=25.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    # threshold=25 filters amount > 25, leaving only user_id=3,amount=30
    assert "3,30" in result.output
    assert "1,10" not in result.output
    assert "2,20" not in result.output


def test_run_params_after_rename(runner, catalog_with_parameterized_entries):
    """--params values bind to the renamed names, not the original."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "ptrn,threshold,trn_threshold",
            "-p",
            "trn_threshold=25.0",
            "-p",
            "threshold=5.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3,30" in result.output
    assert "1,10" not in result.output


def test_run_params_bad_format(runner, catalog_with_parameterized_entries):
    """-p without '=' should report a usage error."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "-p",
            "no_equals",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code != 0
    assert "Expected key=value" in result.output


def test_run_params_unknown_name(runner, catalog_with_parameterized_entries):
    """-p with a name not in the expr should error with available names."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "-p",
            "not_a_param=1.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown parameter" in result.output
    assert "threshold" in result.output


# --- run with ExprBuilder entries ---


def test_run_expr_builder_entry(runner, catalog_path, saved_registry):
    """ExprBuilder entries should be runnable via `catalog run`."""
    _reset_registry()
    handler = TagHandler(
        tag_names=("test_cli_builder",),
        extract_metadata=lambda tag_node: {"type": "test_cli_builder"},
    )
    register_tag_handler(handler)

    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    source = xo.memtable({"x": [1, 2, 3], "y": [4, 5, 6]}, name="builder_src")
    tagged = source.tag("test_cli_builder")
    catalog.add(tagged, aliases=("bld",), sync=False)

    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "bld", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "x" in result.output


# --- run-cached command ---


def test_run_cached_single_entry(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output
    assert cache_dir.exists()
    assert any(cache_dir.iterdir())


def test_run_cached_two_entries(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "trn",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_cached_snapshot_type(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "--cache-type",
            "snapshot",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_cached_ttl(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "--ttl",
            "60",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_cached_with_params(runner, catalog_with_parameterized_entries, tmp_path):
    """run-cached with -p binds a NamedScalarParameter before caching."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "psrc",
            "-p",
            "threshold=25.0",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3,30" in result.output
    assert "1,10" not in result.output
    assert "2,20" not in result.output


def test_run_cached_no_entries(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code != 0
    assert "At least one entry is required" in result.output
