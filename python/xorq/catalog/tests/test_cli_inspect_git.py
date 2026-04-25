# Tests for: info, list, list-aliases, get, check, log, replay, clone,
#            push/pull/sync, CLI option validation, tui, schema commands
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from xorq.catalog.catalog import (
    Catalog,
    CatalogAddition,
)
from xorq.catalog.cli import cli
from xorq.catalog.tests.conftest import (
    TEST_WHEEL_NAME,
    compare_repo_and_catalog,
    make_build_zip,
)
from xorq.catalog.zip_utils import (
    BuildZip,
    write_zip,
)
from xorq.ibis_yaml.enums import REQUIRED_ARCHIVE_NAMES


@pytest.fixture
def runner():
    yield CliRunner()


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
        dict.fromkeys((*REQUIRED_ARCHIVE_NAMES, TEST_WHEEL_NAME), b""),
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
