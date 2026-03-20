"""Tests for ``xorq run --alias`` / ``--name``."""

import shutil
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.catalog.annex import Annex, GitAnnex
from xorq.catalog.catalog import Catalog
from xorq.catalog.expr_utils import build_expr_context_zip
from xorq.cli import _resolve_alias, cli


# --- fixtures ---


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def catalog(tmp_path):
    repo = Catalog.init_repo_path(tmp_path / "catalog")
    repo_path = Path(repo.working_dir)
    git_annex = GitAnnex(repo=repo, annex=Annex(repo_path=repo_path))
    return Catalog(git_annex=git_annex)


@pytest.fixture
def catalog_with_alias(catalog, tmp_path):
    expr = xo.memtable({"col": [1, 2, 3]})
    with build_expr_context_zip(expr) as zip_path:
        target = tmp_path / zip_path.name
        shutil.copy(zip_path, target)
        entry = catalog.add(target, aliases=("my-alias",))
    return catalog, entry


# --- validation tests ---


def test_run_no_args(runner):
    result = runner.invoke(cli, ["run"])
    assert result.exit_code != 0
    assert "Provide either BUILD_PATH or --alias" in result.output


def test_run_build_path_and_alias_mutually_exclusive(runner):
    result = runner.invoke(cli, ["run", "./some/path", "--alias", "foo"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_run_name_without_alias(runner):
    result = runner.invoke(cli, ["run", "./some/path", "--name", "ns"])
    assert result.exit_code != 0
    assert "--name is only valid with --alias" in result.output


# --- alias resolution tests ---


def test_resolve_alias(catalog_with_alias):
    catalog, entry = catalog_with_alias
    with patch.object(Catalog, "from_kwargs", return_value=catalog):
        resolved = _resolve_alias("my-alias")
    assert resolved.name == entry.name


def test_resolve_alias_unknown(catalog_with_alias):
    catalog, _ = catalog_with_alias
    with patch.object(Catalog, "from_kwargs", return_value=catalog):
        with pytest.raises(click.ClickException, match="Unknown alias"):
            _resolve_alias("no-such")


def test_resolve_alias_lists_available(catalog_with_alias):
    catalog, _ = catalog_with_alias
    with patch.object(Catalog, "from_kwargs", return_value=catalog):
        with pytest.raises(click.ClickException, match="my-alias"):
            _resolve_alias("no-such")


def test_resolve_alias_with_name(catalog_with_alias):
    catalog, entry = catalog_with_alias
    with patch.object(Catalog, "from_kwargs", return_value=catalog) as mock:
        resolved = _resolve_alias("my-alias", name="my-ns")
    mock.assert_called_once_with(name="my-ns", init=False)
    assert resolved.name == entry.name


# --- integration tests ---


def test_run_alias_invokes_run_command(runner, catalog_with_alias):
    catalog, _ = catalog_with_alias
    with (
        patch.object(Catalog, "from_kwargs", return_value=catalog),
        patch("xorq.cli.run_command") as mock_run,
    ):
        result = runner.invoke(cli, ["run", "--alias", "my-alias"])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    build_dir = mock_run.call_args[0][0]
    assert isinstance(build_dir, Path)


def test_run_alias_with_name_passes_name(runner, catalog_with_alias):
    catalog, _ = catalog_with_alias
    with (
        patch.object(Catalog, "from_kwargs", return_value=catalog) as mock_catalog,
        patch("xorq.cli.run_command") as mock_run,
    ):
        result = runner.invoke(cli, ["run", "--alias", "my-alias", "--name", "my-ns"])
    assert result.exit_code == 0, result.output
    mock_catalog.assert_called_once_with(name="my-ns", init=False)
    mock_run.assert_called_once()


def test_run_alias_forwards_options(runner, catalog_with_alias):
    catalog, _ = catalog_with_alias
    with (
        patch.object(Catalog, "from_kwargs", return_value=catalog),
        patch("xorq.cli.run_command") as mock_run,
    ):
        result = runner.invoke(
            cli,
            [
                "run",
                "--alias",
                "my-alias",
                "--format",
                "csv",
                "--limit",
                "10",
            ],
        )
    assert result.exit_code == 0, result.output
    _, output_path, output_format, cache_dir, limit = mock_run.call_args[0]
    assert output_format == "csv"
    assert limit == 10
