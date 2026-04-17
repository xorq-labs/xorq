import os
import subprocess
import sys

import pytest
from click.testing import CliRunner

from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import default_catalog
from xorq.catalog.constants import (
    DEFAULT_CATALOG_NAME,
)
from xorq.vendor.ibis.config import env_config


ENV_CONFIG_PATH = "xorq.vendor.ibis.config.env_config"


def _env_config_with(name):
    """Return an env_config clone with XORQ_DEFAULT_CATALOG set."""
    return env_config.clone(XORQ_DEFAULT_CATALOG=name)


def _env_config_without():
    """Return an env_config clone with XORQ_DEFAULT_CATALOG empty."""
    return env_config.clone(XORQ_DEFAULT_CATALOG="")


# ---------------------------------------------------------------------------
# _resolve_default_name – resolution order
# ---------------------------------------------------------------------------


def test_resolve_default_name_hardcoded(monkeypatch, tmp_path):
    """Falls back to the hardcoded name when env and config are absent."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        tmp_path / "nonexistent",
    )
    assert Catalog._resolve_default_name() == DEFAULT_CATALOG_NAME


def test_resolve_default_name_config_file(monkeypatch, tmp_path):
    """Config file wins over the hardcoded default."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    config_path = tmp_path / "catalog-default"
    config_path.write_text("my-catalog\n")
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        config_path,
    )
    assert Catalog._resolve_default_name() == "my-catalog"


def test_resolve_default_name_env_over_config(monkeypatch, tmp_path):
    """Env var wins over config file."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_with("env-catalog"))
    config_path = tmp_path / "catalog-default"
    config_path.write_text("config-catalog\n")
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        config_path,
    )
    assert Catalog._resolve_default_name() == "env-catalog"


def test_resolve_default_name_empty_config_file(monkeypatch, tmp_path):
    """Empty or whitespace-only config file falls back to hardcoded default."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    config_path = tmp_path / "catalog-default"
    config_path.write_text("  \n")
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        config_path,
    )
    assert Catalog._resolve_default_name() == DEFAULT_CATALOG_NAME


def test_resolve_default_name_env_without_config(monkeypatch, tmp_path):
    """Env var works even when no config file exists."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_with("env-catalog"))
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        tmp_path / "nonexistent",
    )
    assert Catalog._resolve_default_name() == "env-catalog"


# ---------------------------------------------------------------------------
# from_default – integration with _resolve_default_name
# ---------------------------------------------------------------------------


def test_from_default_uses_resolved_name(monkeypatch, tmp_path):
    """from_default() opens/creates the catalog whose name was resolved."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        tmp_path / "nonexistent",
    )
    monkeypatch.setattr(Catalog, "by_name_base_path", tmp_path)
    catalog = Catalog.from_default()
    assert catalog.repo_path == tmp_path / DEFAULT_CATALOG_NAME


def test_from_default_respects_config(monkeypatch, tmp_path):
    """from_default() opens the catalog named in the config file."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    config_path = tmp_path / "catalog-default"
    config_path.write_text("custom\n")
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        config_path,
    )
    monkeypatch.setattr(Catalog, "by_name_base_path", tmp_path)
    catalog = Catalog.from_default()
    assert catalog.repo_path == tmp_path / "custom"


def test_from_default_respects_env(monkeypatch, tmp_path):
    """from_default() opens the catalog named in the env var."""
    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_with("env-catalog"))
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        tmp_path / "nonexistent",
    )
    monkeypatch.setattr(Catalog, "by_name_base_path", tmp_path)
    catalog = Catalog.from_default()
    assert catalog.repo_path == tmp_path / "env-catalog"


# ---------------------------------------------------------------------------
# CLI: xorq catalog default [--set | --unset]
# ---------------------------------------------------------------------------


@pytest.fixture
def config_path(monkeypatch, tmp_path):
    path = tmp_path / "catalog-default"
    monkeypatch.setattr(
        "xorq.catalog.constants.DEFAULT_CATALOG_CONFIG",
        path,
    )
    return path


def test_cli_default_show_builtin(config_path, monkeypatch):
    """Bare `default` shows the built-in name when nothing is configured."""

    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    result = CliRunner().invoke(default_catalog)
    assert result.exit_code == 0
    assert DEFAULT_CATALOG_NAME in result.output
    assert "built-in" in result.output


def test_cli_default_set_and_show(config_path, monkeypatch):
    """--set writes the config file, then bare show reflects it."""

    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    runner = CliRunner()

    result = runner.invoke(default_catalog, ["--set", "my-catalog"])
    assert result.exit_code == 0
    assert "Default catalog set to 'my-catalog'" in result.output
    assert config_path.read_text().strip() == "my-catalog"

    result = runner.invoke(default_catalog)
    assert result.exit_code == 0
    assert "my-catalog" in result.output
    assert f"config ({config_path})" in result.output


def test_cli_default_unset(config_path, monkeypatch):
    """--unset removes the config file."""

    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    config_path.write_text("something\n")
    result = CliRunner().invoke(default_catalog, ["--unset"])
    assert result.exit_code == 0
    assert not config_path.exists()


def test_cli_default_unset_no_file(config_path, monkeypatch):
    """--unset when no config exists doesn't fail."""

    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_without())
    result = CliRunner().invoke(default_catalog, ["--unset"])
    assert result.exit_code == 0
    assert "No persisted default" in result.output


def test_cli_default_set_and_unset_mutually_exclusive(config_path):
    """--set and --unset together is an error."""

    result = CliRunner().invoke(default_catalog, ["--set", "x", "--unset"])
    assert result.exit_code != 0


def test_cli_default_show_env(config_path, monkeypatch):
    """Show reports env source when XORQ_DEFAULT_CATALOG is set."""

    monkeypatch.setattr(ENV_CONFIG_PATH, _env_config_with("from-env"))
    result = CliRunner().invoke(default_catalog)
    assert result.exit_code == 0
    assert "from-env" in result.output
    assert "env" in result.output


# ---------------------------------------------------------------------------
# CLI subprocess tests – real cold-start through `xorq catalog default`
# ---------------------------------------------------------------------------


@pytest.fixture
def subprocess_config_path(tmp_path):
    return tmp_path / "catalog-default"


def _run_cli(*args, env_extra=None, config_path=None):
    """Run `xorq catalog default` in a subprocess with controlled config."""
    env = os.environ.copy()
    env.pop("XORQ_DEFAULT_CATALOG", None)
    if env_extra:
        env.update(env_extra)
    code = (
        "import xorq.catalog.constants as c; "
        f"c.DEFAULT_CATALOG_CONFIG = __import__('pathlib').Path({str(config_path)!r}); "
        "from xorq.cli import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code, "catalog", "default", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_subprocess_default_show_builtin(subprocess_config_path):
    """Subprocess: bare `default` shows built-in name."""
    result = _run_cli(config_path=subprocess_config_path)
    assert result.returncode == 0
    assert DEFAULT_CATALOG_NAME in result.stdout
    assert "built-in" in result.stdout


def test_subprocess_default_set_and_show(subprocess_config_path):
    """Subprocess: --set persists, then show reflects it."""
    result = _run_cli("--set", "sub-catalog", config_path=subprocess_config_path)
    assert result.returncode == 0
    assert "sub-catalog" in result.stdout

    result = _run_cli(config_path=subprocess_config_path)
    assert result.returncode == 0
    assert "sub-catalog" in result.stdout
    assert "config" in result.stdout


def test_subprocess_default_env_override(subprocess_config_path):
    """Subprocess: XORQ_DEFAULT_CATALOG env var takes precedence."""
    result = _run_cli(
        env_extra={"XORQ_DEFAULT_CATALOG": "from-env"},
        config_path=subprocess_config_path,
    )
    assert result.returncode == 0
    assert "from-env" in result.stdout
    assert "env" in result.stdout
