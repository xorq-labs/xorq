from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import tomlkit

from xorq.cli import init_command
from xorq.init_templates import (
    InitTemplateError,
    has_latest_placeholder,
    resolve_xorq_spec,
    rewrite_template_xorq_dep,
)


def _mock_distribution(direct_url, version="0.3.25"):
    dist = MagicMock()
    dist.version = version
    dist.read_text.return_value = None if direct_url is None else json.dumps(direct_url)
    return dist


def test_resolve_xorq_spec_pypi():
    with patch(
        "xorq.init_templates.distribution",
        return_value=_mock_distribution(None, version="0.3.25"),
    ):
        assert resolve_xorq_spec() == "xorq == 0.3.25"
        assert resolve_xorq_spec(extras="[duckdb]") == "xorq[duckdb] == 0.3.25"


def test_resolve_xorq_spec_editable():
    direct_url = {
        "url": "file:///workspace/xorq",
        "dir_info": {"editable": True},
    }
    with patch(
        "xorq.init_templates.distribution",
        return_value=_mock_distribution(direct_url),
    ):
        spec = resolve_xorq_spec(extras="[duckdb]")
        assert spec == "xorq[duckdb] @ file:///workspace/xorq"


def test_resolve_xorq_spec_local_dir_non_editable():
    direct_url = {
        "url": "file:///workspace/xorq",
        "dir_info": {},
    }
    with patch(
        "xorq.init_templates.distribution",
        return_value=_mock_distribution(direct_url),
    ):
        assert resolve_xorq_spec() == "xorq @ file:///workspace/xorq"


def test_resolve_xorq_spec_archive():
    direct_url = {
        "url": "https://files.pythonhosted.org/.../xorq-0.3.25.tar.gz",
        "archive_info": {"hash": "sha256=abc"},
    }
    with patch(
        "xorq.init_templates.distribution",
        return_value=_mock_distribution(direct_url),
    ):
        spec = resolve_xorq_spec()
        assert spec == "xorq @ https://files.pythonhosted.org/.../xorq-0.3.25.tar.gz"


def test_resolve_xorq_spec_vcs():
    direct_url = {
        "url": "https://github.com/xorq-labs/xorq",
        "vcs_info": {"vcs": "git", "commit_id": "abc123"},
    }
    with patch(
        "xorq.init_templates.distribution",
        return_value=_mock_distribution(direct_url),
    ):
        spec = resolve_xorq_spec(extras="[duckdb]")
        assert spec == "xorq[duckdb] @ git+https://github.com/xorq-labs/xorq@abc123"


def test_resolve_xorq_spec_unknown_shape_raises():
    direct_url = {"url": "weird://", "mystery_info": {}}
    with patch(
        "xorq.init_templates.distribution",
        return_value=_mock_distribution(direct_url),
    ):
        with pytest.raises(InitTemplateError, match="unrecognized direct_url"):
            resolve_xorq_spec()


def test_resolve_xorq_spec_override_verbatim():
    # No mocking; override short-circuits any detection.
    override = "xorq[duckdb] @ git+https://github.com/xorq-labs/xorq@main"
    assert resolve_xorq_spec(override=override) == override


@pytest.fixture
def latest_template(tmp_path):
    pyproject = tmp_path.joinpath("pyproject.toml")
    pyproject.write_text(
        """
[project]
name = "demo"
version = "0.0.1"
dependencies = [
    "xorq[duckdb] @ LATEST",
    "pandas >= 2.0",
]

[tool.uv.sources]
xorq = { git = "https://github.com/xorq-labs/xorq" }
""".lstrip()
    )
    tmp_path.joinpath("uv.lock").write_text("# stale\n")
    tmp_path.joinpath("requirements.txt").write_text("# stale\n")
    return tmp_path


def test_has_latest_placeholder_true(latest_template):
    assert has_latest_placeholder(latest_template)


def test_has_latest_placeholder_false(tmp_path):
    tmp_path.joinpath("pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0.0.1"\ndependencies = ["xorq == 0.1.0"]\n'
    )
    assert not has_latest_placeholder(tmp_path)


def test_rewrite_template_xorq_dep(latest_template):
    spec = "xorq[duckdb] == 0.3.25"
    rewrite_template_xorq_dep(latest_template, spec)

    data = tomlkit.loads(latest_template.joinpath("pyproject.toml").read_text())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert spec in deps
    assert not any("LATEST" in d for d in deps)
    assert "xorq" not in data.get("tool", {}).get("uv", {}).get("sources", {})
    assert not latest_template.joinpath("uv.lock").exists()
    assert not latest_template.joinpath("requirements.txt").exists()


def test_rewrite_template_xorq_dep_no_match_raises(tmp_path):
    tmp_path.joinpath("pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0.0.1"\ndependencies = ["xorq == 0.1.0"]\n'
    )
    with pytest.raises(InitTemplateError, match="LATEST"):
        rewrite_template_xorq_dep(tmp_path, "xorq == 0.3.25")


def test_init_command_old_template_fallback(tmp_path, capsys):
    """Old-format templates (no `LATEST`) keep working with a warning."""
    target = tmp_path.joinpath("out")

    def fake_download(path, template, branch=None):
        target.mkdir()
        target.joinpath("pyproject.toml").write_text(
            '[project]\nname="x"\nversion="0.0.1"\ndependencies = ["xorq == 0.1.0"]\n'
        )
        return target

    with patch(
        "xorq.common.utils.download_utils.download_unpacked_xorq_template",
        side_effect=fake_download,
    ):
        result = init_command(path=str(target), template="cached-fetcher")

    assert result == target
    captured = capsys.readouterr()
    assert "does not contain `xorq @ LATEST`" in captured.err


def test_init_command_latest_template_rewrites(tmp_path, capsys):
    """LATEST templates get their dep rewritten and uv.lock generated."""
    target = tmp_path.joinpath("out")

    def fake_download(path, template, branch=None):
        target.mkdir()
        target.joinpath("pyproject.toml").write_text(
            '[project]\nname = "x"\nversion = "0.0.1"\n'
            'dependencies = ["xorq[duckdb] @ LATEST"]\n'
        )
        return target

    override = "xorq[duckdb] == 0.3.25"
    with patch(
        "xorq.common.utils.download_utils.download_unpacked_xorq_template",
        side_effect=fake_download,
    ):
        init_command(
            path=str(target),
            template="cached-fetcher",
            xorq_spec=override,
            no_lock=True,  # skip the real `uv lock` subprocess
        )

    data = tomlkit.loads(target.joinpath("pyproject.toml").read_text())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert override in deps
    assert not any("LATEST" in d for d in deps)
    captured = capsys.readouterr()
    assert override in captured.out
