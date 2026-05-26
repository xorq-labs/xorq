from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import click
import pytest
import tomlkit

from xorq.cli import init_command
from xorq.init_templates import (
    InitTemplateError,
    find_latest_dep,
    has_latest_placeholder,
    resolve_xorq_spec,
    rewrite_template_xorq_dep,
    run_uv_lock,
)


def _patch_direct_url(direct_url: dict | None, version: str = "0.3.25"):
    """Return a patcher for the deferred `_read_direct_url` call."""
    return patch(
        "xorq.init_templates._read_direct_url",
        return_value=(direct_url, version),
    )


def test_resolve_xorq_spec_pypi() -> None:
    with _patch_direct_url(None, version="0.3.25"):
        assert resolve_xorq_spec() == "xorq == 0.3.25"
        assert resolve_xorq_spec(extras="[duckdb]") == "xorq[duckdb] == 0.3.25"


def test_resolve_xorq_spec_editable() -> None:
    direct_url = {
        "url": "file:///workspace/xorq",
        "dir_info": {"editable": True},
    }
    with _patch_direct_url(direct_url):
        spec = resolve_xorq_spec(extras="[duckdb]")
        assert spec == "xorq[duckdb] @ file:///workspace/xorq"


def test_resolve_xorq_spec_local_dir_non_editable() -> None:
    direct_url = {
        "url": "file:///workspace/xorq",
        "dir_info": {},
    }
    with _patch_direct_url(direct_url):
        assert resolve_xorq_spec() == "xorq @ file:///workspace/xorq"


def test_resolve_xorq_spec_archive() -> None:
    direct_url = {
        "url": "https://files.pythonhosted.org/.../xorq-0.3.25.tar.gz",
        "archive_info": {"hash": "sha256=abc"},
    }
    with _patch_direct_url(direct_url):
        spec = resolve_xorq_spec()
        assert spec == "xorq @ https://files.pythonhosted.org/.../xorq-0.3.25.tar.gz"


def test_resolve_xorq_spec_vcs() -> None:
    direct_url = {
        "url": "https://github.com/xorq-labs/xorq",
        "vcs_info": {"vcs": "git", "commit_id": "abc123"},
    }
    with _patch_direct_url(direct_url):
        spec = resolve_xorq_spec(extras="[duckdb]")
        assert spec == "xorq[duckdb] @ git+https://github.com/xorq-labs/xorq@abc123"


def test_resolve_xorq_spec_unknown_shape_raises() -> None:
    direct_url = {"url": "weird://", "mystery_info": {}}
    with _patch_direct_url(direct_url):
        with pytest.raises(InitTemplateError, match="unrecognized direct_url"):
            resolve_xorq_spec()


def test_resolve_xorq_spec_override_verbatim() -> None:
    # No mocking; override short-circuits any detection.
    override = "xorq[duckdb] @ git+https://github.com/xorq-labs/xorq@main"
    assert resolve_xorq_spec(override=override) == override


def test_resolve_xorq_spec_override_extras_match_no_warning() -> None:
    override = "xorq[duckdb] == 0.3.25"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_xorq_spec(override=override, extras="[duckdb]") == override


def test_resolve_xorq_spec_override_extras_mismatch_warns() -> None:
    override = "xorq[postgres] == 0.3.25"
    with pytest.warns(UserWarning, match="do not match the template"):
        assert resolve_xorq_spec(override=override, extras="[duckdb]") == override


def test_resolve_xorq_spec_override_missing_extras_warns() -> None:
    override = "xorq == 0.3.25"
    with pytest.warns(UserWarning, match=r"\[\] do not match"):
        assert resolve_xorq_spec(override=override, extras="[duckdb]") == override


def test_resolve_xorq_spec_override_no_template_extras_no_warning() -> None:
    """Template without extras + override with extras: silent (user opted in)."""
    override = "xorq[duckdb] == 0.3.25"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_xorq_spec(override=override, extras="") == override


@pytest.fixture
def latest_template(tmp_path: Path) -> Path:
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


def test_has_latest_placeholder_true(latest_template: Path) -> None:
    assert has_latest_placeholder(latest_template)


def test_has_latest_placeholder_false(tmp_path: Path) -> None:
    tmp_path.joinpath("pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0.0.1"\ndependencies = ["xorq == 0.1.0"]\n'
    )
    assert not has_latest_placeholder(tmp_path)


def test_rewrite_template_xorq_dep(latest_template: Path) -> None:
    spec = "xorq[duckdb] == 0.3.25"
    rewrite_template_xorq_dep(latest_template, spec)

    data = tomlkit.loads(latest_template.joinpath("pyproject.toml").read_text())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert spec in deps
    assert not any("LATEST" in d for d in deps)
    assert "xorq" not in data.get("tool", {}).get("uv", {}).get("sources", {})
    assert not latest_template.joinpath("uv.lock").exists()
    assert not latest_template.joinpath("requirements.txt").exists()


def test_rewrite_template_xorq_dep_no_match_raises(tmp_path: Path) -> None:
    tmp_path.joinpath("pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0.0.1"\ndependencies = ["xorq == 0.1.0"]\n'
    )
    with pytest.raises(InitTemplateError, match="LATEST"):
        rewrite_template_xorq_dep(tmp_path, "xorq == 0.3.25")


def test_init_command_old_template_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Old-format templates (no `LATEST`) keep working with a warning."""
    target = tmp_path.joinpath("out")

    def fake_download(path: str, template: str, branch: str | None = None) -> Path:
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


def test_init_command_latest_template_rewrites(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """LATEST templates get their dep rewritten and uv.lock generated."""
    target = tmp_path.joinpath("out")

    def fake_download(path: str, template: str, branch: str | None = None) -> Path:
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


def test_resolve_xorq_spec_vcs_non_git() -> None:
    """`direct_url.json`'s vcs_info may declare hg/svn; honor it."""
    direct_url = {
        "url": "https://example.org/repo",
        "vcs_info": {"vcs": "hg", "commit_id": "deadbeef"},
    }
    with _patch_direct_url(direct_url):
        spec = resolve_xorq_spec()
        assert spec == "xorq @ hg+https://example.org/repo@deadbeef"


def test_resolve_xorq_spec_url_missing_raises() -> None:
    """A malformed direct_url.json without `url` must error, not yield None."""
    direct_url = {"vcs_info": {"vcs": "git", "commit_id": "abc"}}
    with _patch_direct_url(direct_url):
        with pytest.raises(InitTemplateError, match="no `url` field"):
            resolve_xorq_spec()


def test_find_latest_dep_returns_extras(latest_template: Path) -> None:
    has_placeholder, extras = find_latest_dep(latest_template)
    assert has_placeholder is True
    assert extras == "[duckdb]"


def test_find_latest_dep_no_pyproject(tmp_path: Path) -> None:
    assert find_latest_dep(tmp_path) == (False, "")


def test_rewrite_sets_hatch_allow_direct_references(latest_template: Path) -> None:
    """Substituting an `@ URL` spec must opt into hatchling direct refs."""
    rewrite_template_xorq_dep(latest_template, "xorq[duckdb] @ file:///path")
    data = tomlkit.loads(latest_template.joinpath("pyproject.toml").read_text())
    assert data["tool"]["hatch"]["metadata"]["allow-direct-references"] is True


def test_rewrite_skips_hatch_flag_for_version_pin(latest_template: Path) -> None:
    """PyPI `==` pins are not direct references; don't set the flag."""
    rewrite_template_xorq_dep(latest_template, "xorq[duckdb] == 0.3.25")
    data = tomlkit.loads(latest_template.joinpath("pyproject.toml").read_text())
    hatch = data.get("tool", {}).get("hatch", {})
    assert "metadata" not in hatch


def test_run_uv_lock_failure_raises(tmp_path: Path) -> None:
    """`uv lock` non-zero exit must raise InitTemplateError with stderr surfaced."""
    fake = type(
        "R",
        (),
        {"returncode": 1, "stderr": "No solution found: xorq @ borked"},
    )()
    with patch("subprocess.run", return_value=fake):
        with pytest.raises(InitTemplateError, match="No solution found"):
            run_uv_lock(tmp_path)


def test_init_command_wraps_init_error_as_click(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A failed `uv lock` surfaces as a clean ClickException, not a traceback."""
    target = tmp_path.joinpath("out")

    def fake_download(path: str, template: str, branch: str | None = None) -> Path:
        target.mkdir()
        target.joinpath("pyproject.toml").write_text(
            '[project]\nname = "x"\nversion = "0.0.1"\n'
            'dependencies = ["xorq @ LATEST"]\n'
        )
        return target

    fake_proc = type("R", (), {"returncode": 1, "stderr": "broken"})()
    with (
        patch(
            "xorq.common.utils.download_utils.download_unpacked_xorq_template",
            side_effect=fake_download,
        ),
        patch("subprocess.run", return_value=fake_proc),
    ):
        with pytest.raises(click.ClickException, match="broken"):
            init_command(
                path=str(target),
                template="cached-fetcher",
                xorq_spec="xorq @ git+https://example.com/x@abc",
            )
