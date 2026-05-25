from __future__ import annotations

import json
import re
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import tomlkit

from xorq.common.exceptions import XorqError
from xorq.common.utils import classproperty
from xorq.common.utils.logging_utils import get_logger


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


logger = get_logger(__name__)


default_branch = "main"
LATEST_PLACEHOLDER = "LATEST"
_LATEST_RE = re.compile(r"^xorq(?P<extras>\[[^\]]+\])?\s*@\s*LATEST$")


class InitTemplateError(XorqError):
    """Raised when xorq init cannot prepare a template."""


class InitTemplates(StrEnum):
    cached_fetcher = "cached-fetcher"
    sklearn = "sklearn"
    penguins = "penguins"

    @classproperty
    def default(self):
        return self.cached_fetcher

    def get_default_branch(template, default_branch=default_branch):
        return dict(templates_branches).get(template, default_branch)


# NOTE: These are commit hashes from when the template update occurred.
# Currently pointing at the open draft PRs that switch each template to the
# `xorq @ LATEST` placeholder scheme (see docs/plans/template_system_redesign.md).
# Rotate to the merged-main SHA once those PRs land.
templates_branches = (
    (InitTemplates.cached_fetcher, "963723a6fdcf66a034e1b9631cd0604a7ee6fdd2"),
    (InitTemplates.sklearn, "c4bb98075c80c700d829c7986cb1876bd617c641"),
    (InitTemplates.penguins, "f87520e7a8d9cf541f3d74c8947bb8954fb3142d"),
)


def _read_direct_url(dist_name: str = "xorq") -> tuple[dict | None, str]:
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError as exc:
        raise InitTemplateError(
            f"package `{dist_name}` is not installed; cannot resolve a spec for `xorq init`. "
            "Pass --xorq-spec explicitly."
        ) from exc
    raw = dist.read_text("direct_url.json")
    version = dist.version
    if raw is None:
        return None, version
    return json.loads(raw), version


def resolve_xorq_spec(
    override: str | None = None,
    extras: str | None = None,
    dist_name: str = "xorq",
) -> str:
    """Return a PEP 508 spec that pins xorq to the currently-running install.

    The optional `extras` (e.g. `"[duckdb]"`) is preserved on the substituted
    spec. `override`, if given, is returned verbatim — the user opts in.
    """
    if override is not None:
        return override

    direct_url, version = _read_direct_url(dist_name)
    extras_str = extras or ""

    if direct_url is None:
        # PyPI release path: no direct_url.json
        return f"xorq{extras_str} == {version}"

    url = direct_url.get("url")
    if "vcs_info" in direct_url:
        vcs = direct_url["vcs_info"]
        commit = vcs.get("commit_id", "")
        at = f"@{commit}" if commit else ""
        return f"xorq{extras_str} @ git+{url}{at}"
    if "archive_info" in direct_url:
        return f"xorq{extras_str} @ {url}"
    if "dir_info" in direct_url:
        # editable or local-dir install; uv accepts the file:// URL form
        return f"xorq{extras_str} @ {url}"

    raise InitTemplateError(
        f"unrecognized direct_url.json shape for `{dist_name}`:\n"
        f"{json.dumps(direct_url, indent=2)}\n"
        "Pass --xorq-spec explicitly."
    )


def has_latest_placeholder(template_dir: Path | str) -> bool:
    """True iff the template's pyproject.toml declares xorq[…] @ LATEST."""
    pyproject = Path(template_dir).joinpath("pyproject.toml")
    if not pyproject.exists():
        return False
    data = tomlkit.loads(pyproject.read_text())
    deps = data.get("project", {}).get("dependencies", [])
    return any(_LATEST_RE.match(str(d).strip()) for d in deps)


def rewrite_template_xorq_dep(template_dir: Path | str, spec: str) -> None:
    """Substitute `xorq[extras] @ LATEST` with `spec` in the template.

    Also strips `[tool.uv.sources].xorq` (legacy git source) and deletes any
    shipped `uv.lock` and `requirements.txt`. Errors loudly if no matching
    dependency entry is found.
    """
    template_dir = Path(template_dir)
    pyproject = template_dir.joinpath("pyproject.toml")
    if not pyproject.exists():
        raise InitTemplateError(
            f"template at `{template_dir}` is missing pyproject.toml"
        )

    data = tomlkit.loads(pyproject.read_text())
    project = data.get("project")
    if project is None or "dependencies" not in project:
        raise InitTemplateError(
            f"template at `{template_dir}` has no [project].dependencies"
        )

    deps = project["dependencies"]
    matched = False
    for idx, entry in enumerate(deps):
        if _LATEST_RE.match(str(entry).strip()):
            deps[idx] = spec
            matched = True
            break
    if not matched:
        raise InitTemplateError(
            f"template at `{template_dir}` does not declare `xorq[…] @ LATEST`; "
            "nothing to rewrite."
        )

    # Strip the legacy [tool.uv.sources].xorq entry, if present.
    tool = data.get("tool")
    if tool is not None:
        sources = tool.get("uv", {}).get("sources")
        if sources is not None and "xorq" in sources:
            del sources["xorq"]

    pyproject.write_text(tomlkit.dumps(data))

    for stale in ("uv.lock", "requirements.txt"):
        stale_path = template_dir.joinpath(stale)
        if stale_path.exists():
            stale_path.unlink()


def run_uv_lock(template_dir: Path | str) -> subprocess.CompletedProcess:
    """Run `uv lock` in the template dir. Raises on failure with stderr surfaced."""
    template_dir = Path(template_dir)
    result = subprocess.run(
        ["uv", "lock"],
        cwd=str(template_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Per design: leave the substituted pyproject.toml in place so the
        # half-done state is inspectable.
        raise InitTemplateError(
            "`uv lock` failed in the substituted template "
            f"(cwd={template_dir}):\n{result.stderr}"
        )
    return result
