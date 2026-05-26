from __future__ import annotations

import json
import re
import subprocess
import warnings
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


# `xorq.init_templates` is imported by `xorq.cli` on every CLI invocation,
# including fast-path commands. The stdlib imports above are essentially free
# (already loaded transitively via xorq.common); the heavy `tomlkit` import
# (~15ms cold) is deferred inside the functions that need it.

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from xorq.common.exceptions import XorqError
from xorq.common.utils import classproperty


default_branch = "main"
_LATEST_RE = re.compile(r"^xorq(?P<extras>\[[^\]]+\])?\s*@\s*LATEST$")
_SPEC_EXTRAS_RE = re.compile(r"^\s*xorq(?P<extras>\[[^\]]+\])?")


def _extract_spec_extras(spec: str) -> str:
    """Return the ``[…]`` block from a ``xorq[…] …`` spec, or ``""``."""
    m = _SPEC_EXTRAS_RE.match(spec)
    return (m.group("extras") or "") if m else ""


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

    The optional `extras` (e.g. ``"[duckdb]"``) is preserved on the
    substituted spec. ``override``, if given, is returned verbatim — the user
    opts in.
    """
    if override is not None:
        template_extras = extras or ""
        override_extras = _extract_spec_extras(override)
        if template_extras and override_extras != template_extras:
            warnings.warn(
                f"--xorq-spec extras {override_extras or '[]'} do not match the "
                f"template's xorq extras {template_extras}; proceeding with the "
                "override as-is. The generated project may be missing extras the "
                "template expects.",
                stacklevel=2,
            )
        return override

    direct_url, version = _read_direct_url(dist_name)
    extras_str = extras or ""

    if direct_url is None:
        return f"xorq{extras_str} == {version}"

    url = direct_url.get("url")
    if url is None:
        raise InitTemplateError(
            f"direct_url.json for `{dist_name}` has no `url` field:\n"
            f"{json.dumps(direct_url, indent=2)}\n"
            "Pass --xorq-spec explicitly."
        )

    if "vcs_info" in direct_url:
        vcs = direct_url["vcs_info"]
        vcs_type = vcs.get("vcs", "git")
        commit = vcs.get("commit_id", "")
        at = f"@{commit}" if commit else ""
        return f"xorq{extras_str} @ {vcs_type}+{url}{at}"
    if "archive_info" in direct_url:
        return f"xorq{extras_str} @ {url}"
    if "dir_info" in direct_url:
        return f"xorq{extras_str} @ {url}"

    raise InitTemplateError(
        f"unrecognized direct_url.json shape for `{dist_name}`:\n"
        f"{json.dumps(direct_url, indent=2)}\n"
        "Pass --xorq-spec explicitly."
    )


def find_latest_dep(template_dir: Path | str) -> tuple[bool, str]:
    """Return ``(has_placeholder, extras_str)`` for the template's pyproject.

    ``extras_str`` is the literal ``[…]`` block (or ``""``) when present.
    Reads the pyproject exactly once.
    """
    import tomlkit  # noqa: PLC0415

    pyproject = Path(template_dir).joinpath("pyproject.toml")
    if not pyproject.exists():
        return False, ""
    data = tomlkit.loads(pyproject.read_text())
    deps = data.get("project", {}).get("dependencies", [])
    for entry in deps:
        m = _LATEST_RE.match(str(entry).strip())
        if m:
            return True, m.group("extras") or ""
    return False, ""


def has_latest_placeholder(template_dir: Path | str) -> bool:
    """True iff the template's pyproject.toml declares xorq[…] @ LATEST."""
    return find_latest_dep(template_dir)[0]


def rewrite_template_xorq_dep(template_dir: Path | str, spec: str) -> None:
    """Substitute ``xorq[extras] @ LATEST`` with ``spec`` in the template.

    Also:

    - strips ``[tool.uv.sources].xorq`` (legacy git source);
    - sets ``[tool.hatch.metadata].allow-direct-references = true`` when the
      substituted spec uses a direct-reference (``@ URL``) form, so hatchling
      will accept it during ``uv build --wheel``;
    - deletes any shipped ``uv.lock`` and ``requirements.txt``.

    Errors loudly if no matching dependency entry is found.
    """
    import tomlkit  # noqa: PLC0415

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

    # Hatchling forbids direct references in [project.dependencies] unless
    # explicitly opted in. The substituted spec is a direct reference whenever
    # it contains ` @ ` (PyPI `==` pins don't trigger this).
    if " @ " in spec:
        tool = data.setdefault("tool", tomlkit.table())
        hatch_metadata = tool.setdefault("hatch", tomlkit.table()).setdefault(
            "metadata", tomlkit.table()
        )
        hatch_metadata["allow-direct-references"] = True

    pyproject.write_text(tomlkit.dumps(data))

    for stale in ("uv.lock", "requirements.txt"):
        stale_path = template_dir.joinpath(stale)
        if stale_path.exists():
            stale_path.unlink()


def run_uv_lock(template_dir: Path | str):
    """Run ``uv lock`` in the template dir, streaming stdout to the user.

    Stderr is captured and surfaced via ``InitTemplateError`` on non-zero
    exit. The half-done substituted ``pyproject.toml`` is intentionally left
    in place so the failure state is inspectable.
    """
    template_dir = Path(template_dir)
    result = subprocess.run(
        ["uv", "lock"],
        cwd=str(template_dir),
        check=False,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise InitTemplateError(
            "`uv lock` failed in the substituted template "
            f"(cwd={template_dir}):\n{result.stderr}"
        )
    return result
