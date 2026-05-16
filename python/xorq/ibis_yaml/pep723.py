import contextlib
import importlib.metadata
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import tomlkit
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from xorq.ibis_yaml.packager import _cap_requires_python


INLINE_METADATA_REGEX = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)


def read_inline_metadata(script: str) -> dict | None:
    """Parse PEP 723 inline metadata from a script string.

    Returns the parsed TOML as a dict, or None if no ``# /// script`` block
    is present.  Reference implementation from PEP 723.
    """
    matches = [
        m for m in INLINE_METADATA_REGEX.finditer(script) if m.group("type") == "script"
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("multiple # /// script blocks found")
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in matches[0].group("content").splitlines(keepends=True)
    )
    return tomlkit.loads(content)


def synthesize_project(
    script_path: Path, xorq_version: str | None = None
) -> TemporaryDirectory:
    """Create an ephemeral uv project from a PEP 723-annotated script.

    Reads inline metadata, writes a minimal ``pyproject.toml``, and runs
    ``uv lock``.  Returns the ``TemporaryDirectory`` — the
    caller owns its lifetime.

    Raises ``ValueError`` if the script has no PEP 723 inline metadata.
    """

    script_path = Path(script_path)
    meta = read_inline_metadata(script_path.read_text())
    if meta is None:
        raise ValueError(f"script has no PEP 723 inline metadata: {script_path}")

    dependencies = list(meta.get("dependencies", []))

    if xorq_version is None:
        xorq_version = importlib.metadata.version("xorq")
    for d in dependencies:
        try:
            Requirement(d)
        except Exception as e:
            raise ValueError(
                f"invalid dependency {d!r} in PEP 723 metadata of {script_path}"
            ) from e

    if not any(canonicalize_name(Requirement(d).name) == "xorq" for d in dependencies):
        dependencies.append(f"xorq=={xorq_version}")

    requires_python = _cap_requires_python(meta.get("requires-python"))

    tmpdir = TemporaryDirectory()
    tmp = Path(tmpdir.name)

    sanitized_stem = re.sub(r"[^a-z0-9]+", "-", script_path.stem.lower()).strip("-")
    project_name = f"xorq-script-{sanitized_stem}"
    pkg_dir = project_name.replace("-", "_")
    pyproject = {
        "project": {
            "name": project_name,
            "version": "0.0.0",
            "requires-python": requires_python,
            "dependencies": dependencies,
        },
        "build-system": {
            "requires": ["hatchling"],
            "build-backend": "hatchling.build",
        },
        "tool": {
            "hatch": {
                "build": {
                    "targets": {
                        "wheel": {
                            "packages": [pkg_dir],
                        },
                    },
                },
            },
        },
    }
    (tmp / "pyproject.toml").write_text(tomlkit.dumps(pyproject))
    (tmp / pkg_dir).mkdir()
    (tmp / pkg_dir / "__init__.py").touch()

    try:
        subprocess.run(
            ("uv", "lock", "--directory", str(tmp)),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        with contextlib.suppress(Exception):
            tmpdir.cleanup()
        raise RuntimeError(
            f"failed to resolve PEP 723 dependencies from {script_path}:\n{e.stderr}"
        ) from e

    return tmpdir
