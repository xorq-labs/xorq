"""Refuse to publish a release tag that is not the tested release commit.

Guard: the `verify` job of `ci-release.yml` runs it after `uv build` and before
the `publish` job, which alone holds the PyPI credentials. It fails unless

- the tag is `vX` and names the checked-out commit,
- that commit is on `main` and is the squash merge of a `release: X (#N)` PR,
- `pyproject.toml` at that commit says version X, and
- the files in the dist directory are byte-identical to the ones TestPyPI holds
  for X, which ci-pre-release uploaded from the release branch.

The last check is what ties PyPI to the build that was tested: the build is
reproducible from a clean checkout, so any difference in the tree changes a
digest. Run it from a clean checkout; untracked files can change the sdist.

Stdlib only, so CI runs it with the runner's bare `python3` (3.11+ for tomllib).

    uv build && python3 scripts/release_verify.py --tag v0.4.8 --dist dist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import tomllib


TESTPYPI_JSON = "https://test.pypi.org/pypi/xorq/{version}/json"


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), capture_output=True, text=True, check=True
    ).stdout.strip()


def is_ancestor(commit: str, ref: str) -> bool:
    return (
        subprocess.run(("git", "merge-base", "--is-ancestor", commit, ref)).returncode
        == 0
    )


def local_digests(dist: Path) -> dict[str, str]:
    # uv build also writes a .gitignore into the output directory
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dist.iterdir())
        if path.name.endswith((".whl", ".tar.gz"))
    }


def index_digests(version: str) -> dict[str, str] | None:
    """The files TestPyPI holds for `version`, or None when it holds none."""
    try:
        with urllib.request.urlopen(TESTPYPI_JSON.format(version=version)) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    return {url["filename"]: url["digests"]["sha256"] for url in data["urls"]}


def check(
    tag: str,
    dist: Path,
    main_ref: str,
    fetch_index: Callable[[str], dict[str, str] | None] = index_digests,
) -> list[str]:
    """Every reason `tag` must not be published; empty when it may be."""
    match = re.fullmatch(r"v(\d+\.\d+\.\d+)", tag)
    if match is None:
        return [f"tag {tag!r} is not vMAJOR.MINOR.PATCH"]
    version = match.group(1)
    errors = []

    head = git("rev-parse", "HEAD")
    tagged = git("rev-parse", f"{tag}^{{commit}}")
    if tagged != head:
        errors.append(f"{tag} is {tagged}, but the checkout is {head}")
    if not is_ancestor(head, main_ref):
        errors.append(f"{head} is not on {main_ref}")
    subject = git("log", "-1", "--format=%s", head)
    if re.fullmatch(rf"release: {re.escape(version)} \(#\d+\)", subject) is None:
        errors.append(
            f"{head} is {subject!r}, not the squash merge of a 'release: {version}' PR"
        )

    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    if project["version"] != version:
        errors.append(
            f"pyproject.toml says {project['version']}, the tag says {version}"
        )

    built = local_digests(dist)
    if not built:
        errors.append(f"{dist} holds no files to compare")
    tested = fetch_index(version)
    if tested is None:
        errors.append(f"TestPyPI has no xorq {version}; run ci-pre-release first")
    elif built != tested:
        for name in sorted(built.keys() | tested.keys()):
            ours, theirs = built.get(name, "missing"), tested.get(name, "missing")
            if ours != theirs:
                errors.append(f"{name}: built {ours}, TestPyPI {theirs}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--main-ref", default="origin/main")
    args = parser.parse_args(argv)

    errors = check(args.tag, args.dist, args.main_ref)
    for error in errors:
        sys.stderr.write(f"release_verify: error: {error}\n")
    if not errors:
        sys.stdout.write(f"release_verify: {args.tag} is the tested release commit\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
