"""Rewrite named ADR references to the numeric form, once a number exists.

    python3 scripts/adr_sweep_refs.py <slug> <number>

Called by `just adr-rename`. An ADR is citable as ``ADR-<slug>`` from the
moment it is written and as ``ADR-<number>`` once its pull request exists;
both resolve forever, so this sweep is a readability pass and never a
correctness one. A reference it misses stays valid.

Stdlib only, matching scripts/adr_check.py.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


def tracked_files_containing(token: str) -> list[pathlib.Path]:
    """Files carrying the token, tracked or newly created but not ignored."""
    result = subprocess.run(
        ["git", "grep", "--untracked", "-lIF", token],
        capture_output=True,
        text=True,
    )
    # git grep exits 1 when nothing matches, which is not an error here.
    return [pathlib.Path(line) for line in result.stdout.splitlines() if line]


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(f"usage: {sys.argv[0]} <slug> <number>\n")
        return 2
    slug, number = sys.argv[1], sys.argv[2]

    # The lookahead stops a slug that prefixes a longer one from matching it:
    # sweeping `tee-node` must leave `ADR-tee-node-deferred-writes` alone.
    pattern = re.compile(rf"ADR-{re.escape(slug)}(?![a-z0-9-])")

    changed = 0
    for path in tracked_files_containing(f"ADR-{slug}"):
        text = path.read_text(encoding="utf-8")
        rewritten = pattern.sub(f"ADR-{number}", text)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed += 1

    sys.stdout.write(
        f"rewrote ADR-{slug} to ADR-{number} in {changed} file(s)\n"
        if changed
        else f"no named references to ADR-{slug} to rewrite\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
