"""Static contract check: every ``xorq ...`` command referenced in the docs
must resolve against the real click command tree.

This is the doc-side counterpart to the ``*.snippets.py`` smoke test. The
smoke test proves the tutorial *runs*; this proves the tutorial *talks about
commands that exist* — so a renamed flag or removed subcommand breaks the docs
in CI instead of silently rotting. It does not execute anything, so it is
immune to the placeholders (``<you>``, ``$BUILD_A``, ``~/work``), GitHub steps,
and tabset duplicates that make the prose non-executable.

Pure functions, no Sybil/pytest import — ``docs/conftest.py`` wires these into
Sybil, and they're directly callable for unit testing.
"""

from __future__ import annotations

import re
import shlex


# A command-position token that doesn't look like this is treated as a
# placeholder/metavar (e.g. ``<command>``, ``COMMAND``, ``$VAR``) and stops
# validation of that invocation rather than being flagged as a typo.
_COMMAND_NAME = re.compile(r"^[a-z][a-z0-9-]+$")

# Options click adds implicitly or that are universally safe to reference.
_ALWAYS_VALID_OPTS = frozenset({"--help", "-h", "--version"})

# A simple command runs until one of these shell operators.
_INVOCATION = re.compile(r"(?:(?<=\s)|(?<=\()|^)xorq\s+([^|&;)\n]+)")

_root = None


def _logical_lines(block_text):
    """Yield logical lines, joining shell ``\\`` continuations into one. Without
    this, options on a continuation line (which doesn't start with ``xorq``) are
    silently skipped — see ``serve-unbound.qmd``'s multi-line invocation.
    """
    buf = ""
    for raw_line in block_text.splitlines():
        stripped = raw_line.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
        else:
            yield buf + raw_line
            buf = ""
    if buf:
        yield buf


def iter_xorq_invocations(block_text):
    """Yield the token list (after ``xorq``) for each ``xorq`` invocation in a
    bash code block. Strips a leading ``uv run`` and digs inside ``$(...)``;
    non-``xorq`` lines (git, gh, uv add, mkdir, ...) and comments are skipped.
    Shell ``\\`` line continuations are joined first so options on the second
    line are validated too.
    """
    invocations = []
    for logical_line in _logical_lines(block_text):
        line = logical_line.strip()
        if not line or line.startswith("#"):
            continue
        for match in _INVOCATION.finditer(line):
            tail = match.group(1).strip()
            if not tail:
                continue
            try:
                tokens = shlex.split(tail, comments=True)
            except ValueError:
                tokens = tail.split()
            if tokens:
                invocations.append(tokens)
    return invocations


def _get_root():
    global _root
    if _root is None:
        from xorq.cli import cli  # noqa: PLC0415

        # `cli` is a PdbGroup that lazily registers the `catalog` subgroup on
        # first lookup. Going through the public get_command avoids depending on
        # the private `_load_catalog_cli` helper.
        cli.get_command(None, "catalog")
        _root = cli
    return _root


def _option_names(node):
    names = set(_ALWAYS_VALID_OPTS)
    for param in node.params:
        for opt in list(param.opts) + list(param.secondary_opts):
            if opt.startswith("-"):
                names.add(opt)
    return names


def _takes_value(node, opt):
    for param in node.params:
        if opt in param.opts or opt in param.secondary_opts:
            return not getattr(param, "is_flag", False) and not getattr(
                param, "count", False
            )
    return False


def validate_invocation(tokens):
    """Walk the click tree for one ``xorq`` invocation; raise AssertionError on
    an unknown subcommand or option. Placeholders and positional arguments are
    ignored.
    """
    import click  # noqa: PLC0415

    node = _get_root()
    path = ["xorq"]
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "--":
            i += 1
            continue
        if tok.startswith("-"):
            name = tok.split("=", 1)[0]
            valid = _option_names(node)
            assert name in valid, (
                f"`{' '.join(path)}` has no option {name!r} "
                f"(valid: {', '.join(sorted(valid))})"
            )
            if "=" not in tok and _takes_value(node, name):
                i += 1  # skip the option's value token
            i += 1
            continue
        if isinstance(node, click.Group):
            sub = node.commands.get(tok)
            if sub is not None:
                node = sub
                path.append(tok)
                i += 1
                continue
            if _COMMAND_NAME.match(tok):
                raise AssertionError(
                    f"`{' '.join(path)}` has no subcommand {tok!r} "
                    f"(valid: {', '.join(sorted(node.commands))})"
                )
            # Placeholder / metavar (e.g. <command>, COMMAND, $VAR): can't
            # validate further — stop here rather than false-flag it.
            return
        # Leaf command: remaining non-option tokens are positional args.
        i += 1
