"""Unit tests for the docs CLI-contract checker (see cli_contract.py).

These exercise the pure parsing/validation functions directly, covering the
edge cases the Sybil ``.qmd`` sweep does not pin down on its own: line
continuations, ``=``-attached option values, the ``--`` separator, placeholder
metavars, and the unknown-subcommand / unknown-option failure paths.
"""

from __future__ import annotations

import os
import sys

import pytest


sys.path.insert(0, os.path.dirname(__file__))  # make cli_contract importable

from cli_contract import iter_xorq_invocations, validate_invocation  # noqa: E402


pytestmark = pytest.mark.library


class TestIterXorqInvocations:
    def test_plain(self):
        assert iter_xorq_invocations("xorq catalog init") == [["catalog", "init"]]

    def test_strips_uv_run(self):
        # `uv run` is shell, not a xorq token — the regex matches at `xorq`.
        assert iter_xorq_invocations("uv run xorq catalog list-aliases") == [
            ["catalog", "list-aliases"]
        ]

    def test_skips_comments_and_blanks(self):
        text = "# a comment\n\nxorq build foo.py\n"
        assert iter_xorq_invocations(text) == [["build", "foo.py"]]

    def test_skips_non_xorq_lines(self):
        text = "git push -u origin main\nmkdir -p ~/work\nxorq catalog push"
        assert iter_xorq_invocations(text) == [["catalog", "push"]]

    def test_digs_inside_command_substitution(self):
        text = "BUILD_A=$(uv run xorq uv build flights_model.py -e expr | tail -1)"
        assert iter_xorq_invocations(text) == [
            ["uv", "build", "flights_model.py", "-e", "expr"]
        ]

    def test_joins_line_continuation(self):
        # The Medium finding: options on the continuation line must be seen.
        text = (
            "xorq serve-unbound builds/7061dd65ff3c --host 0.0.0.0 --port 8001 \\\n"
            "  --cache-dir cache --to-unbind-hash b2370a29c19df8e1e639c63252dacd0e"
        )
        assert iter_xorq_invocations(text) == [
            [
                "serve-unbound",
                "builds/7061dd65ff3c",
                "--host",
                "0.0.0.0",
                "--port",
                "8001",
                "--cache-dir",
                "cache",
                "--to-unbind-hash",
                "b2370a29c19df8e1e639c63252dacd0e",
            ]
        ]


class TestValidateInvocation:
    def test_known_subcommand_ok(self):
        validate_invocation(["catalog", "init"])

    def test_unknown_subcommand_raises(self):
        with pytest.raises(AssertionError, match="no subcommand 'frobnicate'"):
            validate_invocation(["frobnicate"])

    def test_unknown_option_raises(self):
        with pytest.raises(AssertionError, match="no option"):
            validate_invocation(["catalog", "--definitely-not-a-flag"])

    def test_equals_attached_option_value(self):
        # `--name=foo` must split on `=` so the option resolves and the value
        # isn't treated as a stray token.
        validate_invocation(["catalog", "--name=foo", "init"])

    def test_double_dash_separator_ignored(self):
        validate_invocation(["catalog", "--", "init"])

    def test_placeholder_metavar_stops_validation(self):
        # A non-command-looking token (`<command>`, `COMMAND`, `$VAR`) is a
        # metavar; validation stops rather than false-flagging it.
        validate_invocation(["<command>"])
        validate_invocation(["$VAR"])
        validate_invocation(["COMMAND", "--whatever"])

    def test_continuation_then_validate_catches_underscore_regression(self):
        # End-to-end: the parsed continuation tokens flow into validation, so a
        # future `--to_unbind_hash` regression on line 2 would now be caught.
        (tokens,) = iter_xorq_invocations(
            "xorq serve-unbound builds/abc --host 0.0.0.0 \\\n"
            "  --to_unbind_hash deadbeef"
        )
        with pytest.raises(AssertionError, match="no option '--to_unbind_hash'"):
            validate_invocation(tokens)
