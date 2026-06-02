"""Sybil wiring for the docs CLI-contract check (see cli_contract.py).

The sybil import is guarded so environments without sybil installed (e.g. the
core test matrix) simply collect nothing from ``docs/`` instead of erroring at
import time.
"""

import os
import sys


sys.path.insert(0, os.path.dirname(__file__))  # make cli_contract importable

try:
    from cli_contract import iter_xorq_invocations, validate_invocation
    from sybil import Sybil
    from sybil.parsers.markdown import CodeBlockParser
except ImportError:
    pass
else:

    def _evaluate(example):
        for tokens in iter_xorq_invocations(example.parsed):
            validate_invocation(tokens)

    pytest_collect_file = Sybil(
        parsers=[CodeBlockParser(language="bash", evaluator=_evaluate)],
        patterns=["*.qmd"],
    ).pytest()
