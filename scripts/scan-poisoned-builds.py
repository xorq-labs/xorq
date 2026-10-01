#!/usr/bin/env python3
"""Scan xorq build artifacts for #2233-poisoned pickle payloads.

Builds written before the #2233 fix embed ``MetricComputation`` *by value*
in their pickled payloads and are unloadable, but their build hash is
identical to a healthy rebuild. This scanner tells them apart from the raw
bytes alone.

Detection signals, per build directory:

1. pickle-by-value: a base64 pickle payload in expr.yaml contains the
   cloudpickle by-value marker (``_make_skeleton_class``) together with a
   ``xorq.``-prefixed module string. A xorq-library class captured by value
   is always wrong; the payload cannot be reconstructed by a fixed library.
2. yaml-module: a metric UDF node (``__func_name__`` containing
   ``metric_``) whose ``__module__`` is not ``xorq.expr.ml.metrics``
   (pre-fix builds record the metric fn's module, e.g.
   ``sklearn.metrics._regression``).

SAFETY: this script never unpickles and never imports artifact code. Pickle
payloads are only byte-scanned and opcode-walked with ``pickletools.genops``
(a pure parser -- no object construction). #2155 documents SIGSEGVs on
load, so "just try to load it" is not a safe scanner.

stdlib-only on purpose: runnable anywhere, xorq need not be installed.

Usage:
    python scan-poisoned-builds.py DIR [DIR ...]

Each DIR may be a single build directory (contains expr.yaml) or any tree
of them (a builds/ dir, a catalog checkout, ...). Exit status is 1 if any
poisoned build is found, 2 if no build directories were found, else 0.
"""

from __future__ import annotations

import base64
import binascii
import pickletools
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator


SKELETON_MARKER = b"_make_skeleton_class"
XORQ_MODULE_PREFIX = "xorq."
HEALTHY_METRIC_MODULE = "xorq.expr.ml.metrics"
B64_TOKEN = re.compile(rb"[A-Za-z0-9+/]{100,}={0,2}")
PICKLE_MAGIC = b"\x80"
MODULE_LINE = re.compile(r"^(\s*)__module__:\s*(\S+)\s*$")
FUNC_NAME_LINE = re.compile(r"^(\s*)__func_name__:\s*(\S+)\s*$")


def echo(line: str) -> None:
    sys.stdout.write(line + "\n")


def iter_build_dirs(root: Path) -> Iterator[Path]:
    """Yield every directory under root (inclusive) containing expr.yaml."""
    if (root / "expr.yaml").is_file():
        yield root
        return
    if root.is_dir():
        for expr_yaml in sorted(root.rglob("expr.yaml")):
            yield expr_yaml.parent


def iter_pickle_payloads(data: bytes) -> Iterator[bytes]:
    """Yield decoded pickle payloads from base64 tokens in raw file bytes."""
    for match in B64_TOKEN.finditer(data):
        try:
            raw = base64.b64decode(match.group(0), validate=True)
        except (binascii.Error, ValueError):
            continue
        if raw.startswith(PICKLE_MAGIC):
            yield raw


def skeleton_class_names(raw: bytes) -> list[str]:
    """Names of by-value (skeleton) classes whose payload also carries a
    xorq. module string. Opcode walk only -- nothing is constructed."""
    strings: list[str] = []
    try:
        for _, arg, _ in pickletools.genops(raw):
            if isinstance(arg, str):
                strings.append(arg)
    except Exception:  # truncated/corrupt payload: fall back to byte scan
        if SKELETON_MARKER in raw and XORQ_MODULE_PREFIX.encode() in raw:
            return ["<unparsable payload>"]
        return []
    if not any(s.startswith(XORQ_MODULE_PREFIX) for s in strings):
        return []
    names = []
    skip = {"builtins", "type", "cloudpickle", "cloudpickle.cloudpickle"}
    for i, s in enumerate(strings):
        if s == "_make_skeleton_class":
            for candidate in strings[i + 1 : i + 5]:
                if candidate not in skip:
                    names.append(candidate)
                    break
    return names


def yaml_metric_module_mismatches(text: str) -> list[str]:
    """Modules recorded on metric UDF nodes that are not the healthy one."""
    lines = text.splitlines()
    mismatches = []
    for i, line in enumerate(lines):
        fn_match = FUNC_NAME_LINE.match(line)
        if fn_match is None or "metric_" not in fn_match.group(2):
            continue
        indent = fn_match.group(1)
        lo, hi = max(0, i - 10), min(len(lines), i + 10)
        for other in lines[lo:hi]:
            mod_match = MODULE_LINE.match(other)
            if mod_match and mod_match.group(1) == indent:
                if mod_match.group(2) != HEALTHY_METRIC_MODULE:
                    mismatches.append(mod_match.group(2))
    return mismatches


def scan_build(build_dir: Path) -> tuple[str, str]:
    """Return (verdict, detail) for one build directory."""
    data = (build_dir / "expr.yaml").read_bytes()
    signals = []
    payload_count = 0
    for raw in iter_pickle_payloads(data):
        payload_count += 1
        names = skeleton_class_names(raw)
        if SKELETON_MARKER in raw and names:
            signals.append(
                f"pickle-by-value({', '.join(sorted(set(names)))}, {len(raw)} B)"
            )
    for module in yaml_metric_module_mismatches(data.decode(errors="replace")):
        signals.append(f"yaml-module({module})")
    if signals:
        return "POISONED", "signals=[" + "; ".join(signals) + "]"
    if payload_count == 0:
        return "NOT-APPLICABLE", "no pickled payloads"
    return "CLEAN", f"{payload_count} pickled payload(s), no by-value xorq class"


def main(argv: Iterable[str]) -> int:
    roots = [Path(a) for a in argv]
    if not roots:
        echo(__doc__.strip().splitlines()[0])
        echo("usage: scan-poisoned-builds.py DIR [DIR ...]")
        return 2
    found_builds = False
    found_poisoned = False
    for root in roots:
        for build_dir in iter_build_dirs(root):
            found_builds = True
            verdict, detail = scan_build(build_dir)
            found_poisoned |= verdict == "POISONED"
            echo(f"{verdict:<15} {build_dir}  [{detail}]")
    if not found_builds:
        echo("no build directories (containing expr.yaml) found")
        return 2
    return 1 if found_poisoned else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
