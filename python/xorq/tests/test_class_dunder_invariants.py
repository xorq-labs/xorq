"""Invariant sweep: no class may shadow a pickle-critical dunder (#2235).

#2233 was caused by ``MetricComputation`` defining ``__module__`` as a
``@property``: ``type.__module__``'s getter returns
``cls.__dict__["__module__"]`` verbatim, so the property object leaked out at
class level, cloudpickle fell back to pickling the class by value, and
reconstruction died in ``_class_setstate`` on
``setattr(cls, "__name__", <property>)`` -- every build embedding a metric
became unloadable while ``build`` reported success.

This test walks every importable module in the xorq package (vendor included)
and asserts the dunders cloudpickle's by-reference path relies on are the
plain strings Python and cloudpickle assume:

* the class-level *value* of each dunder is a ``str``, and
* the class's own ``__dict__`` entry for each dunder (when present) is a
  ``str``, not a property or any other object.  The second check is the one
  with teeth for ``__name__``/``__qualname__``: those are data descriptors on
  ``type`` itself, so a shadowing class-dict entry is masked at class level
  and a value check alone never sees it.

Both count assertions at the end are load-bearing (see #2181, where a CI
guard was a silent no-op): a bare try/except-continue over imports is how
this test silently stops checking anything, so every skip is counted and
bounded, and the checked-class count has a floor.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

import pytest

import xorq


PICKLE_CRITICAL_DUNDERS = ("__module__", "__name__", "__qualname__")

# Floor on classes swept: the walk found 1436 classes with all extras
# installed (#2235) and 1160 in a plain `uv sync --group test` env; if the
# count ever collapses below this, the walk itself broke and the test must
# fail rather than silently pass on a handful of classes.
MIN_CLASSES_CHECKED = 1000

# Ceiling on modules that failed to import: measured 19 with all extras
# installed (#2235) and 47 in a plain `uv sync --group test` env (optional
# backends: psycopg, datafusion, snowflake, bigquery, ...).  Small headroom
# on top of the leanest measured env; a walk collapse (hundreds of skips)
# must fail, not pass.
MAX_IMPORT_SKIPS = 60

# (module, qualname, dunder) entries that deliberately shadow a dunder.
#
# MetricComputation.__name__ is a property out of necessity: agg.pandas_df
# reads fn.__name__ unconditionally via _make_udf_name(fn.__name__) to name
# the generated node type (see the property's docstring in
# python/xorq/expr/ml/metrics.py).  It is safe only *given by-reference
# pickling*, which keeping __module__ a plain str preserves.  Tracked as
# #2238 -- when #2238 removes the property, this entry must be deleted (the
# test enforces that: an allowlist entry whose module imported but whose
# shadow no longer exists fails the sweep).
KNOWN_DUNDER_SHADOWERS = frozenset(
    {
        ("xorq.expr.ml.metrics", "MetricComputation", "__name__"),
    }
)


def _import_or_none(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except (Exception, pytest.skip.Exception):
        # pytest.skip.Exception derives from BaseException, not Exception: a
        # module whose import path hits pytest.importorskip (e.g. backend
        # tests' conftest) would otherwise mark this whole test as skipped --
        # exactly the silent-no-op failure mode this test exists to prevent.
        return None


def test_no_class_shadows_a_pickle_critical_dunder() -> None:
    checked = 0
    skipped: set[str] = set()
    violations: list[str] = []
    found_shadowers: set[tuple[str, str, str]] = set()
    imported_modules: set[str] = set()
    seen: set[type] = set()

    def record_walk_error(name: str) -> None:
        # walk_packages swallows ImportError from subpackage __init__ by
        # default, silently dropping the whole subtree; count it as a skip.
        skipped.add(name)

    for info in pkgutil.walk_packages(
        xorq.__path__, prefix="xorq.", onerror=record_walk_error
    ):
        module = _import_or_none(info.name)
        if module is None:
            skipped.add(info.name)
            continue
        imported_modules.add(info.name)
        for obj in vars(module).values():
            if not isinstance(obj, type) or obj in seen:
                continue
            module_value = getattr(obj, "__module__", None)
            if isinstance(module_value, str) and module_value != info.name:
                # Re-export from another module; checked where it is defined.
                # Only a *str* mismatch may be excluded: a class shadowing
                # __module__ (the #2233 shape) leaks a non-str here, and
                # filtering on equality alone would hide the very bug this
                # test exists to catch.
                continue
            seen.add(obj)
            checked += 1
            for dunder in PICKLE_CRITICAL_DUNDERS:
                value = getattr(obj, dunder, None)
                if not isinstance(value, str):
                    violations.append(
                        f"{info.name}.{obj.__qualname__}: {dunder} is {value!r}, not str"
                    )
                entry = obj.__dict__.get(dunder)
                if entry is None or isinstance(entry, str):
                    continue
                key = (info.name, obj.__qualname__, dunder)
                found_shadowers.add(key)
                if key not in KNOWN_DUNDER_SHADOWERS:
                    violations.append(
                        f"{info.name}.{obj.__qualname__}: __dict__[{dunder!r}] is "
                        f"{type(entry).__name__}, not str -- shadowing a "
                        "pickle-critical dunder breaks cloudpickle (#2233)"
                    )

    stale_allowlist = sorted(
        key
        for key in KNOWN_DUNDER_SHADOWERS - found_shadowers
        if key[0] in imported_modules
    )

    assert not violations, "\n".join(violations)
    assert not stale_allowlist, (
        f"allowlist entries no longer shadow anything, delete them: {stale_allowlist}"
    )
    assert checked > MIN_CLASSES_CHECKED, (
        f"walk collapsed to {checked} classes (floor {MIN_CLASSES_CHECKED})"
    )
    assert len(skipped) <= MAX_IMPORT_SKIPS, (
        f"{len(skipped)} modules failed to import (max {MAX_IMPORT_SKIPS}): "
        f"{sorted(skipped)}"
    )
