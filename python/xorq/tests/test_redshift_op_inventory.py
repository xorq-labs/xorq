"""Every function name the Redshift compiler can reach through INHERITED code.

The Redshift compiler subclasses the PostgreSQL one, so any op it does not
override compiles to whatever PostgreSQL would emit -- 324 of the 341 ops it can
compile, when this file was written. Every "Redshift has no such function"
defect found in review sat in that inherited surface, and each was found by
sampling it. This test makes the surface a classified list instead:

* ``test_every_inherited_function_is_classified`` fails on a name nobody has
  classified (an ibis bump, or a newly reachable path) and on a stale entry.
* ``test_no_compilable_op_reaches_a_measured_absent_function`` fails when any
  op still compiles through a function measured absent. That covers every path
  to the function, not just the one a fix was made at: ``time(h, m, s)`` still
  reached ``MAKE_TIME`` through ``SIMPLE_OPS`` after the literal path was fixed.

Names are read statically from each inherited visitor's source (``self.f.X``,
``self.f["X"]``, ``self.agg.X``) and from ``SIMPLE_OPS``, BEFORE the dialect
renders them -- ``datefromparts`` is what reaches the wire as
``DATE_FROM_PARTS``. A name reached only through a helper method the visitor
calls is not seen; the rendered-SQL check in ``redshift_evidence`` covers the
wire side. A hand-written override that delegates with ``super()`` is followed
into the implementation it delegates to, because that is inherited code too:
``visit_Cast`` handles the binary casts itself and passes every other cast on
to PostgreSQL's.
"""

from __future__ import annotations

import inspect
import re
from collections import defaultdict

import pytest


# Must run before the xorq.backends.redshift import; see the matching guard in
# test_redshift_dialect.py for why both names are needed.
pytest.importorskip("adbc_driver_manager")
pytest.importorskip("psycopg")

import xorq.vendor.ibis.expr.operations as ops  # noqa: E402
from xorq.backends.redshift.compiler import RedshiftCompiler  # noqa: E402
from xorq.tests.redshift_evidence import MEASURED_ABSENT_FUNCTIONS  # noqa: E402


PRESENT = "present"  # ran on a compute node, reading a table
UNVERIFIED = "unverified"  # inherited and never asked of the warehouse

_GEO = "geospatial; not probed"

INVENTORY: dict[str, tuple[str, str]] = {
    "akeys": (UNVERIFIED, ""),
    "array": (UNVERIFIED, ""),
    "array_position": (UNVERIFIED, ""),
    "array_remove": (UNVERIFIED, ""),
    "avals": (UNVERIFIED, ""),
    "avg": (UNVERIFIED, ""),
    "bit_and": (UNVERIFIED, ""),
    "bit_or": (UNVERIFIED, ""),
    "bool_and": (UNVERIFIED, ""),
    "bool_or": (PRESENT, "arbitrary() over a boolean, 2026-09-28"),
    "cardinality": (UNVERIFIED, ""),
    "ceil": (UNVERIFIED, ""),
    "coalesce": (UNVERIFIED, ""),
    "concat_ws": (UNVERIFIED, ""),
    "corr": (UNVERIFIED, ""),
    "date_trunc": (UNVERIFIED, ""),
    "exist": (UNVERIFIED, ""),
    "exists": (UNVERIFIED, ""),
    "exp": (UNVERIFIED, ""),
    "extract": (UNVERIFIED, ""),
    "floor": (UNVERIFIED, ""),
    "generate_series": (UNVERIFIED, ""),
    "greatest": (UNVERIFIED, ""),
    "jsonb_extract_path": (UNVERIFIED, ""),
    "jsonb_extract_path_text": (UNVERIFIED, ""),
    "least": (UNVERIFIED, ""),
    "length": (PRESENT, "counts trailing blanks, 2026-09-28"),
    "log": (UNVERIFIED, ""),
    "ltrim": (UNVERIFIED, ""),
    "make_interval": (UNVERIFIED, ""),
    "map": (UNVERIFIED, ""),
    "max": (PRESENT, "arbitrary(where=), 2026-09-28"),
    "min": (UNVERIFIED, ""),
    "nullif": (UNVERIFIED, ""),
    "pg_typeof": (UNVERIFIED, ""),
    "rand": (UNVERIFIED, ""),
    "regexp_like": (UNVERIFIED, ""),
    "regexp_replace": (UNVERIFIED, ""),
    "round": (UNVERIFIED, ""),
    "row": (UNVERIFIED, ""),
    "rtrim": (UNVERIFIED, ""),
    "sign": (UNVERIFIED, ""),
    "strpos": (UNVERIFIED, ""),
    "substr": (UNVERIFIED, ""),
    "substring": (UNVERIFIED, ""),
    "sum": (UNVERIFIED, ""),
    "timezone": (UNVERIFIED, "reached via visit_Cast's super()"),
    "to_char": (UNVERIFIED, ""),
    "to_jsonb": (UNVERIFIED, ""),
    "to_timestamp": (UNVERIFIED, "reached via visit_Cast's super()"),
    "trim": (UNVERIFIED, ""),
    **dict.fromkeys(
        "st_area st_asbinary st_asewkb st_asewkt st_astext st_azimuth "
        "st_buffer st_centroid st_contains st_coveredby st_covers st_crosses "
        "st_dfullywithin st_difference st_disjoint st_distance st_dwithin "
        "st_endpoint st_envelope st_equals st_geometryn st_geometrytype "
        "st_intersection st_intersects st_isvalid st_length "
        "st_linelocatepoint st_linemerge st_linesubstring st_npoints "
        "st_orderingequals st_overlaps st_perimeter st_setsrid st_simplify "
        "st_srid st_startpoint st_touches st_transform st_union st_within "
        "st_x st_y".split(),
        (UNVERIFIED, _GEO),
    ),
}

# Names a hand-written override reaches only through ``super()``, but which it
# intercepts before delegating. A static scan cannot see the branch, so each one
# is listed with the override that shadows it.
SHADOWED_BY_OVERRIDE: dict[str, str] = {
    "decode": "visit_Cast lowers string->binary to TO_VARBYTE before super()",
    "encode": "visit_Cast lowers binary->string to FROM_VARBYTE before super()",
}

_CALL = re.compile(r"""self\.(?:f|agg)\.(\w+)\s*\(|self\.(?:f|agg)\[["'](\w+)["']\]""")


def _compilable_ops():
    refused = set(RedshiftCompiler.UNSUPPORTED_OPS)
    for op in vars(ops).values():
        if not (inspect.isclass(op) and issubclass(op, ops.Node)) or op in refused:
            continue
        if hasattr(RedshiftCompiler, f"visit_{op.__name__}"):
            yield op


def _inherited_sources(op):
    """The source of every inherited implementation this op's visitor can run:
    the visitor itself when it is inherited, and each implementation a
    hand-written one reaches through ``super()``."""
    name = f"visit_{op.__name__}"
    mro = RedshiftCompiler.__mro__
    for i, cls in enumerate(mro):
        if name not in vars(cls):
            continue
        source = inspect.getsource(vars(cls)[name])
        if cls.__module__ != RedshiftCompiler.__module__:
            yield source
            return
        if "super()." not in source:
            return
        mro = mro[i + 1 :]
        for parent in mro:
            if name in vars(parent):
                yield inspect.getsource(vars(parent)[name])
                return
        return


def _compilable_inherited_ops():
    for op in _compilable_ops():
        visitor = getattr(RedshiftCompiler, f"visit_{op.__name__}")
        if visitor.__module__ != RedshiftCompiler.__module__:
            yield op, visitor


def _reachable_functions() -> dict[str, set[str]]:
    reached = defaultdict(set)
    for op in _compilable_ops():
        if op in RedshiftCompiler.SIMPLE_OPS:
            names = [RedshiftCompiler.SIMPLE_OPS[op]]
        else:
            names = [
                a or b
                for source in _inherited_sources(op)
                for a, b in _CALL.findall(source)
            ]
        for name in names:
            if name.lower() not in SHADOWED_BY_OVERRIDE:
                reached[name.lower()].add(op.__name__)
    return reached


def test_the_sweep_sees_the_inherited_surface():
    """Guards the two tests below against a scan that silently finds nothing."""
    reached = _reachable_functions()
    assert len(list(_compilable_inherited_ops())) > 200
    assert {"coalesce", "date_trunc", "st_area"} <= set(reached)


def test_every_inherited_function_is_classified():
    reached = set(_reachable_functions())
    unclassified = sorted(reached - set(INVENTORY))
    stale = sorted(set(INVENTORY) - reached)
    assert not unclassified, (
        f"newly reachable through inherited code, and never classified: "
        f"{unclassified}. Probe each on a compute node and add it to INVENTORY."
    )
    assert not stale, f"INVENTORY names no inherited visitor reaches: {stale}"


def test_no_compilable_op_reaches_a_measured_absent_function():
    reached = _reachable_functions()
    live = {
        name: sorted(reached[name])
        for name in MEASURED_ABSENT_FUNCTIONS
        if name in reached
    }
    assert not live, (
        "these ops still compile through a function Redshift lacks; override "
        f"or refuse them: {live}"
    )


def test_no_absent_function_is_also_classified():
    both = sorted(set(INVENTORY) & set(MEASURED_ABSENT_FUNCTIONS))
    assert not both, f"classified as reachable AND measured absent: {both}"


def test_every_tag_is_known():
    assert {tag for tag, _ in INVENTORY.values()} <= {PRESENT, UNVERIFIED}
