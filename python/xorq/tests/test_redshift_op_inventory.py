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

Names are read statically, BEFORE the dialect renders them -- ``datefromparts``
is what reaches the wire as ``DATE_FROM_PARTS``. A visitor generated from a
``SIMPLE_OPS`` entry, at any level of the hierarchy, carries its name as the
``_name`` default; any other inherited implementation is read from its source
(``self.f.X``, ``self.f["X"]``, ``self.agg.X``). The scan follows what an
implementation calls, because that is inherited code too: a helper method
(``visit_IntervalFromInteger`` emits nothing itself, ``_make_interval`` emits
``MAKE_INTERVAL``), and ``super()``, into the implementation it delegates to
(``visit_Cast`` handles the binary casts itself and passes every other cast on
to PostgreSQL's). Code in the Redshift module is followed but not read: its
names are the Redshift compiler's own choices, which the dialect tests assert.
A name built some other way (an ``sge`` node rendered by the dialect) is not
seen; the rendered-SQL check in ``redshift_evidence`` covers the wire side.
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
_SIMPLE = "inherited SIMPLE_OPS target; not probed"

INVENTORY: dict[str, tuple[str, str]] = {
    "array": (PRESENT, "ARRAY() builds a SUPER on compute; live corpus, 2026-09-28"),
    "avg": (PRESENT, "live corpus, 2026-09-28"),
    "bit_and": (PRESENT, "live corpus, 2026-09-28"),
    "bit_or": (PRESENT, "live corpus, 2026-09-28"),
    "bool_and": (PRESENT, "live corpus, 2026-09-28"),
    "bool_or": (PRESENT, "arbitrary() over a boolean, 2026-09-28"),
    "ceil": (PRESENT, "live corpus, 2026-09-28"),
    "coalesce": (PRESENT, "live corpus, 2026-09-28"),
    "concat_ws": (PRESENT, "live corpus, 2026-09-28"),
    "date_trunc": (PRESENT, "live corpus, 2026-09-28"),
    "exists": (
        PRESENT,
        "in WHERE; a correlated EXISTS in a projection is rejected; live corpus, 2026-09-28",
    ),
    "exp": (PRESENT, "live corpus, 2026-09-28"),
    "extract": (PRESENT, "live corpus, 2026-09-28"),
    "floor": (PRESENT, "live corpus, 2026-09-28"),
    "greatest": (PRESENT, "live corpus, 2026-09-28"),
    "jsonb_extract_path": (
        PRESENT,
        "renders JSON_EXTRACT_PATH_TEXT; live corpus, 2026-09-28",
    ),
    "least": (PRESENT, "live corpus, 2026-09-28"),
    "length": (PRESENT, "counts trailing blanks, 2026-09-28"),
    "ltrim": (PRESENT, "live corpus, 2026-09-28"),
    "max": (PRESENT, "arbitrary(where=), 2026-09-28"),
    "min": (PRESENT, "live corpus, 2026-09-28"),
    "rand": (PRESENT, "live corpus, 2026-09-28"),
    "regexp_like": (PRESENT, "live corpus, 2026-09-28"),
    "round": (PRESENT, "live corpus, 2026-09-28"),
    "rtrim": (PRESENT, "live corpus, 2026-09-28"),
    "strpos": (PRESENT, "live corpus, 2026-09-28"),
    "substr": (PRESENT, "live corpus, 2026-09-28"),
    "substring": (PRESENT, "live corpus, 2026-09-28"),
    "sum": (PRESENT, "live corpus, 2026-09-28"),
    "to_char": (PRESENT, "live corpus, 2026-09-28"),
    "trim": (PRESENT, "live corpus, 2026-09-28"),
    **dict.fromkeys(
        "st_area st_asbinary st_asewkb st_asewkt st_astext st_azimuth "
        "st_buffer st_centroid st_contains st_coveredby st_covers st_crosses "
        "st_dfullywithin st_difference st_disjoint st_distance st_dwithin "
        "st_endpoint st_envelope st_equals st_geometryn st_geometrytype "
        "st_intersection st_intersects st_isvalid st_length "
        "st_linelocatepoint st_linemerge st_linesubstring st_npoints "
        "st_orderingequals st_overlaps st_perimeter st_setsrid st_simplify "
        "st_srid st_startpoint st_touches st_transform st_union st_within "
        "st_x st_y st_geomfromtext".split(),
        (UNVERIFIED, _GEO),
    ),
    **dict.fromkeys(
        "abs acos asin atan atan2 cos cot degrees pi radians sign sin sqrt tan "
        "ascii lower lpad repeat replace reverse right rpad translate upper "
        "count nullif cume_dist dense_rank nth_value ntile percent_rank rank "
        "row_number".split(),
        (UNVERIFIED, _SIMPLE),
    ),
    "array_size": (UNVERIFIED, f"renders GET_ARRAY_LENGTH; {_SIMPLE}"),
    "date": (UNVERIFIED, f"renders DATE(...); {_SIMPLE}"),
    "if": (UNVERIFIED, f"renders CASE WHEN; {_SIMPLE}"),
    "json_extract": (
        UNVERIFIED,
        "renders JSON_EXTRACT_PATH_TEXT; reached through PostgreSQL's super()",
    ),
    "split": (UNVERIFIED, f"renders SPLIT_TO_ARRAY; {_SIMPLE}"),
    "str_to_date": (UNVERIFIED, f"renders TO_DATE; {_SIMPLE}"),
    "str_to_time": (UNVERIFIED, f"renders TO_TIMESTAMP(s, format); {_SIMPLE}"),
}

# Names an op reaches only through a hand-written override that intercepts
# them before delegating. A static scan cannot see the branch, so each is
# listed with the op and the override that shadows it; keying on the op keeps
# the name visible on every other path to it.
_CASTS = ("Cast", "TryCast")
SHADOWED_BY_OVERRIDE: dict[tuple[str, str], str] = {
    **{
        (op, name): reason
        for op in _CASTS
        for name, reason in {
            "decode": "visit_Cast lowers string->binary to TO_VARBYTE before super()",
            "encode": "visit_Cast lowers binary->string to FROM_VARBYTE before super()",
            "to_timestamp": "visit_Cast lowers integer->timestamp to DATEADD before super()",
            "timezone": "visit_Cast refuses integer->timestamptz before super()",
        }.items()
    },
    ("Literal", "datefromparts"): "visit_NonNullLiteral casts a date string first",
    ("Literal", "make_time"): "visit_NonNullLiteral casts a time string first",
    ("Literal", "map"): "visit_NonNullLiteral refuses a map before super()",
}

_CALL = re.compile(r"""self\.(?:f|agg)\.(\w+)\s*\(|self\.(?:f|agg)\[["'](\w+)["']\]""")


def _compilable_ops():
    refused = set(RedshiftCompiler.UNSUPPORTED_OPS)
    for op in vars(ops).values():
        if not (inspect.isclass(op) and issubclass(op, ops.Node)) or op in refused:
            continue
        if hasattr(RedshiftCompiler, f"visit_{op.__name__}"):
            yield op


_HELPER = re.compile(r"\bself\.(\w+)\s*\(")
_SUPER = re.compile(r"\bsuper\(\)\.(\w+)\s*\(")
# Attributes that build SQL nodes rather than name methods, and the dispatcher,
# which reaches every visitor.
_NOT_FOLLOWED = frozenset({"f", "agg", "v", "visit_node"})


def _resolve(name, start=0):
    """The first definition of ``name`` in the MRO from ``start``, with its index."""
    mro = RedshiftCompiler.__mro__
    for i in range(start, len(mro)):
        if name in vars(mro[i]):
            return i, vars(mro[i])[name]
    return None, None


def _inherited_names(op):
    """Every function name this op's visitor can emit through inherited code:
    the visitor, each helper it calls, and each implementation it reaches
    through ``super()``, transitively."""
    names = []
    seen = set()
    pending = [(f"visit_{op.__name__}", 0)]
    while pending:
        name, start = pending.pop()
        i, impl = _resolve(name, start)
        if impl is None or not inspect.isfunction(impl) or (name, i) in seen:
            continue
        seen.add((name, i))
        if "_name" in (impl.__kwdefaults__ or {}):
            names.append(impl.__kwdefaults__["_name"])
            continue
        source = inspect.getsource(impl)
        if RedshiftCompiler.__mro__[i].__module__ != RedshiftCompiler.__module__:
            names.extend(a or b for a, b in _CALL.findall(source))
        pending.extend(
            (helper, 0)
            for helper in _HELPER.findall(source)
            if helper not in _NOT_FOLLOWED
        )
        pending.extend((parent, i + 1) for parent in _SUPER.findall(source))
    return names


def _compilable_inherited_ops():
    for op in _compilable_ops():
        visitor = getattr(RedshiftCompiler, f"visit_{op.__name__}")
        if visitor.__module__ != RedshiftCompiler.__module__:
            yield op, visitor


def _reachable_functions() -> dict[str, set[str]]:
    reached = defaultdict(set)
    for op in _compilable_ops():
        for name in _inherited_names(op):
            if (op.__name__, name.lower()) not in SHADOWED_BY_OVERRIDE:
                reached[name.lower()].add(op.__name__)
    return reached


def test_the_sweep_sees_the_inherited_surface():
    """Guards the two tests below against a scan that silently finds nothing:
    one name read from source, one from a generated ``SIMPLE_OPS`` visitor
    defined on ``SQLGlotCompiler`` itself, and one reached only by following
    ``super()`` and a helper (``visit_Literal`` -> ``visit_NonNullLiteral`` ->
    ``visit_DefaultLiteral``)."""
    reached = _reachable_functions()
    assert len(list(_compilable_inherited_ops())) > 200
    assert {"coalesce", "lower", "st_geomfromtext"} <= set(reached)


def test_every_shadowed_name_is_reached():
    """A shadow entry for a path the scan no longer finds would hide the name
    the day a new path reaches it."""
    scanned = {
        (op.__name__, name.lower())
        for op in _compilable_ops()
        for name in _inherited_names(op)
    }
    stale = sorted(set(SHADOWED_BY_OVERRIDE) - scanned)
    assert not stale, f"SHADOWED_BY_OVERRIDE names a path nothing reaches: {stale}"


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
