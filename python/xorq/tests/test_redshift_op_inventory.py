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
import sqlglot.expressions as sge
from sqlglot.dialects import Redshift as SqlglotRedshift


# Must run before the xorq.backends.redshift import; see the matching guard in
# test_redshift_dialect.py for why both names are needed.
pytest.importorskip("adbc_driver_manager")
pytest.importorskip("psycopg")

import xorq.vendor.ibis.expr.operations as ops  # noqa: E402
from xorq.backends.redshift.compiler import RedshiftCompiler  # noqa: E402
from xorq.tests.redshift_evidence import MEASURED_ABSENT_FUNCTIONS  # noqa: E402
from xorq.vendor.ibis.backends.sql.dialects import Redshift  # noqa: E402


PRESENT = "present"  # ran on a compute node, reading a table
UNVERIFIED = "unverified"  # inherited and never asked of the warehouse

_GEO = "geospatial; not probed"

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
        "st_x st_y".split(),
        (UNVERIFIED, _GEO),
    ),
}

# Function names the xorq Redshift dialect's own TRANSFORMS rename a typed
# sqlglot node to. A visitor that builds ``sge.Split`` names no function, so
# the static read above never sees what reaches the wire; these are read from
# the dialect instead, by rendering each transform the xorq class adds.
DIALECT_RENAMED: dict[str, tuple[str, str]] = {
    "split_to_array": (PRESENT, "sge.Split; live corpus, 2026-09-30"),
    "get_array_length": (
        PRESENT,
        "sge.ArraySize; over ARRAY() and over SPLIT_TO_ARRAY; live corpus, 2026-09-30",
    ),
}

# Names a hand-written override reaches only through ``super()``, but which it
# intercepts before delegating. A static scan cannot see the branch, so each one
# is listed with the override that shadows it.
SHADOWED_BY_OVERRIDE: dict[str, str] = {
    "decode": "visit_Cast lowers string->binary to TO_VARBYTE before super()",
    "encode": "visit_Cast lowers binary->string to FROM_VARBYTE before super()",
    "to_timestamp": "visit_Cast lowers integer->timestamp to DATEADD before super()",
    "timezone": "visit_Cast refuses integer->timestamptz before super()",
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


def _dialect_renamed_functions() -> dict[str, str]:
    """Each function name a transform in the xorq ``Redshift`` generator renders,
    for every transform that class adds or replaces over sqlglot's own."""
    ours = Redshift.Generator.TRANSFORMS
    theirs = SqlglotRedshift.Generator.TRANSFORMS
    generator = Redshift().generator()
    renamed = {}
    for node, transform in ours.items():
        if theirs.get(node) is transform:
            continue
        sql = transform(
            generator, node(this=sge.column("a"), expression=sge.column("b"))
        )
        match = re.match(r"(\w+)\(", sql)
        if match:
            renamed[match.group(1).lower()] = node.__name__
    return renamed


def test_every_dialect_renamed_function_is_classified():
    """A transform the xorq class adds emits a name no inherited visitor names,
    so the inventory above cannot see it: ``SPLIT_TO_ARRAY`` and
    ``GET_ARRAY_LENGTH`` shipped unprobed that way."""
    renamed = _dialect_renamed_functions()
    assert renamed, "the dialect scan found no renames; the scan is broken"
    assert set(renamed) == set(DIALECT_RENAMED), (
        f"dialect renames {sorted(renamed)} against classified "
        f"{sorted(DIALECT_RENAMED)}. Probe each new one on a compute node."
    )
    absent = sorted(set(renamed) & set(MEASURED_ABSENT_FUNCTIONS))
    assert not absent, f"the dialect renames to a measured-absent function: {absent}"


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
    tags = {tag for tag, _ in (*INVENTORY.values(), *DIALECT_RENAMED.values())}
    assert tags <= {PRESENT, UNVERIFIED}
