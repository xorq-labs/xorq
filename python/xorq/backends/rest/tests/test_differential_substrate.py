"""Differential observation harness pinning the REST execution substrate.

WHY THIS EXISTS
---------------
It gated the substrate swap; it now pins the result. `RestBackend` (and its
`github`/`mixpanel` subclasses) USED TO execute on `PandasBackend`: the
`make_dt` boundary (`fetch_resource`) paginated an API into a pandas frame and
stashed it in `self.dictionary`. ADR-2216 (Accepted) replaced that substrate
with an owned xorq-DataFusion connection into which resource reads register as
*lazy* tables, and deleted the streaming workaround
(`RestBackend.read_to_pyarrow_batches`, the `RestBackend.to_pyarrow_batches`
override, and the api-level `_maybe_streaming_read_reader` interceptor).

pandas and Arrow/DataFusion disagree about null representation, integer
widening and string types (`utf8` vs `large_utf8`); and laziness, multi-scan
behaviour and expression identity could all have shifted. The pre-existing rest
suites had been written against pandas behaviour by someone not hunting for
semantic divergence, so *them* passing was not evidence of equivalence. This
module records a structured observation record per expression and compares it
against a committed baseline, so every difference the swap produced had to be
adjudicated as intended-or-bug rather than discovered in production.

The baseline now holds the POST-swap values, and that is the harness's standing
job: the records that gated the swap are what pin the substrate's observable
behaviour from here on. Any later change to it -- a fix, a dependency bump, the
next substrate -- surfaces as a per-field diff that has to be adjudicated the
same way. This is permanent infrastructure, not scaffolding to delete now that
the swap has landed.

The harness stays deliberately assertion-free about what behaviour *ought* to
be. It records what *is*, including behaviour that is arguably wrong -- an
explicit `chunk_size` is now inert, because batch boundaries follow HTTP pages
rather than a rebatching step -- because a baseline that encodes opinions
cannot distinguish "the substrate changed" from "the opinion changed".

WHAT IS OBSERVED (per case, see `observe`)
------------------------------------------
- `lifecycle`  HTTP call counts at four separate stages: after `con.read(...)`
  construction, after `make_dt` materialisation, after `to_pyarrow_batches`
  construction but before any pull, and after a full drain. Plus the `make_dt`
  boundary facts ADR-2216 makes claims about: the resolved table's op type, its
  source backend, and whether `dt.source is read.source` (the ADR admits this
  invariant breaks; that is mechanically visible here). Plus the workaround
  seams -- which now record their own ABSENCE, so a reintroduction is a diff
  (see `observe_lifecycle`).
- `identity`   `tokenize(expr)`, `get_expr_hash(expr)`, and the 12-char build
  directory name via `ArtifactStore`. ADR-2216 claimed identity was untouched
  even though `make_dt`'s table source became the owned connection; these
  fields are how that claim was checked, and they keep it checked. Read hashes
  were deliberately changed by the fix folding resolved `base_urls` and auth
  shape into read identity, so these are current values, not historical ones.
- `schema`     the expression's ibis schema and its Arrow schema (`utf8` vs
  `large_utf8` lives here), plus the Arrow schema of the first transported
  record batch, which need not agree with either.
- `materialized`  row count; full content compared canonically and
  order-independently (rows as sorted JSON objects keyed by column name, so
  neither row order nor column order can perturb it); the *materialised*
  pandas dtypes (a second, independent view of types); per-column null counts;
  how a null actually appears (`None` vs `nan` vs `<NA>` vs `NaT`); and
  ordering recorded separately from content, so an ordering-only change is
  distinguishable from a content change.
- `batches`    batch count and per-batch row counts from `to_pyarrow_batches`,
  three times: no `chunk_size`, `chunk_size=2` and `chunk_size=7` over the
  ~150-row fixture.

MULTI-SCAN (the observation most likely to catch a silent break)
---------------------------------------------------------------
A lazy `pa.RecordBatchReader` registered as a DataFusion table is *one-shot*.
If the physical plan scans that table twice, the second scan yields nothing --
silently, as zero rows rather than an error. That is what the `StreamCache`
replay layer (`RestBackend._replay_cache`) exists to prevent, and these three
shapes are how the prevention stays checked:

- `multiscan_union_all` -- one `Read` op referenced twice
  (`t.union(t, distinct=False)`). Recorded: 4 HTTP calls (ONE pagination) and
  300 rows. The 300 is the tripwire -- with a bare one-shot reader the second
  scan contributes nothing and this drops to 150 -- and the 4 is the other
  half: the replay serves the second scan from the buffer instead of
  re-paginating. Under the pandas substrate the same numbers came from a
  different mechanism (the executor memoised by op node, `Node.map_clear`,
  into a single materialisation), which is why the case is worth keeping: same
  observation, and it now describes replay rather than memoisation.
- `multiscan_two_reads` -- two independent reads of the same resource with the
  same params, joined. Two `Read` ops are two registrations, so this genuinely
  scans twice: 8 HTTP calls (two complete paginations) and 150 joined rows,
  versus 4 calls for a bare read. That is how "two scans really happen" stays
  confirmed, and it is the shape an exhausted-source perturbation goes red on.
  It is also the concurrent-pagination shape: DataFusion polls the two
  reader-backed scans on separate threads, which is why the fixture is
  page-addressed (see `WidgetSession`) and why the engine builds one HTTP
  session per fetch.
- `multiscan_self_join` -- `t.join(t.view(), "id")`. This was UNEXECUTABLE on
  the pandas substrate (`OperationNotDefinedError: No translation rule for
  SelfReference`) and the recorded error was the pre-swap baseline; recording
  the error rather than skipping the case is exactly what made the swap making
  it work show up as a diff to adjudicate instead of passing unnoticed. It now
  records 150 rows on 4 HTTP calls -- one pagination, replayed for the second
  scan.

RETENTION (INFORMATIONAL-ONLY)
------------------------------
`tracemalloc` peak during a full drain is measured and attached with
`record_property` as `tracemalloc_peak_kib`, but it is deliberately NOT part
of the compared record: it varies with interpreter, pandas/pyarrow build and
allocator state, and no honest threshold exists. The gated retention proxies
are the deterministic ones: `materialized_before_first_pull` (did anything hit
the network before the first batch was pulled) and `peak_batch_rows` (the
largest single batch held). Those two are what "bounded memory" actually
means here.

REGENERATING THE BASELINE
-------------------------
One command, from the repository root. Never hand-edit the snapshots.

    uv run --no-sync pytest --import-mode=importlib --snapshot-update \
        python/xorq/backends/rest/tests/test_differential_substrate.py

(Outside `/workspaces/src`, `uv run --no-sync` cannot be used because the
venv's `.pth` hard-codes that path; substitute
`PYTHONPATH=$PWD/python /workspaces/src/.venv/bin/pytest` for `uv run
--no-sync pytest`, with the same remaining arguments.)

Baselines live in `snapshots/test_differential_substrate/test_observation/`,
one `observation.json` per case. On mismatch the failure names every
divergent field path with expected-vs-actual, so a diff can be read as
intended-vs-bug; "hashes differ" is never the whole message.
"""

from __future__ import annotations

import json
import math
import tracemalloc
from typing import TYPE_CHECKING, Any, Callable

import attr
import pandas as pd
import pytest

import xorq.api as xo
from xorq.backends.mixpanel.client import MixpanelClient
from xorq.backends.rest.engines import NativeEngine
from xorq.backends.rest.tests.test_rest import (
    FakeResponse,
    FakeSession,
)
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.provenance_utils import get_expr_hash
from xorq.expr.relations import Read
from xorq.ibis_yaml.compiler import ArtifactStore


if TYPE_CHECKING:
    import pathlib

    import xorq.vendor.ibis.expr.types as ir
    from xorq.vendor.ibis.backends import BaseBackend


# -- fixture data ------------------------------------------------------------
#
# One ~150-row resource over three 50-row pages plus the terminating empty
# page the page_number paginator needs. Every column carries nulls at a
# different stride so that null representation, integer widening (a nullable
# int64 is the classic float64-instead-of-Int64 trap) and boolean nullability
# are each independently observable.

PAGE_SIZE = 50
PAGE_COUNT = 3
ROW_COUNT = PAGE_SIZE * PAGE_COUNT

RICH_SCHEMA = {
    "id": "int64",
    "qty": "int64",  # nulls here are the integer-widening canary
    "name": "string",
    "score": "float64",
    "flag": "boolean",
}

REST_CONFIG_DICT = {
    "base_urls": {"default": "https://harness.example.com"},
    "auth": {"kind": "bearer", "fields": ["token"], "optional_fields": ["token"]},
    "resources": [
        {
            "name": "widgets",
            "path": "/widgets/{bucket}",
            "paginator": "page_number",
            "schema": RICH_SCHEMA,
            "params": [{"name": "bucket", "required": True}],
        },
        {
            "name": "nothing",
            "path": "/nothing",
            "schema": RICH_SCHEMA,
        },
    ],
}

HARNESS_ENV = {
    "XORQ_HARNESS_TOKEN": "fake-harness-token",
    "MIXPANEL_SERVICE_ACCOUNT_USERNAME": "fake-user.abc123",
    "MIXPANEL_SERVICE_ACCOUNT_SECRET": "fake-secret-value",
    "MIXPANEL_PROJECT_ID": "1234567",
}


def widget_records(start: int, count: int) -> list[dict]:
    return [
        {
            "id": i,
            "qty": None if i % 13 == 0 else i * 3,
            "name": None if i % 5 == 0 else f"widget-{i:03d}",
            "score": None if i % 7 == 0 else round(i / 4, 3),
            "flag": None if i % 11 == 0 else bool(i % 2),
        }
        for i in range(start, start + count)
    ]


class WidgetSession:
    """A ``page_number`` server that answers the page it was *asked* for.

    `FakeSession` serves one global FIFO and ignores the request entirely,
    which is exactly right for a single sequential pagination. It cannot
    represent CONCURRENT pagination: two independent reads in one expression
    are paginated on separate threads (an execution engine polls the two table
    scans concurrently -- no session/partition option changes that), and a FIFO
    then hands read B a page that was fetched for read A. Each read still asks
    for its own pages 1, 2, 3, ... in order -- verified by the recorded
    ``page`` params -- so the misassignment is the fixture's, and the recorded
    row count becomes a coin flip across runs (50/100/150 observed) that
    describes the fixture rather than the substrate.

    Addressing the response by the ``page`` param the paginator actually sent
    fixes that without weakening a single observation: one pagination still
    sees pages 1..PAGE_COUNT+1 in order with the same records, the same URL
    and params are still ledgered in ``calls``, and an exhausted one-shot
    source still goes red on ``row_count`` (it would return no rows at all).
    """

    def __init__(self) -> None:
        self.calls: list = []

    def get(
        self, url: str, params: dict | None = None, **kwargs: object
    ) -> FakeResponse:
        params = dict(params or {})
        self.calls.append((url, params))
        page = int(params.get("page", 1))
        if not 1 <= page <= PAGE_COUNT:
            # the empty page that terminates page_number pagination
            return FakeResponse([])
        return FakeResponse(widget_records((page - 1) * PAGE_SIZE, PAGE_SIZE))


def empty_session() -> FakeSession:
    """A present-and-empty page: the single_page paginator's one request. Not
    page-addressed -- single_page sends no page param and asks exactly once."""
    return FakeSession((FakeResponse([]),))


def github_issue_records(start: int, count: int) -> list[dict]:
    return [
        {
            "number": i,
            "title": None if i % 3 == 0 else f"issue {i}",
            "state": "open" if i % 2 else "closed",
            "created_at": f"2026-01-{i:02d}T00:00:00Z",
            "user": {"login": f"user{i}"},
            "unmapped": {"labels": [f"l{i}"]},
        }
        for i in range(start, start + count)
    ]


def github_pages() -> tuple:
    """Two header-link pages, then a single-record repo response, repeated so
    multiple scans are served."""
    return (
        FakeResponse(
            github_issue_records(1, 3),
            links={"next": {"url": "https://api.github.com/issues?page=2"}},
        ),
        FakeResponse(github_issue_records(4, 2)),
    ) * 4


def github_repo_pages() -> tuple:
    return (
        FakeResponse(
            [],
            body={
                "full_name": "xorq-labs/xorq",
                "default_branch": "main",
                "stargazers_count": 7,
                "open_issues_count": None,
                "extra": {"nested": True},
            },
        ),
    ) * 4


MIXPANEL_ENGAGE_FRAME = pd.DataFrame(
    {
        "distinct_id": pd.array(["a", None, "c"], dtype="object"),
        "properties": pd.array(['{"x": 1}', "{}", None], dtype="object"),
    }
)


# -- connections -------------------------------------------------------------
#
# Every case gets a freshly connected backend whose engine is an in-process
# fake session, so no request ever leaves the process and `session.calls` is
# the HTTP ledger. `_engine` is the same injection point
# `test_rest.connect_paged_backend` uses. The single-read cases use
# `test_rest.FakeSession` (a canned FIFO) directly; the widget resource, which
# the multi-scan cases read more than once, needs the page-addressed
# `WidgetSession` above.


def connect_rest(
    make_session: Callable[[], Any] = WidgetSession,
) -> tuple[BaseBackend, Any]:
    con = xo.load_backend("rest").connect(
        token="${XORQ_HARNESS_TOKEN}", config=REST_CONFIG_DICT
    )
    session = make_session()
    con._engine = NativeEngine(session=session)
    return con, session


def connect_github(pages: Callable[[], tuple]) -> tuple[BaseBackend, FakeSession]:
    con = xo.load_backend("github").connect()
    session = FakeSession(pages())
    con._engine = NativeEngine(session=session)
    return con, session


def connect_mixpanel() -> tuple[BaseBackend, FakeSession]:
    """Mixpanel's resources are `fetch_override`s, so no session is involved at
    all -- the override reaches `MixpanelClient`, which the fixture replaces.
    An empty FakeSession is still returned so the ledger reads zero, which is
    itself the observation: an override resource makes no engine requests."""
    con = xo.load_backend("mixpanel").connect(
        username="${MIXPANEL_SERVICE_ACCOUNT_USERNAME}",
        secret="${MIXPANEL_SERVICE_ACCOUNT_SECRET}",
        project_id="${MIXPANEL_PROJECT_ID}",
    )
    session = FakeSession(())
    con._engine = NativeEngine(session=session)
    return con, session


# -- cases -------------------------------------------------------------------


@attr.frozen
class Case:
    """One expression to observe: how to connect, and how to build it."""

    name = attr.field()
    what = attr.field()
    connect = attr.field()
    build = attr.field()


def widgets(con: BaseBackend, table_name: str = "widgets_read") -> ir.Table:
    # an explicit table_name is mandatory for determinism: the default is
    # gen_name(), which is random and lands in the Read op's identity
    return con.read("widgets", bucket="b1", table_name=table_name)


CASES = (
    Case(
        name="bare_read",
        what="con.read('widgets', bucket='b1') -- the unmodified Read",
        connect=connect_rest,
        build=widgets,
    ),
    Case(
        name="filtered_projected",
        what="bare read, filtered on id and projected to three columns",
        connect=connect_rest,
        build=lambda con: (
            widgets(con).filter(xo._.id >= 100).select("id", "qty", "score")
        ),
    ),
    Case(
        name="aggregated",
        what="group_by('flag') with count, sum and max aggregates",
        connect=connect_rest,
        build=lambda con: (
            widgets(con)
            .group_by("flag")
            .agg(
                n=xo._.count(),
                total_score=xo._.score.sum(),
                max_qty=xo._.qty.max(),
            )
        ),
    ),
    Case(
        name="limit_one",
        what="bare read .limit(1) -- the limit stops the pagination early",
        connect=connect_rest,
        build=lambda con: widgets(con).limit(1),
    ),
    Case(
        name="empty_result",
        what="a resource whose page is present and empty (the "
        "empty-plus-declared-dtypes canary, cf. "
        "test_fetch_empty_result_keeps_declared_dtypes)",
        connect=lambda: connect_rest(empty_session),
        build=lambda con: con.read("nothing", table_name="nothing_read"),
    ),
    Case(
        name="multiscan_union_all",
        what="t.union(t, distinct=False) -- ONE Read op referenced twice, so "
        "one pagination and a replayed second scan. The row count is the "
        "tripwire for a one-shot reader scanned twice",
        connect=connect_rest,
        build=lambda con: widgets(con).union(widgets(con), distinct=False),
    ),
    Case(
        name="multiscan_two_reads",
        what="two independent reads of the same resource with the same "
        "params, joined -- genuinely two scans, so HTTP doubles, and the "
        "shape whose two paginations run concurrently",
        connect=connect_rest,
        build=lambda con: widgets(con, "scan_a").join(
            widgets(con, "scan_b"), "id", how="inner"
        ),
    ),
    Case(
        name="multiscan_self_join",
        what="t.join(t.view(), 'id') -- was unexecutable on the pandas "
        "substrate; the swap made it execute, on a replayed second scan",
        connect=connect_rest,
        build=lambda con: widgets(con).join(widgets(con).view(), "id"),
    ),
    Case(
        name="github_issues_paginated",
        what="curated backend, header_link pagination, residual_column",
        connect=lambda: connect_github(github_pages),
        build=lambda con: con.read(
            "issues", owner="xorq-labs", repo="xorq", table_name="issues_read"
        ),
    ),
    Case(
        name="github_repo_single_record",
        what="curated backend, single-record (dict) response, nullable int",
        connect=lambda: connect_github(github_repo_pages),
        build=lambda con: con.read(
            "repo", owner="xorq-labs", repo="xorq", table_name="repo_read"
        ),
    ),
    Case(
        name="mixpanel_override_resource",
        what="a fetch_override resource -- ADR-2216 claims the override "
        "case 'folds in uniformly'",
        connect=connect_mixpanel,
        build=lambda con: con.read("engage", where="", table_name="engage_read"),
    ),
)


# -- canonicalisation --------------------------------------------------------


def canonical_cell(value: Any) -> Any:
    """A JSON-safe, machine-independent rendering of one cell.

    Every flavour of null collapses to `null` here on purpose: WHICH null
    arrived is recorded separately (`materialized.null_representation`) so a
    null-representation change is not smeared across 150 content rows.
    """
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, sort_keys=True, default=str)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int,)) or type(value).__name__.startswith("int"):
        return int(value)
    if isinstance(value, float) or type(value).__name__.startswith("float"):
        # int-vs-float divergence stays visible: 3 and 3.0 render differently
        return float(value)
    if type(value).__name__.startswith("bool"):
        return bool(value)
    return str(value)


def canonical_rows(df: pd.DataFrame) -> list[str]:
    """Rows as JSON objects keyed by column name, so neither row order nor
    column order can perturb the content comparison."""
    columns = list(df.columns)
    return [
        json.dumps(
            {c: canonical_cell(v) for c, v in zip(columns, values)}, sort_keys=True
        )
        for values in df.itertuples(index=False, name=None)
    ]


def null_representation(df: pd.DataFrame) -> dict:
    """For each column that has a null, what object actually landed there:
    None vs nan vs pd.NA vs NaT. This is the pandas/Arrow disagreement that
    silently changes downstream `is None` and `isna()` behaviour."""
    out = {}
    for column in df.columns:
        mask = df[column].isna()
        if not bool(mask.any()):
            out[column] = None
            continue
        value = df[column][mask].iloc[0]
        out[column] = {"type": type(value).__name__, "repr": repr(value)}
    return out


def arrow_schema_dict(schema: Any) -> dict:
    return {field.name: str(field.type) for field in schema}


def error_repr(exc: BaseException) -> str:
    """A stable one-line rendering of a failure, so an expression that cannot
    execute today is a recorded observation rather than a skipped case."""
    message = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {message}"


def http_log(session: Any) -> list[str]:
    return [
        f"{url} {json.dumps(params, sort_keys=True)}" for url, params in session.calls
    ]


# -- observations ------------------------------------------------------------


def observe_lifecycle(case: Case) -> dict:
    """Stage-separated HTTP counts up to `make_dt`, plus the `make_dt`
    boundary facts ADR-2216 makes claims about, plus the workaround seams.

    The seam fields (`read_to_pyarrow_batches_seam`,
    `tables_registered_on_source_dictionary`, `to_pyarrow_batches_defined_by`)
    recorded which of the pre-swap workarounds still existed; they are kept
    because they now record their own absence, which makes REINTRODUCTION a
    diff. Someone reaching for a materialising `fetch_resource`, or restoring a
    `to_pyarrow_batches` override, cannot do it quietly.
    """
    con, session = case.connect()
    expr = case.build(con)
    after_read_construction = len(session.calls)
    reads = tuple(expr.op().find(Read))
    resolved = []
    for read in reads:
        dt = read.make_dt()
        resolved.append(
            {
                "op_type": type(dt).__name__,
                "source_backend_name": getattr(dt.source, "name", None),
                # ADR-2216 admits this invariant breaks (the resolved table's
                # source becomes the owned DataFusion connection). It must be
                # mechanically visible, not a prose claim.
                "source_is_read_source": dt.source is read.source,
                "name_equals_read_name": dt.name == read.name,
                "schema": dict(zip(dt.schema.names, map(str, dt.schema.types))),
            }
        )
    seam_con, _ = case.connect()
    seam_expr = case.build(seam_con)
    seam_method = getattr(seam_con, "read_to_pyarrow_batches", None)
    if seam_method is None:
        seam = "absent"
    else:
        seam = "reader" if seam_method(seam_expr) is not None else "none"
    return {
        "http_after_read_construction": after_read_construction,
        "http_after_make_dt": len(session.calls),
        "read_ops_in_expression": len(reads),
        # pre-swap, each fetch_resource stashed one frame here, so this was the
        # substrate's own materialisation ledger and therefore its scan count.
        # ADR-2216 made the storage vestigial and this reads 0 everywhere: a
        # nonzero value means something materialised onto the pandas table
        # store again.
        "tables_registered_on_source_dictionary": len(getattr(con, "dictionary", {})),
        "make_dt": resolved,
        "read_to_pyarrow_batches_seam": seam,
        # which class actually serves to_pyarrow_batches: ADR-2216 deleted
        # RestBackend's override, so this is BasePandasBackend's, and the
        # deletion is visible rather than implied -- as would be its return
        "to_pyarrow_batches_defined_by": type(con).to_pyarrow_batches.__qualname__,
        "request_log_after_make_dt": http_log(session),
    }


def observe_identity(case: Case, builds_dir: pathlib.Path) -> dict:
    """tokenize, the build hash, and the 12-char build directory name.

    Read hashes were deliberately changed by the fix folding resolved
    `base_urls` and the auth shape into read identity; these are the CURRENT
    values. They were byte-identical across the substrate swap, which is how
    ADR-2216's "identity is untouched" claim was checked. A divergence here is
    therefore an identity
    change: a bug unless it was deliberately one, in which case it needs its
    own adjudicated baseline (every build directory and cache entry in the
    world moves with it).
    """
    con, session = case.connect()
    expr = case.build(con)
    record: dict = {"tokenize": tokenize(expr)}
    try:
        record["expr_hash"] = get_expr_hash(expr)
    except Exception as exc:  # noqa: BLE001 - the failure IS the observation
        record["expr_hash"] = f"UNREACHABLE {error_repr(exc)}"
    try:
        store = ArtifactStore.from_path_and_expr(builds_dir, expr)
        record["build_dir_name"] = store.root_path.name
        record["build_dir_name_length"] = len(store.root_path.name)
    except Exception as exc:  # noqa: BLE001 - the failure IS the observation
        record["build_dir_name"] = f"UNREACHABLE {error_repr(exc)}"
        record["build_dir_name_length"] = None
    record["read_source_backend_names"] = sorted(
        {read.source.name for read in expr.op().find(Read)}
    )
    record["read_method_names"] = sorted(
        {read.method_name for read in expr.op().find(Read)}
    )
    # computing identity must never touch the network
    record["http_during_identity"] = len(session.calls)
    return record


def observe_schema(case: Case) -> dict:
    con, _ = case.connect()
    expr = case.build(con)
    schema = expr.schema()
    return {
        "ibis": dict(zip(schema.names, map(str, schema.types))),
        # utf8 vs large_utf8 lives here
        "arrow": arrow_schema_dict(schema.to_pyarrow()),
    }


def observe_materialized(case: Case) -> dict:
    """Row count, full canonical content, materialised pandas dtypes, null
    counts, null representation, and ordering -- ordering kept separate from
    content so an ordering-only change is distinguishable."""
    con, session = case.connect()
    expr = case.build(con)
    try:
        df = expr.execute()
    except Exception as exc:  # noqa: BLE001 - the failure IS the observation
        return {
            "error": error_repr(exc),
            "http_after_execute": len(session.calls),
        }
    rows = canonical_rows(df)
    ordered = sorted(rows)
    return {
        "http_after_execute": len(session.calls),
        "row_count": int(len(df)),
        "column_order": list(map(str, df.columns)),
        # the second, independent view of types: what the MATERIALIZED result
        # is, which need not agree with the expression's declared schema
        "pandas_dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
        "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "null_representation": null_representation(df),
        "ordering": {
            "first_row_as_returned": rows[0] if rows else None,
            "last_row_as_returned": rows[-1] if rows else None,
            "first_row_sorted": ordered[0] if ordered else None,
            "last_row_sorted": ordered[-1] if ordered else None,
            "returned_in_sorted_order": rows == ordered,
        },
        "content_canonical_sorted": ordered,
    }


def observe_batches(case: Case, chunk_size: int | None) -> tuple[dict, float]:
    """Batch structure, with the pre-pull HTTP count that is the real laziness
    signal. Returns (gated record, informational tracemalloc peak in KiB)."""
    con, session = case.connect()
    expr = case.build(con)
    kwargs = {} if chunk_size is None else {"chunk_size": chunk_size}
    try:
        reader = expr.to_pyarrow_batches(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure IS the observation
        return {"error": error_repr(exc)}, 0.0
    http_after_reader_construction = len(session.calls)
    tracemalloc.start()
    try:
        batches = list(reader)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    row_counts = [int(batch.num_rows) for batch in batches]
    return {
        "http_after_reader_construction": http_after_reader_construction,
        "http_after_full_drain": len(session.calls),
        # the deterministic laziness/retention proxies (see module docstring)
        "materialized_before_first_pull": http_after_reader_construction > 0,
        "peak_batch_rows": max(row_counts) if row_counts else 0,
        "batch_count": len(batches),
        "batch_row_counts": row_counts,
        "total_rows": sum(row_counts),
        "first_batch_arrow_schema": (
            arrow_schema_dict(batches[0].schema) if batches else None
        ),
    }, peak / 1024.0


def observe(case: Case, builds_dir: pathlib.Path) -> tuple[dict, dict]:
    """The full observation record for one case, plus the informational-only
    measurements that are deliberately excluded from it."""
    batch_records = {}
    informational = {}
    for label, chunk_size in (
        ("no_chunk_size", None),
        ("chunk_size_2", 2),
        ("chunk_size_7", 7),
    ):
        record, peak_kib = observe_batches(case, chunk_size)
        batch_records[label] = record
        informational[f"tracemalloc_peak_kib.{label}"] = round(peak_kib, 1)
    return (
        {
            "case": case.name,
            "what": case.what,
            "lifecycle": observe_lifecycle(case),
            "identity": observe_identity(case, builds_dir),
            "schema": observe_schema(case),
            "materialized": observe_materialized(case),
            "batches": batch_records,
        },
        informational,
    )


# -- per-field comparison ----------------------------------------------------


def flatten(record: Any, prefix: str = "") -> dict:
    if isinstance(record, dict):
        out = {}
        for key, value in record.items():
            out.update(flatten(value, f"{prefix}{key}."))
        return out
    if isinstance(record, list) and record and all(isinstance(v, dict) for v in record):
        out = {}
        for index, value in enumerate(record):
            out.update(flatten(value, f"{prefix}[{index}]."))
        return out
    return {prefix.rstrip("."): record}


def brief(value: Any, limit: int = 200) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def describe_list_divergence(expected: list, actual: list) -> str:
    parts = [f"length expected={len(expected)} actual={len(actual)}"]
    expected_set, actual_set = set(map(repr, expected)), set(map(repr, actual))
    missing = [v for v in expected if repr(v) not in actual_set]
    added = [v for v in actual if repr(v) not in expected_set]
    if missing:
        parts.append(f"{len(missing)} missing, first: {brief(missing[0])}")
    if added:
        parts.append(f"{len(added)} added, first: {brief(added[0])}")
    if not missing and not added:
        # this is the ORDERING-ONLY verdict, and it must not be claimed when
        # the lengths differ (that is a multiplicity change, not a reordering)
        parts.append(
            "SAME elements in a DIFFERENT order"
            if len(expected) == len(actual)
            else "same distinct elements, different multiplicity"
        )
    return "; ".join(parts)


def format_divergences(name: str, expected: dict, actual: dict) -> str | None:
    """A failure message that names WHICH observation diverged, per field,
    with expected vs actual -- so the diff can be adjudicated as
    intended-vs-bug rather than merely noticed."""
    flat_expected, flat_actual = flatten(expected), flatten(actual)
    lines = []
    for key in sorted(set(flat_expected) | set(flat_actual)):
        if key not in flat_expected:
            lines.append(f"  + {key}: NEW observation -> {brief(flat_actual[key])}")
        elif key not in flat_actual:
            lines.append(f"  - {key}: GONE (baseline had {brief(flat_expected[key])})")
        elif flat_expected[key] != flat_actual[key]:
            was, now = flat_expected[key], flat_actual[key]
            if isinstance(was, list) and isinstance(now, list):
                lines.append(f"  ! {key}: {describe_list_divergence(was, now)}")
            else:
                lines.append(
                    f"  ! {key}:\n"
                    f"        expected: {brief(was)}\n"
                    f"        actual:   {brief(now)}"
                )
    if not lines:
        return None
    return "\n".join(
        (
            f"{len(lines)} observation(s) diverged from the committed baseline "
            f"for case {name!r}:",
            *lines,
            "",
            "Adjudicate each line as INTENDED (then regenerate the baseline "
            "with the one command in this module's docstring) or as a BUG in "
            "the substrate change.",
        )
    )


# -- the test ----------------------------------------------------------------


@pytest.fixture
def harness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env-var references only -- raw secret values are rejected at expression
    construction -- and a MixpanelClient whose override returns a fixed frame,
    so the override case runs without a network call."""
    for name, value in HARNESS_ENV.items():
        monkeypatch.setenv(name, value)

    def fake_engage(
        self: MixpanelClient, where: str = "", page_size: int | None = None
    ) -> pd.DataFrame:
        return MIXPANEL_ENGAGE_FRAME.copy()

    def fake_export(self: MixpanelClient, from_date: str, to_date: str) -> pd.DataFrame:
        return MIXPANEL_ENGAGE_FRAME.copy()

    monkeypatch.setattr(MixpanelClient, "engage", fake_engage)
    monkeypatch.setattr(MixpanelClient, "export", fake_export)


@pytest.mark.snapshot_check
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.name) for case in CASES],
)
def test_observation(
    case: Case,
    harness_env: None,
    tmp_path: pathlib.Path,
    snapshot: Any,
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
) -> None:
    record, informational = observe(case, tmp_path)
    for key, value in sorted(informational.items()):
        # informational-only: measured, reported, never gated (see docstring)
        record_property(key, value)
    actual = json.dumps(record, indent=2, sort_keys=True)
    baseline = snapshot.snapshot_dir / "observation.json"
    if baseline.is_file() and not request.config.getoption("snapshot_update"):
        message = format_divergences(
            case.name, json.loads(baseline.read_text()), record
        )
        if message is not None:
            pytest.fail(message, pytrace=False)
    snapshot.assert_match(actual, "observation.json")
