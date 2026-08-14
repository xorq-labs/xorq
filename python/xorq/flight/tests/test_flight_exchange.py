from __future__ import annotations

import functools
import operator
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cloudpickle
import pandas as pd
import pyarrow as pa
import pytest
import toolz
from pytest import param

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.common.utils.rbr_utils import streaming_split_exchange
from xorq.expr.relations import (
    FlightExpr,
    FlightUDXF,
)
from xorq.flight import FlightServer
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import (
    AbstractExchanger,
    UnboundExprExchanger,
    make_udxf,
)
from xorq.flight.tests.conftest import (
    do_agg,
    field_name,
    my_udf,
    my_udf_on_expr,
    return_type,
)


@pytest.mark.uv_export
def test_unbound_exchanger_command_stable_across_reduce(tmp_path: Path) -> None:
    """`UnboundExprExchanger.command` embeds the expression's token.

    The exchanger's ``__reduce__`` builds a zip and reloads on unpickle,
    producing a fresh extract dir each time. If the token depends on that
    dir (via DataFusion's execution plan string), two round-trips produce
    different commands and client/server registration breaks.
    """
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    bound = xo.deferred_read_parquet(parquet_path, xo.connect(), "bound")
    unbound = xo.table(bound.schema(), name="unbound")
    joined = unbound.join(bound, unbound.x == bound.x).select(unbound.x)

    exchanger_once = cloudpickle.loads(cloudpickle.dumps(UnboundExprExchanger(joined)))
    exchanger_twice = cloudpickle.loads(cloudpickle.dumps(exchanger_once))
    assert exchanger_once.command == exchanger_twice.command


@pytest.mark.uv_export
def test_flight_expr(
    con: xo.Backend, diamonds: xo.Table, baseline: pd.DataFrame
) -> None:
    unbound_expr = (
        xo.table(diamonds.schema()).pipe(do_agg).mutate(my_udf_on_expr).order_by("cut")
    )
    expr = xo.expr.relations.flight_expr(
        diamonds,
        unbound_expr,
        inner_name="flight-expr",
        name="remote-expr",
        con=con,
    )
    df = expr.execute()
    pd.testing.assert_frame_equal(
        baseline.sort_values("cut", ignore_index=True),
        df.sort_values("cut", ignore_index=True),
        check_exact=False,
    )


def test_flight_udxf(con, diamonds, baseline):
    input_expr = diamonds.pipe(do_agg)
    process_df = operator.methodcaller("assign", **{field_name: my_udf.fn})
    maybe_schema_in = input_expr.schema()
    maybe_schema_out = xo.schema(input_expr.schema() | {field_name: return_type})
    expr = xo.expr.relations.flight_udxf(
        input_expr,
        process_df=process_df,
        maybe_schema_in=maybe_schema_in,
        maybe_schema_out=maybe_schema_out,
        con=con,
        # operator.methodcaller doesn't have name, so must explicitly pass
        make_udxf_kwargs={"name": my_udf.__name__},
    ).order_by("cut")
    df = expr.execute()
    actual = df.sort_values("cut", ignore_index=True)
    expected = baseline.sort_values("cut", ignore_index=True)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
    )


def test_make_udxf_fails():
    def dummy(df: pd.DataFrame):
        return pd.DataFrame({"row_count": [42]})

    with pytest.raises(ValueError):
        make_udxf(
            dummy,
            xo.schema({"dummy": "int64"}),
            pa.schema(
                [
                    ("row_count", pa.int64()),
                ]
            ),
        )

    with pytest.raises(ValueError):
        make_udxf(
            dummy,
            pa.schema(
                [
                    ("dummy", pa.int64()),
                ]
            ),
            xo.schema({"row_count": "int64"}),
        )


def test_flight_serve_unbound_finds_con(parquet_dir):
    batting = xo.deferred_read_parquet(
        parquet_dir.joinpath("batting.parquet"), xo.connect()
    )
    awards_players = xo.deferred_read_parquet(
        parquet_dir.joinpath("awards_players.parquet"),
        xo.connect(),
    )

    awards_players_unbound = xo.table(
        name="awards_players", schema=awards_players.schema()
    )
    predicates = tuple(
        set(batting.columns).intersection(awards_players_unbound.columns)
    )
    joined = batting.select(predicates).join(
        awards_players_unbound.select(predicates), predicates=predicates
    )
    _, do_exchange = xo.expr.relations.flight_serve_unbound(joined)
    actual = do_exchange(awards_players).read_pandas()
    expected = batting.execute()[list(predicates)].merge(
        awards_players.execute()[list(predicates)], on=predicates
    )
    assert not actual.empty
    assert actual.sort_values(list(actual.columns), ignore_index=True).equals(
        expected.sort_values(list(expected.columns), ignore_index=True)
    )


@pytest.mark.parametrize(
    "i,j",
    (
        (0, 2),
        (1, 2),
        (2, 2),
    ),
)
def test_flight_serve_unbound_finds_con_complex(i, j, parquet_dir, tmpdir):
    def do_join(left, right, predicates):
        match con := toolz.excepts(xo.api.XorqError, right._find_backend)():
            case None:
                return left.join(right, predicates=predicates)
            case xo.Backend():
                return left.into_backend(con).join(right, predicates=predicates)
            case _:
                raise ValueError(f"unexpected backend type: {type(con)}")

    name = "batting"
    path = Path(tmpdir).joinpath(f"{name}.parquet")
    predicates = ("playerID", "yearID", "teamID")
    xo.deferred_read_parquet(parquet_dir.joinpath(f"{name}.parquet")).select(
        predicates
    ).distinct().to_parquet(path)

    unbound_batting = xo.table(
        schema=xo.deferred_read_parquet(path).schema(), name=name
    )
    (*battings, to_exchange) = tuple(
        xo.deferred_read_parquet(path, xo.connect()) for _ in range(j + 1)
    )
    (batting0, batting1, *rest) = (*battings[:i], unbound_batting, *battings[i:])
    joined = do_join(batting0, batting1, predicates)
    for other in rest:
        joined = do_join(joined, other, predicates)

    _, do_exchange = xo.expr.relations.flight_serve_unbound(joined)
    actual = do_exchange(to_exchange).read_pandas()
    expected = xo.deferred_read_parquet(path).execute()
    assert not actual.empty
    assert actual.sort_values(list(actual.columns), ignore_index=True).equals(
        expected.sort_values(list(expected.columns), ignore_index=True)
    )


@pytest.mark.uv_export
def test_bare_flight_expr_binds_params_through_to_rbr() -> None:
    """A bare ``FlightExpr`` root (the case the execute/to_pyarrow_batches early
    return fires on) is routed through the transform passes before ``to_rbr`` (R3),
    so a ``xorq.param`` inside ``input_expr`` is bound.

    Before R3 the Flight branch short-circuited ``_transform_expr`` and ``to_rbr``
    re-entered ``input_expr.to_pyarrow_batches()`` with no ``params`` -- so this
    raised ``ValueError: Missing required parameters: cutoff``. Pins both the
    ``execute`` and ``to_pyarrow_batches`` paths (the two early-return sites).
    """
    con = xo.connect()
    t = con.register(xo.memtable({"a": [1, 2, 3], "b": [10, 20, 30]}), table_name="t0")
    p = xo.param("cutoff", "int64")
    input_expr = t.filter(t.a > p)
    # identity remote program: passes the (already-filtered) input through
    unbound = xo.table(input_expr.schema(), name="unbound")
    fe = FlightExpr.from_exprs(input_expr, unbound).to_expr()
    assert isinstance(fe.op(), FlightExpr), "bare Flight root hits the early return"

    df = fe.execute(params={"cutoff": 1})
    assert sorted(df["a"].tolist()) == [2, 3]

    reader = xo.to_pyarrow_batches(fe, params={"cutoff": 1})
    assert reader.read_all().num_rows == 2


def test_bare_flight_udxf_binds_params_through_to_rbr() -> None:
    """The ``FlightUDXF`` sibling of the FlightExpr early-return path.

    Both ``_pandas_execute`` and ``to_pyarrow_batches`` branch on
    ``isinstance(node, (FlightExpr, FlightUDXF))``, so ``FlightUDXF`` has the
    identical param-binding path and must be covered too. A bare ``FlightUDXF``
    root re-enters ``input_expr.to_pyarrow_batches()`` in ``to_rbr`` with no
    ``params``; binding must happen before that.
    """
    con = xo.connect()
    t = con.register(xo.memtable({"a": [1, 2, 3], "b": [10, 20, 30]}), table_name="t0")
    p = xo.param("cutoff", "int64")
    input_expr = t.filter(t.a > p)
    # identity remote program: schema in == schema out, passthrough process_df
    udxf = make_udxf(
        lambda df: df,
        input_expr.schema(),
        input_expr.schema(),
        name="identity",
    )
    fu = FlightUDXF.from_expr(input_expr=input_expr, udxf=udxf).to_expr()
    assert isinstance(fu.op(), FlightUDXF), "bare FlightUDXF root hits early return"

    df = fu.execute(params={"cutoff": 1})
    assert sorted(df["a"].tolist()) == [2, 3]

    reader = xo.to_pyarrow_batches(fu, params={"cutoff": 1})
    assert reader.read_all().num_rows == 2


def execute_with_deadline(expr: xo.Table, seconds: int = 90) -> pd.DataFrame:
    """``xo.execute`` on a daemon thread, so a deadlock fails instead of hanging.

    A hang inside pytest is a 300s faulthandler dump rather than a failure, and
    a daemon thread is abandonable, so the interpreter can still exit.
    """
    box = {}

    def run() -> None:
        try:
            box["value"] = xo.execute(expr)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise AssertionError(f"exchange did not finish within {seconds}s: deadlocked")
    if "error" in box:
        raise box["error"]
    return box["value"]


def make_failing_udxf(exc_type: type[BaseException], message: str) -> xo.Table:
    """A UDXF whose ``process_df`` raises ``exc_type(message)``.

    Built inside the fetcher, not closed over: an exception *instance* in the
    closure is not hashable by dasher, so the expression would fail to build.
    """
    schema = xo.schema({"unit": "int64"})

    def boom(df: pd.DataFrame) -> pd.DataFrame:
        raise exc_type(message)

    return xo.expr.relations.flight_udxf(
        process_df=boom,
        maybe_schema_in=schema,
        maybe_schema_out=schema,
        name="Boom",
    )(xo.memtable([{"unit": 1}], name="unit_tbl"))


@pytest.mark.parametrize(
    ("exc_type", "message"),
    (
        param(ValueError, "plain exception", id="exception"),
        # not an Exception, so it used to escape the excepts wrapper and leave
        # the client's reader in queue.get() forever
        param(SystemExit, "missing credential", id="systemexit"),
    ),
)
def test_udxf_failure_raises_instead_of_hanging(
    exc_type: type[BaseException], message: str
) -> None:
    """A failed exchange must surface as an error to whoever pulls the batches.

    Both halves matter: the client forwards the exception to the consumer (or
    it deadlocks), and the server re-raises rather than swallows (or the
    consumer sees a clean, empty, cacheable stream).
    """
    expr = make_failing_udxf(exc_type, message)
    with pytest.raises(Exception, match=message):
        execute_with_deadline(expr)


def test_failed_udxf_writes_no_cache(tmp_path: Path) -> None:
    """A swallowed failure used to be cached: zero rows, stored as the answer."""
    expr = make_failing_udxf(ValueError, "plain exception").cache(
        ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    )
    with pytest.raises(Exception, match="plain exception"):
        execute_with_deadline(expr)
    assert not tuple(tmp_path.glob("*.parquet"))


# `make_batch`/`make_split_f` are shared with the unit-layer pins of the abort
# policy in xorq.common.utils.tests.test_rbr_utils; duplicated rather than
# imported across test packages.
SPLIT_KEY = "split"


def make_batch(split: int, n_rows: int) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {SPLIT_KEY: [split] * n_rows, "a": list(range(n_rows))}
    )


def make_split_f(fail_on: int) -> Callable:
    """A per-split ``f`` that raises on the ``fail_on`` split."""

    def f(split_reader: pa.RecordBatchReader) -> pa.RecordBatch:
        table = split_reader.read_all()
        (split,) = set(table[SPLIT_KEY].to_pylist())
        if split == fail_on:
            raise ValueError(f"boom on split {split}")
        return pa.RecordBatch.from_pydict({SPLIT_KEY: [split], "n": [table.num_rows]})

    return f


class FailingSplitExchanger(AbstractExchanger):
    """`streaming_split_exchange` exchanger whose ``f`` raises on split 1."""

    @property
    def exchange_f(self) -> Callable:
        return functools.partial(streaming_split_exchange, SPLIT_KEY, make_split_f(1))

    @property
    def schema_in_required(self) -> None:
        return None

    @property
    def schema_in_condition(self) -> Callable:
        def condition(schema_in: Any) -> bool:
            return any(name == SPLIT_KEY for name in schema_in)

        return condition

    @property
    def calc_schema_out(self) -> Callable:
        def f(schema_in: Any) -> Any:
            return xo.schema({SPLIT_KEY: "int64", "n": "int64"})

        return f

    @property
    def description(self) -> str:
        return "raises on split 1"

    @property
    def command(self) -> str:
        return "failing-split-exchange"

    @property
    def query_result(self) -> dict:
        return {
            "schema-in-required": self.schema_in_required,
            "schema-in-condition": self.schema_in_condition,
            "calc-schema-out": self.calc_schema_out,
            "description": self.description,
            "command": self.command,
        }


def test_streaming_split_exchange_flight_failure_aborts() -> None:
    """End-to-end: a failing split surfaces as an error on the client's reader.

    The unit-layer pins of the abort policy live in
    ``xorq.common.utils.tests.test_rbr_utils``. Runs on a daemon thread with a
    hard deadline (the ``execute_with_deadline`` scaffolding, hand-rolled here
    because this drives ``do_exchange_batches`` directly rather than an expr)
    so a regression to swallowing or deadlocking fails instead of wedging CI.
    Also pins the non-atomicity caveat end-to-end: the split-0 batch is
    delivered before the raise.
    """
    batches = (make_batch(0, 3), make_batch(1, 2), make_batch(2, 1))
    rbr_in = pa.RecordBatchReader.from_batches(batches[0].schema, iter(batches))
    exchanger = FailingSplitExchanger()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            with FlightServer() as server:
                client = server.client
                client.do_action(
                    AddExchangeAction.name, exchanger, options=client._options
                )
                (_, rbr_out) = client.do_exchange_batches(exchanger.command, rbr_in)
                delivered: list[pa.RecordBatch] = []
                try:
                    for batch in rbr_out:
                        delivered.append(batch)
                except BaseException as e:  # noqa: BLE001
                    box["error"] = e
                finally:
                    box["delivered"] = delivered
        except BaseException as e:  # noqa: BLE001
            box["setup_error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(90)
    assert not thread.is_alive(), "exchange did not finish within 90s: deadlocked"
    assert "setup_error" not in box, box.get("setup_error")
    assert "error" in box, "failing split produced a clean stream instead of an error"
    assert "boom on split 1" in str(box["error"])
    assert [batch[SPLIT_KEY].to_pylist() for batch in box["delivered"]] == [[0]]
