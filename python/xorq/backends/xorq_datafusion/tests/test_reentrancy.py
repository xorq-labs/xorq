"""Regression tests for the SessionContext re-entrancy hazard.

xorq's execution teardown drops the views it registered from inside the result
``RecordBatchReader``'s generator ``finally`` block (see
``rbr_wrapper``/``clean_up`` in ``xorq/expr/api.py``). That fires
``con.sql("DROP VIEW ...")`` while the reader from the same datafusion
``SessionContext`` is still draining. When the context method held an exclusive
borrow across the GIL-releasing future, the re-entrant ``sql()`` raised
``RuntimeError: Already borrowed``.

Fixed upstream in xorq-labs/xorq-datafusion#37 (methods take ``&self``); xorq
additionally only bounds a ``StreamCache`` with ``max_readers`` on backends that
scan it directly (DuckDB), leaving datafusion-fed caches unbounded so cache
eviction never shifts the stream-exhaustion timing that made this deterministic.

These tests guard against reintroducing the hazard from the xorq side.
"""

import gc
import io
import threading

import pyarrow as pa

import xorq.api as xo
from xorq.vendor.ibis.expr import types as ir


def _teardown_expr(con) -> ir.Table:
    """An expr whose execution registers datafusion views that the result
    reader's ``finally`` drops while the reader is still draining."""
    df = pa.table({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    rt = xo.memtable(df).into_backend(con, "src")
    trn = rt.filter(rt.amount > 0).select("user_id", "amount").into_backend(con, "trn")
    return trn.limit(1)


def test_execute_teardown_no_already_borrowed() -> None:
    """Repeated execute on one shared context must not raise 'Already borrowed'.

    Reusing the context is what made the borrow race deterministic: teardown
    re-enters ``con.sql`` (DROP VIEW) while the prior reader is finalizing.
    """
    con = xo.connect()
    for _ in range(50):
        result = _teardown_expr(con).execute()
        assert len(result) == 1


def test_concurrent_execute_on_shared_context() -> None:
    """Many threads executing on one shared datafusion backend must not panic."""
    con = xo.connect()
    errors = []
    barrier = threading.Barrier(6)

    def worker():
        try:
            barrier.wait()
            for _ in range(15):
                assert len(_teardown_expr(con).execute()) == 1
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"execute raised under concurrency: {errors}"


def test_abandoned_reader_teardown_no_already_borrowed() -> None:
    """A reader read partially then dropped runs cleanup on GC.

    The generator's ``finally`` (DROP VIEW) fires from ``__del__``/GC while the
    underlying datafusion stream is mid-flight — the borrow is still live.
    """
    con = xo.connect()
    for _ in range(50):
        reader = _teardown_expr(con).to_pyarrow_batches()
        next(reader)  # read one batch, leave the stream undrained
        del reader
        gc.collect()  # finally -> clean_up -> ctx.sql("DROP VIEW ...")


def test_nested_datafusion_teardown_no_already_borrowed() -> None:
    """Chained into_backend on one context registers several views; the result
    reader's teardown drops them all while the stream is still finalizing."""
    con = xo.connect()
    df = pa.table({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    for _ in range(50):
        a = xo.memtable(df).into_backend(con, "a")
        b = a.filter(a.amount > 0).into_backend(con, "b")
        c = b.select("user_id").into_backend(con, "c")
        assert len(c.limit(1).execute()) == 1


def test_to_pyarrow_teardown_no_already_borrowed() -> None:
    """The to_pyarrow (read_all) entry point shares the same teardown path."""
    con = xo.connect()
    for _ in range(50):
        assert _teardown_expr(con).to_pyarrow().num_rows == 1


def test_to_pyarrow_stream_teardown_no_already_borrowed() -> None:
    """The to_pyarrow_stream entry point shares the same teardown path."""
    con = xo.connect()
    for _ in range(50):
        sink = io.BytesIO()
        xo.to_pyarrow_stream(_teardown_expr(con), sink)
        assert sink.getbuffer().nbytes > 0
