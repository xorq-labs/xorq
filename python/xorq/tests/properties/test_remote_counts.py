import functools

import pandas as pd
import pytest
from hypothesis import example, given

import xorq.api as xo
from xorq.tests.strategies import build_shape, kv_frames, shapes
from xorq.vendor.ibis.util import gen_name


CONNECT = {"duckdb": lambda: xo.duckdb.connect(), "xorq": lambda: xo.connect()}


@functools.cache
def connection(role, name):
    # one connection per role and backend: a function-scoped fixture under
    # @given is a Hypothesis health-check error
    return CONNECT[name]()


def rows(table):
    # row order is not part of any shape's result, and a mean's fraction may
    # differ in its last bits with the order the engine adds in
    return sorted(
        (
            (k, None if pd.isna(v) else round(v, 6))
            for k, v in table.execute()[["k", "v"]].itertuples(index=False)
        ),
        key=repr,
    )


@pytest.mark.parametrize(
    ("source", "target"),
    (("duckdb", "duckdb"), ("duckdb", "xorq"), ("xorq", "duckdb"), ("xorq", "xorq")),
)
@given(shape=shapes, frame=kv_frames())
# a CTE referenced twice under a LIMIT: duckdb inlines it into each reference
# instead of materialising it, so its table is scanned once per reference
@example(
    shape=("top", ("intersect", ("minus_mean", ("base",))), 1),
    frame=pd.DataFrame({"k": [0, 1], "v": [1.0, 2.0]}),
)
@example(
    shape=("top", ("twice", ("filter", ("base",), 0)), 1),
    frame=pd.DataFrame({"k": [0, 1], "v": [1.0, 2.0]}),
)
def test_remote_table_reads_like_a_plain_table(source, target, shape, frame):
    # the RemoteTable's cache is sized by its scan count: an under-count raises
    # "Maximum number of readers", a wrong replay drops or repeats rows
    src, tgt = connection("source", source), connection("target", target)
    plain_name, remote_name = gen_name("plain"), gen_name("remote")
    try:
        plain = tgt.create_table(plain_name, frame)
        remote = src.create_table(remote_name, frame).into_backend(tgt)
        assert rows(build_shape(shape, remote)) == rows(build_shape(shape, plain))
    finally:
        tgt.drop_table(plain_name, force=True)
        src.drop_table(remote_name, force=True)


def test_nested_self_joins_do_not_fan_out():
    # every row shares one key, so a key-only self-join would square the rows
    # at each step and the property would materialise all of them
    frame = pd.DataFrame({"k": [0] * 12, "v": [float(i) for i in range(12)]})
    con, name = connection("target", "duckdb"), gen_name("fanout")
    shape = ("self_join", ("self_join", ("self_join", ("base",))))
    try:
        table = con.create_table(name, frame)
        assert build_shape(shape, table).count().execute() <= len(frame)
    finally:
        con.drop_table(name, force=True)
