import pathlib

import pytest
import toolz

import xorq.api as xo
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
)
from xorq.caching.strategy import (
    SnapshotStrategy,
)
from xorq.common.utils.func_utils import (
    return_constant,
)
from xorq.common.utils.provenance_utils import (
    get_expr_hash,
)
from xorq.expr.relations import (
    DatabaseTableView,
    gen_name,
)


echo_udxf = xo.expr.relations.flight_udxf(
    process_df=toolz.identity,
    maybe_schema_in=return_constant(True),
    maybe_schema_out=toolz.identity,
)


def get_other_path(name, tmp_path):
    path = pathlib.Path(xo.options.pins.get_path(name))
    other_path = tmp_path.joinpath(f"{name}.parquet")
    other_path.write_bytes(path.read_bytes())
    return other_path


def test_flight_expr_name_doesnt_matter():
    con = xo.connect()
    name = "diamonds"
    t = xo.examples.get_table_from_name(name, con)
    expr0, expr1 = (
        xo.expr.relations.flight_expr(
            t,
            xo.table(t.schema(), name=_name),
        ).cache(ParquetCache.from_kwargs(source=con))
        for _name in ("name-a", "name-b")
    )
    assert expr0.ls.get_key() == expr1.ls.get_key()


def test_flight_udxf_name_doesnt_matter():
    name = "diamonds"
    (con, other_con) = (xo.connect(), xo.connect())
    path = pathlib.Path(xo.options.pins.get_path(name))
    expr0, expr1 = (
        c.read_parquet(path, name)
        .pipe(echo_udxf, name=_name, inner_name="inner_name")
        .cache(ParquetCache.from_kwargs(source=c))
        for (c, _name) in (
            (con, "name-a"),
            (other_con, "name-b"),
        )
    )
    assert expr0.ls.get_key() == expr1.ls.get_key()


def test_flight_udxf_inner_name_doesnt_matter():
    name = "diamonds"
    (con, other_con) = (xo.connect(), xo.connect())
    path = pathlib.Path(xo.options.pins.get_path(name))
    expr0, expr1 = (
        c.read_parquet(path, name)
        .pipe(echo_udxf, name="echo", inner_name=inner_name)
        .cache(ParquetCache.from_kwargs(source=c))
        for (c, inner_name) in (
            (con, "inner_name-a"),
            (other_con, "inner_name-b"),
        )
    )
    assert expr0.ls.get_key() == expr1.ls.get_key()


# The three tests above assert name-neutrality through `ls.get_key()`, which for
# ParquetCache means ModificationTimeStrategy -> the global hasher. That path was
# always correct. The build hash and every SnapshotStrategy-backed cache go
# through SnapshotStrategy's own DatabaseTable override instead, which used to
# miss FlightExpr/FlightUDXF and fold their generated names in -- so `xorq build`
# named a fresh directory per process and snapshot caches never hit (gh-2229).
# These cover the two paths the tests above do not.


def test_flight_udxf_inner_name_doesnt_matter_build_hash():
    name = "diamonds"
    (con, other_con) = (xo.connect(), xo.connect())
    path = pathlib.Path(xo.options.pins.get_path(name))
    expr0, expr1 = (
        c.read_parquet(path, name).pipe(echo_udxf, name="echo", inner_name=inner_name)
        for (c, inner_name) in (
            (con, "inner_name-a"),
            (other_con, "inner_name-b"),
        )
    )
    assert get_expr_hash(expr0) == get_expr_hash(expr1)


def test_flight_udxf_inner_name_doesnt_matter_snapshot_key():
    name = "diamonds"
    (con, other_con) = (xo.connect(), xo.connect())
    path = pathlib.Path(xo.options.pins.get_path(name))
    expr0, expr1 = (
        c.read_parquet(path, name)
        .pipe(echo_udxf, name="echo", inner_name=inner_name)
        .cache(ParquetSnapshotCache.from_kwargs(source=c))
        for (c, inner_name) in (
            (con, "inner_name-a"),
            (other_con, "inner_name-b"),
        )
    )
    assert expr0.ls.get_key() == expr1.ls.get_key()


def test_flight_expr_inner_name_doesnt_matter_build_hash():
    con = xo.connect()
    name = "diamonds"
    t = xo.examples.get_table_from_name(name, con)
    expr0, expr1 = (
        xo.expr.relations.flight_expr(
            t,
            xo.table(t.schema(), name="unbound"),
            inner_name=inner_name,
        )
        for inner_name in ("inner_name-a", "inner_name-b")
    )
    assert get_expr_hash(expr0) == get_expr_hash(expr1)


def test_snapshot_strategy_rejects_unnormalized_databasetableview():
    """A new DatabaseTableView must fail loudly, not leak its generated name."""

    class UnhandledView(DatabaseTableView):
        pass

    dt = UnhandledView(
        name=gen_name(),
        schema=xo.schema({"a": "int64"}),
        source=xo.connect(),
    )
    with pytest.raises(NotImplementedError, match="UnhandledView"):
        SnapshotStrategy.normalize_databasetable(dt)


def test_flight_udxf_path_matters(tmp_path):
    name = "diamonds"
    (con, other_con) = (xo.connect(), xo.connect())
    path = pathlib.Path(xo.options.pins.get_path(name))
    other_path = get_other_path(name, tmp_path)
    expr0, expr1 = (
        c.read_parquet(p, name)
        .pipe(echo_udxf, name="echo", inner_name="inner-echo")
        .cache(ParquetCache.from_kwargs(source=c))
        for c, p in (
            (con, path),
            (other_con, other_path),
        )
    )
    assert expr0.ls.get_key() != expr1.ls.get_key()
