import pathlib

import toolz

import xorq as xo
from xorq.caching import (
    ParquetStorage,
)
from xorq.common.utils.func_utils import (
    return_constant,
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
        ).cache(ParquetStorage(con))
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
        .cache(ParquetStorage(c))
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
        .cache(ParquetStorage(c))
        for (c, inner_name) in (
            (con, "inner_name-a"),
            (other_con, "inner_name-b"),
        )
    )
    assert expr0.ls.get_key() == expr1.ls.get_key()


def test_flight_udxf_path_matters(tmp_path):
    name = "diamonds"
    (con, other_con) = (xo.connect(), xo.connect())
    path = pathlib.Path(xo.options.pins.get_path(name))
    other_path = get_other_path(name, tmp_path)
    expr0, expr1 = (
        c.read_parquet(p, name)
        .pipe(echo_udxf, name="echo", inner_name="inner-echo")
        .cache(ParquetStorage(c))
        for c, p in (
            (con, path),
            (other_con, other_path),
        )
    )
    assert expr0.ls.get_key() != expr1.ls.get_key()
