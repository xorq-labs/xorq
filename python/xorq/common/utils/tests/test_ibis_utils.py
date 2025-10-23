import pytest

import xorq.api as xo
from xorq.common.utils.ibis_utils import from_ibis


ibis = pytest.importorskip("ibis")


@pytest.fixture(scope="function")
def con():
    return xo.connect()


def test_basic_ops():
    t = ibis.memtable({"id": [1, 2, 3]})
    xorq_t = from_ibis(t)
    assert xorq_t is not None


def test_select():
    t = ibis.memtable({"id": [1, 2, 3]})
    expr = t.select(ibis._.id).filter(ibis._.id == 1)
    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_read_parquet(con, parquet_dir):
    astronauts_parquet = parquet_dir / "astronauts.parquet"
    t = con.read_parquet(astronauts_parquet, table_name="astronauts")
    expr = t.join(t, "id")
    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
