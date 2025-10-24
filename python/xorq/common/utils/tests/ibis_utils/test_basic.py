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


@pytest.mark.parametrize(
    "field,predicate", (("id", ibis._.id == 1), ("name", ibis._.name == "Bob"))
)
def test_select(field, predicate, con):
    t = ibis.memtable({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    expr = t.select(field).filter(predicate)
    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty


def test_read_parquet(con, parquet_dir):
    astronauts_parquet = parquet_dir / "astronauts.parquet"
    t = con.read_parquet(astronauts_parquet, table_name="astronauts")
    expr = t.join(t, "id")
    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
