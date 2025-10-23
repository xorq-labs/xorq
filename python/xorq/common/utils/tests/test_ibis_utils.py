import ibis

from xorq.common.utils.ibis_utils import from_ibis


def test_basic_ops():
    t = ibis.memtable({"id": [1, 2, 3]})
    xorq_t = from_ibis(t)
    assert xorq_t is not None
