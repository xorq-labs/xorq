import pyarrow as pa
import pytest

from xorq.conftest import make_xo_con


@pytest.fixture(scope="session")
def xo_con(parquet_dir, ddl_file):
    # this directory's own connection: its tests register and cache tables
    return make_xo_con(parquet_dir, ddl_file)


@pytest.fixture(scope="session")
def con(xo_con):
    return xo_con


@pytest.fixture
def df():
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5, 6]),
            pa.array([7, 4, 3, 8, 9, 1, 6]),
            pa.array(["A", "A", "A", "A", "B", "B", "B"]),
        ],
        names=["a", "b", "c"],
    )
    return batch.to_pandas()
