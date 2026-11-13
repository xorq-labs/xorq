"""into_backend fan-out edge cases on datafusion.

The shapes live in ``xorq.backends.tests.into_backend_fanout``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.backends.tests.into_backend_fanout import *  # noqa: F403


@pytest.fixture
def target() -> xo.Backend:
    return xo.datafusion.connect()


def test_into_backend_reads_upstream_lazily(target) -> None:
    # the remote table streams through a TeeTableProvider: a limit stops
    # pulling the upstream instead of reading it whole first
    schema = pa.schema([("a", pa.int64())])
    pulled = []
    batches = (
        pulled.append(i) or pa.record_batch([[i]], schema=schema) for i in range(50)
    )
    src = xo.connect().read_record_batches(
        pa.RecordBatchReader.from_batches(schema, batches), "src"
    )

    assert len(src.into_backend(target, "rt").limit(1).execute()) == 1
    assert len(pulled) < 50
