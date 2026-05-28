"""
Regression test for GIL+mutex deadlock in StreamCache fan-out via DataFusion.

Root cause:
  DataFusion executes two concurrent scans over the same Python-backed table
  (e.g. two scalar subqueries from .as_scalar().as_table().mutate(...)) by
  spawning Tokio async worker threads.  Each worker thread called
  Python::attach (GIL acquisition) inside PyRecordBatchProviderExec::execute()
  in order to pull the next batch from the Python-backed StreamCache reader.
  At the same time, batchcorder's ingest_up_to() holds the DatasetInner mutex
  while reading batches.  This produces a classic A-B / B-A circular wait:

    Worker thread A: holds GIL, blocks waiting for DatasetInner mutex
    Worker thread B: holds DatasetInner mutex, blocks on PyGILState_Ensure()

  → deadlock.

Fix (xorq-datafusion ef86e2b):
  Move Python::attach into spawn_blocking (a dedicated thread pool) so the
  GIL is never acquired on Tokio async worker threads.  Also drop the GIL
  across reader.next() calls so a blocking reader never starves the async
  executor while holding the GIL.  With these changes no async worker thread
  ever holds the GIL, breaking the circular wait.
  See: https://github.com/xorq-labs/xorq-datafusion/commit/ef86e2b10190c5f0276d4d4aa4dd9135a8dcb82a
"""

import subprocess
import sys

import pytest


_DEADLOCK_SCRIPT = """
import pyarrow as pa
import xorq.api as xo
from batchcorder import StreamCache

schema = pa.schema([("x", pa.int64())])

def gen():
    for i in range(5):
        yield pa.record_batch({"x": [i]})

cache = StreamCache(pa.RecordBatchReader.from_batches(schema, gen()))
con = xo.connect()
t = con.read_record_batches(cache, table_name="deadlock_t")

# Two scalar subqueries over the same StreamCache-backed table.
# DataFusion executes both via Tokio worker threads that each call
# ingest_up_to() -> with_gil_acquired() while holding the inner Mutex.
s1 = t.x.sum().as_scalar()
s2 = t.x.count().as_scalar()
result = s1.as_table().mutate(cnt=s2).execute()
assert result["cnt"].iloc[0] == 5
assert result.iloc[0, 0] == 10
print("ok")
"""


def test_stream_cache_two_scan_no_deadlock():
    """Two scalar subqueries over a StreamCache-backed table must not deadlock."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DEADLOCK_SCRIPT],
            timeout=8,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "deadlock detected: two-scan StreamCache query hung for 8 s "
            "(GIL + Mutex inversion in ingest_up_to / with_gil_acquired)"
        )

    assert proc.returncode == 0, proc.stderr
