from contextlib import contextmanager

import pyarrow as pa
import pytest
from adbc_driver_manager import ProgrammingError

from xorq.common.utils.adbc_utils import ADBCBase


INGEST_UNSUPPORTED = ProgrammingError(
    "INVALID_ARGUMENT: unknown statement string type option `adbc.ingest.target_table`",
    status_code=3,
)


class FakeStatement:
    def __init__(self, supports_ingest):
        self.supports_ingest = supports_ingest
        self.options = {}

    def set_options(self, **kwargs):
        if not self.supports_ingest:
            raise INGEST_UNSUPPORTED
        self.options |= kwargs


class FakeCursor:
    def __init__(self, supports_ingest, ingest_exc=None):
        self.adbc_statement = FakeStatement(supports_ingest)
        self.ingest_exc = ingest_exc
        self.ingested = None

    def adbc_ingest(self, table_name, reader, **kwargs):
        if self.ingest_exc is not None:
            raise self.ingest_exc
        self.ingested = (table_name, reader)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, supports_ingest, ingest_exc=None):
        self._cursor = FakeCursor(supports_ingest, ingest_exc)
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_adbc(supports_ingest, hint="", ingest_exc=None):
    class FakeADBC(ADBCBase):
        ingest_install_hint = hint

        def __init__(self):
            self.conn = FakeConn(supports_ingest, ingest_exc)

        def get_conn(self, **kwargs):
            return self.conn

    return FakeADBC()


def make_reader():
    schema = pa.schema([("id", pa.int64())])
    return pa.RecordBatchReader.from_batches(
        schema, [pa.record_batch([[1]], schema=schema)]
    )


def test_adbc_ingest_probe_passes_through():
    adbc = make_adbc(supports_ingest=True)
    adbc.adbc_ingest("t", make_reader())
    cur = adbc.conn._cursor
    assert cur.adbc_statement.options == {"adbc.ingest.target_table": "t"}
    assert cur.ingested[0] == "t"
    assert adbc.conn.committed


def test_adbc_ingest_probe_rejects_with_hint():
    adbc = make_adbc(supports_ingest=False, hint="install it with `dbc install x`")
    with pytest.raises(RuntimeError, match="does not support bulk ingest") as exc_info:
        adbc.adbc_ingest("t", make_reader())
    assert "dbc install x" in str(exc_info.value)
    assert exc_info.value.__cause__ is INGEST_UNSUPPORTED
    assert adbc.conn._cursor.ingested is None


def test_adbc_ingest_probe_rejects_without_hint():
    adbc = make_adbc(supports_ingest=False)
    with pytest.raises(RuntimeError, match="does not support bulk ingest$"):
        adbc.adbc_ingest("t", make_reader())


def test_adbc_ingest_unrelated_error_propagates():
    # a driver whose probe passes but whose ingest fails for an unrelated
    # reason (auth, bad table) must raise the original error untranslated
    exc = ProgrammingError(
        "PERMISSION_DENIED: caller lacks bigquery.tables.create", status_code=1
    )
    adbc = make_adbc(supports_ingest=True, ingest_exc=exc)
    with pytest.raises(ProgrammingError, match="PERMISSION_DENIED"):
        adbc.adbc_ingest("t", make_reader())
    assert not adbc.conn.committed
