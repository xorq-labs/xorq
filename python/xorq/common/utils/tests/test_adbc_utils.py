import pyarrow as pa
import pytest


pytest.importorskip("adbc_driver_manager")

from adbc_driver_manager import (  # noqa: E402
    NotSupportedError,
    ProgrammingError,
)

from xorq.common.utils.adbc_utils import ADBCBase  # noqa: E402


INGEST_UNSUPPORTED = ProgrammingError(
    "INVALID_ARGUMENT: unknown statement string type option `adbc.ingest.target_table`",
    status_code=3,
)
# a driver that reports the unknown option via NOT_IMPLEMENTED instead
INGEST_NOT_IMPLEMENTED = NotSupportedError("NOT_IMPLEMENTED: option not supported")


class FakeStatement:
    def __init__(
        self, supports_ingest: bool, probe_exc: Exception | None = None
    ) -> None:
        self.supports_ingest = supports_ingest
        self.probe_exc = probe_exc if probe_exc is not None else INGEST_UNSUPPORTED
        self.options = {}

    def set_options(self, **kwargs: object) -> None:
        if not self.supports_ingest:
            raise self.probe_exc
        self.options |= kwargs


class FakeCursor:
    def __init__(
        self,
        supports_ingest: bool,
        ingest_exc: Exception | None = None,
        probe_exc: Exception | None = None,
    ) -> None:
        self.adbc_statement = FakeStatement(supports_ingest, probe_exc)
        self.ingest_exc = ingest_exc
        self.ingested = None

    def adbc_ingest(
        self, table_name: str, reader: pa.RecordBatchReader, **kwargs: object
    ) -> None:
        if self.ingest_exc is not None:
            raise self.ingest_exc
        self.ingested = (table_name, reader)

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeConn:
    def __init__(
        self,
        supports_ingest: bool,
        ingest_exc: Exception | None = None,
        probe_exc: Exception | None = None,
    ) -> None:
        self._cursor = FakeCursor(supports_ingest, ingest_exc, probe_exc)
        self.committed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def __enter__(self) -> "FakeConn":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def make_adbc(
    supports_ingest: bool,
    hint: str = "",
    ingest_exc: Exception | None = None,
    probe_exc: Exception | None = None,
) -> ADBCBase:
    class FakeADBC(ADBCBase):
        ingest_install_hint = hint

        def __init__(self) -> None:
            self.conn = FakeConn(supports_ingest, ingest_exc, probe_exc)

        def get_conn(self, **kwargs: object) -> FakeConn:
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


def test_adbc_ingest_probe_rejects_not_implemented() -> None:
    # a driver that reports the unknown option via NOT_IMPLEMENTED
    # (NotSupportedError) must be translated just like the INVALID_ARGUMENT
    # (ProgrammingError) case
    adbc = make_adbc(supports_ingest=False, probe_exc=INGEST_NOT_IMPLEMENTED)
    with pytest.raises(RuntimeError, match="does not support bulk ingest"):
        adbc.adbc_ingest("t", make_reader())
    assert adbc.conn._cursor.ingested is None


def test_adbc_ingest_unrelated_probe_error_propagates() -> None:
    # a supporting driver that rejects set_options for a reason unrelated to
    # the ingest option (e.g. a malformed target-table value) must not be
    # mislabeled as lacking bulk-ingest capability
    exc = ProgrammingError(
        "INVALID_ARGUMENT: invalid table name `bad name`", status_code=3
    )
    adbc = make_adbc(supports_ingest=False, probe_exc=exc)
    with pytest.raises(ProgrammingError, match="invalid table name"):
        adbc.adbc_ingest("t", make_reader())
    assert not adbc.conn.committed
