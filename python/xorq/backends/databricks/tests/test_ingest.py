from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator

    from xorq.backends.databricks import Backend


N_ROWS = 2_000


def make_table(offset: int = 0) -> pa.Table:
    ids = list(range(offset, offset + N_ROWS))
    return pa.table(
        {
            "id": ids,
            "value": [i * 0.5 for i in ids],
            "name": [f"name-{i}" for i in ids],
            "flag": [i % 2 == 0 for i in ids],
            "bucket": [i % 7 for i in ids],
            "label": [f"label-{i % 13}" for i in ids],
        }
    )


@pytest.fixture
def table_name(con: Backend) -> Iterator[str]:
    name = f"test_ingest_{time.monotonic_ns()}"
    yield name
    con.drop_table(name, force=True)


def staged_files(con: Backend) -> list[str]:
    with con.con.cursor() as cur:
        rows = cur.execute(f"LIST '{con._memtable_volume_path}'").fetchall()
    return [row[0] for row in rows if "ingest" in row[0]]


def test_read_record_batches_thousands_of_rows(con: Backend, table_name: str) -> None:
    data = make_table()
    start = time.monotonic()
    t = con.read_record_batches(data.to_reader(), table_name=table_name)
    elapsed = time.monotonic() - start
    assert t.count().execute() == N_ROWS
    assert t.id.sum().execute() == sum(range(N_ROWS))
    assert elapsed < 120, f"ingest of {N_ROWS} rows took {elapsed:.0f}s"
    assert staged_files(con) == []


def test_read_record_batches_modes(con: Backend, table_name: str) -> None:
    con.read_record_batches(make_table().to_reader(), table_name=table_name)
    with pytest.raises(Exception, match="TABLE_OR_VIEW_ALREADY_EXISTS"):
        con.read_record_batches(make_table().to_reader(), table_name=table_name)
    t = con.read_record_batches(
        make_table(N_ROWS).to_reader(), table_name=table_name, mode="append"
    )
    assert t.count().execute() == 2 * N_ROWS
    reordered = make_table().select(["label", "id", "value", "name", "flag", "bucket"])
    t = con.read_record_batches(
        reordered.to_reader(), table_name=table_name, mode="append"
    )
    assert t.id.sum().execute() == 2 * sum(range(N_ROWS)) + sum(
        range(N_ROWS, 2 * N_ROWS)
    )
    t = con.read_record_batches(
        make_table().to_reader(), table_name=table_name, mode="replace"
    )
    assert t.count().execute() == N_ROWS


def test_read_record_batches_bad_mode_uploads_nothing(
    con: Backend, table_name: str
) -> None:
    with pytest.raises(ValueError, match="bogus"):
        con.read_record_batches(
            make_table().to_reader(), table_name=table_name, mode="bogus"
        )
    assert staged_files(con) == []
