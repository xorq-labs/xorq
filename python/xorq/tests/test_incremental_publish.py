"""Round-trip tests for the Phase 2 publish mechanisms (ADR-0017 / XOR-444).

The two architectural extremes:
  * NATIVE_MERGE (duckdb)  — one ``MERGE INTO`` statement
  * REWRITE     (datafusion) — anti-join + union-all + replace

Each is exercised for upsert, merge-with-deletes, and first-run final creation,
plus one end-to-end pass through ``make_backend_wap_expr``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.writes.enums import PublishMode
from xorq.writes.publish import (
    _publish_native_merge,
    _publish_parquet_merge,
    _publish_rewrite,
    _publish_statement_dml,
    publish,
    publish_expr,
    publish_parquet,
)
from xorq.writes.wap import make_backend_wap_expr, make_parquet_wap_expr


def _read(con, name: str) -> pd.DataFrame:
    return con.table(name).execute().sort_values("id").reset_index(drop=True)


FINAL_SEED = pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]})
# changeset: update id=2, delete id=3, insert id=4
MERGE_DELTA = pd.DataFrame(
    {"id": [2, 3, 4], "v": ["B", "x", "d"], "_op": ["U", "D", "I"]}
)
UPSERT_DELTA = pd.DataFrame({"id": [2, 3], "v": ["B", "c"]})  # update 2, insert 3


@pytest.fixture
def ddb():
    pytest.importorskip("duckdb")
    return xo.duckdb.connect()


@pytest.fixture
def dff():
    pytest.importorskip("datafusion")
    return xo.datafusion.connect()


# --- NATIVE_MERGE (duckdb) --------------------------------------------------


def test_native_merge_upsert(ddb) -> None:
    ddb.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    ddb.create_table("staging", UPSERT_DELTA)
    _publish_native_merge(
        ddb, "staging", "final", ["id"], ["id", "v"], PublishMode.UPSERT
    )
    out = _read(ddb, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]
    assert "staging" not in ddb.list_tables()


def test_native_merge_with_deletes(ddb) -> None:
    ddb.create_table("final", FINAL_SEED)
    ddb.create_table("staging", MERGE_DELTA)
    _publish_native_merge(
        ddb, "staging", "final", ["id"], ["id", "v", "_op"], PublishMode.MERGE
    )
    out = _read(ddb, "final")
    assert out["id"].tolist() == [1, 2, 4]  # 3 deleted, 4 inserted
    assert out["v"].tolist() == ["a", "B", "d"]  # 2 updated


def test_native_merge_creates_final_first_run(ddb) -> None:
    ddb.create_table("staging", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    _publish_native_merge(
        ddb, "staging", "final", ["id"], ["id", "v"], PublishMode.UPSERT
    )
    assert _read(ddb, "final")["id"].tolist() == [1, 2]


# --- REWRITE (datafusion) ---------------------------------------------------


def test_rewrite_upsert(dff) -> None:
    dff.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    dff.create_table("staging", UPSERT_DELTA)
    _publish_rewrite(dff, "staging", "final", ["id"], ["id", "v"], PublishMode.UPSERT)
    out = _read(dff, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]
    assert "staging" not in dff.list_tables()


def test_rewrite_with_deletes(dff) -> None:
    dff.create_table("final", FINAL_SEED)
    dff.create_table("staging", MERGE_DELTA)
    _publish_rewrite(
        dff, "staging", "final", ["id"], ["id", "v", "_op"], PublishMode.MERGE
    )
    out = _read(dff, "final")
    assert out["id"].tolist() == [1, 2, 4]
    assert out["v"].tolist() == ["a", "B", "d"]


def test_rewrite_creates_final_first_run(dff) -> None:
    dff.create_table("staging", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    _publish_rewrite(dff, "staging", "final", ["id"], ["id", "v"], PublishMode.UPSERT)
    assert _read(dff, "final")["id"].tolist() == [1, 2]


# --- end-to-end through make_backend_wap_expr -------------------


def test_wap_integration_upsert_duckdb(ddb) -> None:
    ddb.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    src = xo.connect().register(xo.memtable(UPSERT_DELTA), table_name="src")
    wap = src.pipe(
        make_backend_wap_expr(ddb, key=["id"], mode=PublishMode.UPSERT),
        "staging",
        "final",
    )
    out = wap.execute()
    assert out["published"].iloc[0]
    final = _read(ddb, "final")
    assert final["id"].tolist() == [1, 2, 3]
    assert final["v"].tolist() == ["a", "B", "c"]


# --- UPSERT_DELETE (iceberg), end-to-end via two WAP runs -------------------


@pytest.fixture
def ice(tmp_path):
    pytest.importorskip("pyiceberg")
    return xo.pyiceberg.connect(warehouse_path=str(tmp_path / "wh"))


def _run_backend_wap(con, data: pd.DataFrame, key, mode) -> pd.DataFrame:
    src = xo.connect().register(xo.memtable(data), table_name="src")
    return src.pipe(
        make_backend_wap_expr(con, key=key, mode=mode), "staging", "final"
    ).execute()


def _read_iceberg_final(con) -> pd.DataFrame:
    df = con.catalog.load_table(f"{con.namespace}.final").scan().to_arrow().to_pandas()
    return df.sort_values("id").reset_index(drop=True)


def test_iceberg_upsert(ice) -> None:
    _run_backend_wap(
        ice, pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}), ["id"], PublishMode.UPSERT
    )
    out = _run_backend_wap(ice, UPSERT_DELTA, ["id"], PublishMode.UPSERT)
    assert out["published"].iloc[0]
    final = _read_iceberg_final(ice)
    assert final["id"].tolist() == [1, 2, 3]
    assert final["v"].tolist() == ["a", "B", "c"]


def test_iceberg_merge_with_deletes(ice) -> None:
    _run_backend_wap(ice, FINAL_SEED, ["id"], PublishMode.UPSERT)
    out = _run_backend_wap(ice, MERGE_DELTA, ["id"], PublishMode.MERGE)
    assert out["published"].iloc[0]
    final = _read_iceberg_final(ice)
    assert final["id"].tolist() == [1, 2, 4]
    assert final["v"].tolist() == ["a", "B", "d"]


# --- STATEMENT_DML (sqlite) -------------------------------------------------


@pytest.fixture
def lite():
    pytest.importorskip("adbc_driver_sqlite")
    return xo.sqlite.connect()


def test_statement_dml_upsert(lite) -> None:
    lite.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    lite.create_table("staging", UPSERT_DELTA)
    _publish_statement_dml(
        lite, "staging", "final", ["id"], ["id", "v"], PublishMode.UPSERT
    )
    out = _read(lite, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]
    assert "staging" not in lite.list_tables()


def test_statement_dml_with_deletes(lite) -> None:
    lite.create_table("final", FINAL_SEED)
    lite.create_table("staging", MERGE_DELTA)
    _publish_statement_dml(
        lite, "staging", "final", ["id"], ["id", "v", "_op"], PublishMode.MERGE
    )
    out = _read(lite, "final")
    assert out["id"].tolist() == [1, 2, 4]
    assert out["v"].tolist() == ["a", "B", "d"]


def test_statement_dml_creates_final_first_run(lite) -> None:
    lite.create_table("staging", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    _publish_statement_dml(
        lite, "staging", "final", ["id"], ["id", "v"], PublishMode.UPSERT
    )
    assert _read(lite, "final")["id"].tolist() == [1, 2]


# --- parquet (file target) --------------------------------------------------


def test_parquet_upsert(tmp_path) -> None:
    pytest.importorskip("datafusion")
    final_p, staging_p = tmp_path / "final.parquet", tmp_path / "staging.parquet"
    pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}).to_parquet(final_p)
    UPSERT_DELTA.to_parquet(staging_p)
    _publish_parquet_merge(staging_p, final_p, ["id"], PublishMode.UPSERT)
    out = pd.read_parquet(final_p).sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]
    assert not staging_p.exists()


def test_parquet_with_deletes(tmp_path) -> None:
    pytest.importorskip("datafusion")
    final_p, staging_p = tmp_path / "final.parquet", tmp_path / "staging.parquet"
    FINAL_SEED.to_parquet(final_p)
    MERGE_DELTA.to_parquet(staging_p)
    _publish_parquet_merge(staging_p, final_p, ["id"], PublishMode.MERGE)
    out = pd.read_parquet(final_p).sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 2, 4]
    assert out["v"].tolist() == ["a", "B", "d"]


def test_parquet_creates_final_first_run(tmp_path) -> None:
    pytest.importorskip("datafusion")
    final_p, staging_p = tmp_path / "final.parquet", tmp_path / "staging.parquet"
    pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}).to_parquet(staging_p)
    _publish_parquet_merge(staging_p, final_p, ["id"], PublishMode.UPSERT)
    assert pd.read_parquet(final_p).sort_values("id")["id"].tolist() == [1, 2]


def test_parquet_integration_merge(tmp_path) -> None:
    pytest.importorskip("datafusion")
    final_p = str(tmp_path / "final.parquet")
    staging_p = str(tmp_path / "staging.parquet")
    src1 = xo.connect().register(xo.memtable(FINAL_SEED), table_name="s1")
    src1.pipe(
        make_parquet_wap_expr(key=["id"], mode=PublishMode.UPSERT),
        staging_p,
        final_p,
    ).execute()
    src2 = xo.connect().register(xo.memtable(MERGE_DELTA), table_name="s2")
    out = src2.pipe(
        make_parquet_wap_expr(key=["id"], mode=PublishMode.MERGE),
        staging_p,
        final_p,
    ).execute()
    assert out["published"].iloc[0]
    final = pd.read_parquet(final_p).sort_values("id").reset_index(drop=True)
    assert final["id"].tolist() == [1, 2, 4]
    assert final["v"].tolist() == ["a", "B", "d"]


# --- APPEND (no key) across backends ----------------------------------------
# append keeps duplicates: final [1, 2] + staging [2, 3] -> [1, 2, 2, 3].

APPEND_DELTA = pd.DataFrame({"id": [2, 3], "v": ["b2", "c"]})


def test_native_merge_append(ddb) -> None:
    ddb.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    ddb.create_table("staging", APPEND_DELTA)
    _publish_native_merge(ddb, "staging", "final", [], ["id", "v"], PublishMode.APPEND)
    assert _read(ddb, "final")["id"].tolist() == [1, 2, 2, 3]


def test_rewrite_append(dff) -> None:
    dff.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    dff.create_table("staging", APPEND_DELTA)
    _publish_rewrite(dff, "staging", "final", [], ["id", "v"], PublishMode.APPEND)
    assert _read(dff, "final")["id"].tolist() == [1, 2, 2, 3]


def test_statement_dml_append(lite) -> None:
    lite.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    lite.create_table("staging", APPEND_DELTA)
    _publish_statement_dml(
        lite, "staging", "final", [], ["id", "v"], PublishMode.APPEND
    )
    assert _read(lite, "final")["id"].tolist() == [1, 2, 2, 3]


def test_iceberg_append(ice) -> None:
    # exercises the router + _publish_append delegation end-to-end
    _run_backend_wap(
        ice, pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}), [], PublishMode.APPEND
    )
    out = _run_backend_wap(ice, APPEND_DELTA, [], PublishMode.APPEND)
    assert out["published"].iloc[0]
    assert _read_iceberg_final(ice)["id"].tolist() == [1, 2, 2, 3]


def test_parquet_append(tmp_path) -> None:
    pytest.importorskip("datafusion")
    final_p, staging_p = tmp_path / "final.parquet", tmp_path / "staging.parquet"
    pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}).to_parquet(final_p)
    APPEND_DELTA.to_parquet(staging_p)
    _publish_parquet_merge(staging_p, final_p, [], PublishMode.APPEND)
    out = pd.read_parquet(final_p).sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 2, 2, 3]


# --- REWRITE end-to-end via the tee (not just direct _publish_rewrite) ------
# pandas is the in-process REWRITE sink with read_record_batches; the tee
# bulk-registers staging, so this exercises _drop_staging's force-drop and the
# anti-join/union over a tee-registered (re-scannable) staging. (xorq_datafusion
# can also host REWRITE, but datafusion async re-entrancy panics when the let
# engine is teed into itself in-process, so pandas is the integration target.)


@pytest.fixture
def pdb():
    return xo.pandas.connect()


def test_rewrite_integration_upsert_via_tee(pdb) -> None:
    pdb.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    src = xo.connect().register(xo.memtable(UPSERT_DELTA), table_name="src")
    out = src.pipe(
        make_backend_wap_expr(pdb, key=["id"], mode=PublishMode.UPSERT),
        "staging",
        "final",
    ).execute()
    assert out["published"].iloc[0]
    final = pdb.table("final").execute().sort_values("id").reset_index(drop=True)
    assert final["id"].tolist() == [1, 2, 3]
    assert final["v"].tolist() == ["a", "B", "c"]
    assert "staging" not in pdb.list_tables()  # _drop_staging force-drops bulk staging


def test_rewrite_integration_merge_via_tee(pdb) -> None:
    src1 = xo.connect().register(xo.memtable(FINAL_SEED), table_name="s1")
    src1.pipe(
        make_backend_wap_expr(pdb, key=["id"], mode=PublishMode.UPSERT),
        "staging",
        "final",
    ).execute()
    src2 = xo.connect().register(xo.memtable(MERGE_DELTA), table_name="s2")
    out = src2.pipe(
        make_backend_wap_expr(pdb, key=["id"], mode=PublishMode.MERGE),
        "staging",
        "final",
    ).execute()
    assert out["published"].iloc[0]
    final = pdb.table("final").execute().sort_values("id").reset_index(drop=True)
    assert final["id"].tolist() == [1, 2, 4]
    assert final["v"].tolist() == ["a", "B", "d"]


# --- #3: '_op' is reserved only for MERGE; APPEND keeps it as data ----------


def test_native_merge_append_keeps_op_data_column(ddb) -> None:
    ddb.create_table("final", pd.DataFrame({"id": [1], "_op": ["x"]}))
    ddb.create_table("staging", pd.DataFrame({"id": [2], "_op": ["y"]}))
    _publish_native_merge(
        ddb, "staging", "final", [], ["id", "_op"], PublishMode.APPEND
    )
    out = _read(ddb, "final")
    assert out["id"].tolist() == [1, 2]
    assert out["_op"].tolist() == ["x", "y"]  # data column preserved, not stripped


def test_parquet_append_keeps_op_data_column(tmp_path) -> None:
    pytest.importorskip("datafusion")
    final_p, staging_p = tmp_path / "final.parquet", tmp_path / "staging.parquet"
    pd.DataFrame({"id": [1], "_op": ["x"]}).to_parquet(final_p)
    pd.DataFrame({"id": [2], "_op": ["y"]}).to_parquet(staging_p)
    _publish_parquet_merge(staging_p, final_p, [], PublishMode.APPEND)
    out = pd.read_parquet(final_p).sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 2]
    assert out["_op"].tolist() == ["x", "y"]


# --- #4: NULL _op upserts (only literal 'D' deletes) ------------------------
# id=2 carries _op=None (must upsert -> v becomes 'B'); id=3 inserts.
NULL_OP_DELTA = pd.DataFrame({"id": [2, 3], "v": ["B", "c"], "_op": [None, "I"]})


def test_native_merge_null_op_upserts(ddb) -> None:
    ddb.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    ddb.create_table("staging", NULL_OP_DELTA)
    _publish_native_merge(
        ddb, "staging", "final", ["id"], ["id", "v", "_op"], PublishMode.MERGE
    )
    out = _read(ddb, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]


def test_rewrite_null_op_upserts(dff) -> None:
    dff.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    dff.create_table("staging", NULL_OP_DELTA)
    _publish_rewrite(
        dff, "staging", "final", ["id"], ["id", "v", "_op"], PublishMode.MERGE
    )
    out = _read(dff, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]


def test_statement_dml_null_op_upserts(lite) -> None:
    lite.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    lite.create_table("staging", NULL_OP_DELTA)
    _publish_statement_dml(
        lite, "staging", "final", ["id"], ["id", "v", "_op"], PublishMode.MERGE
    )
    out = _read(lite, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]


def test_parquet_null_op_upserts(tmp_path) -> None:
    pytest.importorskip("datafusion")
    final_p, staging_p = tmp_path / "final.parquet", tmp_path / "staging.parquet"
    pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}).to_parquet(final_p)
    NULL_OP_DELTA.to_parquet(staging_p)
    _publish_parquet_merge(staging_p, final_p, ["id"], PublishMode.MERGE)
    out = pd.read_parquet(final_p).sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]


def test_iceberg_null_op_upserts(ice) -> None:
    _run_backend_wap(
        ice, pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}), ["id"], PublishMode.UPSERT
    )
    out = _run_backend_wap(ice, NULL_OP_DELTA, ["id"], PublishMode.MERGE)
    assert out["published"].iloc[0]
    final = _read_iceberg_final(ice)
    assert final["id"].tolist() == [1, 2, 3]
    assert final["v"].tolist() == ["a", "B", "c"]


# --- #5 key-only table + #6 identifiers needing quotes ----------------------


def test_native_merge_key_only_upsert(ddb) -> None:
    # No data columns: matched keys are no-ops, unmatched insert (no empty SET).
    ddb.create_table("final", pd.DataFrame({"id": [1, 2]}))
    ddb.create_table("staging", pd.DataFrame({"id": [2, 3]}))
    _publish_native_merge(ddb, "staging", "final", ["id"], ["id"], PublishMode.UPSERT)
    assert _read(ddb, "final")["id"].tolist() == [1, 2, 3]


def test_native_merge_identifier_needs_quoting(ddb) -> None:
    # a column whose name needs quoting (a space) must not break the SQL.
    ddb.create_table("final", pd.DataFrame({"id": [1], "a b": ["x"]}))
    ddb.create_table("staging", pd.DataFrame({"id": [2], "a b": ["y"]}))
    _publish_native_merge(
        ddb, "staging", "final", ["id"], ["id", "a b"], PublishMode.UPSERT
    )
    out = _read(ddb, "final")
    assert out["id"].tolist() == [1, 2]
    assert out["a b"].tolist() == ["x", "y"]


# --- standalone publish (the non-WAP entry point) ---------------


def test_publish_standalone_upsert(ddb) -> None:
    # No tee, no audit: a staging table already exists; publish it into final.
    ddb.create_table("final", pd.DataFrame({"id": [1, 2], "v": ["a", "b"]}))
    ddb.create_table("staging", UPSERT_DELTA)
    publish(ddb, "staging", "final", key=["id"], mode=PublishMode.UPSERT)
    out = _read(ddb, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]
    assert "staging" not in ddb.list_tables()  # consumed


def test_publish_standalone_first_run_creates_final(ddb) -> None:
    # final absent: the primitive creates it from staging's data schema.
    ddb.create_table("staging", UPSERT_DELTA)
    publish(ddb, "staging", "final", key=["id"], mode=PublishMode.UPSERT)
    assert _read(ddb, "final")["id"].tolist() == [2, 3]


def test_publish_standalone_merge_rewrite(dff) -> None:
    # REWRITE backend, merge-with-deletes, through the standalone entry point.
    dff.create_table("final", FINAL_SEED)
    dff.create_table("staging", MERGE_DELTA)
    publish(dff, "staging", "final", key=["id"], mode=PublishMode.MERGE)
    out = _read(dff, "final")
    assert out["id"].tolist() == [1, 2, 4]
    assert out["v"].tolist() == ["a", "B", "d"]


def test_publish_standalone_validates(ddb) -> None:
    ddb.create_table("staging", UPSERT_DELTA)
    with pytest.raises(ValueError, match="requires a non-empty key"):
        publish(ddb, "staging", "final", mode=PublishMode.UPSERT)


def test_publish_parquet_standalone(tmp_path) -> None:
    pytest.importorskip("datafusion")  # UPSERT registers the files in datafusion
    staging = tmp_path / "staging.parquet"
    final = tmp_path / "final.parquet"
    FINAL_SEED.to_parquet(final)
    UPSERT_DELTA.to_parquet(staging)
    publish_parquet(staging, final, key=["id"], mode=PublishMode.UPSERT)
    out = pd.read_parquet(final).sort_values("id").reset_index(drop=True)
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]
    assert not staging.exists()  # consumed


# --- default no-duplicate-keys audit gate (ADR-0017 open Q5) -----------------


def test_backend_wap_default_audit_rejects_duplicate_keys(ddb) -> None:
    # UPSERT/MERGE default to a no-duplicate-keys gate: a delta with two rows for
    # one key fails the audit, so nothing publishes — instead of native MERGE
    # raising a cardinality error while REWRITE silently keeps both rows.
    dup = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "a2", "b"]})
    out = _run_backend_wap(ddb, dup, ["id"], PublishMode.UPSERT)
    assert not out["published"].iloc[0]
    assert "final" not in ddb.list_tables()  # gate ran before any tier


def test_backend_wap_default_audit_allows_unique_keys(ddb) -> None:
    # Regression guard: the default gate passes a unique-keyed delta.
    out = _run_backend_wap(ddb, UPSERT_DELTA, ["id"], PublishMode.UPSERT)
    assert out["published"].iloc[0]
    assert "final" in ddb.list_tables()


# --- publish_expr: fully-remote W-A-P (CTAS staging, ADR-0017) ----------------


def test_publish_expr_upsert(ddb) -> None:
    # The changeset is an expr over a warehouse-resident table; W is a CTAS,
    # A a remote scalar, P the same MERGE the tee path resolves to.
    ddb.create_table("final", FINAL_SEED)
    ddb.create_table("src", UPSERT_DELTA)
    changeset = ddb.table("src").filter(ddb.table("src").id > 0)
    result = publish_expr(
        ddb, changeset, "staging", "final", key=["id"], mode=PublishMode.UPSERT
    )
    assert result == {
        "passed": True,
        "published": True,
        "staging": "staging",
        "final": "final",
    }
    out = _read(ddb, "final")
    assert out["id"].tolist() == [1, 2, 3]
    assert out["v"].tolist() == ["a", "B", "c"]
    assert "staging" not in ddb.list_tables()  # consumed


def test_publish_expr_creates_final_first_run(ddb) -> None:
    ddb.create_table("src", UPSERT_DELTA)
    result = publish_expr(
        ddb, ddb.table("src"), "staging", "final", key=["id"], mode=PublishMode.UPSERT
    )
    assert result["published"]
    assert _read(ddb, "final")["id"].tolist() == [2, 3]


def test_publish_expr_append(ddb) -> None:
    ddb.create_table("final", FINAL_SEED)
    ddb.create_table("src", pd.DataFrame({"id": [4], "v": ["d"]}))
    result = publish_expr(
        ddb, ddb.table("src"), "staging", "final", mode=PublishMode.APPEND
    )
    assert result["published"]
    assert _read(ddb, "final")["id"].tolist() == [1, 2, 3, 4]


def test_publish_expr_default_audit_retains_staging_on_failure(ddb) -> None:
    # Duplicate keys fail the remote no-duplicate-keys gate: final untouched,
    # staging retained for forensics (and blocking an identical rerun).
    ddb.create_table("src", pd.DataFrame({"id": [1, 1, 2], "v": ["a", "a2", "b"]}))
    result = publish_expr(
        ddb, ddb.table("src"), "staging", "final", key=["id"], mode=PublishMode.UPSERT
    )
    assert result == {
        "passed": False,
        "published": False,
        "staging": "staging",
        "final": "final",
    }
    assert "final" not in ddb.list_tables()
    assert "staging" in ddb.list_tables()


def test_publish_expr_custom_audit(ddb) -> None:
    # audit_fn takes the staged Table and judges it remotely.
    ddb.create_table("src", UPSERT_DELTA)
    result = publish_expr(
        ddb,
        ddb.table("src"),
        "staging",
        "final",
        key=["id"],
        mode=PublishMode.UPSERT,
        audit_fn=lambda staged: (
            not int(staged.filter(staged.id.isnull()).count().execute())
        ),
    )
    assert result["published"]


def test_publish_expr_rejects_cross_backend_changeset(ddb, dff) -> None:
    # CTAS staging never moves rows through the client, so a changeset with a
    # source on another connection must fail before anything is created.
    dff.create_table("src", UPSERT_DELTA)
    with pytest.raises(ValueError, match="not fully resident"):
        publish_expr(
            ddb,
            dff.table("src"),
            "staging",
            "final",
            key=["id"],
            mode=PublishMode.UPSERT,
        )
    assert "staging" not in ddb.list_tables()
