"""Routing + changeset validation for incremental WAP publish.

ADR-0017 / XOR-444, Phase 1. The per-strategy publish functions are covered in
later phases; here we pin the ``PublishMode`` -> ``PublishStrategy`` routing and
the build-time ``_validate`` contract. Backend-routing cases skip when the
backend (or its driver) is not installed.
"""

from __future__ import annotations

import sqlite3
import warnings

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.writes
from xorq.writes.enums import PublishMode, PublishStrategy, StagingStrategy
from xorq.writes.publish import (
    _COMPOSITE_DELETE_WARN_ROWS,
    _key_filter,
    _merge_query,
    _validate,
    publish_parquet,
    resolve_strategy,
)
from xorq.writes.wap import make_backend_wap_expr, make_publish_with_backend


# --- _validate: pure, no backend required -----------------------------------


def test_validate_append_ok() -> None:
    _validate(PublishMode.APPEND, [], ["a", "b"])  # no raise


def test_validate_append_rejects_key() -> None:
    with pytest.raises(ValueError, match="APPEND takes no key"):
        _validate(PublishMode.APPEND, ["a"], ["a", "b"])


def test_validate_append_allows_op() -> None:
    # _op is reserved only for MERGE; APPEND keeps it as ordinary data.
    _validate(PublishMode.APPEND, [], ["a", "_op"])  # no raise


def test_validate_upsert_ok() -> None:
    _validate(PublishMode.UPSERT, ["a"], ["a", "b"])


def test_validate_upsert_requires_key() -> None:
    with pytest.raises(ValueError, match="requires a non-empty key"):
        _validate(PublishMode.UPSERT, [], ["a", "b"])


def test_validate_key_must_be_subset_of_columns() -> None:
    with pytest.raises(ValueError, match="not in changeset columns"):
        _validate(PublishMode.UPSERT, ["z"], ["a", "b"])


def test_validate_upsert_forbids_op() -> None:
    with pytest.raises(ValueError, match="UPSERT forbids"):
        _validate(PublishMode.UPSERT, ["a"], ["a", "_op"])


def test_validate_merge_ok() -> None:
    _validate(PublishMode.MERGE, ["a"], ["a", "_op"])


def test_validate_merge_requires_op() -> None:
    with pytest.raises(ValueError, match="MERGE requires an '_op'"):
        _validate(PublishMode.MERGE, ["a"], ["a", "b"])


# --- resolve_strategy -------------------------------------------------------


def test_append_short_circuits_before_dispatch() -> None:
    # APPEND is a mode-level fact; it never inspects the backend, so any object
    # resolves without a real connection.
    assert resolve_strategy(object(), PublishMode.APPEND) is PublishStrategy.APPEND


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("xorq.backends.duckdb", PublishStrategy.NATIVE_MERGE),
        ("xorq.backends.snowflake", PublishStrategy.NATIVE_MERGE),
        ("xorq.backends.sqlite", PublishStrategy.STATEMENT_DML),
        ("xorq.backends.pyiceberg", PublishStrategy.UPSERT_DELETE),
        ("xorq.backends.databricks", PublishStrategy.NATIVE_MERGE),
        ("xorq.backends.gizmosql", PublishStrategy.NATIVE_MERGE),
        ("xorq.backends.xorq_datafusion", PublishStrategy.REWRITE),
        ("xorq.backends.pandas", PublishStrategy.REWRITE),
    ],
)
def test_resolve_strategy_routes_by_backend_type(module, expected) -> None:
    mod = pytest.importorskip(module)
    backend_cls = mod.Backend
    # An instance of the backend type is enough for dispatch; no connection or
    # driver is needed to resolve the strategy. (postgres is omitted: its handler
    # reads the live server version, so it needs a real connection — see the
    # gated postgres integration test.)
    con = backend_cls.__new__(backend_cls)
    for mode in (PublishMode.UPSERT, PublishMode.MERGE):
        assert resolve_strategy(con, mode) is expected


def test_sqlite_strategy_drops_to_rewrite_below_3_33(monkeypatch) -> None:
    # STATEMENT_DML's `UPDATE ... FROM` needs SQLite >= 3.33; below that the
    # declaration drops to the universal REWRITE floor instead of raising —
    # the same shape as postgres < 15 dropping a tier.
    mod = pytest.importorskip("xorq.backends.sqlite")
    con = mod.Backend.__new__(mod.Backend)
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 32, 2))
    assert resolve_strategy(con, PublishMode.UPSERT) is PublishStrategy.REWRITE


# --- SQL generation: identifier quoting + key-only tables -------------------


def test_merge_query_quotes_and_transpiles_identifiers() -> None:
    # _q escapes embedded quotes, and identifiers transpile per dialect —
    # double-quote escaping on postgres, backticks on databricks.
    query = _merge_query(
        "final", "staging", ["id"], ["id", 'we"ird'], PublishMode.UPSERT
    )
    assert '"we""ird" = s."we""ird"' in query.sql(dialect="postgres")
    assert '`we"ird` = s.`we"ird`' in query.sql(dialect="databricks")


def test_merge_query_parenthesizes_not_delete_condition() -> None:
    # The not-delete condition lands after the implicit AND in `WHEN MATCHED AND
    # <cond>`; without explicit parens the OR would bind looser than that AND,
    # silently changing merge semantics ((MATCHED AND a) OR b).
    sql = _merge_query(
        "final", "staging", ["id"], ["id", "b", "_op"], PublishMode.MERGE
    ).sql(dialect="duckdb")
    assert """AND (s."_op" <> 'D' OR s."_op" IS NULL)""" in sql


def test_merge_query_key_only_omits_update() -> None:
    # No non-key columns -> no SET clause to emit; the MATCHED-UPDATE branch is
    # omitted so the MERGE stays valid on engines that reject UPDATE without SET.
    sql = _merge_query("final", "staging", ["id"], ["id"], PublishMode.UPSERT).sql(
        dialect="postgres"
    )
    assert "UPDATE" not in sql.upper()
    assert "WHEN NOT MATCHED" in sql.upper()


# --- BRANCH staging: factory-time validation (pure, no backend) --------------
#
# All three checks fire before the con is used for anything but a type() lookup,
# so a plain object() stands in for the connection.


def test_branch_staging_rejects_keyed_mode() -> None:
    with pytest.raises(ValueError, match="fast-forward"):
        make_backend_wap_expr(
            object(),
            key=["id"],
            mode=PublishMode.UPSERT,
            staging_strategy=StagingStrategy.BRANCH,
        )


def test_branch_staging_rejects_key() -> None:
    with pytest.raises(ValueError, match="takes no key"):
        make_backend_wap_expr(
            object(),
            key=["id"],
            mode=PublishMode.APPEND,
            staging_strategy=StagingStrategy.BRANCH,
        )


def test_branch_staging_requires_publish_branch() -> None:
    # Capability is a backend fact: type(con) must declare publish_branch.
    with pytest.raises(ValueError, match="does not support BRANCH staging"):
        make_backend_wap_expr(
            object(), mode=PublishMode.APPEND, staging_strategy=StagingStrategy.BRANCH
        )


# --- publish UDF naming: session-context registration is by name -------------


def test_publish_udf_name_disambiguates_by_closure() -> None:
    # UDFs register into the execution session by name and compile by name, so
    # two publish UDFs sharing a name in one plan would resolve to a single
    # closure. The name folds in the closure identity (con name + key) with a
    # deterministic token, so build hashes stay reproducible.
    class DuckCon:
        name = "duckdb"

    class SnowCon:
        name = "snowflake"

    t = xo.memtable({"staging": ["s"], "final": ["f"], "passed": [True]})

    def name_of(udf):
        return udf.on_expr(t).op().__func_name__

    by_id = make_publish_with_backend(DuckCon(), key=["id"], mode=PublishMode.UPSERT)
    again = make_publish_with_backend(DuckCon(), key=["id"], mode=PublishMode.UPSERT)
    by_name = make_publish_with_backend(
        DuckCon(), key=["name"], mode=PublishMode.UPSERT
    )
    other_con = make_publish_with_backend(
        SnowCon(), key=["id"], mode=PublishMode.UPSERT
    )

    assert name_of(by_id) == name_of(again)  # deterministic across constructions
    assert len({name_of(by_id), name_of(by_name), name_of(other_con)}) == 3


# --- _key_filter: composite-key bulk-delete predicate warning ----------------


def test_key_filter_warns_on_bulk_composite_delete() -> None:
    pytest.importorskip("pyiceberg")
    n = _COMPOSITE_DELETE_WARN_ROWS + 1
    rows = pa.table({"a": list(range(n)), "b": list(range(n))})
    with pytest.warns(UserWarning, match="composite key"):
        _key_filter(rows, ["a", "b"])
    # Single-column keys use In and never hit the Or-of-And tree: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _key_filter(rows, ["a"])


# --- parquet publish: structural validation precedes the full read -----------


def test_publish_parquet_validates_before_read(tmp_path, monkeypatch) -> None:
    # A structural mismatch (here: MERGE without _op) must fail from parquet
    # metadata alone, before the changeset is deserialized into pandas.
    staging = tmp_path / "staging.parquet"
    pd.DataFrame({"id": [1], "v": [2]}).to_parquet(staging, index=False)

    def _fail_read(*args, **kwargs):
        raise AssertionError("read_parquet ran before _validate")

    monkeypatch.setattr(pd, "read_parquet", _fail_read)
    with pytest.raises(ValueError, match="MERGE requires an '_op'"):
        publish_parquet(
            staging, tmp_path / "final.parquet", key=["id"], mode=PublishMode.MERGE
        )


# --- public API surface -----------------------------------------------------


def test_public_api_exports() -> None:
    # The incremental publish surface (WAP builders + standalone primitives +
    # PublishMode) is importable from xorq.writes and listed in __all__.
    for name in (
        "PublishMode",
        "StagingStrategy",
        "publish",
        "publish_parquet",
        "make_backend_wap_expr",
        "make_parquet_wap_expr",
    ):
        assert name in xorq.writes.__all__, f"{name} missing from __all__"
        assert hasattr(xorq.writes, name), f"{name} not importable from xorq.writes"
