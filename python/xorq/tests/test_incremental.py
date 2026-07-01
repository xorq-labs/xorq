"""Routing + changeset validation for incremental WAP publish.

ADR-0017 / XOR-444, Phase 1. The per-strategy publish functions are covered in
later phases; here we pin the ``PublishMode`` -> ``PublishStrategy`` routing and
the build-time ``_validate`` contract. Backend-routing cases skip when the
backend (or its driver) is not installed.
"""

from __future__ import annotations

import pytest

import xorq.writes
from xorq.writes.enums import PublishMode, PublishStrategy
from xorq.writes.publish import _merge_query, _q, _validate, resolve_strategy


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


# --- SQL generation: identifier quoting + key-only tables -------------------


def test_q_escapes_embedded_quote() -> None:
    assert _q("plain") == '"plain"'
    assert _q('a"b') == '"a""b"'


def test_merge_query_key_only_omits_update() -> None:
    # No non-key columns -> no SET clause to emit; the MATCHED-UPDATE branch is
    # omitted so the MERGE stays valid on engines that reject UPDATE without SET.
    sql = _merge_query("final", "staging", ["id"], ["id"], PublishMode.UPSERT).sql(
        dialect="postgres"
    )
    assert "UPDATE" not in sql.upper()
    assert "WHEN NOT MATCHED" in sql.upper()


# --- public API surface -----------------------------------------------------


def test_public_api_exports() -> None:
    # The incremental publish surface (WAP builders + standalone primitives +
    # PublishMode) is importable from xorq.writes and listed in __all__.
    for name in (
        "PublishMode",
        "publish",
        "publish_parquet",
        "make_backend_wap_expr",
        "make_parquet_wap_expr",
    ):
        assert name in xorq.writes.__all__, f"{name} missing from __all__"
        assert hasattr(xorq.writes, name), f"{name} not importable from xorq.writes"
