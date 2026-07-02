"""Routing + changeset validation for incremental WAP publish.

ADR-0017 / XOR-444, Phase 1. The per-strategy publish functions are covered in
later phases; here we pin the ``PublishMode`` -> ``PublishStrategy`` routing and
the build-time ``_validate`` contract. Backend-routing cases skip when the
backend (or its driver) is not installed.
"""

from __future__ import annotations

import pytest

import xorq.writes
from xorq.writes.enums import PublishMode, PublishStrategy, StagingStrategy
from xorq.writes.publish import _merge_query, _validate, resolve_strategy
from xorq.writes.wap import make_backend_wap_expr


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


def test_merge_query_quotes_and_transpiles_identifiers() -> None:
    # Quoting/escaping come from the sqlglot AST (quoted=True), and identifiers
    # transpile per dialect — double-quote escaping on postgres, backticks on
    # databricks — with no hand-rolled quoter.
    query = _merge_query(
        "final", "staging", ["id"], ["id", 'we"ird'], PublishMode.UPSERT
    )
    assert '"we""ird" = "s"."we""ird"' in query.sql(dialect="postgres")
    assert '`we"ird` = `s`.`we"ird`' in query.sql(dialect="databricks")


def test_merge_query_parenthesizes_not_delete_condition() -> None:
    # The not-delete condition lands after the implicit AND in `WHEN MATCHED AND
    # <cond>`; without explicit parens the OR would bind looser than that AND,
    # silently changing merge semantics ((MATCHED AND a) OR b).
    sql = _merge_query(
        "final", "staging", ["id"], ["id", "b", "_op"], PublishMode.MERGE
    ).sql(dialect="duckdb")
    assert """AND ("s"."_op" <> 'D' OR "s"."_op" IS NULL)""" in sql


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
