from __future__ import annotations

import pytest

import xorq.api as xo
from xorq.ibis_yaml.combine import (
    _build_join_predicates,
    join_exprs,
    union_exprs,
)


@pytest.fixture
def duckdb_left() -> xo.Expr:
    return xo.duckdb.connect().create_table(
        "left_t", {"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}
    )


@pytest.fixture
def duckdb_right() -> xo.Expr:
    return xo.duckdb.connect().create_table(
        "right_t", {"id": [1, 2, 4], "name": ["a", "b", "d"]}
    )


@pytest.fixture
def datafusion_left() -> xo.Expr:
    return xo.datafusion.connect().create_table(
        "left_t", {"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}
    )


@pytest.fixture
def datafusion_right() -> xo.Expr:
    return xo.datafusion.connect().create_table(
        "right_t", {"id": [1, 2, 4], "name": ["a", "b", "d"]}
    )


@pytest.fixture
def left() -> xo.Expr:
    return xo.memtable({"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}, name="left_t")


@pytest.fixture
def right() -> xo.Expr:
    return xo.memtable({"id": [1, 2, 4], "name": ["a", "b", "d"]}, name="right_t")


def test_build_join_predicates_on() -> None:
    assert _build_join_predicates("id", None, None, "inner") == ("id",)


def test_build_join_predicates_on_multiple_columns() -> None:
    assert _build_join_predicates("id, amount", None, None, "inner") == (
        "id",
        "amount",
    )


def test_build_join_predicates_left_right_on() -> None:
    assert _build_join_predicates(None, "a", "b", "inner") == (("a", "b"),)


def test_build_join_predicates_left_right_on_multiple() -> None:
    assert _build_join_predicates(None, "a,b", "c,d", "inner") == (
        ("a", "c"),
        ("b", "d"),
    )


def test_build_join_predicates_on_mutually_exclusive_with_left_on() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_join_predicates("id", "a", None, "inner")


def test_build_join_predicates_left_on_requires_right_on() -> None:
    with pytest.raises(ValueError, match="must be given together"):
        _build_join_predicates(None, "a", None, "inner")


def test_build_join_predicates_mismatched_column_counts() -> None:
    with pytest.raises(ValueError, match="column"):
        _build_join_predicates(None, "a,b", "c", "inner")


def test_build_join_predicates_none_requires_cross() -> None:
    with pytest.raises(ValueError, match="must specify --on"):
        _build_join_predicates(None, None, None, "inner")


def test_build_join_predicates_none_allowed_for_cross() -> None:
    assert _build_join_predicates(None, None, None, "cross") == ()


def test_join_exprs_inner(left: xo.Expr, right: xo.Expr) -> None:
    joined = join_exprs(left, right, on="id", how="inner")
    assert set(joined.columns) == {"id", "amount", "name"}
    assert joined.count().execute() == 2


def test_join_exprs_left_on_right_on(left: xo.Expr, right: xo.Expr) -> None:
    joined = join_exprs(left, right, left_on="id", right_on="id", how="left")
    assert joined.count().execute() == 3


def test_union_exprs_default_all(left: xo.Expr) -> None:
    other = xo.memtable({"id": [1, 2], "amount": [10.0, 20.0]}, name="dup")
    unioned = union_exprs(left, other)
    assert unioned.count().execute() == 5


def test_union_exprs_distinct(left: xo.Expr) -> None:
    other = xo.memtable({"id": [1, 2], "amount": [10.0, 20.0]}, name="dup")
    unioned = union_exprs(left, other, distinct=True)
    assert unioned.count().execute() == 3


def test_union_exprs_requires_at_least_two(left: xo.Expr) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        union_exprs(left)


def test_join_exprs_different_backend_classes_raises(
    duckdb_left: xo.Expr, datafusion_right: xo.Expr
) -> None:
    with pytest.raises(ValueError, match="different backend"):
        join_exprs(duckdb_left, datafusion_right, on="id")


def test_join_exprs_same_backend_different_profile_raises(
    datafusion_left: xo.Expr, datafusion_right: xo.Expr
) -> None:
    # Two separate `xo.datafusion.connect()` calls: same backend class, but
    # distinct connections (profile idx N vs N+1) -- not interchangeable.
    with pytest.raises(ValueError, match="different backend"):
        join_exprs(datafusion_left, datafusion_right, on="id")


def test_union_exprs_different_backend_classes_raises(
    duckdb_left: xo.Expr, datafusion_right: xo.Expr
) -> None:
    # Backend check runs before the schema check, so mismatched columns here
    # (amount/name) don't matter -- the backend error fires first.
    with pytest.raises(ValueError, match="different backend"):
        union_exprs(duckdb_left, datafusion_right)


def test_union_exprs_same_backend_different_profile_raises(
    datafusion_left: xo.Expr, datafusion_right: xo.Expr
) -> None:
    with pytest.raises(ValueError, match="different backend"):
        union_exprs(datafusion_left, datafusion_right)
