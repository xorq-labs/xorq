from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.ibis_yaml.combine import (
    _build_join_predicates,
    join_builds,
    join_exprs,
    union_exprs,
)
from xorq.ibis_yaml.compiler import build_expr, load_expr


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
def shared_parquet_path(tmp_path: Path) -> str:
    path = str(tmp_path / "shared.parquet")
    pd.DataFrame({"id": [1, 2, 3], "a": ["x", "y", "z"]}).to_parquet(path)
    return path


@pytest.fixture
def file_backed_left(shared_parquet_path: str) -> xo.Expr:
    # A fresh `xo.duckdb.connect()` -- same profile params as the one in
    # `file_backed_right`, but a distinct connection object (this is what
    # two independent `load_expr()` calls of the same source look like).
    return deferred_read_parquet(shared_parquet_path, xo.duckdb.connect(), table_name="t")


@pytest.fixture
def file_backed_right(shared_parquet_path: str) -> xo.Expr:
    return deferred_read_parquet(
        shared_parquet_path, xo.duckdb.connect(), table_name="t"
    ).rename(b="a")


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


def test_join_exprs_same_profile_in_memory_data_raises(
    datafusion_left: xo.Expr, datafusion_right: xo.Expr
) -> None:
    # Two separate `xo.datafusion.connect()` calls with matching params share
    # a `Profile` (idx aside), but each `.create_table(...)` registers data
    # only on its own session -- same profile does not imply same data here,
    # so rebind_backends (on by default) correctly declines to touch it and
    # this still raises.
    with pytest.raises(ValueError, match="different backend"):
        join_exprs(datafusion_left, datafusion_right, on="id")


def test_join_exprs_same_profile_file_backed_rebinds_by_default(
    file_backed_left: xo.Expr, file_backed_right: xo.Expr
) -> None:
    # Same scenario as above, but reading a real file: same profile really
    # does mean the same physical data, so this is exactly the case
    # `rebind_backends` (on by default) fixes.
    joined = join_exprs(file_backed_left, file_backed_right, on="id")
    result = joined.execute()
    assert set(result.columns) == {"id", "a", "b"}
    assert len(result) == 3


def test_join_builds_rebind_survives_colliding_saved_idx(
    shared_parquet_path: str, tmp_path: Path
) -> None:
    """`Profile.idx` is a session-local counter, but a *single-source* build
    always canonicalizes its one backend to idx=0 before serializing (see
    `normalize_profiles`) -- so any two independently-built, single-source
    builds are saved with the *same* idx=0 regardless of their actual
    content. Loading both back reconstructs two backends that: (a) may
    coincidentally share idx=0, and (b) are grouped for rebinding purely by
    content (idx excluded from the comparison already), not by idx. This
    pins that the merge produces exactly one surviving profile -- no
    duplicate/colliding entries -- and the build round-trips correctly.
    """
    left = deferred_read_parquet(shared_parquet_path, xo.duckdb.connect(), table_name="t")
    right = deferred_read_parquet(
        shared_parquet_path, xo.duckdb.connect(), table_name="t"
    ).rename(b="a")
    left_path = build_expr(left, builds_dir=tmp_path / "builds")
    right_path = build_expr(right, builds_dir=tmp_path / "builds")

    # Confirm the premise: both independently saved with idx=0.
    for path in (left_path, right_path):
        assert "idx: 0" in (path / "profiles.yaml").read_text()

    result_path = join_builds(left_path, right_path, on="id", builds_dir=tmp_path / "joined")
    profiles_text = (result_path / "profiles.yaml").read_text()
    assert profiles_text.count("idx:") == 1

    result = load_expr(result_path).execute()
    assert set(result.columns) == {"id", "a", "b"}
    assert len(result) == 3


def test_join_exprs_no_rebind_backends_still_raises(
    file_backed_left: xo.Expr, file_backed_right: xo.Expr
) -> None:
    with pytest.raises(ValueError, match="different backend"):
        join_exprs(file_backed_left, file_backed_right, on="id", rebind_backends=False)


def test_join_exprs_into_backend_remote_table_rebinds_by_default(
    shared_parquet_path: str,
) -> None:
    """A `RemoteTable` (from `.into_backend(...)`) sharing a profile with
    another source is safe to rebind, same as a plain `Read`: `into_backend`
    is lazy (stores `source`/`remote_expr`, no registration happens until
    execution), so repointing `.source` onto a same-profile connection is
    equivalent to letting it register there in the first place. Confirmed
    with the actual hub backend (`xo.connect()`, not a bare `xo.datafusion
    .connect()`, which doesn't implement `read_record_batches` at all and
    would fail identically with zero rebinding involved)."""
    left = deferred_read_parquet(shared_parquet_path, xo.connect(), table_name="t")
    right_raw = deferred_read_parquet(
        shared_parquet_path, xo.duckdb.connect(), table_name="t"
    ).rename(b="a")
    right = right_raw.into_backend(xo.connect(), name="t_remote")

    joined = join_exprs(left, right, on="id")
    result = joined.execute()
    assert set(result.columns) == {"id", "a", "b"}
    assert len(result) == 3


def test_join_exprs_into_backend_no_rebind_backends_still_raises(
    shared_parquet_path: str,
) -> None:
    left = deferred_read_parquet(shared_parquet_path, xo.connect(), table_name="t")
    right = deferred_read_parquet(
        shared_parquet_path, xo.duckdb.connect(), table_name="t"
    ).into_backend(xo.connect(), name="t_remote")

    with pytest.raises(ValueError, match="different backend"):
        join_exprs(left, right, on="id", rebind_backends=False)


def test_union_exprs_different_backend_classes_raises(
    duckdb_left: xo.Expr, datafusion_right: xo.Expr
) -> None:
    # Backend check runs before the schema check, so mismatched columns here
    # (amount/name) don't matter -- the backend error fires first.
    with pytest.raises(ValueError, match="different backend"):
        union_exprs(duckdb_left, datafusion_right)


def test_union_exprs_same_profile_in_memory_data_raises(
    datafusion_left: xo.Expr, datafusion_right: xo.Expr
) -> None:
    with pytest.raises(ValueError, match="different backend"):
        union_exprs(datafusion_left, datafusion_right)


def test_union_exprs_same_profile_file_backed_rebinds_by_default(
    shared_parquet_path: str,
) -> None:
    left = deferred_read_parquet(shared_parquet_path, xo.duckdb.connect(), table_name="t")
    right = deferred_read_parquet(shared_parquet_path, xo.duckdb.connect(), table_name="t")
    unioned = union_exprs(left, right)
    assert unioned.count().execute() == 6


def test_union_exprs_no_rebind_backends_still_raises(
    shared_parquet_path: str,
) -> None:
    left = deferred_read_parquet(shared_parquet_path, xo.duckdb.connect(), table_name="t")
    right = deferred_read_parquet(shared_parquet_path, xo.duckdb.connect(), table_name="t")
    with pytest.raises(ValueError, match="different backend"):
        union_exprs(left, right, rebind_backends=False)
