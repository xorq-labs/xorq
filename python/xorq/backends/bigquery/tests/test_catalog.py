from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import xorq.api as xo
from xorq.catalog.backend import GitBackend
from xorq.catalog.catalog import Catalog
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis.expr import operations as ops


if TYPE_CHECKING:
    from pathlib import Path

    from xorq.backends.bigquery import Backend
    from xorq.vendor.ibis.expr import types as ir


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    # catalog round-trips rebind to a fresh backend, so row order is not
    # guaranteed; sort by every column to compare on content alone
    return (
        df.sort_values(by=list(df.columns))
        .reset_index(drop=True)
        .reindex(sorted(df.columns), axis=1)
    )


def _assert_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(_norm(actual), _norm(expected))


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    # a plain-git catalog rooted in the test's tmp dir; storage backend is
    # pinned to git deliberately — the annex/pointer matrix is orthogonal to
    # what a bigquery-backed *source* exercises and is covered elsewhere
    repo = Catalog.init_repo_path(tmp_path / "bq-catalog")
    return Catalog(backend=GitBackend(repo=repo))


@pytest.fixture
def source_expr(batting: ir.Table) -> ir.Table:
    # a small, deterministic projection over a live bigquery table
    return batting.select("playerID", "yearID", "G")


@pytest.fixture
def transform_expr(source_expr: ir.Table) -> ir.Table:
    # an unbound transform shaped by the source schema (mirrors the catalog
    # bind tests): filter to one season, keep two columns
    schema = source_expr.schema()
    unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
    return unbound.filter(unbound.yearID == 2015).select("playerID", "G")


# ---------------------------------------------------------------------------
# add -> load -> execute round-trip
# ---------------------------------------------------------------------------


@pytest.mark.bigquery
def test_add_load_execute_roundtrip(
    catalog: Catalog, source_expr: ir.Table, con: Backend
) -> None:
    # the core feature: a bigquery-backed expr serializes into a catalog
    # archive and reloads into an expr that executes to the same data
    entry = catalog.add(source_expr)
    assert entry.name in catalog.list()

    loaded = catalog.load(entry.name, con=con)
    _assert_equal(loaded.execute(), source_expr.execute())


@pytest.mark.bigquery
def test_load_reconnects_from_profile_without_con(
    catalog: Catalog, source_expr: ir.Table
) -> None:
    # con=None rebuilds the source backend from the serialized profile. The
    # bigquery profile strips credentials but keeps project_id/dataset_id, so
    # the reconstructed backend must reconnect via ADC and resolve the table.
    entry = catalog.add(source_expr)
    loaded = catalog.load(entry.name)
    _assert_equal(loaded.execute(), source_expr.execute())


# ---------------------------------------------------------------------------
# aliasing
# ---------------------------------------------------------------------------


@pytest.mark.bigquery
def test_load_by_alias(catalog: Catalog, source_expr: ir.Table, con: Backend) -> None:
    entry = catalog.add(source_expr, aliases=("batting-src",))
    assert "batting-src" in catalog.list_aliases()

    resolved = catalog.get_catalog_entry("batting-src", maybe_alias=True)
    assert resolved.name == entry.name

    loaded = catalog.load("batting-src", con=con)
    _assert_equal(loaded.execute(), source_expr.execute())


@pytest.mark.bigquery
def test_add_alias_after_the_fact(
    catalog: Catalog, source_expr: ir.Table, con: Backend
) -> None:
    entry = catalog.add(source_expr)
    catalog.add_alias(entry.name, "late-alias")

    assert "late-alias" in catalog.list_aliases()
    loaded = catalog.load("late-alias", con=con)
    _assert_equal(loaded.execute(), source_expr.execute())


# ---------------------------------------------------------------------------
# composing (bind)
# ---------------------------------------------------------------------------


@pytest.mark.bigquery
def test_bind_source_and_transform_executes(
    catalog: Catalog,
    source_expr: ir.Table,
    transform_expr: ir.Table,
    con: Backend,
) -> None:
    # the most common compose usage: a cataloged bigquery source threaded
    # through a cataloged unbound transform
    source_entry = catalog.add(source_expr, aliases=("src",))
    transform_entry = catalog.add(transform_expr, aliases=("trn",))

    bound = catalog.bind(source_entry, transform_entry, con=con)
    expected = source_expr.filter(source_expr.yearID == 2015).select("playerID", "G")
    _assert_equal(bound.execute(), expected.execute())


@pytest.mark.bigquery
def test_bind_variadic_chain(
    catalog: Catalog,
    source_expr: ir.Table,
    transform_expr: ir.Table,
    con: Backend,
) -> None:
    # chaining two transforms: bind(source, t1, t2)
    source_entry = catalog.add(source_expr)
    t1_entry = catalog.add(transform_expr)

    t1_schema = xo.Schema({"playerID": "string", "G": "int64"})
    unbound2 = ops.UnboundTable(name="ph2", schema=t1_schema).to_expr()
    t2_entry = catalog.add(unbound2.filter(unbound2.G > 0))

    bound = catalog.bind(source_entry, t1_entry, t2_entry, con=con)
    expected = (
        source_expr.filter(source_expr.yearID == 2015)
        .select("playerID", "G")
        .filter(lambda t: t.G > 0)
    )
    _assert_equal(bound.execute(), expected.execute())


@pytest.mark.bigquery
def test_bind_result_roundtrips_into_catalog(
    catalog: Catalog,
    source_expr: ir.Table,
    transform_expr: ir.Table,
    con: Backend,
) -> None:
    # a bound (composed) result can itself be cataloged and reloaded
    source_entry = catalog.add(source_expr)
    transform_entry = catalog.add(transform_expr)

    bound = catalog.bind(source_entry, transform_entry, con=con)
    bound_entry = catalog.add(bound, aliases=("bound-result",))
    assert bound_entry.kind == ExprKind.Composed
    assert len(bound_entry.composed_from) == 2

    reloaded = catalog.load("bound-result", con=con)
    _assert_equal(reloaded.execute(), bound.execute())


# ---------------------------------------------------------------------------
# archive export / import portability
# ---------------------------------------------------------------------------


@pytest.mark.bigquery
def test_get_zip_export_import_roundtrip(
    catalog: Catalog, source_expr: ir.Table, con: Backend, tmp_path: Path
) -> None:
    # a bigquery-backed entry exports to a portable archive that imports into a
    # fresh catalog and still loads/executes — proving the serialized profile
    # (no credentials) is self-contained across catalogs
    entry = catalog.add(source_expr)
    zip_path = catalog.get_zip(entry.name, dir_path=tmp_path)
    assert zip_path.exists()

    other_repo = Catalog.init_repo_path(tmp_path / "other-catalog")
    other = Catalog(backend=GitBackend(repo=other_repo))
    imported = other.add(zip_path)

    loaded = other.load(imported.name, con=con)
    _assert_equal(loaded.execute(), source_expr.execute())
