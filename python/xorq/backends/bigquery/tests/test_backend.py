from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import xorq.api as xo
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends import _get_backend_names
from xorq.backends.bigquery import Backend
from xorq.vendor.ibis.backends.bigquery import Backend as IbisBigQueryBackend


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr import types as ir


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")

from google.auth.credentials import (  # noqa: E402
    AnonymousCredentials,
    Credentials,
)
from xorq_dasher import fqn  # noqa: E402

from xorq.common.utils.dasher import tokenize  # noqa: E402
from xorq.common.utils.dasher._gap_rules import (  # noqa: E402
    normalize_google_credentials,
)


def test_backend_registered() -> None:
    assert "bigquery" in _get_backend_names()


def test_backend_subclasses_vendored() -> None:
    assert issubclass(Backend, IbisBigQueryBackend)


def test_api_exposes_backend() -> None:
    assert xo.bigquery.name == "bigquery"
    assert callable(xo.bigquery.connect)
    assert callable(xo.bigquery.compile)


def test_compile_offline() -> None:
    # compilation needs no live connection or credentials
    con = Backend()
    t = ibis.table({"a": "int64", "b": "string"}, name="t")
    sql = con.compile(t.select(t.a + 1))
    assert "SELECT" in sql
    assert "`a`" in sql


def test_compile_aggregate_offline() -> None:
    con = Backend()
    t = ibis.table({"playerID": "string", "G": "int64"}, name="batting")
    sql = con.compile(t.group_by("playerID").agg(total=t.G.sum()))
    assert "GROUP BY" in sql
    assert "SUM(`t0`.`G`)" in sql


def test_compile_join_offline() -> None:
    con = Backend()
    batting = ibis.table({"playerID": "string", "yearID": "int64"}, name="batting")
    awards = ibis.table({"playerID": "string", "awardID": "string"}, name="awards")
    sql = con.compile(batting.join(awards, "playerID").select("playerID", "awardID"))
    assert "INNER JOIN" in sql


@pytest.mark.parametrize(
    "op",
    (
        pytest.param(ops.Project, id="project"),
        pytest.param(ops.Filter, id="filter"),
        pytest.param(ops.Sort, id="sort"),
        pytest.param(ops.Aggregate, id="aggregate"),
        pytest.param(ops.JoinChain, id="join-chain"),
        pytest.param(ops.Cast, id="cast"),
        pytest.param(ops.Sum, id="sum"),
        pytest.param(ops.Mean, id="mean"),
    ),
)
def test_has_operation(op: type[ops.Value]) -> None:
    # has_operation is a compile-time property; no connection required
    assert Backend().has_operation(op)


@pytest.mark.bigquery
def test_read_parquet_and_execute(batting: ir.Table) -> None:
    result = batting.filter(batting.yearID == 2015).select("playerID").execute()
    assert len(result) > 0
    assert list(result.columns) == ["playerID"]


@pytest.mark.bigquery
def test_google_credentials_normalizer() -> None:
    # the bigquery connection profile carries a google credentials object;
    # without the registered normalizer, tokenizing it (via Profile.hash_name)
    # raises "No normalizer registered for ...Credentials"

    # the FQN string registered in _EXTRA_RULES must match the real base class
    assert fqn(Credentials) == "google.auth.credentials.Credentials"

    token = normalize_google_credentials(AnonymousCredentials())
    assert token[0] == "google.auth.credentials.Credentials"
    assert token[1] == "AnonymousCredentials"

    # tokenizes cleanly and stably through the HASHER
    assert tokenize(AnonymousCredentials()) == tokenize(AnonymousCredentials())
