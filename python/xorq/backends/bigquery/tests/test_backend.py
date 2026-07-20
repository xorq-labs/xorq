from __future__ import annotations

import types
import warnings
from typing import TYPE_CHECKING

import pytest

import xorq.api as xo
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends import _get_backend_names
from xorq.backends.bigquery import Backend
from xorq.common.utils.bigquery_utils import BigQueryADBC
from xorq.vendor.ibis.backends.bigquery import Backend as IbisBigQueryBackend


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr import types as ir


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")


def _mock_con(con_kwargs: dict, credentials: object) -> Backend:
    # an unconnected Backend with just enough attributes stubbed for
    # BigQueryADBC.db_kwargs (which needs the client, project, and dataset)
    con = Backend()
    con._con_kwargs = con_kwargs
    con.data_project = "proj"
    con.billing_project = "proj"
    con.dataset = "ds"
    con.client = types.SimpleNamespace(_credentials=credentials)
    return con


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


def test_read_record_batches_rejects_invalid_mode() -> None:
    # mode is validated before the ADBC driver is touched, so no connection
    # or record batches are needed to exercise the guard
    con = Backend()
    with pytest.raises(ValueError, match="not a valid IngestMode"):
        con.read_record_batches(None, table_name="t", mode="bogus")


def test_profile_strips_runtime_objects() -> None:
    # the profile is hashed and serialized into build artifacts, so it must not
    # retain live credential/client objects; the live connection keeps them via
    # _con_kwargs. Guards against a rename of _profile_exclude_kwargs silently
    # leaking credentials into build YAML.
    class UnpicklableCred:
        def __reduce__(self):
            raise TypeError("credentials are not serializable")

    con = Backend(project_id="proj", dataset_id="ds", credentials=UnpicklableCred())
    profile_kwargs = con._profile.kwargs_dict
    for key in ("credentials", "client", "storage_client"):
        assert key not in profile_kwargs
    assert profile_kwargs["project_id"] == "proj"
    # the live connection still carries the credential for reconnection
    assert "credentials" in con._con_kwargs
    # hashing must not touch the credential object (the vendored db_identity hash
    # would embed it)
    assert isinstance(hash(con), int)


def test_hash_eq_contract_holds_with_stripped_profile() -> None:
    # two backends that compare equal (share a profile) must hash equal even when
    # their live credential objects differ — the vendored db_identity hash embeds
    # the credential and would break this
    con1 = Backend(project_id="proj", dataset_id="ds", credentials=object())
    con2 = Backend(project_id="proj", dataset_id="ds", credentials=object())
    # independent instances get distinct profile idx, so they are not equal
    assert con1 != con2
    # a shared profile (as reconnect / Profile.from_con would preserve) makes
    # them equal, and the hash must follow
    con2._profile = con1._profile
    assert con1 == con2
    assert hash(con1) == hash(con2)


def test_adbc_warns_when_dropping_explicit_non_user_credentials() -> None:
    # an explicit non-user credential can't be forwarded to the ADBC driver, so
    # ingest falls back to ADC discovery — warn so it isn't a silent identity swap
    cred = types.SimpleNamespace(
        service_account_email="svc@proj.iam.gserviceaccount.com"
    )
    con = _mock_con({"credentials": cred}, cred)
    with pytest.warns(UserWarning, match="cannot reuse the explicit credentials"):
        db_kwargs = BigQueryADBC(con).db_kwargs
    assert db_kwargs["adbc.bigquery.sql.auth_type"].endswith("auth_bigquery")


def test_adbc_warns_for_prebuilt_client_credentials() -> None:
    # connecting via a prebuilt `client=` carries the credential under the
    # `client` kwarg, not `credentials`; the drop must still warn
    cred = types.SimpleNamespace(
        service_account_email="svc@proj.iam.gserviceaccount.com"
    )
    con = _mock_con({"client": object()}, cred)
    with pytest.warns(UserWarning, match="cannot reuse the explicit credentials"):
        BigQueryADBC(con).db_kwargs


def test_adbc_quiet_for_adc_credentials() -> None:
    # ADC-authenticated backends carry no `credentials`/`client` kwarg, so the
    # fallback to ADC discovery is expected and must not warn
    cred = types.SimpleNamespace(
        service_account_email="svc@proj.iam.gserviceaccount.com"
    )
    con = _mock_con({}, cred)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BigQueryADBC(con).db_kwargs


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
