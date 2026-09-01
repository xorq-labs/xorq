"""An API-shaped fixture backend: test support, never a declared entry point.

This is the in-tree consumer of the plugin contract an out-of-tree REST
backend relies on, so core can pin that contract without shipping any vendor
integration:

- credentials arrive as env var references and live in the Profile;
- static `_secret_keys` is the only secret-key declaration, made live by the
  tier-2 class read on the imported backend -- there is deliberately no
  `con_name_to_secret_keys` mirror entry, because an out-of-tree backend can
  never have one;
- deferred reads capture a client built from `expr_safe_profile_kwargs()`,
  never from the resolved state `do_connect` received.

Tests install it as a plugin via `installed_mid_process(..., module=__name__)`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from attr import frozen
from attr.validators import instance_of

import xorq.common.exceptions as com
from xorq import __version__
from xorq.common.utils.attr_utils import secret_field
from xorq.common.utils.env_utils import maybe_substitute_env_var
from xorq.vendor import ibis
from xorq.vendor.ibis.backends import (
    BaseBackend,
    NoUrl,
)


if TYPE_CHECKING:
    import pyarrow as pa

    import xorq.vendor.ibis.expr.operations as ops
    import xorq.vendor.ibis.expr.schema as sch
    import xorq.vendor.ibis.expr.types as ir


records_schema_in = ibis.schema({"query": "string"})
records_schema_out = ibis.schema({"id": "int64", "payload": "string"})


@frozen
class FakeApiClient:
    """A stand-in for a REST client: fields may hold env var references,
    resolved per call, and never printed by repr."""

    username = secret_field(validator=instance_of(str))
    secret = secret_field(validator=instance_of(str))

    @property
    def _auth(self) -> tuple[str, str]:
        return (
            maybe_substitute_env_var(self.username),
            maybe_substitute_env_var(self.secret),
        )

    def fetch(self, query: str) -> pd.DataFrame:
        # deterministic and offline; resolving _auth is the point where a
        # missing env var fails, as it would on a real request
        self._auth
        return pd.DataFrame(
            {"id": [0], "payload": [json.dumps({"query": query})]}
        ).astype({"id": "Int64", "payload": "string"})

    def fetch_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            (self.fetch(row.query) for row in df.itertuples(index=False)),
            ignore_index=True,
        )


class Backend(BaseBackend, NoUrl):
    name = "fakeapi"
    dialect = None
    _secret_keys = ("secret",)

    @property
    def version(self) -> str:
        return __version__

    def do_connect(
        self, *, username: str | None = None, secret: str | None = None
    ) -> None:
        if username is None or secret is None:
            raise com.XorqError("fakeapi backend requires username and secret")
        self._client = FakeApiClient(username=username, secret=secret)

    def disconnect(self) -> None:
        pass

    def read_records(self, query: str) -> ir.Table:
        import xorq.api as xo  # noqa: PLC0415
        from xorq.expr.relations import flight_udxf  # noqa: PLC0415

        client = FakeApiClient(**self.expr_safe_profile_kwargs())
        return xo.memtable(
            ({"query": query},),
            name="fakeapi_params",
        ).pipe(
            flight_udxf(
                process_df=client.fetch_batch,
                maybe_schema_in=records_schema_in,
                maybe_schema_out=records_schema_out,
                name="FakeApiRecords",
            )
        )

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        return self._filter_with_like(["records"], like)

    def table(
        self, name: str, /, *, database: str | None = None, **params: Any
    ) -> ir.Table:
        if name != "records":
            raise com.XorqError(f"fakeapi backend has no resource {name!r}")
        return self.read_records(**params)

    def create_table(
        self,
        name: str,
        obj: pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        raise com.XorqError("the fakeapi backend is read-only")

    def drop_table(self, name: str, *, force: bool = False) -> None:
        raise com.XorqError("the fakeapi backend is read-only")

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        raise com.XorqError("the fakeapi backend is read-only")

    def drop_view(self, name: str, *, force: bool = False) -> None:
        raise com.XorqError("the fakeapi backend is read-only")

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        return False
