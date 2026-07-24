from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import xorq.common.exceptions as com
from xorq import __version__
from xorq.backends.mixpanel.client import (
    MixpanelClient,
    engage_schema_in,
    engage_schema_out,
    export_schema_in,
    export_schema_out,
)
from xorq.vendor.ibis.backends import (
    BaseBackend,
    NoUrl,
)


if TYPE_CHECKING:
    import pyarrow as pa

    import xorq.vendor.ibis.expr.operations as ops
    import xorq.vendor.ibis.expr.schema as sch
    import xorq.vendor.ibis.expr.types as ir


__all__ = [
    "Backend",
]


resource_schemas = {
    "events": export_schema_out,
    "engage": engage_schema_out,
}


class Backend(BaseBackend, NoUrl):
    """The Mixpanel API as a read-only xorq backend.

    Resources are exposed as deferred relations built on `flight_udxf`:
    construction never fetches, and the fetcher captures only the
    connection's Profile values (env var references), never resolved
    credentials.
    """

    name = "mixpanel"
    dialect = None
    _secret_keys = ("secret",)

    @property
    def version(self) -> str:
        return __version__

    def do_connect(
        self,
        *,
        username: str | None = None,
        secret: str | None = None,
        project_id: str | int | None = None,
        region: str = "us",
    ) -> None:
        """Create a Mixpanel connection from service-account credentials.

        Pass env var references (e.g. ``secret="${MIXPANEL_SERVICE_ACCOUNT_SECRET}"``)
        rather than raw values: references keep credentials out of saved
        profiles and build artifacts, and `Profile.save` rejects a raw
        `secret`.
        """
        missing = tuple(
            name
            for name, value in (
                ("username", username),
                ("secret", secret),
                ("project_id", project_id),
            )
            if value is None
        )
        if missing:
            raise com.XorqError(
                f"mixpanel backend requires {', '.join(missing)} to connect"
            )
        # do_connect receives env-substituted values; this client serves
        # interactive/metadata calls only and is never captured in expressions
        self._client = MixpanelClient(
            username=username,
            secret=secret,
            project_id=project_id,
            region=region,
        )

    def disconnect(self) -> None:
        pass

    @property
    def _expr_client(self) -> MixpanelClient:
        from xorq.vendor.ibis.backends.profiles import (  # noqa: PLC0415
            check_for_exposed_secrets,
        )

        # expressions capture this client (and serialize it into build
        # artifacts), so secret values must be env var references; raw values
        # are rejected here, not just at Profile.save
        check_for_exposed_secrets(self.name, self._profile.kwargs_dict)
        return MixpanelClient.from_profile(self._profile)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        return self._filter_with_like(sorted(resource_schemas), like)

    def get_schema(self, table_name: str, *, database: str | None = None) -> sch.Schema:
        try:
            return resource_schemas[table_name]
        except KeyError:
            raise com.XorqError(
                f"mixpanel backend has no resource {table_name!r}; "
                f"available: {sorted(resource_schemas)}"
            ) from None

    def table(
        self, name: str, /, *, database: str | None = None, **params: str
    ) -> ir.Table:
        readers = {
            "events": self.read_events,
            "engage": self.read_engage,
        }
        try:
            reader = readers[name]
        except KeyError:
            raise com.XorqError(
                f"mixpanel backend has no resource {name!r}; "
                f"available: {sorted(readers)}"
            ) from None
        return reader(**params)

    def read_events(self, from_date: str, to_date: str) -> ir.Table:
        """Deferred raw-event export for the (inclusive, UTC) date range."""
        import xorq.api as xo  # noqa: PLC0415
        from xorq.expr.relations import flight_udxf  # noqa: PLC0415

        return xo.memtable(
            ({"from_date": from_date, "to_date": to_date},),
            name="mixpanel_export_params",
        ).pipe(
            flight_udxf(
                process_df=self._expr_client.export_batch,
                maybe_schema_in=export_schema_in,
                maybe_schema_out=export_schema_out,
                name="MixpanelExport",
            )
        )

    def read_engage(self, where: str = "", page_size: int | None = None) -> ir.Table:
        """Deferred user-profile query (all profiles when `where` is empty)."""
        import xorq.api as xo  # noqa: PLC0415
        from xorq.expr.relations import flight_udxf  # noqa: PLC0415

        process_df = self._expr_client.engage_batch
        if page_size is not None:
            # update_wrapper because make_udxf reads process_df.__name__
            process_df = functools.update_wrapper(
                functools.partial(process_df, page_size=page_size), process_df
            )
        return xo.memtable(
            ({"where": where},),
            name="mixpanel_engage_params",
        ).pipe(
            flight_udxf(
                process_df=process_df,
                maybe_schema_in=engage_schema_in,
                maybe_schema_out=engage_schema_out,
                name="MixpanelEngage",
            )
        )

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
        raise com.XorqError("the mixpanel backend is read-only")

    def drop_table(self, name: str, *, force: bool = False) -> None:
        raise com.XorqError("the mixpanel backend is read-only")

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        raise com.XorqError("the mixpanel backend is read-only")

    def drop_view(self, name: str, *, force: bool = False) -> None:
        raise com.XorqError("the mixpanel backend is read-only")

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        return False
