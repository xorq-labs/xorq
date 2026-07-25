"""A REST API as a xorq backend, driven by a declarative config.

Two ways to use it (open question 4's resolution — "identity: always
folded; residence: per-API packaging choice"):

- **curated** (config in code): subclass with a class-level ``config`` and
  its own entry point; the profile stays credentials-shaped.
- **self-service** (config in profile): connect the ``rest`` entry point
  with ``config=<plain dict>`` — ``Profile.from_con`` captures it, so the
  saved profile carries the whole API definition. Config-expressible APIs
  only: ``fetch_override`` cannot ride in a profile.

In both modes resources are path-less ``Read`` ops (ADR-0017) whose
identity folds the per-resource config content hash
(``normalize_read_source_identity``).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import xorq.common.exceptions as com
from xorq import __version__
from xorq.backends.pandas import Backend as PandasBackend
from xorq.backends.rest.config import (
    ResourceConfig,
    RestBackendConfig,
)
from xorq.backends.rest.engines import (
    AUTH_APPLIERS,
    Engine,
    FetchOverrideEngine,
    NativeEngine,
)
from xorq.backends.rest.paginators import PAGINATORS
from xorq.common.utils.file_utils import normalize_read_source_identity


if TYPE_CHECKING:
    import pyarrow as pa

    import xorq.vendor.ibis.expr.schema as sch
    import xorq.vendor.ibis.expr.types as ir
    from xorq.expr.relations import Read


__all__ = [
    "Backend",
    "RestBackend",
]


class RestBackend(PandasBackend):
    name = "rest"
    config: RestBackendConfig | None = None
    # engine extension registries: subclasses extend by merging over these
    # (e.g. `paginators = {**RestBackend.paginators, "acme.cursor": Cls}`);
    # `make_engine` threads them into the default engine
    paginators = PAGINATORS
    auth_appliers = AUTH_APPLIERS

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Profile.from_con and make_read_kwargs flatten a var-keyword bucket
        # by its literal parameter name, "kwargs". A subclass declaring
        # `**creds` would silently mis-shape profiles/read_kwargs; turn the
        # docstring contract into a definition-time error.
        super().__init_subclass__(**kwargs)
        for method_name in ("do_connect", "fetch_resource"):
            fn = cls.__dict__.get(method_name)
            if fn is None:
                continue
            var_keyword = next(
                (
                    p
                    for p in inspect.signature(fn).parameters.values()
                    if p.kind is inspect.Parameter.VAR_KEYWORD
                ),
                None,
            )
            if var_keyword is not None and var_keyword.name != "kwargs":
                raise TypeError(
                    f"{cls.__name__}.{method_name} declares **{var_keyword.name}; "
                    "the var-keyword bucket must be named **kwargs (Profile."
                    "from_con and make_read_kwargs flatten it by that name)"
                )

    @classmethod
    def register_options(cls) -> None:
        # BasePandasBackend brings an Options class, but the vendored config
        # declares per-backend options entries and has none for rest backends
        pass

    @classmethod
    def _get_secret_keys(cls, kwargs: dict | None = None) -> tuple[str, ...]:
        """Secret keys for `check_for_exposed_secrets`: from the config in
        the kwargs being checked (self-service), else the class config
        (curated), else nothing."""
        config = RestBackendConfig.maybe_from_dict(
            (kwargs or {}).get("config") or cls.config
        )
        if config is None:
            return ()
        return config.auth.effective_secret_fields

    @property
    def version(self) -> str:
        return __version__

    def do_connect(
        self,
        *,
        config: RestBackendConfig | dict | None = None,
        **kwargs: str,
    ) -> None:
        """Connect with credentials named by the config's AuthConfig.

        Pass env var references (``token="${MY_TOKEN}"``) rather than raw
        values; raw secret values are rejected at `Profile.save` and at
        expression construction. (The var-kwargs bucket must be named
        ``kwargs``: `Profile.from_con` flattens that name into the profile's
        kwargs_tuple.)
        """
        credentials = kwargs
        config = RestBackendConfig.maybe_from_dict(config) or type(self).config
        if config is None:
            raise com.XorqError(
                f"{self.name} backend requires a config (curated subclasses "
                "set one in code; the rest backend accepts config=...)"
            )
        if config.auth.kind not in self.auth_appliers:
            raise com.XorqError(
                f"{self.name} backend has no auth applier for kind "
                f"{config.auth.kind!r}; available: {sorted(self.auth_appliers)} "
                "(declare one in the class-level auth_appliers registry)"
            )
        missing = tuple(
            name
            for name in config.auth.fields
            if credentials.get(name) is None and name not in config.auth.optional_fields
        )
        if missing:
            raise com.XorqError(
                f"{self.name} backend requires {', '.join(missing)} to connect"
            )
        unknown = tuple(set(credentials) - set(config.auth.fields))
        if unknown:
            raise com.XorqError(
                f"unknown credential kwargs {unknown}; "
                f"config.auth.fields = {config.auth.fields}"
            )
        super().do_connect()
        self._config = config
        self._credentials = dict(credentials)
        self._engine = self.make_engine()

    def make_engine(self) -> Engine:
        """The engine seam: the default is the native paginator engine,
        constructed with this backend's extension registries. Alternative
        engines (e.g. dlt) override here; the obligation is engine
        equivalence — same config, any engine, same rows."""
        return NativeEngine(
            paginators=self.paginators, auth_appliers=self.auth_appliers
        )

    def _engine_for(self, resource_config: ResourceConfig) -> Engine:
        if resource_config.fetch_override is not None:
            return FetchOverrideEngine(self)
        return self._engine

    @property
    def current_config(self) -> RestBackendConfig:
        return self._config

    def read_identity_parts(self, read: Read) -> tuple:
        """This backend's contribution to path-less Read identity
        (`normalize_read_source_identity` delegates here): the per-resource
        config content hash, so editing a resource's declarative config
        changes build/cache hashes without invalidating siblings."""
        resource = self.current_config.get_resource(dict(read.read_kwargs)["resource"])
        return (("config", resource.content_hash),)

    # -- resource surface ---------------------------------------------------

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        return self._filter_with_like(
            sorted(set(self.current_config.resource_names) | set(self.dictionary)),
            like,
        )

    def get_schema(self, table_name: str, *, database: str | None = None) -> sch.Schema:
        if table_name in self.dictionary:
            return super().get_schema(table_name, database=database)
        try:
            return self.current_config.get_resource(table_name).schema
        except KeyError:
            raise com.XorqError(
                f"{self.name} backend has no resource or table {table_name!r}; "
                f"available: {self.list_tables()}"
            ) from None

    def table(
        self, name: str, schema: sch.Schema | None = None, **params: str
    ) -> ir.Table:
        if name in self.dictionary:
            if params:
                raise com.XorqError(
                    f"{name!r} is a fetched table; params are only valid for "
                    f"resources {self.current_config.resource_names}"
                )
            return super().table(name, schema=schema)
        if name not in self.current_config.resource_names:
            raise com.XorqError(
                f"{self.name} backend has no resource or table {name!r}; "
                f"available: {self.list_tables()}"
            )
        return self.read(name, **params)

    def read(
        self, resource: str, table_name: str | None = None, **params: str
    ) -> ir.Table:
        """Deferred read of a resource: construction never fetches."""
        from xorq.common.utils.defer_utils import make_read_kwargs  # noqa: PLC0415
        from xorq.expr.relations import Read  # noqa: PLC0415
        from xorq.ibis_yaml.normalize_registry import validate  # noqa: PLC0415
        from xorq.vendor.ibis.backends.profiles import (  # noqa: PLC0415
            check_for_exposed_secrets,
        )
        from xorq.vendor.ibis.util import gen_name  # noqa: PLC0415

        try:
            resource_config = self.current_config.get_resource(resource)
        except KeyError:
            raise com.XorqError(
                f"{self.name} backend has no resource or table {resource!r}; "
                f"available: {self.list_tables()}"
            ) from None
        missing = tuple(
            name for name in resource_config.required_params if name not in params
        )
        if missing:
            raise com.XorqError(f"resource {resource!r} requires params {missing}")
        unknown = tuple(set(params) - set(resource_config.param_names))
        if unknown:
            raise com.XorqError(
                f"resource {resource!r} does not accept params {unknown}; "
                f"available: {resource_config.param_names}"
            )
        # the Read op serializes this backend's profile into build artifacts,
        # so secret values must be env var references, not raw values
        check_for_exposed_secrets(self.name, self._profile.kwargs_dict)
        validate(normalize_read_source_identity)
        table_name = table_name or gen_name("xorq-fetch_resource")
        read_kwargs = make_read_kwargs(
            self.fetch_resource, resource=resource, table_name=table_name, **params
        )
        return Read(
            method_name="fetch_resource",
            name=table_name,
            schema=resource_config.schema,
            source=self,
            read_kwargs=read_kwargs,
            normalize_method=normalize_read_source_identity,
        ).to_expr()

    # -- execution (Read.make_dt boundary) ----------------------------------

    def fetch_resource(
        self, resource: str, table_name: str | None = None, **kwargs: str
    ) -> ir.Table:
        """Eagerly fetch a resource and serve it as a table.

        The var-kwargs bucket must be named ``kwargs``: `make_read_kwargs`
        flattens that name into the Read op's hashable ``read_kwargs``.
        """
        from xorq.vendor.ibis.util import gen_name  # noqa: PLC0415

        resource_config = self.current_config.get_resource(resource)
        df = self._engine_for(resource_config).fetch(
            self.current_config, resource_config, kwargs, self._credentials
        )
        table_name = table_name or gen_name("xorq-fetch_resource")
        self.dictionary[table_name] = df
        self.schemas[table_name] = resource_config.schema
        return super().table(table_name)

    # -- read-only ----------------------------------------------------------

    def create_table(
        self,
        name: str,
        obj: pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        raise com.XorqError(f"the {self.name} backend is read-only")

    def drop_table(self, name: str, *, force: bool = False) -> None:
        raise com.XorqError(f"the {self.name} backend is read-only")

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        raise com.XorqError(f"the {self.name} backend is read-only")

    def drop_view(self, name: str, *, force: bool = False) -> None:
        raise com.XorqError(f"the {self.name} backend is read-only")


Backend = RestBackend
