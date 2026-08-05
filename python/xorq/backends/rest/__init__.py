"""A REST API as a xorq backend, driven by a declarative config.

Two ways to use it (ADR-2215 — "identity: always folded; residence: per-API
packaging choice"):

- **curated** (config in code): subclass with a class-level ``config`` and
  its own entry point; the profile stays credentials-shaped.
- **self-service** (config in profile): connect the ``rest`` entry point
  with ``config=<plain dict>`` — ``Profile.from_con`` captures it, so the
  saved profile carries the whole API definition. Config-expressible APIs
  only: ``fetch_override`` cannot ride in a profile.

In both modes resources are path-less ``Read`` ops
(ADR-api-relations-are-pathless-read-ops) whose identity folds the per-resource
config content hash (``normalize_read_source_identity``).

Execution is composed, not inherited
(ADR-rest-resource-reads-are-lazy-datafusion-tables): the ``PandasBackend`` base
supplies Backend plumbing (profile machinery, ``do_connect``,
``_filter_with_like``) while a private owned xorq-DataFusion connection
supplies execution and storage. ``fetch_resource`` -- the ``Read.make_dt``
boundary -- registers a *lazy* table over the engine's page stream there, so
nothing is fetched until the engine pulls batches.
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
import weakref
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
from xorq.backends.xorq_datafusion import connect as xorq_datafusion_connect
from xorq.common.utils.file_utils import normalize_read_source_identity
from xorq.internal import SessionConfig


if TYPE_CHECKING:
    import pyarrow as pa
    from batchcorder import StreamCache

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
    # The static floor, mirrored in `con_name_to_secret_keys["rest"]`. The
    # authoritative answer is config-derived (`_secret_key_sources` below), but
    # a source resolves only when the kwargs actually carry a config, and a
    # process validating a hand-authored profile without one would otherwise
    # check `("password",)` alone. Self-service field names are config-defined,
    # so no static tuple can be complete -- this is the conventional set, and
    # the tiers are unioned, so it can only widen the check. Curated subclasses
    # override it from their own config.
    _secret_keys = ("token", "secret", "api_key", "access_token")
    # Where the config-defined credential kwarg names live, as data the
    # profile machinery resolves itself (`get_declared_secret_keys`): explicit
    # `secret_fields` wins, else every declared field is secret -- the same
    # rule as `AuthConfig.effective_secret_fields`, kept in that shape so the
    # two cannot diverge. Inherited by every curated subclass, whose
    # `do_connect` accepts the same `config=` override; mirrored per con_name
    # in `con_name_to_secret_key_sources`.
    _secret_key_sources = (
        ("config", "auth", "secret_fields"),
        ("config", "auth", "fields"),
    )
    # engine extension registries: subclasses extend by merging over these
    # (e.g. `paginators = {**RestBackend.paginators, "acme.cursor": Cls}`);
    # `make_engine` threads them into the default engine
    paginators = PAGINATORS
    auth_appliers = AUTH_APPLIERS
    # Bounds on the replay cache every resource read registers behind
    # (`_replay_cache`). The hot layer is what keeps RAM bounded; the disk
    # budget is a diagnosable ceiling rather than an invitation to fill the
    # volume. Class-level so an API whose reads are known-small (or
    # known-enormous) can retune them without touching the read path.
    spill_memory_capacity = 128 << 20  # 128 MiB retained in RAM
    spill_disk_capacity = 64 << 30  # 64 GiB spilled at most

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
        # Composition, not inheritance (ADR-0019): resource reads register as
        # lazy tables here, so execution and storage are DataFusion's while
        # this class stays a read-only, resource-shaped Backend. Owned and
        # private -- it is never captured in a profile or a build artifact, and
        # nothing outside this module should reach for it.
        #
        # Single-partition on purpose. Every input here is one page-wise
        # RecordBatchReader, so repartitioning parallelizes nothing it can
        # exploit; what it does do is insert a buffering shuffle in front of a
        # stream this ADR exists to keep bounded, and make row order and batch
        # boundaries a function of thread scheduling (a two-read join returned
        # its 150 rows in a different order on every run). Deterministic order
        # is worth more than parallelism on an API-latency-bound read: it is
        # what makes a `.cache()` of a rest expression reproducible rather than
        # merely correct. Parallel compute over a resource read is available
        # the documented way, by `into_backend`-ing it onto a full connection.
        self._df = xorq_datafusion_connect(
            SessionConfig().set("datafusion.execution.target_partitions", "1")
        )
        self._spill_dir = None

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
        (`normalize_read_source_identity` delegates here): two derived hashes.

        - ``api``: the API-wide contract (resolved ``base_urls`` + auth shape).
          Required for correctness in curated mode, where the profile carries
          credentials only: without it, repointing ``base_urls`` from prod to
          staging changed no hash and cached data from the old host was served
          as current data from the new one.
        - ``config``: the per-resource declarative config, so editing one
          resource changes build/cache hashes. In curated mode that also leaves
          siblings untouched; in self-service mode the config rides in the
          profile, whose hash is folded alongside these parts, so siblings move
          too (deliberate -- see `config.py`'s module docstring).
        """
        config = self.current_config
        resource = config.get_resource(dict(read.read_kwargs)["resource"])
        return (("api", config.content_hash), ("config", resource.content_hash))

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
        if schema is not None:
            raise com.XorqError(
                f"{name!r} is a resource; its schema is declared in the "
                "config and cannot be overridden at table()"
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
        """Register a resource as a *lazy* table on the owned connection.

        The ``Read.make_dt`` boundary (ADR-0019). Nothing is fetched here: the
        return value is a DataFusion table backed by a replayable cache over
        the engine's page stream, and the first HTTP request fires when the
        engine drains that cache. Paginated and ``fetch_override`` resources
        take exactly this path -- an override contributes one chunk -- so
        there is no materializing branch.

        The var-kwargs bucket must be named ``kwargs``: `make_read_kwargs`
        flattens that name into the Read op's hashable ``read_kwargs``.
        """
        from xorq.vendor.ibis.util import gen_name  # noqa: PLC0415

        resource_config = self.current_config.get_resource(resource)
        # the declared schema, which every chunk is conformed to below, so the
        # retype-only StreamCache registration never needs to project
        schema = resource_config.schema.to_pyarrow()
        table_name = table_name or gen_name("xorq-fetch_resource")
        return self._df.read_record_batches(
            self._replay_cache(self._resource_reader(resource_config, kwargs, schema)),
            table_name=table_name,
            schema=schema,
        )

    def _resource_reader(
        self,
        resource_config: ResourceConfig,
        params: dict,
        schema: pa.Schema,
    ) -> pa.ipc.RecordBatchReader:
        """A lazy reader over the resource's chunk stream, one RecordBatch per
        engine chunk (one HTTP page, for the native engine).

        Lazy end to end: ``fetch_batches`` is a generator and the batch
        conversion is a generator expression, so constructing this issues no
        request. Every batch carries exactly ``schema``'s fields, in order --
        the engine contract conforms each chunk to the declared schema, and
        ``from_pandas(..., schema=...)`` is what enforces it here.

        Empty chunks are dropped. Most paginators terminate on an empty page,
        so the last chunk of nearly every read is a zero-row frame; forwarding
        it would push a zero-row RecordBatch into the engine, which carries no
        rows (the *schema* travels on the reader, not the batch) and only adds a
        batch boundary the plan has to interleave -- observably making batch
        arrival order nondeterministic where the row-bearing batches alone are
        stable. An all-empty result is still fine: the reader declares the
        schema even with no batches at all.
        """
        import pyarrow as pa  # noqa: PLC0415

        frames = self._engine_for(resource_config).fetch_batches(
            self.current_config, resource_config, params, self._credentials
        )
        return pa.RecordBatchReader.from_batches(
            schema,
            (
                pa.RecordBatch.from_pandas(frame, schema=schema, preserve_index=False)
                for frame in frames
                if len(frame)
            ),
        )

    def _replay_cache(self, reader: pa.ipc.RecordBatchReader) -> StreamCache:
        """Wrap a one-shot reader in a replayable, disk-spilling cache.

        A bare ``pa.RecordBatchReader`` registered as a DataFusion table is
        single-scan: if the physical plan scans it twice (a self-join, one read
        referenced twice, any re-scanning plan) the second scan gets an
        exhausted reader and silently returns *no rows*. `StreamCache` ingests
        the upstream lazily and exactly once while serving independent replay
        handles, so repeated scans share one buffer.

        Replay means retention, which is the honest cost of not being silently
        wrong. It is paid to disk rather than to RAM: the hot layer is capped
        at ``spill_memory_capacity`` and ``write_policy="on_eviction"`` only
        writes when that cap is exceeded, so a result smaller than the cap
        never touches disk while a larger one keeps RAM bounded instead of
        growing with the result. ``max_readers`` is deliberately left unset
        (retain everything): the scan count is not knowable here -- unlike the
        RemoteTable path, which derives it from a *compiled* plan -- and an
        under-estimate would evict batches a later scan still needs.
        """
        from batchcorder import StreamCache  # noqa: PLC0415

        return StreamCache(
            reader,
            memory_capacity=self.spill_memory_capacity,
            disk_path=self._spill_root(),
            disk_capacity=self.spill_disk_capacity,
            write_policy="on_eviction",
        )

    def _spill_root(self) -> str:
        """The per-connection spill directory, created on first use and removed
        when this backend is garbage-collected. Each cache takes its own
        subdirectory of it, so one root per connection is enough."""
        if self._spill_dir is None:
            self._spill_dir = tempfile.mkdtemp(prefix=f"xorq-{self.name}-spill-")
            weakref.finalize(self, shutil.rmtree, self._spill_dir, True)
        return self._spill_dir

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
