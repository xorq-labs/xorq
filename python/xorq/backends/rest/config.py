"""The declarative REST contract: what a REST API *is*, engine-independent.

The config is the stable contract (see plans/udxf-source-api-backend.md,
Phase 3): extraction engines (native paginators, later dlt) are swappable
behind it. Three deliberate omissions vs dlt's RESTAPIConfig:

1. no incremental/cursor block — incrementality is a param in ``read_kwargs``
   (``ParamSpec(kind="range")``), never backend state;
2. auth names profile *fields*, never values — resolution happens at
   execution via the profile/env machinery (ADR-0016);
3. ``fetch_override`` — config-first, code-fallback; bespoke resources
   (NDJSON export, nonstandard pagination) supply a callable. Overrides are
   code-path-only: they cannot ride in a self-service profile.

Identity: each resource carries a ``content_hash`` over its *declarative*
fields (``fetch_override`` excluded — code is refactorable, per ADR-0017's
line), folded into ``normalize_read_source_identity`` so editing a
resource's path/params changes build and cache hashes (ADR-0015) without
invalidating sibling resources.
"""

from __future__ import annotations

from types import MappingProxyType

from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    in_,
    instance_of,
    is_callable,
    optional,
)

from xorq.vendor import ibis
from xorq.vendor.ibis.expr.schema import Schema


auth_kinds = ("basic", "bearer", "none")

_dtype_to_pandas = MappingProxyType(
    {
        "int8": "Int8",
        "int16": "Int16",
        "int32": "Int32",
        "int64": "Int64",
        "float32": "Float32",
        "float64": "Float64",
        "boolean": "boolean",
        "string": "string",
    }
)


def schema_to_nullable_dtypes(schema: Schema) -> dict[str, str]:
    """Pandas nullable dtypes for a schema: an empty API result would
    otherwise dtype as all-float64 and violate the declared schema."""
    return {
        name: _dtype_to_pandas.get(str(dtype), "object")
        for name, dtype in schema.items()
    }


def _tuplify(value: object) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


@frozen
class ParamSpec:
    """A read-time parameter: appears in ``read_kwargs`` (identity-bearing),
    substituted into the path template when named there, else sent as a
    query param. ``kind="range"`` marks date/cursor-style params — the
    no-cursor-state rule means ranges are always explicit here, never
    backend state; chunking is constructing multiple reads."""

    name = field(validator=instance_of(str))
    required = field(validator=instance_of(bool), default=False)
    kind = field(validator=in_(("scalar", "range")), default="scalar")

    @classmethod
    def from_dict(cls, dct: dict) -> ParamSpec:
        return cls(**dct)


@frozen
class AuthConfig:
    """Names the profile fields that carry credentials (as env-var refs).

    ``fields`` are the ``do_connect`` kwargs the auth scheme consumes;
    ``secret_fields`` (default: all fields) are the subset enforced as
    env-var references by ``check_for_exposed_secrets`` via
    ``Backend._get_secret_keys``.

    Which field fills which credential *role* is named explicitly, never
    inferred from declaration order (an order-based mapping silently swaps
    credentials when a self-service config lists fields in a different
    order): ``basic`` requires ``username_field`` and ``password_field``;
    ``bearer`` uses ``token_field`` (defaulting to the sole field when
    exactly one is declared, where there is no ambiguity).

    ``optional_fields`` names credentials the API can omit (e.g. GitHub
    serves unauthenticated, rate-limited requests) — those are not required
    at ``do_connect``.
    """

    kind = field(validator=in_(auth_kinds))
    fields = field(
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
        converter=tuple,
        default=(),
    )
    secret_fields = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        converter=lambda v: tuple(v) if v is not None else None,
        default=None,
    )
    optional_fields = field(
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
        converter=tuple,
        default=(),
    )
    username_field = field(validator=optional(instance_of(str)), default=None)
    password_field = field(validator=optional(instance_of(str)), default=None)
    token_field = field(validator=optional(instance_of(str)), default=None)

    def __attrs_post_init__(self) -> None:
        # role fields must name declared fields; roles are mapped by name,
        # not position, so credential meaning is stable across configs
        roles = {
            "username_field": self.username_field,
            "password_field": self.password_field,
            "token_field": self.resolved_token_field,
        }
        for role, name in roles.items():
            if name is not None and name not in self.fields:
                raise ValueError(
                    f"{role}={name!r} is not one of auth.fields {self.fields}"
                )
        for name in self.optional_fields:
            if name not in self.fields:
                raise ValueError(
                    f"optional field {name!r} is not one of auth.fields {self.fields}"
                )
        if self.kind == "basic" and (
            self.username_field is None or self.password_field is None
        ):
            raise ValueError(
                "basic auth requires username_field and password_field "
                "(mapping fields by position is a silent-swap hazard)"
            )
        if self.kind == "bearer" and self.resolved_token_field is None:
            raise ValueError(
                "bearer auth requires token_field (or a single-element fields)"
            )

    @property
    def resolved_token_field(self) -> str | None:
        if self.token_field is not None:
            return self.token_field
        return self.fields[0] if len(self.fields) == 1 else None

    @property
    def effective_secret_fields(self) -> tuple[str, ...]:
        return self.secret_fields if self.secret_fields is not None else self.fields

    @classmethod
    def from_dict(cls, dct: dict) -> AuthConfig:
        return cls(**dct)


@frozen
class ResourceConfig:
    """One API resource exposed as a relation."""

    name = field(validator=instance_of(str))
    schema = field(validator=instance_of(Schema))
    path = field(validator=instance_of(str), default="")
    record_path = field(validator=instance_of(str), default="")
    paginator = field(validator=optional(instance_of(str)), default=None)
    paginator_kwargs = field(validator=instance_of(tuple), converter=tuple, default=())
    params = field(
        validator=deep_iterable(instance_of(ParamSpec), instance_of(tuple)),
        converter=tuple,
        default=(),
    )
    fetch_override = field(validator=optional(is_callable()), default=None)

    @property
    def dtypes(self) -> dict[str, str]:
        return schema_to_nullable_dtypes(self.schema)

    @property
    def required_params(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def content_hash(self) -> str:
        """Identity over declarative fields only: ``fetch_override`` is code
        and stays out (refactoring it must not invalidate builds/caches,
        the same line ADR-0017 draws for fetch code)."""
        from xorq.common.utils.dasher import tokenize  # noqa: PLC0415

        return tokenize(
            (
                self.name,
                self.schema,
                self.path,
                self.record_path,
                self.paginator,
                self.paginator_kwargs,
                tuple((p.name, p.required, p.kind) for p in self.params),
            )
        )

    @classmethod
    def from_dict(cls, dct: dict) -> ResourceConfig:
        # yaml-safe plain data: schemas arrive as {name: dtype-string} and
        # fetch_override cannot be expressed (code-path-only)
        dct = dict(dct)
        if "fetch_override" in dct:
            raise ValueError(
                "fetch_override is code-path-only; it cannot be provided "
                "via plain-data config (e.g. a self-service rest profile)"
            )
        dct["schema"] = ibis.schema(dct["schema"])
        dct["params"] = tuple(
            ParamSpec.from_dict(p) if isinstance(p, dict) else p
            for p in dct.get("params", ())
        )
        if "paginator_kwargs" in dct:
            dct["paginator_kwargs"] = tuple(
                (k, v) for k, v in dict(dct["paginator_kwargs"]).items()
            )
        return cls(**dct)


@frozen
class RestBackendConfig:
    """The whole API: base urls, auth shape, resources."""

    base_urls = field(
        converter=lambda v: MappingProxyType(dict(v)),
        validator=instance_of(MappingProxyType),
    )
    auth = field(validator=instance_of(AuthConfig))
    resources = field(
        validator=deep_iterable(instance_of(ResourceConfig), instance_of(tuple)),
        converter=tuple,
    )

    def __attrs_post_init__(self) -> None:
        names = tuple(r.name for r in self.resources)
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate resource names: {names}")

    @property
    def resource_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.resources)

    def get_resource(self, name: str) -> ResourceConfig:
        by_name = {r.name: r for r in self.resources}
        try:
            return by_name[name]
        except KeyError:
            raise KeyError(
                f"no resource {name!r}; available: {sorted(by_name)}"
            ) from None

    def base_url(self, which: str = "default") -> str:
        return self.base_urls[which]

    @classmethod
    def from_dict(cls, dct: dict) -> RestBackendConfig:
        """Parse yaml-safe plain data (the self-service profile path)."""
        return cls(
            base_urls=dict(dct["base_urls"]),
            auth=AuthConfig.from_dict(dct["auth"]),
            resources=tuple(
                ResourceConfig.from_dict(r) if isinstance(r, dict) else r
                for r in dct["resources"]
            ),
        )

    @classmethod
    def maybe_from_dict(
        cls, config: RestBackendConfig | dict | None
    ) -> RestBackendConfig | None:
        match config:
            case None | RestBackendConfig():
                return config
            case dict():
                return cls.from_dict(config)
            case _:
                raise TypeError(f"cannot build a config from {type(config)}")
