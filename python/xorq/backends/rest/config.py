"""The declarative REST contract: what a REST API *is*, engine-independent.

The config is the stable contract (ADR-0018): extraction engines
(``engines.Engine`` — ``NativeEngine`` today, dlt later) are swappable
behind it. Three deliberate omissions vs dlt's RESTAPIConfig:

1. no incremental/cursor block — incrementality is a param in ``read_kwargs``
   (``ParamSpec(kind="range")``), never backend state;
2. auth names profile *fields*, never values — resolution happens at
   execution via the profile/env machinery (ADR-0016);
3. ``fetch_override`` — config-first, code-fallback; bespoke resources
   (NDJSON export, nonstandard pagination) supply a callable. Overrides are
   code-path-only: they cannot ride in a self-service profile.

Identity: both config classes carry a ``content_hash`` DERIVED from their
attrs declaration (``identity_field_names``) rather than a hand-written
tuple — a field is identity-bearing by default and opting out takes an
explicit ``non_identity_field(...)`` declaration. ``ResourceConfig`` covers the
resource's declarative fields (``fetch_override`` excluded — code is
refactorable, per ADR-0025's line); ``RestBackendConfig`` covers the
API-wide contract, the resolved ``base_urls`` and the auth shape. The
backend folds both into ``normalize_read_source_identity``, so editing a
resource's path/params, or repointing a base URL, changes build and cache
hashes (ADR-0015).

Sibling independence — editing one resource not invalidating the others —
is **mode-dependent**, and the unconditional claim this docstring used to
make was true of curated backends only:

- **curated** (config in code): the profile carries credentials only, so the
  per-resource hash is the only place a resource's config enters identity.
  Editing resource B leaves A's read identity byte-identical. This is the
  property the ``resources`` identity exclusion exists to deliver.
- **self-service** (config in profile): the whole config dict rides in the
  profile, and ``normalize_read_source_identity`` folds the profile's content
  hash (``file_utils``), so editing resource B changes *every* sibling read's
  identity — and the touched resource's config is counted twice, once through
  the profile and once through its own hash.

The failure direction is safe (spurious invalidation, never stale data served
as current) and ADR-0018 records the trade as deliberate. Making the property
uniform would mean ``dissoc``-ing ``config`` from the profile's contribution
to read identity, which is itself an identity change — every build directory
and cache entry for every self-service rest read moves — so it needs its own
adjudicated baseline and is deliberately not done here.
"""

from __future__ import annotations

import string
from types import MappingProxyType

from attr import (
    field,
    fields,
    frozen,
    has,
)
from attr.validators import (
    deep_iterable,
    in_,
    instance_of,
    is_callable,
    optional,
)

from xorq.common.utils.attr_utils import (
    IDENTITY_METADATA_KEY,
    non_identity_field,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.schema import Schema


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


# -- derived identity --------------------------------------------------------
#
# Identity membership is DERIVED from the attrs declaration, never
# hand-enumerated. A hand-written tuple beside a growing attrs class drifts
# silently in the dangerous direction: a field that is identity-bearing but
# unhashed makes two different configs share a hash, so cached data from one
# is served as current data for the other. Deriving flips the failure
# direction -- a new field is identity-bearing BY DEFAULT, and the worst a
# mistake can do is a spurious cache miss.


def identity_field_names(cls: type) -> tuple[str, ...]:
    """Identity-bearing field names of an attrs class: every declared field
    except those declared with ``non_identity_field`` (which is the annotation
    ``metadata={"identity": False}``, spelled as what it means -- see that
    helper for why the raw annotation is a polarity trap).

    Opting a field out is therefore a conscious declaration carrying its own
    justification, not an omission from a list nobody re-reads.
    """
    return tuple(
        f.name for f in fields(cls) if f.metadata.get(IDENTITY_METADATA_KEY, True)
    )


def _identity_value(value: object) -> object:
    if has(type(value)):
        return identity_parts(value)
    if isinstance(value, (MappingProxyType, dict)):
        return tuple(sorted((k, _identity_value(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_identity_value(v) for v in value)
    return value


def identity_parts(inst: object) -> tuple[tuple[str, object], ...]:
    """``(field_name, value)`` pairs over an attrs instance's identity fields.

    Named pairs, not a bare tuple: the encoding stays injective under field
    renames and reorderings. Nested attrs values (``ParamSpec``,
    ``AuthConfig``) recurse through the same rule, so they need no bespoke
    normalization; mappings are sorted so declaration order is not
    identity-bearing.
    """
    return tuple(
        (name, _identity_value(getattr(inst, name)))
        for name in identity_field_names(type(inst))
    )


def content_hash(inst: object) -> str:
    """Tokenized digest of an attrs instance's derived identity parts."""
    from xorq.common.utils.dasher import tokenize  # noqa: PLC0415

    return tokenize(identity_parts(inst))


def _freeze_kv_pairs(value: object) -> tuple:
    """Normalize kwargs-shaped input (a dict or an iterable of pairs) to a
    sorted tuple of pairs: dicts survive direct construction (a bare
    ``tuple(dict)`` would keep only the keys and silently poison
    ``content_hash``) and ordering stops being identity-bearing."""
    pairs = dict(value).items() if isinstance(value, dict) else tuple(value)
    return tuple(sorted((k, v) for k, v in pairs))


@frozen
class ParamSpec:
    """A read-time parameter: appears in ``read_kwargs`` (identity-bearing),
    substituted into the path template when named there, else sent as a
    query param. ``kind="range"`` marks date/cursor-style params — the
    no-cursor-state rule means ranges are always explicit here, never
    backend state; chunking is constructing multiple reads.

    ``kind="scope"`` marks a **caller-scope discriminator**: a non-secret
    value naming whose data this read is (an account, org, or workspace). It
    is identity-only — never sent as a query param, so declaring one cannot
    change what goes on the wire — and it is what
    ``ResourceConfig(caller_scoped=True)`` requires. When the name also appears
    in the path template it is substituted there as usual.
    """

    name = field(validator=instance_of(str))
    required = field(validator=instance_of(bool), default=False)
    kind = field(validator=in_(("scalar", "range", "scope")), default="scalar")

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

    ``kind`` is an open set, and this class deliberately does not police it.
    The role rules below apply to the two kinds that have roles (``basic``,
    ``bearer``); any other kind is accepted here as declarative and is
    resolved to an applier at ``do_connect``, which rejects unknown kinds
    against the backend's class-level ``auth_appliers`` registry (defaulting
    to ``engines.AUTH_APPLIERS``, the single source of truth for which kinds
    can actually be applied). A curated backend extends the set by declaring
    an applier there, not by editing this module.
    """

    kind = field(validator=instance_of(str))
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
    """One API resource exposed as a relation.

    ``caller_scoped=True`` declares that what the endpoint returns depends on
    *who is asking* — ``/user/repos``, an org-scoped listing, a "my
    workspaces" endpoint. Identity deliberately carries env-var *references*,
    never credential values (ADR-0024), so two users with the same reference
    names and different credentials otherwise produce the same read identity
    and can serve each other's data out of a shared cache. Marking the
    resource makes the config **fail assembly** unless a required
    ``ParamSpec(kind="scope")`` is declared, turning that silent leak into a
    config-time error; the scope value is identity-only and never hits the
    wire, so declaring one changes no request.
    """

    name = field(validator=instance_of(str))
    schema = field(validator=instance_of(Schema))
    path = field(validator=instance_of(str), default="")
    # which of the API's base_urls this resource is served from (e.g. a
    # data vs query host); validated against base_urls at config assembly
    base_url_key = field(validator=instance_of(str), default="default")
    record_path = field(validator=instance_of(str), default="")
    paginator = field(validator=optional(instance_of(str)), default=None)
    paginator_kwargs = field(
        validator=instance_of(tuple), converter=_freeze_kv_pairs, default=()
    )
    params = field(
        validator=deep_iterable(instance_of(ParamSpec), instance_of(tuple)),
        converter=tuple,
        default=(),
    )
    # the overflow column: unmapped record fields land here as sorted JSON.
    # Declared, not name-sniffed, so an API with a genuine field named
    # "properties" can still have it as a typed column.
    residual_column = field(validator=optional(instance_of(str)), default=None)
    # "whose data is this?" — see the class docstring and __attrs_post_init__
    caller_scoped = field(validator=instance_of(bool), default=False)
    # the one identity opt-out on this class: fetch_override is code, and
    # refactoring code must not invalidate builds/caches (ADR-0025's line)
    fetch_override = non_identity_field(
        validator=optional(is_callable()),
        default=None,
    )

    def __attrs_post_init__(self) -> None:
        if (
            self.residual_column is not None
            and self.residual_column not in self.schema.names
        ):
            raise ValueError(
                f"resource {self.name!r}: residual_column "
                f"{self.residual_column!r} is not a schema column"
            )
        # a mistyped placeholder or an optional placeholder param would
        # otherwise surface as a bare KeyError at fetch time
        undeclared = self.path_placeholders - set(self.param_names)
        if undeclared:
            raise ValueError(
                f"resource {self.name!r}: path placeholders {sorted(undeclared)} "
                f"are not declared params {self.param_names}"
            )
        not_required = self.path_placeholders - set(self.required_params)
        if not_required:
            raise ValueError(
                f"resource {self.name!r}: path placeholders {sorted(not_required)} "
                "must be required params (the path cannot format without them)"
            )
        optional_scope = tuple(
            p.name for p in self.params if p.kind == "scope" and not p.required
        )
        if optional_scope:
            raise ValueError(
                f"resource {self.name!r}: scope params {sorted(optional_scope)} "
                "must be required -- a discriminator the caller may omit does "
                "not discriminate"
            )
        if self.caller_scoped and not self.scope_params:
            raise ValueError(
                f"resource {self.name!r}: caller_scoped=True requires a "
                'required ParamSpec(kind="scope") naming a non-secret '
                "discriminator (an account, org, or workspace). Identity "
                "carries env-var references, never credential values "
                "(ADR-0024), so without one two callers with the same "
                "reference names and different credentials share a read "
                "identity -- and a cache entry"
            )

    @property
    def path_placeholders(self) -> frozenset[str]:
        return frozenset(
            name for _, name, *_ in string.Formatter().parse(self.path) if name
        )

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
    def scope_params(self) -> tuple[str, ...]:
        """Caller-scope discriminators: identity-only, never sent on the wire."""
        return tuple(p.name for p in self.params if p.kind == "scope")

    @property
    def content_hash(self) -> str:
        """Identity over declarative fields only, DERIVED from the attrs
        declaration (``identity_field_names``): every field but
        ``fetch_override``, which is code and stays out (refactoring it must
        not invalidate builds/caches, the same line ADR-0025 draws for fetch
        code).

        Note what this hash does *not* cover: the resolved base URL. It folds
        ``base_url_key``, the name of a route; the URL that name resolves to
        lives in ``RestBackendConfig.base_urls`` and is folded by
        ``RestBackendConfig.content_hash``, which the backend contributes to
        read identity alongside this one.
        """
        return content_hash(self)

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
        # paginator_kwargs: dicts are normalized by the field converter
        return cls(**dct)


@frozen
class RestBackendConfig:
    """The whole API: base urls, auth shape, resources."""

    base_urls = field(
        converter=lambda v: MappingProxyType(dict(v)),
        validator=instance_of(MappingProxyType),
    )
    auth = field(validator=instance_of(AuthConfig))
    # excluded from *this* class's content_hash, not from identity: each
    # resource contributes its own `ResourceConfig.content_hash` per read, so
    # editing one resource does not invalidate its siblings (ADR-0018). Folding
    # the whole tuple here would undo that.
    #
    # This delivers sibling independence in CURATED mode only. In self-service
    # mode the config rides in the profile, whose hash every read folds, so
    # siblings move anyway -- see the module docstring for why that stands.
    resources = non_identity_field(
        validator=deep_iterable(instance_of(ResourceConfig), instance_of(tuple)),
        converter=tuple,
    )

    def __attrs_post_init__(self) -> None:
        names = tuple(r.name for r in self.resources)
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate resource names: {names}")
        unrouted = tuple(
            (r.name, r.base_url_key)
            for r in self.resources
            if r.base_url_key not in self.base_urls
        )
        if unrouted:
            raise ValueError(
                f"resources reference unknown base_urls keys: {unrouted}; "
                f"available: {sorted(self.base_urls)}"
            )

    @property
    def content_hash(self) -> str:
        """Identity of the API-wide contract: the resolved ``base_urls`` and
        the whole auth shape, DERIVED from the attrs declaration.

        This is what makes *where the data came from* identity-bearing. For a
        curated backend the profile carries credentials only, so without this
        hash repointing ``base_urls`` from prod to staging changed nothing a
        read hashed on -- and cached data from the old host was served as
        current data from the new one. ``resources`` is excluded here because
        it is folded per-resource (see the field comment).
        """
        return content_hash(self)

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
