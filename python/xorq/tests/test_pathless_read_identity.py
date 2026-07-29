"""Path-less Read ops (no ``hash_path``) take identity from a registered
normalize_method instead of a file path.

This is the core enabler for API-backed sources (backends whose resources
are Read ops with no filesystem anchor): tokenize, snapshot caching, and
build canonicalization must all accept a Read whose identity is
``normalize_read_source_identity`` — the source profile's content hash,
the source's optional ``read_identity_parts`` contribution, and the
declarative read kwargs.
"""

from __future__ import annotations

import xorq.api as xo
from xorq.caching.strategy import snapshot_normalize_read
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.file_utils import normalize_read_source_identity
from xorq.expr.relations import Read
from xorq.ibis_yaml.normalize_registry import (
    serialize_normalize_method,
    validate,
)


schema = xo.schema({"a": "int64"})


def make_read(
    con: object,
    read_kwargs: tuple = (("resource", "things"),),
    normalize_method: object = normalize_read_source_identity,
    name: str = "t",
) -> object:
    kwargs = {} if normalize_method is None else {"normalize_method": normalize_method}
    return Read(
        method_name="fetch_resource",
        name=name,
        schema=schema,
        source=con,
        read_kwargs=read_kwargs,
        **kwargs,
    ).to_expr()


def test_registered_by_name() -> None:
    validate(normalize_read_source_identity)
    assert serialize_normalize_method(normalize_read_source_identity) == {
        "kind": "named",
        "name": "read_source_identity",
    }


def test_tokenize_is_declarative() -> None:
    (one, two) = (make_read(xo.connect()) for _ in range(2))
    # same profile content, same kwargs -> same identity, across connections
    assert tokenize(one) == tokenize(two)
    # the gen_name'd op name is excluded from identity
    assert tokenize(make_read(xo.connect(), name="other")) == tokenize(one)
    # declarative kwargs are identity
    changed = make_read(xo.connect(), read_kwargs=(("resource", "other"),))
    assert tokenize(changed) != tokenize(one)


def test_source_contributes_identity_parts() -> None:
    (con_one, con_two) = (xo.connect() for _ in range(2))
    con_one.read_identity_parts = lambda read: (("config", "hash-one"),)
    con_two.read_identity_parts = lambda read: (("config", "hash-two"),)
    assert tokenize(make_read(con_one)) != tokenize(make_read(con_two))
    con_two.read_identity_parts = lambda read: (("config", "hash-one"),)
    assert tokenize(make_read(con_one)) == tokenize(make_read(con_two))


def test_snapshot_strategy_accepts_pathless_read() -> None:
    expr = make_read(xo.connect())
    normalized = snapshot_normalize_read(expr.op())
    assert normalized[0] == "snapshot_normalize_read"
    # stat-free: identical construction normalizes identically
    assert normalized == snapshot_normalize_read(make_read(xo.connect()).op())


# NB: the dasher loud-guard for a path-less Read with normalize_method=None
# is defense-in-depth only: the Read op's Callable annotation rejects None at
# construction, so the state is unreachable through the public constructor.
