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

import subprocess
import sys

import pytest

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


class Cursor:
    """Tokenizable, but inherits the default ``object.__repr__``, so ``str()``
    of any container holding one embeds a memory address."""

    def __init__(self, page: int) -> None:
        self.page = page

    def __dasher_tokenize__(self) -> tuple:
        return ("Cursor", self.page)


_CROSS_PROCESS_SCRIPT = """
import xorq.api as xo
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.graph_utils import replace_nodes
from xorq.common.utils.dasher._opaque import _xorq_opaque_to_placeholder
from xorq.tests.test_pathless_read_identity import Cursor, make_read

expr = make_read(xo.connect(), read_kwargs=(("cursor", Cursor(7)),))
print(replace_nodes(_xorq_opaque_to_placeholder, expr.op()).name)
print(tokenize(expr))
"""


def test_identity_is_stable_across_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opaque placeholder name and the expression token must not vary
    between interpreters.

    ``_stable_opaque_name`` used to ``str()`` each part; for the path-less
    anchor (a nested tuple) that reprs its members, so a member with the
    default ``__repr__`` leaked a memory address into the placeholder name and
    the build hash -- silent permanent cache misses and unstable build
    directory names, with no error raised. Two subprocesses with different
    ``PYTHONHASHSEED`` values pin that down.
    """

    def run(seed: str) -> str:
        monkeypatch.setenv("PYTHONHASHSEED", seed)
        proc = subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_SCRIPT],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    assert run("0") == run("12345")


def test_kwargs_and_parts_do_not_alias() -> None:
    """The identity encoding must be injective across its two groups.

    Flat ``(*parts, *sorted(kwargs))`` concatenation is not: a source
    contributing ``(("resource", "things"),)`` via ``read_identity_parts`` with
    no kwargs would tokenize identically to a read contributing no parts but
    carrying ``resource="things"`` as a kwarg.
    """
    con_kwarg = xo.connect()
    con_parts = xo.connect()
    con_parts.read_identity_parts = lambda read: (("resource", "things"),)
    from_kwarg = make_read(con_kwarg, read_kwargs=(("resource", "things"),))
    from_parts = make_read(con_parts, read_kwargs=())
    assert normalize_read_source_identity(
        from_kwarg.op()
    ) != normalize_read_source_identity(from_parts.op())
    assert tokenize(from_kwarg) != tokenize(from_parts)


def test_snapshot_strategy_accepts_pathless_read() -> None:
    expr = make_read(xo.connect())
    normalized = snapshot_normalize_read(expr.op())
    assert normalized[0] == "snapshot_normalize_read"
    # stat-free: identical construction normalizes identically
    assert normalized == snapshot_normalize_read(make_read(xo.connect()).op())


# NB: the dasher loud-guard for a path-less Read with normalize_method=None
# is defense-in-depth only: the Read op's Callable annotation rejects None at
# construction, so the state is unreachable through the public constructor.
