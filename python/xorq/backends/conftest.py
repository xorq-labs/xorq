import itertools
import os

import _pytest
import pytest

import xorq.api as xo
from xorq.backends import _get_backend_names
from xorq.caching import ParquetSnapshotStorage, SourceSnapshotStorage
from xorq.vendor import ibis


snowflake_credentials_varnames = (
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_USER",
)
have_snowflake_credentials = all(
    os.environ.get(varname) for varname in snowflake_credentials_varnames
)

KEY_PREFIX = xo.config.options.cache.key_prefix


def _get_backend_from_parts(parts: tuple[str, ...]) -> str | None:
    """Return the backend part of a test file's path parts.

    Examples
    --------
    >>> _get_backend_from_parts(("/", "ibis", "backends", "sqlite", "tests"))
    "sqlite"
    """
    try:
        index = parts.index("backends")
    except ValueError:
        return None
    else:
        return parts[index + 1]


def pytest_ignore_collect(collection_path, config):
    # get the backend path part
    backend = _get_backend_from_parts(collection_path.parts)
    if backend is None or backend not in _get_backend_names():
        return False

    # we evaluate the marker early so that we don't trigger
    # an import of conftest files for the backend, which will
    # import the backend and thus require dependencies that may not
    # exist
    #
    # alternatives include littering library code with pytest.importorskips
    # and moving all imports close to their use site
    #
    # the latter isn't tenable for backends that use multiple dispatch
    # since the rules are executed at import time
    mark_expr = config.getoption("-m")
    # we can't let the empty string pass through, since `'' in s` is `True` for
    # any `s`; if no marker was passed don't ignore the collection of `path`
    if not mark_expr:
        return False
    expr = _pytest.mark.expression.Expression.compile(mark_expr)
    # we check the "backend" marker as well since if that's passed
    # any file matching a backend should be skipped
    keep = expr.evaluate(lambda s, **_kw: s in (backend, "backend"))
    return not keep


def pytest_collection_modifyitems(session, config, items):
    all_backends = _get_backend_names()
    additional_markers = []

    unrecognized_backends = set()
    for item in items:
        # add the backend marker to any tests are inside "xorq/backends"
        parts = item.path.parts
        backend = _get_backend_from_parts(parts)
        if backend is not None and backend in all_backends:
            item.add_marker(getattr(pytest.mark, backend))
            item.add_marker(pytest.mark.backend)
        elif "backends" not in parts and not tuple(
            itertools.chain(
                *(item.iter_markers(name=name) for name in all_backends),
                item.iter_markers(name="backend"),
            )
        ):
            # anything else is a "core" test and is run by default
            if not any(item.iter_markers(name="benchmark")):
                item.add_marker(pytest.mark.core)

    if unrecognized_backends:
        raise pytest.PytestCollectionWarning("\n" + "\n".join(unrecognized_backends))

    for item, markers in additional_markers:
        for marker in markers:
            item.add_marker(marker)


def pytest_runtest_setup(item):
    if any(mark.name == "snowflake" for mark in item.iter_markers()):
        pytest.importorskip("snowflake.connector")
        if not have_snowflake_credentials:
            pytest.skip("cannot run snowflake tests without snowflake creds")


def get_storage_uncached(expr):
    assert expr.ls.is_cached
    storage = expr.ls.storage
    uncached = expr.ls.uncached_one
    return (storage, uncached)


def con_snapshot(alltypes_df, ls_con, df_con):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")
    # create a temp table we can mutate
    table = df_con.create_table(name, alltypes_df)
    cached_expr = (
        table.group_by(group_by)
        .agg({f"count_{col}": table[col].count() for col in table.columns})
        .cache(storage=SourceSnapshotStorage(source=ls_con))
    )
    (storage, uncached) = get_storage_uncached(cached_expr)
    # test preconditions
    assert not storage.exists(uncached)
    # test cache creation
    executed0 = cached_expr.execute()
    assert storage.exists(uncached)
    # test cache use
    executed1 = cached_expr.execute()
    assert executed0.equals(executed1)
    # test NO cache invalidation
    df_con.insert(name, alltypes_df)
    executed2 = cached_expr.execute()
    executed3 = cached_expr.ls.uncached.execute()
    assert executed0.equals(executed2)
    assert not executed0.equals(executed3)
    assert storage.get_key(uncached).count(KEY_PREFIX) == 1


def con_cross_source_snapshot(alltypes_df, con, expr_con):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")
    # create a temp table we can mutate
    table = expr_con.create_table(name, alltypes_df)
    storage = ParquetSnapshotStorage(source=con)
    expr = table.group_by(group_by).agg(
        {f"count_{col}": table[col].count() for col in table.columns}
    )
    cached_expr = expr.cache(storage=storage)
    # test preconditions
    assert not storage.exists(expr)  # the expr is not cached
    assert storage.source is not expr_con  # the cache is cross source
    # test cache creation
    df = cached_expr.execute()
    assert not df.empty
    assert storage.exists(expr)
    # test cache use
    executed1 = cached_expr.execute()
    assert df.equals(executed1)
    # test NO cache invalidation
    expr_con.insert(name, alltypes_df)
    executed2 = cached_expr.execute()
    executed3 = cached_expr.ls.uncached.execute()
    assert df.equals(executed2)
    assert not df.equals(executed3)


def con_cache_find_backend(cls, parquet_dir, conn):
    astronauts_path = parquet_dir / "astronauts.parquet"
    storage = cls(source=conn)
    expr = conn.read_parquet(astronauts_path).cache(storage=storage)
    assert expr._find_backend()._profile == conn._profile
