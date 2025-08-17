import itertools
import os

import _pytest
import pytest

from xorq.backends import _get_backend_names


snowflake_credentials_varnames = (
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_USER",
)
have_snowflake_credentials = all(
    os.environ.get(varname) for varname in snowflake_credentials_varnames
)


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
    keep = expr.evaluate(lambda s: s in (backend, "backend"))
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
