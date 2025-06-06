from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import itertools
import operator
from functools import cache
from typing import Any

import _pytest
import pytest
from packaging.requirements import Requirement
from packaging.version import parse as vparse

from xorq.vendor import ibis
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends import (
    CanCreateCatalog,
    CanCreateDatabase,
    CanListSchema,
    _get_backend_names,
)
from xorq.vendor.ibis.conftest import WINDOWS
from xorq.vendor.ibis.util import promote_tuple


def compare_versions(module_name, given_version, op):
    try:
        current_version = importlib.metadata.version(module_name)
        return op(vparse(current_version), vparse(given_version))
    except importlib.metadata.PackageNotFoundError:
        return False


def is_newer_than(module_name, given_version):
    return compare_versions(module_name, given_version, operator.gt)


def is_older_than(module_name, given_version):
    return compare_versions(module_name, given_version, operator.lt)


TEST_TABLES = {
    "functional_alltypes": ibis.schema(
        {
            "id": "int32",
            "bool_col": "boolean",
            "tinyint_col": "int8",
            "smallint_col": "int16",
            "int_col": "int32",
            "bigint_col": "int64",
            "float_col": "float32",
            "double_col": "float64",
            "date_string_col": "string",
            "string_col": "string",
            "timestamp_col": "timestamp",
            "year": "int32",
            "month": "int32",
        }
    ),
    "diamonds": ibis.schema(
        {
            "carat": "float64",
            "cut": "string",
            "color": "string",
            "clarity": "string",
            "depth": "float64",
            "table": "float64",
            "price": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
        }
    ),
    "batting": ibis.schema(
        {
            "playerID": "string",
            "yearID": "int64",
            "stint": "int64",
            "teamID": "string",
            "lgID": "string",
            "G": "int64",
            "AB": "int64",
            "R": "int64",
            "H": "int64",
            "X2B": "int64",
            "X3B": "int64",
            "HR": "int64",
            "RBI": "int64",
            "SB": "int64",
            "CS": "int64",
            "BB": "int64",
            "SO": "int64",
            "IBB": "int64",
            "HBP": "int64",
            "SH": "int64",
            "SF": "int64",
            "GIDP": "int64",
        }
    ),
    "awards_players": ibis.schema(
        {
            "playerID": "string",
            "awardID": "string",
            "yearID": "int64",
            "lgID": "string",
            "tie": "string",
            "notes": "string",
        }
    ),
    "astronauts": ibis.schema(
        {
            "id": "int64",
            "number": "int64",
            "nationwide_number": "int64",
            "name": "string",
            "original_name": "string",
            "sex": "string",
            "year_of_birth": "int64",
            "nationality": "string",
            "military_civilian": "string",
            "selection": "string",
            "year_of_selection": "int64",
            "mission_number": "int64",
            "total_number_of_missions": "int64",
            "occupation": "string",
            "year_of_mission": "int64",
            "mission_title": "string",
            "ascend_shuttle": "string",
            "in_orbit": "string",
            "descend_shuttle": "string",
            "hours_mission": "float64",
            "total_hrs_sum": "float64",
            "field21": "int64",
            "eva_hrs_mission": "float64",
            "total_eva_hrs": "float64",
        }
    ),
}

# We want to check for exceptions in xfail tests for two reasons:
# * xfail tests without exception checking may hide problems.
#   For example, the implementation may work, but we may have an error in the test that needs to be fixed.
# * facilitates code reviews
# For now, many of our tests don't do this, and we're working to change this situation
# by improving all tests file by file. All files that have already been improved are
# added to this list to prevent regression.
FILES_WITH_STRICT_EXCEPTION_CHECK = [
    "ibis/backends/tests/test_api.py",
    "ibis/backends/tests/test_array.py",
    "ibis/backends/tests/test_aggregation.py",
    "ibis/backends/tests/test_binary.py",
    "ibis/backends/tests/test_numeric.py",
    "ibis/backends/tests/test_column.py",
    "ibis/backends/tests/test_string.py",
    "ibis/backends/tests/test_temporal.py",
    "ibis/backends/tests/test_uuid.py",
    "ibis/backends/tests/test_window.py",
]

ALL_BACKENDS = set(_get_backend_names())


def _get_backend_conf(backend_str: str):
    """Convert a backend string to the test class for the backend."""
    conftest = importlib.import_module(f"ibis.backends.{backend_str}.tests.conftest")
    return conftest.TestConf


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
        # Yell loudly if unrecognized backend in notimpl, notyet or never
        for name in ("notimpl", "notyet", "never"):
            for mark in item.iter_markers(name=name):
                if backend := set(util.promote_list(mark.args[0])).difference(
                    ALL_BACKENDS
                ):
                    unrecognized_backends.add(
                        f"""Unrecognize backend(s) {backend} passed to {name} marker in
{item.path}::{item.originalname}"""
                    )

        # add the backend marker to any tests are inside "ibis/backends"
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


@cache
def _get_backends_to_test(
    keep: tuple[str, ...] = (),
    discard: tuple[str, ...] = (),
) -> list[Any]:
    """Get a list of `TestConf` classes of the backends to test."""
    backends = _get_backend_names()

    if discard:
        backends = backends.difference(discard)

    if keep:
        backends = backends.intersection(keep)

    return [
        pytest.param(
            backend,
            marks=[getattr(pytest.mark, backend), pytest.mark.backend],
            id=backend,
        )
        for backend in sorted(backends)
    ]


def pytest_runtest_call(item):
    """Dynamically add various custom markers."""
    backend = [
        getattr(backend, "name", lambda backend=backend: backend)()
        for key, backend in item.funcargs.items()
        if key.endswith(("backend", "backend_name", "backend_no_data"))
    ]
    if len(backend) > 1:
        raise ValueError(
            f"test {item.originalname} was supplied with multiple backend "
            f"objects simultaneously. This is likely due to a leaky fixture."
            f"Backends passed: {(back.name() for back in backend)}"
        )
    if not backend:
        # Check item path to see if test is in backend-specific folder
        backend = set(_get_backend_names()).intersection(item.path.parts)

    if not backend:
        return

    backend = next(iter(backend))

    def _filter_none_from_raises(kwargs):
        # Filter out any None values from kwargs['raises']
        # to cover any missing backend error types as defined in ibis/backends/tests/errors.py
        if (raises := kwargs.get("raises")) is not None:
            raises = tuple(filter(None, promote_tuple(raises)))
            if raises:
                kwargs["raises"] = raises
            else:
                # if filtering removes all of the values of raises pop the
                # argument otherwise it gets passed as an empty tuple and this
                # messes up xfail
                kwargs.pop("raises")
        return kwargs

    # Ibis hasn't exposed existing functionality
    # This xfails so that you know when it starts to pass
    for marker in item.iter_markers(name="notimpl"):
        if backend in marker.args[0]:
            if (
                item.location[0] in FILES_WITH_STRICT_EXCEPTION_CHECK
                and "raises" not in marker.kwargs.keys()
            ):
                raise ValueError("notimpl requires a raises")
            kwargs = marker.kwargs.copy()
            kwargs.setdefault("reason", f"Feature not yet exposed in {backend}")
            kwargs = _filter_none_from_raises(kwargs)
            item.add_marker(pytest.mark.xfail(**kwargs))

    # Functionality is unavailable upstream (but could be)
    # This xfails so that you know when it starts to pass
    for marker in item.iter_markers(name="notyet"):
        if backend in marker.args[0]:
            if (
                item.location[0] in FILES_WITH_STRICT_EXCEPTION_CHECK
                and "raises" not in marker.kwargs.keys()
            ):
                raise ValueError("notyet requires a raises")

            kwargs = marker.kwargs.copy()
            kwargs.setdefault("reason", f"Feature not available upstream for {backend}")
            kwargs = _filter_none_from_raises(kwargs)
            item.add_marker(pytest.mark.xfail(**kwargs))

    for marker in item.iter_markers(name="never"):
        if backend in marker.args[0]:
            if "reason" not in marker.kwargs.keys():
                raise ValueError("never requires a reason")
            kwargs = marker.kwargs.copy()
            kwargs = _filter_none_from_raises(kwargs)
            item.add_marker(pytest.mark.xfail(**kwargs))

    for marker in item.iter_markers(name="xfail_version"):
        kwargs = marker.kwargs.copy()
        kwargs = _filter_none_from_raises(kwargs)
        if backend not in kwargs:
            continue

        provided_reason = kwargs.pop("reason", None)
        specs = kwargs.pop(backend)
        failing_specs = []
        for spec in specs:
            req = Requirement(spec)
            if req.specifier.contains(
                importlib.import_module(req.name).__version__
            ) and ((not req.marker) or req.marker.evaluate()):
                failing_specs.append(spec)
        reason = f"{backend} backend test fails with {backend}{specs}"
        if provided_reason is not None:
            reason += f"; {provided_reason}"
        if failing_specs:
            item.add_marker(pytest.mark.xfail(reason=reason, **kwargs))


@pytest.fixture(params=_get_backends_to_test(), scope="session")
def backend(request, data_dir, tmp_path_factory, worker_id):
    """Return an instance of BackendTest, loaded with data."""

    cls = _get_backend_conf(request.param)
    return cls.load_data(data_dir, tmp_path_factory, worker_id)


@pytest.fixture(scope="session")
def con(backend):
    """Instance of a backend client."""
    return backend.connection


@pytest.fixture(params=_get_backends_to_test(), scope="session")
def backend_no_data(request, data_dir, tmp_path_factory, worker_id):
    """Return an instance of BackendTest, with no data loaded."""
    cls = _get_backend_conf(request.param)
    return cls(data_dir=data_dir, tmpdir=tmp_path_factory, worker_id=worker_id)


@pytest.fixture(scope="session")
def con_no_data(backend_no_data):
    """Return an Ibis backend instance, with no data loaded."""
    return backend_no_data.connection


@pytest.fixture(scope="session")
def con_list_schema(con):
    if isinstance(con, CanListSchema):
        return con
    else:
        pytest.skip(f"{con.name} backend cannot create schemas")


@pytest.fixture(scope="session")
def con_create_catalog(con):
    if isinstance(con, CanCreateCatalog):
        return con
    else:
        pytest.skip(f"{con.name} backend cannot create databases")


@pytest.fixture(scope="session")
def con_create_database(con):
    if isinstance(con, CanCreateDatabase):
        return con
    else:
        pytest.skip(f"{con.name} backend cannot create schemas")


@pytest.fixture(scope="session")
def con_create_catalog_database(con):
    if isinstance(con, CanCreateCatalog) and isinstance(con, CanCreateDatabase):
        return con
    else:
        pytest.skip(f"{con.name} backend cannot create both database and schemas")


def _setup_backend(request, data_dir, tmp_path_factory, worker_id):
    if (backend := request.param) == "duckdb" and WINDOWS:
        pytest.xfail(
            "windows prevents two connections to the same duckdb file "
            "even in the same process"
        )
    else:
        cls = _get_backend_conf(backend)
        return cls.load_data(data_dir, tmp_path_factory, worker_id)


@pytest.fixture(
    params=_get_backends_to_test(discard=("dask", "pandas")),
    scope="session",
)
def ddl_backend(request, data_dir, tmp_path_factory, worker_id):
    """Set up the backends that are SQL-based."""
    return _setup_backend(request, data_dir, tmp_path_factory, worker_id)


@pytest.fixture(scope="session")
def ddl_con(ddl_backend):
    """Instance of Client, already connected to the db (if applies)."""
    return ddl_backend.connection


@pytest.fixture(
    params=_get_backends_to_test(keep=("dask", "pandas", "pyspark")),
    scope="session",
)
def udf_backend(request, data_dir, tmp_path_factory, worker_id):
    """Runs the UDF-supporting backends."""
    cls = _get_backend_conf(request.param)
    return cls.load_data(data_dir, tmp_path_factory, worker_id)


@pytest.fixture(scope="session")
def alltypes(backend):
    return backend.functional_alltypes


@pytest.fixture(scope="session")
def json_t(backend):
    return backend.json_t


@pytest.fixture(scope="session")
def struct(backend):
    return backend.struct


@pytest.fixture(scope="session")
def udf_alltypes(udf_backend):
    return udf_backend.functional_alltypes


@pytest.fixture(scope="session")
def batting(backend):
    return backend.batting


@pytest.fixture(scope="session")
def awards_players(backend):
    return backend.awards_players


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def struct_df(struct):
    return struct.execute()


@pytest.fixture(scope="session")
def udf_df(udf_alltypes):
    return udf_alltypes.execute()


@pytest.fixture(scope="session")
def sorted_df(df):
    return df.sort_values("id").reset_index(drop=True)


@pytest.fixture(scope="session")
def batting_df(batting):
    return batting.execute(limit=None)


@pytest.fixture(scope="session")
def awards_players_df(awards_players):
    return awards_players.execute(limit=None)


@pytest.fixture
def temp_table(con):
    """Return a temporary table name.

    Parameters
    ----------
    con : ibis.backends.Client

    Yields
    ------
    name : string
        Random table name for a temporary usage.
    """
    name = util.gen_name("temp_table")
    yield name
    with contextlib.suppress(NotImplementedError):
        con.drop_table(name, force=True)


@pytest.fixture
def temp_table2(con):
    name = util.gen_name("temp_table2")
    yield name
    with contextlib.suppress(NotImplementedError):
        con.drop_table(name, force=True)


@pytest.fixture
def temp_table_orig(con, temp_table):
    name = f"{temp_table}_orig"
    yield name
    with contextlib.suppress(NotImplementedError):
        con.drop_table(name, force=True)


@pytest.fixture
def temp_view(ddl_con):
    """Return a temporary view name.

    Parameters
    ----------
    ddl_con : backend connection

    Yields
    ------
    name : string
        Random view name for a temporary usage.
    """
    name = util.gen_name("view")
    yield name
    with contextlib.suppress(NotImplementedError):
        ddl_con.drop_view(name, force=True)


@pytest.fixture
@pytest.mark.snapshot_check
def assert_sql(con, snapshot):
    def checker(expr, file_name="out.sql"):
        sql = con.compile(expr, pretty=True)
        snapshot.assert_match(sql, file_name)

    return checker
