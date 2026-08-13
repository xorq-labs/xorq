from __future__ import annotations

import ast
import collections.abc
import importlib.util
import os
import pathlib
import sys
import types

import pytest

import xorq.api as xo
from xorq.common.utils.env_utils import maybe_substitute_env_vars
from xorq.loader import _load_entry_points
from xorq.tests.util import installed_mid_process
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.backends import profiles as profiles_mod
from xorq.vendor.ibis.backends.profiles import (
    Profile,
    Profiles,
    check_for_exposed_secrets,
    con_name_to_secret_key_sources,
    con_name_to_secret_keys,
    get_declared_secret_keys,
)


local_con_names = ("duckdb", "xorq_datafusion", "datafusion", "pandas", "pyiceberg")
remote_connectors = (lambda: xo.postgres.connect_env(),)
local_connectors = tuple(
    lambda: getattr(xo, con_name).connect()  # noqa: B023
    for con_name in local_con_names
)


@pytest.mark.parametrize(
    "con_name",
    local_con_names
    + (
        pytest.param(
            "invalid-con-name", marks=pytest.mark.xfail(reason="only valid con names")
        ),
    ),
)
def test_con_has_profile(con_name):
    con = getattr(xo, con_name).connect()
    assert isinstance(con, BaseBackend)
    profile = getattr(con, "_profile", None)
    assert isinstance(profile, Profile)
    assert profile.almost_equals(Profile.from_con(con))
    #
    other = profile.get_con()
    assert con.name == other.name
    # this doesn't work because _con_args, _con_kwargs doesn't get the defaults which are eventually invoked
    # assert hash(con) == hash(other)
    assert profile.almost_equals(other._profile)


@pytest.mark.postgres
@pytest.mark.parametrize("connect", remote_connectors)
def test_remote_con_works(connect):
    con = connect()
    assert isinstance(con, BaseBackend)
    profile = getattr(con, "_profile", None)
    assert isinstance(profile, Profile)
    assert profile.almost_equals(Profile.from_con(con))
    #
    other = profile.get_con()
    assert con.name == other.name
    # this doesn't work because _con_args, _con_kwargs doesn't get the defaults which are eventually invoked
    # assert hash(con) == hash(other)
    assert profile.almost_equals(other._profile)
    assert con.list_tables() == other.list_tables()


def test_profiles(monkeypatch, tmp_path):
    default_profile_dir = xo.options.profiles.profile_dir
    assert default_profile_dir == pathlib.Path("~/.config/xorq/profiles").expanduser()
    profiles = Profiles()
    assert profiles.profile_dir == default_profile_dir

    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profiles = Profiles()
    assert profiles.profile_dir == tmp_path
    assert not profiles.list()


@pytest.mark.parametrize(
    "connector",
    [pytest.param(c, marks=pytest.mark.postgres) for c in remote_connectors]
    + list(local_connectors),
)
def test_save_load(connector, monkeypatch, tmp_path):
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    # In the is case letsql has a raw passwords so its value is
    # ***elided*** so we can't instantiate it
    os.environ["LETSQL_PASSWORD"] = "letsql"
    con = connector()
    profiles = Profiles()
    profile = con._profile
    profile.save(check_secrets=False)

    others = (
        profiles.get(profile.hash_name),
        profiles[profile.hash_name],
        profile.load(profile.hash_name),
    )

    for other in others:
        assert profile == other
        assert con.list_tables() == other.get_con().list_tables()

    del os.environ["LETSQL_PASSWORD"]


def test_profile_hash_order_independence():
    # different order same kwargs
    profile1 = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost-dum"),
            ("port", 5432),
            ("user", "testuser"),
            ("database", "testdb"),
        ),
    )

    profile2 = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("port", 5432),
            ("database", "testdb"),
            ("user", "testuser"),
            ("host", "localhost-dum"),
        ),
    )

    # check hash order agnostic
    assert profile1.hash_name.split("_")[0] == profile2.hash_name.split("_")[0]
    assert profile1.hash_name.split("_")[1] != profile2.hash_name.split("_")[1]
    # check sort
    assert profile1.kwargs_tuple == profile2.kwargs_tuple

    # check clone
    cloned = profile1.clone()
    assert cloned.hash_name.split("_")[0] == profile1.hash_name.split("_")[0]


def test_profile_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend 'nonexistent'"):
        Profile(con_name="nonexistent", kwargs_tuple=())


def test_parse_env_vars_empty_dict():
    """Test with empty dictionary."""
    assert maybe_substitute_env_vars({}) == {}


def test_parse_env_vars_no_env_vars():
    """Test with dictionary containing no environment variables."""
    input_dict = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "non_string": 123,
        "none_value": None,
        "empty_string": "",
    }
    assert maybe_substitute_env_vars(input_dict) == input_dict


def test_parse_env_vars_dollar_brace_format(monkeypatch):
    """Test with ${VAR} format environment variables."""
    # Set environment variables for testing
    monkeypatch.setenv("TEST_USER", "testuser")
    monkeypatch.setenv("TEST_PASSWORD", "secretpass")

    input_dict = {
        "host": "localhost",
        "port": 5432,
        "user": "${TEST_USER}",
        "password": "${TEST_PASSWORD}",
        "non_env": "regular_value",
    }

    expected = {
        "host": "localhost",
        "port": 5432,
        "user": "testuser",
        "password": "secretpass",
        "non_env": "regular_value",
    }

    assert maybe_substitute_env_vars(input_dict) == expected


def test_parse_env_vars_dollar_format(monkeypatch):
    """Test with $VAR format environment variables."""
    # Set environment variables for testing
    monkeypatch.setenv("TEST_USER", "testuser")
    monkeypatch.setenv("TEST_PASSWORD", "secretpass")

    input_dict = {
        "host": "localhost",
        "port": 5432,
        "user": "$TEST_USER",
        "password": "$TEST_PASSWORD",
        "non_env": "regular_value",
    }

    expected = {
        "host": "localhost",
        "port": 5432,
        "user": "testuser",
        "password": "secretpass",
        "non_env": "regular_value",
    }

    assert maybe_substitute_env_vars(input_dict) == expected


def test_parse_env_vars_mixed_formats(monkeypatch):
    """Test with mixed ${VAR} and $VAR formats."""
    # Set environment variables for testing
    monkeypatch.setenv("TEST_USER", "testuser")
    monkeypatch.setenv("TEST_PASSWORD", "secretpass")

    input_dict = {
        "host": "localhost",
        "port": 5432,
        "user": "${TEST_USER}",
        "password": "$TEST_PASSWORD",
        "non_env": "regular_value",
    }

    expected = {
        "host": "localhost",
        "port": 5432,
        "user": "testuser",
        "password": "secretpass",
        "non_env": "regular_value",
    }

    assert maybe_substitute_env_vars(input_dict) == expected


def test_parse_env_vars_non_string_values(monkeypatch):
    """Test with non-string values."""
    monkeypatch.setenv("TEST_VAR", "test_value")

    input_dict = {
        "string": "${TEST_VAR}",
        "integer": 123,
        "float": 45.67,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
    }

    expected = {
        "string": "test_value",
        "integer": 123,
        "float": 45.67,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
    }

    assert maybe_substitute_env_vars(input_dict) == expected


def test_parse_env_vars_missing_env_var(monkeypatch):
    """Test with missing environment variables - should raise ValueError."""
    monkeypatch.setenv("EXISTING_VAR", "exists")

    input_dict = {
        "existing": "${EXISTING_VAR}",
        "missing": "${MISSING_VAR}",
        "regular": "value",
    }

    with pytest.raises(KeyError, match="'MISSING_VAR'"):
        maybe_substitute_env_vars(input_dict)


def test_parse_env_vars_dollar_sign_in_string(monkeypatch):
    """Test with strings containing dollar signs but not as env vars."""
    input_dict = {
        "code": "a$b$c",
        "text": "This costs $5",
        "complex": "a${non-env}b",  # Not a proper env var syntax
    }

    expected = {
        "code": "a$b$c",
        "text": "This costs $5",
        "complex": "a${non-env}b",
    }

    assert maybe_substitute_env_vars(input_dict) == expected


def test_parse_env_vars_preserve_case(monkeypatch):
    """Test that environment variable case is preserved."""
    monkeypatch.setenv("UPPERCASE", "value1")
    monkeypatch.setenv("lowercase", "value2")
    monkeypatch.setenv("MixedCase", "value3")

    input_dict = {
        "var1": "${UPPERCASE}",
        "var2": "${lowercase}",
        "var3": "${MixedCase}",
    }

    expected = {
        "var1": "value1",
        "var2": "value2",
        "var3": "value3",
    }

    assert maybe_substitute_env_vars(input_dict) == expected


def test_parse_env_vars_nested_structures(monkeypatch):
    """Test how function handles nested structures (should only process top level)."""
    monkeypatch.setenv("TEST_VAR", "test_value")

    input_dict = {
        "top_level": "${TEST_VAR}",
        "nested_dict": {
            "env_var": "${TEST_VAR}",  # This should not be processed
            "normal": "value",
        },
        "list_with_vars": ["${TEST_VAR}", "normal"],  # This should not be processed
    }

    expected = {
        "top_level": "test_value",
        "nested_dict": {"env_var": "${TEST_VAR}", "normal": "value"},
        "list_with_vars": ["${TEST_VAR}", "normal"],
    }

    assert maybe_substitute_env_vars(input_dict) == expected


def test_connection_with_env_vars_preserves_env_vars(monkeypatch, tmp_path):
    """Test that connections instantiated with env vars preserve them in profiles."""

    # Set up test environment
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)

    # Set environment variables to match the existing profile values
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "postgres")
    monkeypatch.setenv("POSTGRES_DB", "ibis_testing")

    con_postgres = xo.postgres.connect(
        host="${POSTGRES_HOST}",
        user="${POSTGRES_USER}",
        password="${POSTGRES_PASSWORD}",
        database="${POSTGRES_DB}",
        port=5432,
    )

    # Get profile from connection
    profile = con_postgres._profile

    # Verify profile has env var references
    assert profile.kwargs_dict["host"] == "${POSTGRES_HOST}"
    assert profile.kwargs_dict["user"] == "${POSTGRES_USER}"
    assert profile.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"
    assert profile.kwargs_dict["database"] == "${POSTGRES_DB}"

    # Save profile
    profile.save(alias="pg_env_var_test", check_secrets=False)

    # Create Profiles instance to load profiles
    profiles = Profiles(profile_dir=tmp_path)

    # Get profiles in different ways
    loaded_profiles = [
        profiles.get(profile.hash_name),
        profiles[profile.hash_name],
        profile.load(profile.hash_name, profile_dir=tmp_path),
    ]

    # Verify all loaded profiles have env var references
    for loaded_profile in loaded_profiles:
        assert loaded_profile.kwargs_dict["host"] == "${POSTGRES_HOST}"
        assert loaded_profile.kwargs_dict["user"] == "${POSTGRES_USER}"
        assert loaded_profile.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"

        # Create connection from loaded profile
        loaded_con = loaded_profile.get_con()

        # Verify the connection's profile still has env var references
        assert loaded_con._profile is not None
        assert loaded_con._profile.kwargs_dict["host"] == "${POSTGRES_HOST}"
        assert loaded_con._profile.kwargs_dict["user"] == "${POSTGRES_USER}"
        assert loaded_con._profile.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"

        # Test that the connection works by comparing to a simple list_tables call
        tables1 = con_postgres.list_tables()
        tables2 = loaded_con.list_tables()
        assert tables1 == tables2


def test_check_for_exposed_secrets_password_no_env_var():
    """Test that a profile with password not using env var is rejected."""
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost"),
            ("port", 5432),
            ("database", "postgres"),
            ("user", "postgres"),
            ("password", "secret"),  # Not using env var
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        profile.check_for_exposed_secrets()

    # Check error message contains password
    assert "'password'" in str(excinfo.value)
    assert "$password or ${password}" in str(excinfo.value)


def test_check_for_exposed_secrets_password_with_env_var_dollar():
    """Test that a profile with password using $ENV_VAR format is accepted."""
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost"),
            ("port", 5432),
            ("database", "postgres"),
            ("user", "postgres"),
            ("password", "$PASSWORD"),  # Using env var
        ),
    )

    # Should not raise an error
    profile.check_for_exposed_secrets()


def test_check_for_exposed_secrets_password_with_env_var_dollar_brace():
    """Test that a profile with password using ${ENV_VAR} format is accepted."""
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost"),
            ("port", 5432),
            ("database", "postgres"),
            ("user", "postgres"),
            ("password", "${PASSWORD}"),  # Using env var
        ),
    )

    # Should not raise an error
    profile.check_for_exposed_secrets()


def test_check_for_exposed_secrets_postgres_specific_secret_keys():
    """Test that postgres-specific secret keys are checked."""
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost"),
            ("port", 5432),
            ("database", "postgres"),
            ("user", "postgres"),
            ("password", "$PASSWORD"),  # Using env var
            ("sslcert", "/path/to/cert"),  # Not using env var
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        profile.check_for_exposed_secrets()

    # Check error message contains sslcert
    assert "'sslcert'" in str(excinfo.value)


def test_check_for_exposed_secrets_snowflake_specific_secret_keys():
    """Test that snowflake-specific secret keys are checked."""
    profile = Profile(
        con_name="snowflake",
        kwargs_tuple=(
            ("host", "localhost"),
            ("database", "snowflake"),
            ("password", "$PASSWORD"),  # Using env var
            (
                "user",
                "snowuser",
            ),  # Not using env var - snowflake treats this as sensitive
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        profile.check_for_exposed_secrets()

    # Check error message contains user
    assert "'user'" in str(excinfo.value)


def test_check_for_exposed_secrets_check_secrets_disabled(tmp_path):
    """Test that check_secrets=False allows profiles with secrets."""
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost"),
            ("port", 5432),
            ("database", "postgres"),
            ("user", "postgres"),
            ("password", "secret"),  # Not using env var
        ),
    )

    # Should not raise an error when check_secrets=False
    profile.save(tmp_path, check_secrets=False)
    with pytest.raises(ValueError, match="Profile contains exposed secret keys"):
        profile.save(tmp_path, check_secrets=True)


def test_check_for_exposed_secrets_multiple_exposed_secrets():
    """Test error message when multiple secrets are exposed."""
    profile = Profile(
        con_name="snowflake",
        kwargs_tuple=(
            ("host", "localhost"),
            ("database", "snowflake"),
            ("password", "secret"),  # Not using env var
            ("user", "admin"),  # Not using env var
            ("account", "acc123"),  # Not using env var
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        profile.check_for_exposed_secrets()

    # Check error message contains all secrets
    error_msg = str(excinfo.value)
    assert "'password'" in error_msg
    assert "'user'" in error_msg
    assert "'account'" in error_msg


def test_check_for_exposed_secrets_unknown_backend_defaults_to_password():
    """Test that unknown backends default to checking password."""
    profile = Profile(
        con_name="duckdb",  # Not in the secret_keys dict
        kwargs_tuple=(
            ("path", "mydb.duckdb"),
            ("password", "secret"),  # Not using env var
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        profile.check_for_exposed_secrets()

    # Check error message contains password
    assert "'password'" in str(excinfo.value)


def test_check_for_exposed_secrets_save_method_calls_check_secrets_fail_then_pass(
    tmp_path, monkeypatch
):
    """Test that save() method calls _check_for_exposed_secrets."""
    profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "localhost"),
            ("port", 5432),
            ("database", "postgres"),
            ("user", "postgres"),
            ("password", "secret"),  # Not using env var
        ),
    )

    # Override the profile directory for testing
    monkeypatch.setattr("xorq.api.options.profiles.profile_dir", tmp_path)

    with pytest.raises(ValueError) as excinfo:
        profile.save()

    assert "'password'" in str(excinfo.value)

    # Should succeed with check_secrets=False
    profile.save(check_secrets=False)


def test_profile_from_con_preserves_env_vars(monkeypatch, tmp_path):
    """Test that Profile.from_con() preserves environment variables from the original profile."""

    # Set up the profile directory for testing
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)

    # Set up environment variables
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "postgres")

    # Create a profile with environment variable references
    original_profile = Profile(
        con_name="postgres",
        kwargs_tuple=(
            ("host", "${POSTGRES_HOST}"),
            ("port", 5432),
            ("database", "postgres"),
            ("user", "${POSTGRES_USER}"),
            ("password", "${POSTGRES_PASSWORD}"),
        ),
    )

    # Create a connection from the profile
    try:
        connection = original_profile.get_con()

        # Create a profile from the connection using from_con
        profile_from_connection = Profile.from_con(connection)

        # Check if environment variables are preserved
        assert profile_from_connection.kwargs_dict["host"] == "${POSTGRES_HOST}"
        assert profile_from_connection.kwargs_dict["user"] == "${POSTGRES_USER}"
        assert profile_from_connection.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"

        # Test saving and loading the profile from connection
        saved_path = profile_from_connection.save(alias="test_profile", clobber=True)
        assert saved_path.exists()
        loaded_profile = Profile.load("test_profile")

        # Check loaded profile still has env vars
        assert loaded_profile.kwargs_dict["host"] == "${POSTGRES_HOST}"
        assert loaded_profile.kwargs_dict["user"] == "${POSTGRES_USER}"
        assert loaded_profile.kwargs_dict["password"] == "${POSTGRES_PASSWORD}"

    except Exception as e:
        if "connection refused" in str(e).lower():
            pytest.skip(f"Database connection failed: {e}")
        else:
            raise


def test_profile_matches_find_backend(data_dir: pathlib.Path) -> None:
    path = data_dir / "parquet" / "diamonds.parquet"
    con = xo.connect()
    t = xo.deferred_read_parquet(path, con)
    assert con._profile == t._find_backend()._profile


@pytest.mark.parametrize("con_name", sorted(con_name_to_secret_keys))
def test_secret_key_mirror_matches_backend_declaration(con_name: str) -> None:
    """The con_name_to_secret_keys mirror must not drift from the keys each
    backend's Backend class declares in _secret_keys. check_for_exposed_secrets
    reads the mirror rather than the backend (so it needn't import the backend),
    so a silent divergence would go unnoticed; this pins the two together."""
    entry_point = next((ep for ep in _load_entry_points() if ep.name == con_name), None)
    assert entry_point is not None, f"no entry point for {con_name!r}"
    try:
        module = entry_point.load()
    except ImportError as e:
        pytest.skip(f"{con_name} backend not importable: {e}")
    declared = getattr(module.Backend, "_secret_keys", None)
    assert declared is not None, (
        f"{con_name} is in con_name_to_secret_keys but its Backend declares no "
        "_secret_keys; declare them so the mirror stays honest"
    )
    assert tuple(declared) == tuple(con_name_to_secret_keys[con_name])


@pytest.mark.parametrize("con_name", sorted(ep.name for ep in _load_entry_points()))
def test_declared_secret_keys_are_mirrored(con_name: str) -> None:
    """Inverse of test_secret_key_mirror_matches_backend_declaration: every
    installed backend that declares _secret_keys must also appear (and match)
    in the con_name_to_secret_keys mirror. Without this a backend could declare
    keys yet be absent from the mirror, so check_for_exposed_secrets would
    silently fall back to just ("password",) for it."""
    entry_point = next(ep for ep in _load_entry_points() if ep.name == con_name)
    try:
        module = entry_point.load()
    except ImportError as e:
        pytest.skip(f"{con_name} backend not importable: {e}")
    declared = getattr(getattr(module, "Backend", None), "_secret_keys", None)
    if declared is None:
        pytest.skip(f"{con_name} declares no _secret_keys")
    assert con_name in con_name_to_secret_keys, (
        f"{con_name} declares _secret_keys but is missing from the "
        "con_name_to_secret_keys mirror; add it so the mirror stays a complete "
        "reflection of all declaring backends"
    )
    assert tuple(declared) == tuple(con_name_to_secret_keys[con_name])


def test_mirrored_secret_key_sources_are_tuples_of_str() -> None:
    """Declaration validity for the mirror itself: every entry is a tuple of
    sources, each source a non-empty tuple of str steps. A malformed
    declaration is an authoring-time bug this test catches; resolution never
    guards for it at runtime."""
    for con_name, sources in con_name_to_secret_key_sources.items():
        assert isinstance(sources, tuple), con_name
        for source in sources:
            assert isinstance(source, tuple) and source, (con_name, source)
            assert all(isinstance(step, str) for step in source), (con_name, source)


@pytest.mark.parametrize("con_name", sorted(con_name_to_secret_key_sources))
def test_secret_key_sources_mirror_matches_backend_declaration(con_name: str) -> None:
    """con_name_to_secret_key_sources must not drift from the sources each
    backend's Backend class declares in _secret_key_sources, exactly as the
    static-keys mirror is pinned to _secret_keys."""
    entry_point = next((ep for ep in _load_entry_points() if ep.name == con_name), None)
    assert entry_point is not None, f"no entry point for {con_name!r}"
    try:
        module = entry_point.load()
    except ImportError as e:
        pytest.skip(f"{con_name} backend not importable: {e}")
    declared = getattr(module.Backend, "_secret_key_sources", None)
    assert declared is not None, (
        f"{con_name} is in con_name_to_secret_key_sources but its Backend "
        "declares no _secret_key_sources; declare them so the mirror stays honest"
    )
    assert tuple(declared) == tuple(con_name_to_secret_key_sources[con_name])


@pytest.mark.parametrize("con_name", sorted(ep.name for ep in _load_entry_points()))
def test_declared_secret_key_sources_are_mirrored(con_name: str) -> None:
    """Inverse of test_secret_key_sources_mirror_matches_backend_declaration:
    every installed backend declaring _secret_key_sources must also appear (and
    match) in the mirror. The mirror is what lets the sources resolve with the
    backend unimported; an unmirrored declaration would silently degrade to the
    import-dependent class read."""
    entry_point = next(ep for ep in _load_entry_points() if ep.name == con_name)
    try:
        module = entry_point.load()
    except ImportError as e:
        pytest.skip(f"{con_name} backend not importable: {e}")
    declared = getattr(getattr(module, "Backend", None), "_secret_key_sources", None)
    if declared is None:
        pytest.skip(f"{con_name} declares no _secret_key_sources")
    assert con_name in con_name_to_secret_key_sources, (
        f"{con_name} declares _secret_key_sources but is missing from the "
        "con_name_to_secret_key_sources mirror; add it so its sources resolve "
        "without the backend imported"
    )
    assert tuple(declared) == tuple(con_name_to_secret_key_sources[con_name])


def _source_declares(module_name: str, name: str) -> bool | None:
    """Whether a module's source defines or assigns `name`, without importing it:
    find_spec locates the file but doesn't execute it. A package is searched
    whole, since a Backend is commonly re-exported from a submodule. None when
    there is no source to read."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    paths = [pathlib.Path(spec.origin)] if str(spec.origin).endswith(".py") else []
    for location in spec.submodule_search_locations or ():
        paths.extend(sorted(pathlib.Path(location).rglob("*.py")))
    if not paths:
        return None
    return any(
        getattr(node, "name", None) == name
        or (isinstance(node, ast.Name) and node.id == name)
        for path in paths
        for node in ast.walk(ast.parse(path.read_text()))
    )


def test_source_declares_finds_names_without_importing() -> None:
    """The fallback the guardrail test leans on for an unimportable backend: names
    are found in the source, and a name that isn't there reads as absent rather
    than as "can't tell"."""
    module_name = "xorq.vendor.ibis.backends.profiles"
    assert _source_declares(module_name, "get_secret_keys") is True
    assert _source_declares(module_name, "con_name_to_secret_key_sources") is True
    assert _source_declares(module_name, "_get_secret_keys") is False
    assert _source_declares("xorq_no_such_module", "_secret_keys") is None


def test_source_declares_searches_a_package_for_a_re_exported_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend package whose Backend is re-exported from a submodule is still
    searched: reading only the entry-point module's own source would report the
    declaration as absent and skip the guardrail for exactly the backends CI
    cannot import."""
    package = tmp_path / "xorq_reexporting_backend"
    package.mkdir()
    (package / "__init__.py").write_text("from .backend import Backend\n")
    (package / "backend.py").write_text(
        "class Backend:\n"
        "    _secret_keys = ('token',)\n"
        "    _secret_key_sources = (('config', 'auth', 'fields'),)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert _source_declares(package.name, "_secret_key_sources") is True
    assert _source_declares(package.name, "_secret_keys") is True
    assert _source_declares(package.name, "_no_such_declaration") is False


@pytest.mark.parametrize("con_name", sorted(ep.name for ep in _load_entry_points()))
def test_declaring_backends_also_mirror_static_keys(con_name: str) -> None:
    """A backend declaring _secret_key_sources must also declare static
    _secret_keys: a source only resolves when the kwargs actually carry what it
    points at, so the always-present names belong in the static mirror, which
    is checked with no kwargs at all.

    A backend whose extras aren't installed is checked against its module source
    rather than skipped -- skipping there would leave the guardrail unenforced
    for exactly the backends CI can't import."""
    hint = (
        f"{con_name} declares _secret_key_sources but no static _secret_keys; "
        "mirror the always-present names, since a source contributes only when "
        "the kwargs carry what it points at"
    )
    entry_point = next(ep for ep in _load_entry_points() if ep.name == con_name)
    try:
        module = entry_point.load()
    except ImportError:
        declares_sources = _source_declares(entry_point.module, "_secret_key_sources")
        if declares_sources is None:
            pytest.skip(f"{con_name} is neither importable nor locatable")
        if not declares_sources:
            pytest.skip(f"{con_name} declares no _secret_key_sources (by source)")
        assert _source_declares(entry_point.module, "_secret_keys"), hint
        return
    backend = getattr(module, "Backend", None)
    if getattr(backend, "_secret_key_sources", None) is None:
        pytest.skip(f"{con_name} declares no _secret_key_sources")
    assert getattr(backend, "_secret_keys", None), hint


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend_cls: type,
    con_name: str = "fakedb",
    module_name: str = "xorq_fake_backend_mod",
    imported: bool = True,
) -> str:
    """Register a throwaway backend so the declared-sources tier finds
    backend_cls through the import-dependent class read: patch the entry-point
    lookup, and mark the module as imported, which that read requires. Pass
    imported=False for the not-imported case."""
    module = types.ModuleType(module_name)
    module.Backend = backend_cls
    entry_point = types.SimpleNamespace(
        name=con_name, module=module_name, load=lambda: module
    )
    monkeypatch.setattr(profiles_mod, "_load_entry_points", lambda: (entry_point,))
    monkeypatch.setattr(
        profiles_mod,
        "_find_entry_point",
        lambda name: entry_point if name == con_name else None,
    )
    if imported:
        monkeypatch.setitem(sys.modules, module_name, module)
    return con_name


_REST_STYLE_SOURCES = (
    ("config", "auth", "secret_fields"),
    ("config", "auth", "fields"),
)


class _DeclaringBackend:
    _secret_key_sources = _REST_STYLE_SOURCES


def test_get_declared_secret_keys_first_resolving_source_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sources are ordered fallback: when the first resolves, the second is
    never consulted -- an explicit secret_fields narrows what fields would
    have said."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = {
        "config": {
            "auth": {"fields": ["user", "api_key"], "secret_fields": ["api_key"]}
        }
    }
    assert get_declared_secret_keys(con_name, kwargs) == ("api_key",)


@pytest.mark.parametrize(
    "auth",
    (
        pytest.param({"fields": ["api_key"]}, id="secret_fields_absent"),
        pytest.param(
            {"fields": ["api_key"], "secret_fields": None}, id="secret_fields_null"
        ),
    ),
)
def test_get_declared_secret_keys_falls_back_in_order(
    monkeypatch: pytest.MonkeyPatch, auth: dict
) -> None:
    """The second source is used exactly when the first doesn't resolve: an
    absent secret_fields falls through, and so does an explicit null -- None
    is unresolved, not "no names"."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    assert get_declared_secret_keys(con_name, {"config": {"auth": auth}}) == (
        "api_key",
    )


def test_empty_secret_fields_resolves_and_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secret_fields: [] resolves to () and wins -- "none of the fields are
    secret" is genuinely different from the key being absent
    (presence-and-not-None, never truthiness) -- and the unconditional default
    still applies: () is not an opt-out."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = {"config": {"auth": {"fields": ["api_key"], "secret_fields": []}}}
    assert get_declared_secret_keys(con_name, kwargs) == ()
    assert profiles_mod.get_secret_keys(con_name, kwargs) == ("password",)
    with pytest.raises(ValueError, match="password"):
        check_for_exposed_secrets(con_name, {**kwargs, "password": "plaintext"})


class _AttrsLike:
    """A config carried as an object rather than a plain dict -- e.g. a
    RestBackendConfig instance handed to connect() -- is never reached into;
    profiles carry configs as plain dicts."""

    auth = {"fields": ["never_reached"]}


@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param({}, id="no_config_at_all"),
        pytest.param({"config": None}, id="config_is_none"),
        pytest.param({"config": {"other": 1}}, id="missing_step"),
        pytest.param({"config": {"auth": "basic"}}, id="non_dict_intermediate"),
        pytest.param({"config": {"auth": ["fields"]}}, id="list_intermediate"),
        pytest.param({"config": _AttrsLike()}, id="attrs_instance_intermediate"),
        pytest.param({"config": {"auth": {"fields": "token"}}}, id="bare_str_leaf"),
        pytest.param({"config": {"auth": {"fields": {"token": 1}}}}, id="dict_leaf"),
        pytest.param({"config": {"auth": {"fields": 42}}}, id="non_sequence_leaf"),
    ),
)
def test_unresolved_sources_contribute_nothing(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict
) -> None:
    """Every shape a source cannot read -- a missing step, a non-dict
    intermediate, a leaf that isn't a list/tuple -- is unresolved: the tier
    contributes nothing, silently, and tiers 1+2 still enforce. The bare-str
    leaf in particular stops being a hazard rather than being guarded: a str
    is not a list/tuple, so `fields: "token"` can never be iterated into
    ('t', 'o', 'k', ...)."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    assert get_declared_secret_keys(con_name, kwargs) == ()
    assert profiles_mod.get_secret_keys(con_name, kwargs) == ("password",)
    with pytest.raises(ValueError, match="password"):
        check_for_exposed_secrets(con_name, {**kwargs, "password": "plaintext"})


def test_mixed_leaf_keeps_the_str_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list mixing str and non-str names keeps the str ones: the leaf is
    user config data, not the declaration, so a declaration test can't catch
    it, and checking "api_key" is strictly safer than dropping the source. A
    non-str name can never match a kwarg, so dropping it needs no warning."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = {"config": {"auth": {"fields": ["api_key", 3, None]}}}
    assert get_declared_secret_keys(con_name, kwargs) == ("api_key",)
    with pytest.raises(ValueError, match="api_key"):
        check_for_exposed_secrets(con_name, {**kwargs, "api_key": "plaintext"})
    check_for_exposed_secrets(con_name, {**kwargs, "api_key": "${API_KEY}"})


def test_dict_subclass_overrides_are_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No backend-authored code runs during resolution: a dict subclass
    forging results from get/__getitem__/__missing__ has those overrides
    bypassed -- the unbound dict.get reads the true underlying data. Applies
    to kwargs itself and to every nested step."""

    class Forging(dict):
        def get(self, key: str, default: object = None) -> object:
            return {"auth": {"fields": ["forged"]}}

        def __getitem__(self, key: str) -> object:
            return {"auth": {"fields": ["forged"]}}

        def __missing__(self, key: str) -> object:
            return {"auth": {"fields": ["forged"]}}

    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = Forging(config=Forging(auth=Forging(fields=["true_key"])))
    assert get_declared_secret_keys(con_name, kwargs) == ("true_key",)


def test_a_mapping_that_is_not_a_dict_is_not_reached_into(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intermediate value that isn't a plain dict is unresolved with no
    call made: resolution never invokes a Mapping's own lookup methods."""

    class RecordingMapping(collections.abc.Mapping):
        def __init__(self, data: dict) -> None:
            self.data = data
            self.lookups: list[str] = []

        def __getitem__(self, key: str) -> object:
            self.lookups.append(key)
            return self.data[key]

        def __iter__(self) -> collections.abc.Iterator:
            return iter(self.data)

        def __len__(self) -> int:
            return len(self.data)

    config = RecordingMapping({"auth": {"fields": ["api_key"]}})
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = {"config": config}
    assert get_declared_secret_keys(con_name, kwargs) == ()
    assert config.lookups == []
    assert profiles_mod.get_secret_keys(con_name, kwargs) == ("password",)


def test_a_lying_class_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """An object whose __class__ lies passes the isinstance check and makes
    the unbound dict.get raise TypeError; the guard converts that into a
    ValueError naming the con_name and source -- fail closed, with no kwarg
    value in the message, since only our own declared source is interpolated.
    (This needs an adversarial object already inside the caller's kwargs, the
    same trust domain as the credentials themselves.)"""

    class Liar:
        @property
        def __class__(self) -> type:
            return dict

    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = {"config": Liar(), "token": "hunter2-secret"}
    with pytest.raises(
        ValueError, match="could not resolve secret-key source"
    ) as excinfo:
        check_for_exposed_secrets(con_name, kwargs)
    message = str(excinfo.value)
    assert con_name in message
    assert "TypeError" in message
    assert "hunter2" not in message
    # and through Profile.save(): the save aborts and writes nothing
    profile = Profile(con_name=con_name, kwargs_tuple=(("config", Liar()),))
    with pytest.raises(ValueError, match="could not resolve secret-key source"):
        profile.save(profile_dir=tmp_path)
    assert not tuple(tmp_path.iterdir())


def test_a_property_declaration_contributes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_secret_key_sources declared as a property never executes:
    inspect.getattr_static reads the class dict without firing descriptors and
    returns the descriptor object, which fails the tuple check. The property
    lives on the metaclass, where a plain getattr on the class *would* run it
    -- and would get back a well-formed tuple, so only the no-execution rule
    keeps it out."""
    executed = []

    class Meta(type):
        @property
        def _secret_key_sources(cls) -> tuple:
            executed.append(True)
            return _REST_STYLE_SOURCES

    class Backend(metaclass=Meta):
        pass

    con_name = _install_fake_backend(monkeypatch, Backend)
    kwargs = {"config": {"auth": {"fields": ["api_key"]}}}
    assert get_declared_secret_keys(con_name, kwargs) == ()
    assert not executed
    assert profiles_mod.get_secret_keys(con_name, kwargs) == ("password",)


def test_a_non_tuple_class_declaration_contributes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed class declaration -- a list here -- fails the tuple check and
    contributes nothing: declaration shape is an authoring-time concern, pinned by
    the mirror tests, never a runtime guard."""

    class Backend:
        _secret_key_sources = [("config", "auth", "fields")]

    con_name = _install_fake_backend(monkeypatch, Backend)
    kwargs = {"config": {"auth": {"fields": ["api_key"]}}}
    assert get_declared_secret_keys(con_name, kwargs) == ()
    assert profiles_mod.get_secret_keys(con_name, kwargs) == ("password",)


@pytest.mark.parametrize(
    "declared",
    (
        pytest.param((42,), id="source_is_not_a_tuple"),
        pytest.param((["config", "auth"],), id="source_is_a_list"),
        pytest.param(((),), id="source_is_empty"),
        pytest.param((("config", 3),), id="step_is_not_str"),
        pytest.param(("config", "auth", "fields"), id="steps_not_nested_in_a_source"),
    ),
)
def test_malformed_sources_contribute_nothing(
    monkeypatch: pytest.MonkeyPatch, declared: tuple
) -> None:
    """A declaration whose *elements* are malformed contributes nothing, like a
    malformed declaration as a whole -- it does not raise, and it does not block
    every save for that backend. The last case is the likely authoring slip: a
    flat tuple of steps, which without the shape check walks single-character
    keys."""

    class Backend:
        _secret_key_sources = declared

    con_name = _install_fake_backend(monkeypatch, Backend)
    kwargs = {"config": {"auth": {"fields": ["api_key"]}}}
    assert get_declared_secret_keys(con_name, kwargs) == ()
    assert profiles_mod.get_secret_keys(con_name, kwargs) == ("password",)
    check_for_exposed_secrets(con_name, {**kwargs, "api_key": "plaintext"})


@pytest.mark.parametrize(
    "base",
    (pytest.param(list, id="list_subclass"), pytest.param(tuple, id="tuple_subclass")),
)
def test_a_leaf_subclass_has_its_true_names_read(
    monkeypatch: pytest.MonkeyPatch, base: type
) -> None:
    """The leaf is read the way the steps are, through an unbound builtin, so a
    list/tuple subclass forging __iter__ has the override bypassed and its true
    data read. Neither believing the forgery (which invents names) nor rejecting
    the subclass (which drops a benign one's real names, and with them the check
    on the credential they name) is safe."""

    class Forging(base):
        def __iter__(self) -> collections.abc.Iterator:
            return iter(["forged"])

    leaf = Forging(["true_key"])
    assert tuple(leaf) == ("forged",), "the forgery is live"
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = {"config": {"auth": {"fields": leaf}}}
    assert get_declared_secret_keys(con_name, kwargs) == ("true_key",)
    with pytest.raises(ValueError, match="true_key"):
        check_for_exposed_secrets(con_name, {**kwargs, "true_key": "plaintext"})


def test_a_forged_empty_leaf_cannot_suppress_the_true_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nor can a forgery narrow the check by resolving to less: a subclass whose
    __iter__ reports nothing still resolves to the names it really holds, rather
    than to an empty set that would win the fallback."""

    class Forging(list):
        def __iter__(self) -> collections.abc.Iterator:
            return iter([])

    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    auth = {"secret_fields": Forging(["api_key"]), "fields": ["other"]}
    kwargs = {"config": {"auth": auth}}
    assert get_declared_secret_keys(con_name, kwargs) == ("api_key",)


def test_a_forged_str_name_cannot_escape_the_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A name is a value out of the kwargs too, so it comes back as an exact str
    copy. A str subclass returned as-is would carry its own __eq__ into the
    membership test that matches it against a kwarg -- the source resolves, the
    key is named, and the literal saves anyway."""

    class Ghost(str):
        def __eq__(self, other: object) -> bool:
            return False

        def __hash__(self) -> int:
            return str.__hash__(self)

        def __str__(self) -> str:
            return "not_a_kwarg"

    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    config = {"auth": {"fields": [Ghost("api_key")]}}
    names = get_declared_secret_keys(con_name, {"config": config})
    assert names == ("api_key",)
    assert all(type(name) is str for name in names)
    with pytest.raises(ValueError, match="api_key"):
        check_for_exposed_secrets(con_name, {"config": config, "api_key": "plaintext"})
    profile = Profile(
        con_name=con_name,
        kwargs_tuple=(("config", config), ("api_key", "plaintext")),
    )
    with pytest.raises(ValueError, match="api_key"):
        profile.save(profile_dir=tmp_path)
    assert not tuple(tmp_path.iterdir())


def test_a_name_whose_hash_raises_cannot_escape_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nor can a name reach the dedupe in get_secret_keys, which hashes it outside
    the fail-closed guard: an exact str copy has nothing left to raise."""

    class HashBomb(str):
        def __hash__(self) -> int:
            raise RuntimeError("boom")

    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    config = {"auth": {"fields": [HashBomb("api_key")]}}
    assert profiles_mod.get_secret_keys(con_name, {"config": config}) == (
        "password",
        "api_key",
    )
    with pytest.raises(ValueError, match="api_key"):
        check_for_exposed_secrets(con_name, {"config": config, "api_key": "plaintext"})


def test_forged_items_cannot_hide_a_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enforcement loop reads the kwargs the way the resolver does: a subclass
    hiding an entry from items() cannot hide it from the check that the resolver
    just named it in."""

    class Hiding(dict):
        def items(self) -> list:
            return [(k, v) for k, v in dict.items(self) if k != "api_key"]

    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = Hiding(config={"auth": {"fields": ["api_key"]}}, api_key="plaintext")
    assert get_declared_secret_keys(con_name, kwargs) == ("api_key",)
    with pytest.raises(ValueError, match="api_key"):
        check_for_exposed_secrets(con_name, kwargs)


def test_a_kwargs_subclass_lying_about_len_still_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalizing absent kwargs tests for None, not truthiness: a dict subclass
    whose __len__ reports empty would otherwise disable the whole tier, silently
    and without the fail-closed error, despite every lookup on it reading the true
    data."""

    class LenLying(dict):
        def __len__(self) -> int:
            return 0

    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    kwargs = LenLying(config={"auth": {"fields": ["api_key"]}})
    assert get_declared_secret_keys(con_name, kwargs) == ("api_key",)
    with pytest.raises(ValueError, match="api_key"):
        check_for_exposed_secrets(con_name, LenLying(**kwargs, api_key="plaintext"))


def test_mirrored_sources_resolve_without_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror is what makes this tier no longer best-effort: a mirrored
    backend's sources resolve with the backend never imported, where the old
    callable tier silently contributed nothing."""

    class Backend:
        _secret_key_sources = _REST_STYLE_SOURCES

    con_name = _install_fake_backend(monkeypatch, Backend, imported=False)
    monkeypatch.setattr(
        profiles_mod,
        "con_name_to_secret_key_sources",
        {con_name: _REST_STYLE_SOURCES},
    )
    kwargs = {"config": {"auth": {"fields": ["api_key"]}}}
    assert get_declared_secret_keys(con_name, kwargs) == ("api_key",)
    with pytest.raises(ValueError, match="api_key"):
        check_for_exposed_secrets(con_name, {**kwargs, "api_key": "plaintext"})


def test_unimported_backend_contributes_no_class_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The class read stays best-effort exactly as the callable tier was: an
    out-of-tree backend that isn't imported (and, being out-of-tree, has no
    mirror entry) contributes nothing, and the unconditional default still
    applies."""

    class Backend:
        _secret_key_sources = _REST_STYLE_SOURCES

    con_name = _install_fake_backend(monkeypatch, Backend, imported=False)
    kwargs = {"config": {"auth": {"fields": ["api_key"]}}}
    assert get_declared_secret_keys(con_name, kwargs) == ()
    assert profiles_mod.get_secret_keys(con_name, kwargs) == ("password",)
    with pytest.raises(ValueError, match="password"):
        check_for_exposed_secrets(con_name, {"password": "plaintext"})


def test_class_and_mirror_sources_are_unioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-tree backends cannot add mirror entries, so an imported
    backend's class declaration tops up the mirror's: mirror sources first,
    class sources after."""

    class Backend:
        _secret_key_sources = (("extra_names",),)

    con_name = _install_fake_backend(monkeypatch, Backend)
    monkeypatch.setattr(
        profiles_mod,
        "con_name_to_secret_key_sources",
        {con_name: (("config", "auth", "fields"),)},
    )
    # the mirror's source is consulted first when both could resolve
    kwargs = {
        "config": {"auth": {"fields": ["from_mirror"]}},
        "extra_names": ["from_class"],
    }
    assert get_declared_secret_keys(con_name, kwargs) == ("from_mirror",)
    # the class-declared source is a top-up: used when the mirror's doesn't resolve
    assert get_declared_secret_keys(con_name, {"extra_names": ["from_class"]}) == (
        "from_class",
    )


def test_get_declared_secret_keys_handles_missing_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No kwargs at all is just the unresolved case: no error, nothing
    contributed."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    assert get_declared_secret_keys(con_name) == ()
    assert profiles_mod.get_secret_keys(con_name) == ("password",)


def test_check_for_exposed_secrets_uses_declared_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a key surfaced only by a declared source is enforced by
    check_for_exposed_secrets, on top of the default/static keys, and an
    env-var reference is accepted."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    config = {"auth": {"fields": ["api_key"]}}
    with pytest.raises(ValueError, match="api_key"):
        check_for_exposed_secrets(con_name, {"config": config, "api_key": "plaintext"})
    check_for_exposed_secrets(con_name, {"config": config, "api_key": "${API_KEY}"})


def test_save_is_blocked_by_a_declared_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The declared tier is enforced through Profile.save(), not just the
    module-level helper: the save aborts and writes nothing."""
    con_name = _install_fake_backend(monkeypatch, _DeclaringBackend)
    config = {"auth": {"fields": ["api_key"]}}
    profile = Profile(
        con_name=con_name,
        kwargs_tuple=(("config", config), ("api_key", "plaintext")),
    )
    with pytest.raises(ValueError, match="api_key"):
        profile.save(profile_dir=tmp_path)
    assert not tuple(tmp_path.iterdir())
    # the same profile with an env-var reference saves
    Profile(
        con_name=con_name,
        kwargs_tuple=(("config", config), ("api_key", "${API_KEY}")),
    ).save(profile_dir=tmp_path)
    assert tuple(tmp_path.iterdir())


def test_get_secret_keys_unions_default_mirror_and_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three tiers are unioned, not chained: a backend that is both
    mirrored and declares sources is checked for the default, the mirrored,
    and the declared keys together. Ordering is deterministic (default,
    mirror, declared) and duplicates collapse."""

    class Backend:
        _secret_key_sources = (("secret_kwarg_names",),)

    _install_fake_backend(monkeypatch, Backend, con_name="postgres")
    keys = profiles_mod.get_secret_keys(
        "postgres", {"secret_kwarg_names": ["token", "sslcert"]}
    )
    mirror = con_name_to_secret_keys["postgres"]
    assert set(keys) == {*profiles_mod.default_secret_keys, *mirror, "token"}
    assert keys[0] == "password"  # tier 1, unconditionally
    assert tuple(key for key in keys if key in mirror) == mirror  # tier 2, in order
    assert keys[-1] == "token"  # tier 3, after the tiers it can only widen
    # first occurrence wins, so the source repeating "sslcert" adds no duplicate
    assert len(keys) == len(set(keys))


def test_declared_sources_cannot_shrink_mirrored_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monotonicity: a source resolving to a subset of a mirrored backend's
    keys -- or to () -- can only widen the checked set, never narrow it."""

    class Backend:
        _secret_key_sources = (("secret_kwarg_names",),)

    _install_fake_backend(monkeypatch, Backend, con_name="postgres")
    keys = profiles_mod.get_secret_keys("postgres", {"secret_kwarg_names": []})
    assert set(con_name_to_secret_keys["postgres"]) <= set(keys)
    with pytest.raises(ValueError, match="password"):
        check_for_exposed_secrets("postgres", {"password": "plaintext"})
    with pytest.raises(ValueError, match="sslcert"):
        check_for_exposed_secrets("postgres", {"sslcert": "/path/to/cert"})


class _StaticKeysBackend:
    _secret_keys = ("secret",)


def test_an_imported_backends_static_keys_top_up_the_mirror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Tier 2 is the mirror topped up from an already-imported backend's static
    _secret_keys, exactly as tier 3 tops up the declared sources: an out-of-tree
    backend cannot add a mirror entry, and a fixed kwarg name like "secret"
    lives in no kwargs data for a source to point at, so without the top-up its
    static declaration was dead documentation -- enforced as a convention by
    test_declaring_backends_also_mirror_static_keys, read by nothing."""
    con_name = _install_fake_backend(monkeypatch, _StaticKeysBackend)
    assert profiles_mod.get_secret_keys(con_name) == ("password", "secret")
    with pytest.raises(ValueError, match="'secret'"):
        check_for_exposed_secrets(con_name, {"secret": "plaintext"})
    # an env-var reference is accepted
    check_for_exposed_secrets(con_name, {"secret": "${SECRET}"})
    # and through Profile.save(): the save aborts and writes nothing
    profile = Profile(con_name=con_name, kwargs_tuple=(("secret", "plaintext"),))
    with pytest.raises(ValueError, match="'secret'"):
        profile.save(profile_dir=tmp_path)
    assert not tuple(tmp_path.iterdir())


def test_unimported_backend_contributes_no_static_class_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static-keys class read stays best-effort exactly as the sources read
    is: an out-of-tree backend that isn't imported (and, being out-of-tree, has
    no mirror entry) contributes nothing, and the unconditional default still
    applies."""
    con_name = _install_fake_backend(monkeypatch, _StaticKeysBackend, imported=False)
    assert profiles_mod.get_secret_keys(con_name) == ("password",)
    check_for_exposed_secrets(con_name, {"secret": "plaintext"})
    with pytest.raises(ValueError, match="'password'"):
        check_for_exposed_secrets(con_name, {"password": "plaintext"})


def test_a_property_static_keys_declaration_contributes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_secret_keys declared as a property never executes: getattr_static reads
    the class dict without firing descriptors and returns the descriptor object,
    which fails the tuple check -- the same no-execution rule as the sources
    read, on the same metaclass layout where a plain getattr would run it."""
    executed = []

    class Meta(type):
        @property
        def _secret_keys(cls) -> tuple:
            executed.append(True)
            return ("secret",)

    class Backend(metaclass=Meta):
        pass

    con_name = _install_fake_backend(monkeypatch, Backend)
    assert profiles_mod.get_secret_keys(con_name) == ("password",)
    assert not executed


def test_a_non_tuple_static_keys_declaration_contributes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed class declaration -- a list here -- fails the tuple check
    and contributes nothing: declaration shape is an authoring-time concern,
    never a runtime guard, and tiers 1 and 3 keep enforcing."""

    class Backend:
        _secret_keys = ["secret"]

    con_name = _install_fake_backend(monkeypatch, Backend)
    assert profiles_mod.get_secret_keys(con_name) == ("password",)


def test_a_mixed_static_keys_declaration_keeps_the_str_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-str members are dropped while the well-formed names beside them are
    kept and enforced -- the same rule as a resolved leaf's, because checking
    the good names is strictly safer than dropping the declaration."""

    class Backend:
        _secret_keys = ("secret", 3, None)

    con_name = _install_fake_backend(monkeypatch, Backend)
    assert profiles_mod.get_secret_keys(con_name) == ("password", "secret")
    with pytest.raises(ValueError, match="'secret'"):
        check_for_exposed_secrets(con_name, {"secret": "plaintext"})


def test_a_static_keys_tuple_subclass_has_its_true_names_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The class declaration is plugin-authored data, so it is read the way the
    resolver reads a leaf: a tuple subclass forging __iter__ has the override
    bypassed and its true names read and enforced."""

    class Forging(tuple):
        def __iter__(self) -> collections.abc.Iterator:
            return iter(("forged",))

    declared = Forging(("secret",))
    assert tuple(declared) == ("forged",), "the forgery is live"

    class Backend:
        _secret_keys = declared

    con_name = _install_fake_backend(monkeypatch, Backend)
    assert profiles_mod.get_secret_keys(con_name) == ("password", "secret")
    with pytest.raises(ValueError, match="'secret'"):
        check_for_exposed_secrets(con_name, {"secret": "plaintext"})


def test_a_forged_str_static_key_cannot_escape_the_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A static key comes back as an exact str copy, like a name out of a
    resolved leaf: returned as-is, a str subclass would carry its own __eq__
    into the membership test that matches it against a kwarg -- the key is
    named in no error and the literal saves anyway."""

    class Ghost(str):
        def __eq__(self, other: object) -> bool:
            return False

        def __hash__(self) -> int:
            return str.__hash__(self)

    class Backend:
        _secret_keys = (Ghost("secret"),)

    con_name = _install_fake_backend(monkeypatch, Backend)
    keys = profiles_mod.get_secret_keys(con_name)
    assert keys == ("password", "secret")
    assert all(type(key) is str for key in keys)
    with pytest.raises(ValueError, match="'secret'"):
        check_for_exposed_secrets(con_name, {"secret": "plaintext"})
    profile = Profile(con_name=con_name, kwargs_tuple=(("secret", "plaintext"),))
    with pytest.raises(ValueError, match="'secret'"):
        profile.save(profile_dir=tmp_path)
    assert not tuple(tmp_path.iterdir())


def test_a_static_key_whose_hash_raises_cannot_escape_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nor can a static key reach the dedupe in get_secret_keys, which hashes
    it outside any guard: an exact str copy has nothing left to raise."""

    class HashBomb(str):
        def __hash__(self) -> int:
            raise RuntimeError("boom")

    class Backend:
        _secret_keys = (HashBomb("secret"),)

    con_name = _install_fake_backend(monkeypatch, Backend)
    assert profiles_mod.get_secret_keys(con_name) == ("password", "secret")
    with pytest.raises(ValueError, match="'secret'"):
        check_for_exposed_secrets(con_name, {"secret": "plaintext"})


def test_static_class_keys_top_up_the_mirror_and_cannot_shrink_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror and class keys are unioned, mirror first: a class declaration on
    a mirrored backend can only widen the checked set, never narrow it."""

    class Backend:
        _secret_keys = ("extra_secret",)

    _install_fake_backend(monkeypatch, Backend, con_name="postgres")
    keys = profiles_mod.get_secret_keys("postgres")
    mirror = con_name_to_secret_keys["postgres"]
    assert set(mirror) <= set(keys)
    assert "extra_secret" in keys
    assert tuple(key for key in keys if key in mirror) == mirror
    with pytest.raises(ValueError, match="'sslcert'"):
        check_for_exposed_secrets("postgres", {"sslcert": "/path/to/cert"})
    with pytest.raises(ValueError, match="'extra_secret'"):
        check_for_exposed_secrets("postgres", {"extra_secret": "plaintext"})


def test_validate_con_name_sees_a_backend_installed_mid_process(
    tmp_path: pathlib.Path,
) -> None:
    """A backend installed into a live process is usable without a restart: a
    direct scan of the cached entry points would reject it as "Unknown backend",
    listing a stale set as the installed ones, until the process restarted."""
    with installed_mid_process(tmp_path, "xorqfakeprofilebackend") as con_name:
        profile = Profile(con_name=con_name, kwargs_tuple=())
        assert profile.con_name == con_name
        # a name that really doesn't exist still raises, against the fresh list
        with pytest.raises(ValueError, match="Unknown backend 'xorqnosuchbackend'"):
            Profile(con_name="xorqnosuchbackend", kwargs_tuple=())
