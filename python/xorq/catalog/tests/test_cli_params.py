# Tests for: --rename-params, -p/--params, ExprBuilder entries, run-cached with params
import pytest
from click.testing import CliRunner

import xorq.api as xo
import xorq.expr.builders as _builders_mod
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.expr.builders import (
    _FROM_TAG_NODE_REGISTRY,
    TagHandler,
    _reset_registry,
    register_tag_handler,
)
from xorq.vendor.ibis.expr import operations as ops


@pytest.fixture
def runner():
    yield CliRunner()


@pytest.fixture
def saved_registry():
    """Save and restore the handler registry around a test."""
    saved = dict(_FROM_TAG_NODE_REGISTRY)
    saved_keys = _builders_mod._BUILTIN_KEYS
    saved_init = _builders_mod._initialized
    yield
    _FROM_TAG_NODE_REGISTRY.clear()
    _FROM_TAG_NODE_REGISTRY.update(saved)
    _builders_mod._BUILTIN_KEYS = saved_keys
    _builders_mod._initialized = saved_init


@pytest.fixture
def catalog_with_source_and_transform(catalog_path):
    """Populate a catalog with a source entry and an unbound transform entry."""
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)

    source = xo.memtable(
        {"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0], "name": ["a", "b", "c"]}
    )
    source_entry = catalog.add(source, aliases=("src",))

    schema = source.schema()
    unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
    transform = unbound.filter(unbound.amount > 0).select("user_id", "amount")
    transform_entry = catalog.add(transform, aliases=("trn",))

    return catalog_path, source_entry.name, transform_entry.name


@pytest.fixture
def catalog_with_parameterized_entries(catalog_path):
    """Populate a catalog with source and transform entries containing NamedScalarParameter nodes."""
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)

    # Source: a memtable with a parameterized filter
    threshold = xo.param("threshold", "float64", default=5.0)
    source = xo.memtable(
        {"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0], "name": ["a", "b", "c"]}
    )
    source_filtered = source.filter(source.amount > threshold)
    source_entry = catalog.add(source_filtered, aliases=("psrc",))

    # Transform: an unbound expr with its own NamedScalarParameter
    limit_param = xo.param("threshold", "float64", default=15.0)
    schema = source_filtered.schema()
    unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
    transform = unbound.filter(unbound.amount > limit_param).select("user_id", "amount")
    transform_entry = catalog.add(transform, aliases=("ptrn",))

    return catalog_path, source_entry.name, transform_entry.name


def _csv_data_rows(output):
    """Return non-header, non-empty CSV lines from CLI output."""
    lines = [ln for ln in output.splitlines() if ln.strip()]
    return [ln for ln in lines if ln and ln[0].isdigit()]


# --- --rename-params tests ---


def test_run_with_rename_params(runner, catalog_with_parameterized_entries):
    """run with --rename-params renames a parameter in a transform entry."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "ptrn,threshold,trn_threshold",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_compose_with_rename_params(runner, catalog_with_parameterized_entries):
    """compose with --rename-params renames params before composition."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "compose",
            "psrc",
            "ptrn",
            "--rename-params",
            "ptrn,threshold,trn_threshold",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "user_id" in result.output


def test_rename_params_bad_format(runner, catalog_with_parameterized_entries):
    """--rename-params with wrong format should show an error."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "bad_format",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code != 0
    assert "Expected" in result.output


def test_rename_params_unknown_entry(runner, catalog_with_parameterized_entries):
    """--rename-params with unknown entry name should show an error."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "nonexistent,threshold,new_threshold",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown entry" in result.output


# --- --params tests ---


def test_run_with_params_single_entry(runner, catalog_with_parameterized_entries):
    """run with -p binds a NamedScalarParameter value before execution."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "-p",
            "threshold=25.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    # threshold=25 filters amount > 25, leaving only user_id=3,amount=30,name=c
    rows = _csv_data_rows(result.output)
    assert rows == ['3,30,"c"']


def test_run_params_after_rename(runner, catalog_with_parameterized_entries):
    """--params values bind to the renamed names, not the original."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "ptrn",
            "--rename-params",
            "ptrn,threshold,trn_threshold",
            "-p",
            "trn_threshold=25.0",
            "-p",
            "threshold=5.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _csv_data_rows(result.output)
    assert rows == ["3,30"]


def test_run_params_multiple_distinct(runner, catalog_path):
    """Two distinct -p flags bind to their respective NamedScalarParameters."""
    lo = xo.param("lo", "float64", default=0.0)
    hi = xo.param("hi", "float64", default=1000.0)
    t = xo.memtable({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    expr = t.filter((t.amount > lo) & (t.amount < hi))
    Catalog.from_kwargs(path=catalog_path, init=False).add(expr, aliases=("rng",))

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "rng",
            "-p",
            "lo=15.0",
            "-p",
            "hi=25.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _csv_data_rows(result.output)
    assert rows == ["2,20"]


def test_run_params_bad_format(runner, catalog_with_parameterized_entries):
    """-p without '=' should report a usage error."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "-p",
            "no_equals",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 2
    assert "Expected key=value" in result.output


def test_run_params_bad_value(runner, catalog_with_parameterized_entries):
    """-p with a value that fails dtype coercion should error."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "-p",
            "threshold=not_a_float",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 2
    assert "not_a_float" in result.output


def test_run_params_unknown_name(runner, catalog_with_parameterized_entries):
    """-p with a name not in the expr should error with available names."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "psrc",
            "-p",
            "not_a_param=1.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 2
    assert "Unknown parameter" in result.output
    assert "threshold" in result.output


def test_run_params_no_declared(runner, catalog_path):
    """-p on an expr with no NamedScalarParameter reports 'Available: (none)'."""
    plain = xo.memtable({"x": [1, 2, 3]})
    Catalog.from_kwargs(path=catalog_path, init=False).add(plain, aliases=("plain",))

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "plain",
            "-p",
            "anything=1.0",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 2
    assert "Unknown parameter" in result.output
    assert "(none)" in result.output


# --- run with ExprBuilder entries ---


def test_run_expr_builder_entry(runner, catalog_path, saved_registry):
    """ExprBuilder entries should be runnable via `catalog run`."""
    _reset_registry()
    handler = TagHandler(
        tag_names=("test_cli_builder",),
        extract_metadata=lambda tag_node: {"type": "test_cli_builder"},
    )
    register_tag_handler(handler)

    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    source = xo.memtable({"x": [1, 2, 3], "y": [4, 5, 6]}, name="builder_src")
    tagged = source.tag("test_cli_builder")
    catalog.add(tagged, aliases=("bld",), sync=False)

    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "bld", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "x" in result.output


# --- run-cached with parameterized entries ---


def test_run_cached_with_params(runner, catalog_with_parameterized_entries, tmp_path):
    """run-cached with -p binds a NamedScalarParameter before caching."""
    catalog_path, _, _ = catalog_with_parameterized_entries
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "psrc",
            "-p",
            "threshold=25.0",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _csv_data_rows(result.output)
    assert rows == ['3,30,"c"']
