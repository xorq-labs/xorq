from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner, Result

import xorq.api as xo
from xorq.api import Expr
from xorq.caching import ParquetCache
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.tests.conftest import alias_target_hash
from xorq.catalog.zip_utils import extract_build_zip_to
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.common.utils.process_utils import subprocess_run
from xorq.expr.relations import CachedNode, CacheTag


@pytest.mark.parametrize(
    "command",
    (
        pytest.param("pin", id="pin"),
        pytest.param("unpin", id="unpin"),
    ),
)
def test_catalog_pin_cli_help_smoke_subprocess(command: str) -> None:
    """`xorq catalog {pin,unpin} --help` must load via a real subprocess.

    The other catalog CLI tests here use the in-process CliRunner fixture, which
    never exercises the real entrypoint and so hides an import-time cold-start
    regression from the command's deferred imports. This spawns a real `xorq`.
    """
    (returncode, stdout, _stderr) = subprocess_run(
        ["xorq", "catalog", command, "--help"], text=True
    )
    assert returncode == 0, _stderr
    assert command in stdout


def _cached_expr() -> Expr:
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con)
    t = xo.memtable({"a": [1, 2, 3], "b": [4, 5, 6]}, name="t")
    return t.filter(t.a > 1).cache(cache=cache)


def _add_cached_entry(catalog: Catalog, *aliases: str) -> str:
    entry = catalog.add(_cached_expr(), sync=False, aliases=aliases)
    return entry.name


def _add_file_cached_entry(catalog: Catalog, parquet_path: Path) -> str:
    pq.write_table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), parquet_path)
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con)
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    entry = catalog.add(t.filter(t.a > 1).cache(cache=cache), sync=False)
    return entry.name


def _last_line(result: Result) -> str:
    return result.output.strip().splitlines()[-1]


def test_catalog_pin_creates_new_pinned_entry(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    src_name = _add_cached_entry(catalog)
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)
    assert new_name != src_name

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    expr = reloaded.get_catalog_entry(new_name).load_expr(cache_dir=cache_dir)
    assert walk_nodes((CacheTag,), expr)
    assert not walk_nodes((CachedNode,), expr)


def test_catalog_pin_with_alias(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    src_name = _add_cached_entry(catalog)
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "-a",
            "pinned",
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert alias_target_hash(reloaded, "pinned") == new_name


def test_catalog_pin_move_aliases(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    src_name = _add_cached_entry(catalog, "prod")
    cache_dir = tmp_path / "cache"
    # alias points at the source entry to begin with
    assert alias_target_hash(catalog, "prod") == src_name

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            "prod",
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--move-aliases",
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)
    assert new_name != src_name

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert alias_target_hash(reloaded, "prod") == new_name


def test_catalog_pin_move_aliases_moves_every_alias(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    src_name = _add_cached_entry(catalog, "prod", "latest")
    cache_dir = tmp_path / "cache"
    assert alias_target_hash(catalog, "prod") == src_name
    assert alias_target_hash(catalog, "latest") == src_name

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--move-aliases",
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)
    assert new_name != src_name

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert alias_target_hash(reloaded, "prod") == new_name
    assert alias_target_hash(reloaded, "latest") == new_name


def test_catalog_pin_move_aliases_noop_without_aliases(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    src_name = _add_cached_entry(catalog)
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--move-aliases",
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _last_line(result) != src_name


def test_catalog_pin_noop_without_caches(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    entry = catalog.add(
        xo.memtable({"a": [1, 2, 3]}, name="t").filter(xo._.a > 1), sync=False
    )
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            entry.name,
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output

    new_name = _last_line(result)
    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    pinned = reloaded.get_catalog_entry(new_name).load_expr(cache_dir=cache_dir)
    # no caches to freeze: pin is a no-op, so the entry name is unchanged
    assert not walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)
    assert new_name == entry.name


def test_catalog_unpin_inverts_pin(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    src_name = _add_cached_entry(catalog)
    cache_dir = tmp_path / "cache"

    pin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert pin_result.exit_code == 0, pin_result.output
    pinned_name = _last_line(pin_result)
    assert pinned_name != src_name

    unpin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "unpin",
            pinned_name,
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert unpin_result.exit_code == 0, unpin_result.output
    unpinned_name = _last_line(unpin_result)
    # unpin restores the original unpinned build hash -> same content name
    assert unpinned_name == src_name

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    expr = reloaded.get_catalog_entry(unpinned_name).load_expr(cache_dir=cache_dir)
    assert walk_nodes((CachedNode,), expr)
    assert not walk_nodes((CacheTag,), expr)


def test_catalog_pin_is_portable_across_cache_dirs(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    # A pinned entry is portable via base_path relocation, not bundling: the
    # frozen read is a build-hash leaf keyed on its cache key, so loading the
    # entry against a different cache dir re-points the read at the relocated
    # artifact (see relocate_cache_tag). Moving the cache dir proves the pinned
    # build reads the new location rather than the original.
    src_name = _add_cached_entry(catalog)
    cache_dir = tmp_path / "cacheA"

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--relocate-reads",
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = reloaded.get_catalog_entry(new_name)

    # baseline against the original cache dir
    expected = entry.load_expr(cache_dir=cache_dir).execute()

    # relocate the cache files; the original directory no longer exists
    relocated_dir = tmp_path / "cacheB"
    shutil.move(str(cache_dir), str(relocated_dir))

    relocated = entry.load_expr(cache_dir=relocated_dir)
    assert walk_nodes((CacheTag,), relocated)
    assert not walk_nodes((CachedNode,), relocated)
    assert relocated.execute().equals(expected)


def test_catalog_unpin_relocate_reads_bundles_source(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    src_name = _add_file_cached_entry(catalog, tmp_path / "data.parquet")
    cache_dir = tmp_path / "cache"

    pin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert pin_result.exit_code == 0, pin_result.output
    pinned_name = _last_line(pin_result)

    unpin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "unpin",
            pinned_name,
            "--cache-dir",
            str(cache_dir),
            "--relocate-reads",
            "--no-sync",
        ],
    )
    assert unpin_result.exit_code == 0, unpin_result.output
    unpinned_name = _last_line(unpin_result)

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = reloaded.get_catalog_entry(unpinned_name)
    expr = entry.load_expr(cache_dir=cache_dir)
    # unpin restores the recomputable cache form ...
    assert walk_nodes((CachedNode,), expr)
    assert not walk_nodes((CacheTag,), expr)
    # ... and --relocate-reads bundles the source read behind it so the
    # recomputable form is runnable from anywhere too
    with tempfile.TemporaryDirectory() as td:
        build_dir = extract_build_zip_to(entry.catalog_path, td)
        reads_dir = Path(build_dir) / "reads"
        assert reads_dir.exists()
        assert list(reads_dir.glob("*.parquet"))


def test_catalog_pin_unpin_relocate_roundtrip_file_backed(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    # Regression: an explicit --relocate-reads pin -> unpin round-trip of a
    # file-backed entry must succeed. The pinned/unpinned exprs keep relocatable
    # reads pointing into the loaded entry's temp extract dir; the command must
    # hold the loaded expr alive until the rebuild copies them, or the extract
    # dir is swept and the relocating rebuild fails. (relocate_reads is opt-in
    # for catalog entries now -- default is lean -- so pass it explicitly here.)
    src_name = _add_file_cached_entry(catalog, tmp_path / "data.parquet")
    cache_dir = tmp_path / "cache"

    pin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--relocate-reads",
            "--no-sync",
        ],
    )
    assert pin_result.exit_code == 0, pin_result.output
    pinned_name = _last_line(pin_result)

    unpin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "unpin",
            pinned_name,
            "--cache-dir",
            str(cache_dir),
            "--relocate-reads",
            "--no-sync",
        ],
    )
    assert unpin_result.exit_code == 0, unpin_result.output
    unpinned_name = _last_line(unpin_result)

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = reloaded.get_catalog_entry(unpinned_name)
    expr = entry.load_expr(cache_dir=cache_dir)
    assert walk_nodes((CachedNode,), expr)
    assert not walk_nodes((CacheTag,), expr)
    # the relocated unpinned form bundles its source read
    with tempfile.TemporaryDirectory() as td:
        build_dir = extract_build_zip_to(entry.catalog_path, td)
        assert list((Path(build_dir) / "reads").glob("*.parquet"))


def test_catalog_pin_build_load_unpin_recomputes_from_uncached(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    # Regression for `uncached` reconstruction. 2111 prunes a pin's frozen
    # leaves from the build hash and source scan (_decompose_expr /
    # find_all_sources), but `uncached` must still be fully serialized so unpin
    # rebuilds a *re-computable* CachedNode. Node-type assertions alone would
    # miss a starved reconstruction, so this drives the full circle: pin ->
    # build -> load -> unpin -> clear cache -> execute, and asserts the
    # recompute-from-`uncached` result equals the cached value.
    src_name = _add_file_cached_entry(catalog, tmp_path / "data.parquet")
    cache_dir = tmp_path / "cache"

    pin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert pin_result.exit_code == 0, pin_result.output

    unpin_result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "unpin",
            _last_line(pin_result),
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert unpin_result.exit_code == 0, unpin_result.output
    unpinned_name = _last_line(unpin_result)

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = reloaded.get_catalog_entry(unpinned_name)
    expr = entry.load_expr(cache_dir=cache_dir)
    assert walk_nodes((CachedNode,), expr)
    assert not walk_nodes((CacheTag,), expr)

    from_cache = expr.execute()
    # wipe the cache so the next execute must rebuild from `uncached` (the
    # reconstructed upstream reading the bundled source), not the frozen cache
    shutil.rmtree(cache_dir)
    recomputed = entry.load_expr(cache_dir=cache_dir).execute()
    assert recomputed.equals(from_cache)


def test_catalog_pin_cold_cache_errors_without_ensure_materialized(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    # a freshly added cached entry has never been executed, so its cache is
    # cold; pinning without -e must fail with the actionable hint rather than
    # writing a silently-broken entry.
    src_name = _add_cached_entry(catalog)
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "--cache-dir",
            str(cache_dir),
            "--no-sync",
        ],
    )
    assert result.exit_code != 0
    assert "ensure-materialized" in result.output


def test_catalog_pin_no_relocate_reads_is_lean(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    # --no-relocate-reads must NOT bundle the source file (nor the frozen cache):
    # the result is a lean, machine-local entry whose reads stay absolute paths.
    src_name = _add_file_cached_entry(catalog, tmp_path / "data.parquet")
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(cache_dir),
            "--no-relocate-reads",
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = reloaded.get_catalog_entry(new_name)
    # the entry still pins (caches frozen into CacheTag reads) ...
    expr = entry.load_expr(cache_dir=cache_dir)
    assert walk_nodes((CacheTag,), expr)
    assert not walk_nodes((CachedNode,), expr)
    # ... but nothing is bundled into the build
    with tempfile.TemporaryDirectory() as td:
        build_dir = extract_build_zip_to(entry.catalog_path, td)
        reads_dir = Path(build_dir) / "reads"
        assert not reads_dir.exists() or not list(reads_dir.glob("*.parquet"))


def test_catalog_pin_warns_about_fuse_gap_when_relocated(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    # A pin re-adds with relocate_reads=True by default, which fuses to an empty
    # result until #2133 lands; the command must warn at pin time (not only in
    # docs + an xfail test). --no-relocate-reads produces a fuse-safe entry, so
    # it must NOT warn.
    src_name = _add_cached_entry(catalog)
    cache_dir = tmp_path / "cache"
    base_args = ["--path", catalog_path, "pin", src_name, "-e", "--no-sync"]

    default = runner.invoke(cli, [*base_args, "--cache-dir", str(cache_dir / "a")])
    assert default.exit_code == 0, default.output
    assert "#2133" in default.output
    assert "fuse" in default.output

    lean = runner.invoke(
        cli, [*base_args, "--cache-dir", str(cache_dir / "b"), "--no-relocate-reads"]
    )
    assert lean.exit_code == 0, lean.output
    assert "#2133" not in lean.output
