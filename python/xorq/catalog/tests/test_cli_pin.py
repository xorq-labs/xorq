from __future__ import annotations

import gc
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
from xorq.catalog.bind import bind, fuse_catalog_source
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.enums import RebaseExit
from xorq.catalog.exceptions import RebaseError
from xorq.catalog.rebase import AliasPlan
from xorq.catalog.tests.conftest import (
    alias_target_hash,
    fail_nth_alias_move,
    pull_then,
)
from xorq.catalog.zip_utils import extract_build_zip_to
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.common.utils.process_utils import subprocess_run
from xorq.expr.relations import CachedNode, CacheTag
from xorq.vendor.ibis.expr import operations as ops


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


def _add_file_cached_entry(
    catalog: Catalog, parquet_path: Path, relocate_reads: bool = True
) -> str:
    pq.write_table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), parquet_path)
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con)
    t = deferred_read_parquet(parquet_path, con, table_name="t")
    entry = catalog.add(
        t.filter(t.a > 1).cache(cache=cache),
        sync=False,
        relocate_reads=relocate_reads,
    )
    return entry.name


def _other_entry(catalog: Catalog, *aliases: str) -> str:
    """An entry an alias can be moved to, beside the source."""
    expr = xo.memtable({"a": [1, 2, 3]}, name="t").filter(xo._.a > 1)
    return catalog.add(expr, sync=False, aliases=aliases).name


def _run(runner: CliRunner, catalog_path: str, *args: str) -> Result:
    return runner.invoke(cli, ["--path", catalog_path, *args])


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


def test_catalog_pin_failed_alias_move_rolls_back(
    runner: CliRunner,
    catalog: Catalog,
    catalog_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `prod` move after `latest`'s fails: no new entry, both back on the source."""
    src_name = _add_cached_entry(catalog, "prod", "latest")
    fail_nth_alias_move(monkeypatch, 2)

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--move-aliases",
            "--no-sync",
        ],
    )
    monkeypatch.undo()
    assert result.exit_code == 1, result.output
    assert "alias move failed" in result.output

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert reloaded.list() == [src_name]
    assert alias_target_hash(reloaded, "prod") == src_name
    assert alias_target_hash(reloaded, "latest") == src_name


def test_catalog_pin_refuses_an_alias_on_another_entry(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    """`-a` names another entry's alias: refused, as `rebase` refuses it, not taken."""
    src_name = _add_cached_entry(catalog)
    other = _other_entry(catalog, "taken")

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            str(tmp_path / "cache"),
            "-a",
            "taken",
            "--no-sync",
        ],
    )
    assert result.exit_code == 1, result.output
    assert f"alias 'taken' points at {other}" in result.output

    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert set(reloaded.list()) == {src_name, other}
    assert alias_target_hash(reloaded, "taken") == other


def test_catalog_pin_exits_with_the_refusals_code(
    runner: CliRunner,
    catalog: Catalog,
    catalog_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal exits with its own code and its bare message, as `rebase` does."""
    src_name = _add_cached_entry(catalog)

    def refusing(self: AliasPlan) -> AliasPlan:
        raise RebaseError("refused for the test", RebaseExit.UNREACHABLE)

    monkeypatch.setattr(AliasPlan, "checked", refusing)
    result = _run(
        runner,
        catalog_path,
        "pin",
        src_name,
        "-e",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--no-sync",
    )
    assert result.exit_code == RebaseExit.UNREACHABLE, result.output
    assert result.stderr == "refused for the test\n"


@pytest.mark.parametrize(
    "command", (pytest.param("pin", id="pin"), pytest.param("unpin", id="unpin"))
)
def test_catalog_pin_move_aliases_names_what_the_pull_took(
    runner: CliRunner,
    catalog: Catalog,
    catalog_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """A teammate moved `prod` before the sync's pull: said, not refused or taken."""
    src_name = _add_cached_entry(catalog, "prod", "latest")
    other = _other_entry(catalog)
    cache_dir = str(tmp_path / "cache")
    if command == "unpin":
        # The entry to unpin is the pinned one, with the aliases on it.
        pinned = _run(
            runner,
            catalog_path,
            "pin",
            src_name,
            "-e",
            "--cache-dir",
            cache_dir,
            "--move-aliases",
            "--no-sync",
        )
        assert pinned.exit_code == 0, pinned.output
        src_name = _last_line(pinned)
    pull_then(monkeypatch, lambda c: c.add_alias(other, "prod", sync=False))

    materialize = ("-e",) if command == "pin" else ()
    result = _run(
        runner,
        catalog_path,
        command,
        src_name,
        *materialize,
        "--cache-dir",
        cache_dir,
        "--move-aliases",
    )
    monkeypatch.undo()
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)
    assert f"Moved alias 'latest' -> {new_name}" in result.stderr
    assert (
        f"Alias 'prod' not moved: no longer on {src_name} after the pull"
        in result.stderr
    )
    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert alias_target_hash(reloaded, "latest") == new_name
    assert alias_target_hash(reloaded, "prod") == other


def test_catalog_pin_refuses_an_entry_alias_the_pull_moved(
    runner: CliRunner,
    catalog: Catalog,
    catalog_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ENTRY `prod` meant the source before the pull, not after it: refused."""
    src_name = _add_cached_entry(catalog, "prod")
    other = _other_entry(catalog)
    pull_then(monkeypatch, lambda c: c.add_alias(other, "prod", sync=False))

    result = _run(
        runner,
        catalog_path,
        "pin",
        "prod",
        "-e",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--move-aliases",
    )
    monkeypatch.undo()
    assert result.exit_code == 1, result.output
    assert result.stderr == (
        f"alias 'prod' points at {other} after the pull, not {src_name}; "
        f"rerun with {src_name}\n"
    )
    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert set(reloaded.list()) == {src_name, other}
    assert alias_target_hash(reloaded, "prod") == other


def test_catalog_pin_names_only_a_missing_source_as_one(
    runner: CliRunner,
    catalog: Catalog,
    catalog_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file the sync's pull misses is not a read the build could not bundle."""
    src_name = _add_cached_entry(catalog)
    gone = tmp_path / "gone.parquet"

    def missing(c: Catalog) -> None:
        raise FileNotFoundError(2, "No such file or directory", str(gone))

    pull_then(monkeypatch, missing)
    result = _run(
        runner, catalog_path, "pin", src_name, "-e", "--cache-dir", str(tmp_path / "c")
    )
    monkeypatch.undo()
    assert result.exit_code == 1, result.output
    assert "cannot bundle reads" not in result.output
    assert str(gone) in result.output


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


def _alias_lines(result: Result) -> list[str]:
    """The stderr lines about aliases; the rest is a warning from the build."""
    return [line for line in result.stderr.splitlines() if "lias" in line]


@pytest.mark.parametrize(
    "args, expected",
    (
        pytest.param((), (), id="bare"),
        pytest.param(
            ("--move-aliases",), ("Alias 'prod' already on {name}",), id="move"
        ),
        pytest.param(("-a", "prod"), ("Alias 'prod' already on {name}",), id="alias"),
        pytest.param(("-a", "fresh"), ("Added alias 'fresh' -> {name}",), id="fresh"),
    ),
)
def test_catalog_pin_noop_moves_and_leaves_no_alias(
    runner: CliRunner,
    catalog: Catalog,
    catalog_path: str,
    tmp_path: Path,
    args: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    """The new entry is the old one: nothing moved onto it, nothing left behind."""
    entry = catalog.add(
        xo.memtable({"a": [1, 2, 3]}, name="t").filter(xo._.a > 1),
        sync=False,
        aliases=("prod",),
    )

    result = _run(
        runner,
        catalog_path,
        "pin",
        entry.name,
        "--cache-dir",
        str(tmp_path / "cache"),
        "--no-sync",
        *args,
    )
    assert result.exit_code == 0, result.output
    assert _last_line(result) == entry.name
    assert _alias_lines(result) == [line.format(name=entry.name) for line in expected]
    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    assert alias_target_hash(reloaded, "prod") == entry.name


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
    # The source entry is created lean too -- relocation is irreversible, so a
    # relocated source could not be thinned back to lean by --no-relocate-reads.
    src_name = _add_file_cached_entry(
        catalog, tmp_path / "data.parquet", relocate_reads=False
    )
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


def test_catalog_pin_relocated_entry_fuses_correctly(
    runner: CliRunner, catalog: Catalog, catalog_path: str, tmp_path: Path
) -> None:
    # A relocated (default) pinned entry must consume correctly via fuse -- the
    # default consumption path -- rather than warn or error (#2133).
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
            "--no-sync",
        ],
    )
    assert result.exit_code == 0, result.output
    new_name = _last_line(result)

    # the pinned entry bundles its relocated read ...
    reloaded = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = reloaded.get_catalog_entry(new_name)
    with tempfile.TemporaryDirectory() as td:
        build_dir = extract_build_zip_to(entry.catalog_path, td)
        reads_dir = Path(build_dir) / "reads"
        assert reads_dir.exists() and list(reads_dir.glob("*.parquet"))

    # ... and it fuses to the correct, non-empty result.
    schema = entry.load_expr(cache_dir=cache_dir).schema()
    ub = ops.UnboundTable(name="ph", schema=schema).to_expr()
    transform_entry = reloaded.add(ub.limit(10))
    bound = bind(entry, transform_entry, cache_dir=str(cache_dir))
    fused = fuse_catalog_source(bound)
    gc.collect()
    assert len(fused.execute()) == 2
