import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from click.testing import CliRunner

from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import _merge_joint_wheels_into_build, cli
from xorq.cli import cli as top_cli
from xorq.ibis_yaml.enums import DumpFiles


# --- run command ---


def test_run_single_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_default_output(runner, catalog_with_source_and_transform):
    """Default output (no -o) writes to /dev/null, exits 0."""
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src"],
    )
    assert result.exit_code == 0, result.output


def test_run_with_limit(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "-o", "-", "-f", "csv", "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 2  # header + 1 data row


def test_run_with_code(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "src",
            "-c",
            "source.filter(source.amount > 15)",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output
    # amount <= 15 should be filtered out (only 20.0 and 30.0 remain)
    lines = result.output.strip().splitlines()
    assert len(lines) == 3  # header + 2 data rows


def test_run_rejects_unbound_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "trn"],
    )
    assert result.exit_code != 0
    assert "unbound expression" in result.output


def test_run_two_entries(runner, catalog_with_source_and_transform):
    """run src trn  — compose + execute in one shot."""
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "trn", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_two_entries_with_code(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "src",
            "-c",
            "source.filter(source.amount > 15)",
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 3  # header + 2 data rows


def test_run_two_entries_json(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "trn", "-o", "-", "-f", "json"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_two_entries_with_limit(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run",
            "src",
            "trn",
            "-o",
            "-",
            "-f",
            "csv",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 2  # header + 1 data row


def test_run_no_entries(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run"],
    )
    assert result.exit_code != 0


def test_run_piped_arrow_into_unbound(runner, catalog_with_source_and_transform):
    """run bound -o - -f arrow | run unbound  — simulated via CliRunner input."""

    catalog_path, _, _ = catalog_with_source_and_transform

    # Step 1: run the source entry, capture arrow bytes
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "-o", "-", "-f", "arrow"],
    )
    assert result.exit_code == 0, result.output
    arrow_bytes = result.output_bytes

    # Step 2: pipe arrow bytes into the unbound entry
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "trn", "-o", "-", "-f", "csv"],
        input=arrow_bytes,
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output
    assert "amount" in result.output


def test_run_nonexistent_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "no-such-entry"],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


# --- compose command ---


def test_compose_two_entries(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn"],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output


def test_compose_with_alias(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "composed-result"],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output
    assert "composed-result" in result.output


def test_compose_with_code(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "compose",
            "src",
            "-c",
            "source.filter(source.amount > 15)",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output


def test_compose_code_no_entry(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "-c", "source.limit(1)"],
    )
    assert result.exit_code != 0


def test_compose_no_entries(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose"],
    )
    assert result.exit_code != 0


def test_compose_dry_run(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Schema:" in result.output
    assert "user_id" in result.output
    # dry-run should NOT catalog
    assert "Cataloged" not in result.output


def test_compose_without_alias_catalogs_by_hash(
    runner, catalog_with_source_and_transform
):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn"],
    )
    assert result.exit_code == 0, result.output
    assert "Cataloged as" in result.output


# --- compose + run roundtrip tests ---


def test_compose_then_run_roundtrip(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    # compose with alias
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "my-composed"],
    )
    assert result.exit_code == 0, result.output
    # run the composed entry
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "my-composed", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_compose_then_run_json(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "json-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "json-test", "-f", "json", "-o", "-"],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_compose_then_run_parquet(runner, catalog_with_source_and_transform, tmpdir):
    catalog_path, _, _ = catalog_with_source_and_transform
    out = str(Path(tmpdir).joinpath("out.parquet"))
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "pq-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "pq-test", "-f", "parquet", "-o", out],
    )
    assert result.exit_code == 0, result.output
    assert Path(out).exists()


def test_compose_then_run_csv_file(runner, catalog_with_source_and_transform, tmpdir):
    catalog_path, _, _ = catalog_with_source_and_transform
    out = str(Path(tmpdir).joinpath("out.csv"))
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "csv-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "csv-test", "-f", "csv", "-o", out],
    )
    assert result.exit_code == 0, result.output
    assert Path(out).exists()
    assert "user_id" in Path(out).read_text()


def test_compose_then_run_arrow_file(runner, catalog_with_source_and_transform, tmpdir):
    catalog_path, _, _ = catalog_with_source_and_transform
    out = str(Path(tmpdir).joinpath("out.arrow"))
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "arrow-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "arrow-test", "-f", "arrow", "-o", out],
    )
    assert result.exit_code == 0, result.output
    assert Path(out).exists()
    with open(out, "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    assert len(table) > 0


def test_compose_then_run_arrow_stdout(runner, catalog_with_source_and_transform):
    catalog_path, _, _ = catalog_with_source_and_transform
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "arrow-stdout-test"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        ["--path", catalog_path, "run", "arrow-stdout-test", "-f", "arrow", "-o", "-"],
    )
    assert result.exit_code == 0, result.output
    assert len(result.output) > 0


# --- --pdb flag ---


def test_pdb_flag_invokes_post_mortem(runner, tmp_path, monkeypatch):
    """--pdb should let exceptions propagate to PdbGroup, which calls pdb.post_mortem."""
    mock_pm = MagicMock()
    monkeypatch.setattr("xorq.cli.pdb.post_mortem", mock_pm)

    # "catalog list" on a non-catalog directory fails inside the command body
    # (not at Click arg-parsing time), so it exercises click_context_catalog.
    result = runner.invoke(
        top_cli,
        ["--pdb", "catalog", "--path", str(tmp_path), "list"],
    )
    assert result.exit_code != 0
    assert mock_pm.called, "pdb.post_mortem was not called with --pdb"


def test_no_pdb_flag_wraps_exception(runner, tmp_path):
    """Without --pdb, errors should be wrapped as clean 'Error: ...' messages."""
    result = runner.invoke(
        top_cli,
        ["catalog", "--path", str(tmp_path), "list"],
    )
    assert result.exit_code != 0
    assert "Error:" in result.output


# --- run-cached command ---


def test_run_cached_single_entry(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output
    assert cache_dir.exists()
    assert any(cache_dir.iterdir())


def test_run_cached_two_entries(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "trn",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_cached_snapshot_type(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "--cache-type",
            "snapshot",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_cached_ttl(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "--ttl",
            "60",
            "--cache-dir",
            str(cache_dir),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user_id" in result.output


def test_run_cached_no_entries(runner, catalog_with_source_and_transform, tmp_path):
    catalog_path, _, _ = catalog_with_source_and_transform
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code != 0
    assert "At least one entry is required" in result.output


# --- _merge_joint_wheels_into_build: direct unit tests ---


def test_merge_joint_wheels_empty_entries_is_noop(
    catalog_with_source_and_transform, tmp_path
):
    catalog_path, _, _ = catalog_with_source_and_transform
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    build_path = tmp_path / "build"
    build_path.mkdir()
    _merge_joint_wheels_into_build(catalog, (), build_path)
    assert list(build_path.iterdir()) == []


def test_merge_joint_wheels_single_entry_skips_resolver(
    catalog_with_source_and_transform, tmp_path, monkeypatch
):
    """Single-entry path copies the entry's wheel + requirements verbatim,
    without invoking JointWheelResolver (no `uv lock` subprocess)."""
    catalog_path, _, _ = catalog_with_source_and_transform
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    build_path = tmp_path / "build"
    build_path.mkdir()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("JointWheelResolver must not be called for single entry")

    monkeypatch.setattr("xorq.ibis_yaml.packager.JointWheelResolver", _fail_if_called)

    _merge_joint_wheels_into_build(catalog, ("src",), build_path)

    wheels = list(build_path.glob("*.whl"))
    assert len(wheels) >= 1
    assert (build_path / DumpFiles.requirements).exists()


def test_merge_joint_wheels_unknown_entry_raises(
    catalog_with_source_and_transform, tmp_path
):
    catalog_path, _, _ = catalog_with_source_and_transform
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    build_path = tmp_path / "build"
    build_path.mkdir()
    with pytest.raises(Exception, match="not found in catalog"):
        _merge_joint_wheels_into_build(catalog, ("no-such-entry",), build_path)


@pytest.mark.slow(level=1)
def test_merge_joint_wheels_multi_entry_runs_resolver(
    catalog_with_source_and_transform, tmp_path
):
    """Multi-entry path runs JointWheelResolver and writes a joint
    requirements.txt that excludes the input wheels (those are supplied via
    --with at runtime)."""
    catalog_path, _, _ = catalog_with_source_and_transform
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    build_path = tmp_path / "build"
    build_path.mkdir()

    _merge_joint_wheels_into_build(catalog, ("src", "trn"), build_path)

    wheels = list(build_path.glob("*.whl"))
    assert len(wheels) >= 1
    reqs = build_path / DumpFiles.requirements
    assert reqs.exists()
    text = reqs.read_text()
    assert "file://" not in text


# --- uv-subprocess path coverage ---
#
# Other tests in this file go through the auto-injecting `runner` fixture
# (which adds `--use-this-venv`), so they exercise the in-process branch.
# These tests cover the *uv* branch — the one that builds, merges joint
# wheels, and spawns `uv tool run xorq run`.


def test_catalog_run_invokes_packaged_runner(
    catalog_with_source_and_transform, monkeypatch
):
    """Without --use-this-venv, `catalog run` constructs a PackagedRunner
    pointing at the entry's extracted archive and forwards output options.
    Stub the subprocess so we don't pay for `uv tool run`. Build-path
    assertions run inside the stub because the fast path uses a temp dir
    that gets cleaned up when invoke returns."""
    calls = []

    def fake_run(self):
        # Capture the snapshot while the tmpdir still exists.
        calls.append(
            {
                "output_path": self.output_path,
                "output_format": self.output_format,
                "expr_exists": (self.build_path / DumpFiles.expr).exists(),
                "requirements_exists": (
                    self.build_path / DumpFiles.requirements
                ).exists(),
                "has_wheel": bool(list(self.build_path.glob("*.whl"))),
            }
        )
        return self

    monkeypatch.setattr("xorq.ibis_yaml.packager.PackagedRunner.run", fake_run)

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()  # NOT the auto-injecting runner fixture
    result = bare.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output

    (pr,) = calls
    assert pr["output_path"] == "-"
    assert pr["output_format"] == "csv"
    assert pr["expr_exists"]
    assert pr["requirements_exists"]
    assert pr["has_wheel"]


def test_catalog_run_fast_path_skips_merge(
    catalog_with_source_and_transform, monkeypatch
):
    """Single-entry, no-transform `catalog run` short-circuits to the
    archive — it must not deserialize the expression or invoke the
    merge/rebuild path."""
    merge_calls = []
    resolve_calls = []

    from xorq.catalog import cli as cli_mod  # noqa: PLC0415

    def spy_merge(catalog, entries, build_path):
        merge_calls.append(build_path)

    def spy_resolve(*args, **kwargs):
        resolve_calls.append(args)
        raise AssertionError("fast path must not call _resolve_single_entry")

    monkeypatch.setattr(cli_mod, "_merge_joint_wheels_into_build", spy_merge)
    monkeypatch.setattr(cli_mod, "_resolve_single_entry", spy_resolve)
    monkeypatch.setattr("xorq.ibis_yaml.packager.PackagedRunner.run", lambda self: self)

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    args = ["--path", catalog_path, "run", "src", "-o", "-", "-f", "csv"]

    result = bare.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert merge_calls == []
    assert resolve_calls == []


def test_catalog_run_reinvokes_via_uv_for_transforms(
    catalog_with_source_and_transform, monkeypatch
):
    """Transforms (--limit, -c, -p, --rename-params) under --no-use-this-venv
    must re-invoke `xorq catalog run --use-this-venv` via `uv tool run`
    instead of deserializing the expression in the caller — the caller's
    venv may not have the entry's pinned UDF deps."""
    from xorq.catalog import cli as cli_mod  # noqa: PLC0415

    captured = {}

    def fake_uv_tool_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        class _R:
            returncode = 0

        return _R()

    def spy_resolve(*args, **kwargs):
        raise AssertionError("re-invoke path must not call _resolve_single_entry")

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)
    monkeypatch.setattr(cli_mod, "_resolve_single_entry", spy_resolve)

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "--limit", "1", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output

    assert "args" in captured, "uv_tool_run was not called"
    args = captured["args"]
    assert "xorq" in args
    assert "catalog" in args
    assert "run" in args
    assert "src" in args
    assert "--limit" in args
    assert "1" in args
    # Inner xorq is the entry's pinned version (often older than this branch)
    # and doesn't know --use-this-venv; forwarding it would break the inner
    # parser. Older xorq defaults to in-process, which is what we want.
    assert "--use-this-venv" not in args


def test_catalog_run_cached_reinvokes_via_uv(
    catalog_with_source_and_transform, monkeypatch, tmp_path
):
    """`run-cached` without --use-this-venv goes through the same
    re-invocation path."""
    from xorq.catalog import cli as cli_mod  # noqa: PLC0415

    captured = {}

    def fake_uv_tool_run(*args, **kwargs):
        captured["args"] = args
        return type("R", (), {"returncode": 0})()

    def spy_resolve(*args, **kwargs):
        raise AssertionError("re-invoke path must not call _resolve_single_entry")

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)
    monkeypatch.setattr(cli_mod, "_resolve_single_entry", spy_resolve)

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "run-cached",
            "src",
            "--cache-dir",
            str(tmp_path),
            "-o",
            "-",
            "-f",
            "csv",
        ],
    )
    assert result.exit_code == 0, result.output

    args = captured["args"]
    assert "run-cached" in args
    assert "--cache-dir" in args
    assert str(tmp_path) in args


@pytest.mark.slow(level=1)
def test_catalog_run_uv_path_end_to_end(catalog_with_source_and_transform, tmp_path):
    """Full pipeline: archive ⇒ `uv tool run xorq run`. Slow because uv
    resolves and installs the entry's pinned env."""
    catalog_path, _, _ = catalog_with_source_and_transform
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [
            "xorq",
            "catalog",
            "--path",
            catalog_path,
            "run",
            "src",
            "-o",
            str(out),
            "-f",
            "csv",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "user_id" in out.read_text()
