import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pyarrow as pa
import pytest
from click.testing import CliRunner

from xorq.catalog import cli as cli_mod
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import (
    _assert_requirements_identical,
    _entry_run_bundle,
    _forward_ctx_params,
    _has_expr_modifications,
    _stage_bundle_into_build,
    cli,
)
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


# --- _entry_run_bundle / _stage_bundle_into_build: direct unit tests ---


def test_entry_run_bundle_empty_entries_raises():
    with pytest.raises(click.ClickException, match="at least one entry"):
        with _entry_run_bundle(None, ()):
            pass


def test_stage_bundle_single_entry_copies_verbatim(
    catalog_with_source_and_transform, tmp_path
):
    """Single-entry path copies the entry's wheel + requirements verbatim."""
    catalog_path, _, _ = catalog_with_source_and_transform
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    build_path = tmp_path / "build"
    build_path.mkdir()

    with _entry_run_bundle(catalog, ("src",)) as bundle:
        _stage_bundle_into_build(bundle, build_path)

    wheels = list(build_path.glob("*.whl"))
    assert len(wheels) >= 1
    assert (build_path / DumpFiles.requirements).exists()


def test_entry_run_bundle_unknown_entry_raises(
    catalog_with_source_and_transform, tmp_path
):
    catalog_path, _, _ = catalog_with_source_and_transform
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    with pytest.raises(click.ClickException, match="not found"):
        with _entry_run_bundle(catalog, ("no-such-entry",)):
            pass


def test_stage_bundle_multi_entry_writes_shared_requirements_verbatim(
    catalog_with_source_and_transform, tmp_path
):
    """Multi-entry path requires byte-identical requirements.txt across
    entries and writes it verbatim (joint resolution is deferred to a
    follow-up PR)."""
    catalog_path, _, _ = catalog_with_source_and_transform
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    build_path = tmp_path / "build"
    build_path.mkdir()

    with _entry_run_bundle(catalog, ("src", "trn")) as bundle:
        _stage_bundle_into_build(bundle, build_path)

    wheels = list(build_path.glob("*.whl"))
    assert len(wheels) >= 1
    reqs = build_path / DumpFiles.requirements
    assert reqs.exists()
    text = reqs.read_text()
    assert "file://" not in text


def test_assert_requirements_identical_passes_when_equal():
    _assert_requirements_identical([("a", b"foo==1.0\n"), ("b", b"foo==1.0\n")])
    _assert_requirements_identical([("only", b"")])
    _assert_requirements_identical([])


def test_assert_requirements_identical_raises_on_mismatch():
    with pytest.raises(click.ClickException) as exc_info:
        _assert_requirements_identical([("a", b"foo==1.0\n"), ("b", b"foo==2.0\n")])
    msg = exc_info.value.message
    assert "requirements.txt differs across entries" in msg
    assert "'a'" in msg and "'b'" in msg


# --- uv-subprocess test helpers ---


def _fake_uv_tool_run(captured=None, stdout="", stderr="", side_effect=None):
    def fake(*args, **kwargs):
        if captured is not None:
            captured["args"] = args
            captured["kwargs"] = kwargs
        if side_effect is not None:
            side_effect(*args, **kwargs)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    return fake


def _must_not_call(label):
    def _spy(*args, **kwargs):
        raise AssertionError(f"{label} must not be called")

    return _spy


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
    bare = CliRunner()
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
    monkeypatch.setattr(
        cli_mod, "_stage_bundle_into_build", _must_not_call("_stage_bundle_into_build")
    )
    monkeypatch.setattr(
        cli_mod, "_resolve_single_entry", _must_not_call("_resolve_single_entry")
    )
    monkeypatch.setattr("xorq.ibis_yaml.packager.PackagedRunner.run", lambda self: self)

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    args = ["--path", catalog_path, "run", "src", "-o", "-", "-f", "csv"]

    result = bare.invoke(cli, args)
    assert result.exit_code == 0, result.output


def test_catalog_run_no_fuse_bypasses_fast_path(
    catalog_with_source_and_transform, monkeypatch
):
    """`--no-fuse` must fall through to the re-invoke path; the fast
    path runs PackagedRunner directly and would silently drop the flag."""
    captured = {}
    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run", _fake_uv_tool_run(captured)
    )
    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.PackagedRunner.run",
        _must_not_call("PackagedRunner.run"),
    )

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        ["--path", catalog_path, "run", "src", "--no-fuse", "-o", "-", "-f", "csv"],
    )
    assert result.exit_code == 0, result.output

    args = captured["args"]
    assert "--no-fuse" in args


def test_catalog_run_reinvokes_via_uv_for_transforms(
    catalog_with_source_and_transform, monkeypatch
):
    """Transforms (--limit, -c, -p, --rename-params) under --no-use-this-venv
    must re-invoke `xorq catalog run --use-this-venv` via `uv tool run`
    instead of deserializing the expression in the caller — the caller's
    venv may not have the entry's pinned UDF deps."""
    captured = {}
    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run", _fake_uv_tool_run(captured)
    )
    monkeypatch.setattr(
        cli_mod, "_resolve_single_entry", _must_not_call("_resolve_single_entry")
    )

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
    assert "--use-this-venv" in args


def test_catalog_run_cached_reinvokes_via_uv(
    catalog_with_source_and_transform, monkeypatch, tmp_path
):
    """`run-cached` without --use-this-venv goes through the same
    re-invocation path."""
    captured = {}
    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run", _fake_uv_tool_run(captured)
    )
    monkeypatch.setattr(
        cli_mod, "_resolve_single_entry", _must_not_call("_resolve_single_entry")
    )

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
    """Full pipeline: archive => `uv tool run xorq run`. Slow because uv
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


# --- compose uv-reinvoke path coverage ---
#
# Other tests in this file go through the auto-injecting `runner` fixture
# (which adds `--use-this-venv`), so they exercise the in-process branch
# of compose. These tests cover the *uv* branch — outer spawns the inner
# under `uv tool run` and finalizes (merge wheels + catalog.add) here.


def test_catalog_compose_reinvokes_via_uv(
    catalog_with_source_and_transform, monkeypatch, tmp_path
):
    """Without --use-this-venv, `catalog compose` must (a) spawn the inner
    via `uv tool run`, (b) pass `--use-this-venv --emit-build-path-to <file>`
    to it (otherwise the inner would re-enter the outer path and recurse,
    or would mutate the catalog twice), (c) NOT load .expr in the caller
    shell. The fake uv_tool_run writes pre_built to the requested file."""
    pre_built = tmp_path / "fake-build"
    pre_built.mkdir()
    (pre_built / DumpFiles.expr).write_text("# placeholder\n")
    (pre_built / DumpFiles.requirements).write_text("")

    captured = {}

    def _write_build_path(*args, **kwargs):
        idx = args.index("--emit-build-path-to")
        Path(args[idx + 1]).write_text(str(pre_built))

    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run",
        _fake_uv_tool_run(captured, side_effect=_write_build_path),
    )
    monkeypatch.setattr(cli_mod, "_compose_expr", _must_not_call("_compose_expr"))

    def spy_stage(bundle, build_path):
        captured["merge_build_path"] = build_path
        captured["merge_bundle"] = bundle

    def spy_add(self, build_path, sync=True, aliases=(), exist_ok=False):
        captured["add_build_path"] = build_path
        captured["add_sync"] = sync
        captured["add_aliases"] = aliases
        return MagicMock(name="catalog_entry")

    monkeypatch.setattr(cli_mod, "_stage_bundle_into_build", spy_stage)
    monkeypatch.setattr(Catalog, "add", spy_add)

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "-a", "outer-alias"],
    )
    assert result.exit_code == 0, result.output

    args = captured["args"]
    assert "xorq" in args
    assert "catalog" in args
    assert "compose" in args
    assert "src" in args and "trn" in args
    assert "--use-this-venv" in args
    assert "--emit-build-path-to" in args
    assert "--alias" not in args and "-a" not in args
    assert captured["merge_build_path"] == pre_built
    assert captured["merge_bundle"] is not None
    assert len(captured["merge_bundle"].wheel_paths) > 0
    assert captured["add_build_path"] == pre_built
    assert captured["add_sync"] is True
    assert captured["add_aliases"] == ("outer-alias",)


def test_catalog_compose_no_sync(
    catalog_with_source_and_transform, monkeypatch, tmp_path
):
    """--no-sync must propagate sync=False to catalog.add."""
    pre_built = tmp_path / "fake-build"
    pre_built.mkdir()
    (pre_built / DumpFiles.expr).write_text("# placeholder\n")
    (pre_built / DumpFiles.requirements).write_text("")

    captured = {}

    def _write_build_path(*args, **kwargs):
        idx = args.index("--emit-build-path-to")
        Path(args[idx + 1]).write_text(str(pre_built))

    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run",
        _fake_uv_tool_run(captured, side_effect=_write_build_path),
    )
    monkeypatch.setattr(cli_mod, "_compose_expr", _must_not_call("_compose_expr"))

    def spy_stage(bundle, build_path):
        captured["merge_build_path"] = build_path
        captured["merge_bundle"] = bundle

    def spy_add(self, build_path, sync=True, aliases=(), exist_ok=False):
        captured["add_sync"] = sync
        return MagicMock(name="catalog_entry")

    monkeypatch.setattr(cli_mod, "_stage_bundle_into_build", spy_stage)
    monkeypatch.setattr(Catalog, "add", spy_add)

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        ["--path", catalog_path, "compose", "--no-sync", "src", "trn"],
    )
    assert result.exit_code == 0, result.output
    assert captured["add_sync"] is False


def test_catalog_compose_dry_run_relays_inner_stdout(
    catalog_with_source_and_transform, monkeypatch
):
    """`compose --dry-run` under uv-reinvoke must forward `--dry-run` to the
    inner and relay its stdout verbatim. The outer must not call .expr or
    catalog.add."""

    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run",
        _fake_uv_tool_run(
            stdout="Dry run — composition plan:\n  Entries: src -> trn\n"
        ),
    )
    monkeypatch.setattr(cli_mod, "_compose_expr", _must_not_call("_compose_expr"))
    monkeypatch.setattr(Catalog, "add", _must_not_call("catalog.add"))

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        ["--path", catalog_path, "compose", "src", "trn", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run — composition plan:" in result.output
    assert "src -> trn" in result.output


def test_catalog_compose_outer_unknown_entry_fails_before_uv(
    catalog_with_source_and_transform, monkeypatch
):
    """`_entry_run_bundle` (called by the outer before spawning) must fail
    cleanly when an entry is unknown — never reaches uv_tool_run."""

    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_tool_run", _must_not_call("uv_tool_run")
    )

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        ["--path", catalog_path, "compose", "no-such-entry"],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_catalog_compose_emit_build_path_to_short_circuits(
    catalog_with_source_and_transform, monkeypatch, tmp_path
):
    """With `--use-this-venv --emit-build-path-to <file>` (the flags the
    outer passes when spawning the inner), compose must run _compose_expr +
    build_expr, write the build_path to the file, and exit before
    _stage_bundle_into_build / catalog.add fire."""
    pre_built = tmp_path / "inner-build"
    pre_built.mkdir()
    out_path = tmp_path / "build-path-out.txt"

    monkeypatch.setattr(
        "xorq.ibis_yaml.compiler.build_expr",
        lambda expr, **kw: pre_built,
    )

    monkeypatch.setattr(
        cli_mod, "_stage_bundle_into_build", _must_not_call("_stage_bundle_into_build")
    )
    monkeypatch.setattr(Catalog, "add", _must_not_call("catalog.add"))

    catalog_path, _, _ = catalog_with_source_and_transform
    bare = CliRunner()
    result = bare.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "compose",
            "src",
            "trn",
            "--use-this-venv",
            "--emit-build-path-to",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.read_text() == str(pre_built)


@pytest.mark.slow(level=1)
def test_catalog_compose_uv_path_end_to_end(
    catalog_with_source_and_transform, tmp_path
):
    """Full pipeline: outer spawns inner under `uv tool run`, inner builds
    + emits build_path, outer merges + adds. Slow because uv resolves and
    installs the entries' pinned env."""
    catalog_path, _, _ = catalog_with_source_and_transform
    result = subprocess.run(
        [
            "xorq",
            "catalog",
            "--path",
            catalog_path,
            "compose",
            "src",
            "trn",
            "-a",
            "e2e-composed",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    assert catalog.catalog_yaml.contains_alias("e2e-composed")


# --- _has_expr_modifications ---


class TestHasExprModifications:
    @staticmethod
    def _ctx(**overrides):
        params = {
            "fuse": True,
            "code": None,
            "limit": None,
            "raw_params": (),
            "raw_rename_params": (),
        }
        params.update(overrides)
        return SimpleNamespace(params=params)

    def test_defaults_no_modifications(self):
        assert _has_expr_modifications(self._ctx()) is False

    def test_no_fuse(self):
        assert _has_expr_modifications(self._ctx(fuse=False)) is True

    def test_code_set(self):
        assert _has_expr_modifications(self._ctx(code="source.limit(1)")) is True

    def test_limit_nonzero(self):
        assert _has_expr_modifications(self._ctx(limit=10)) is True

    def test_limit_zero(self):
        assert _has_expr_modifications(self._ctx(limit=0)) is True

    def test_raw_params_set(self):
        assert _has_expr_modifications(self._ctx(raw_params=("k=v",))) is True

    def test_raw_rename_params_set(self):
        assert (
            _has_expr_modifications(self._ctx(raw_rename_params=("e,old,new",))) is True
        )

    def test_empty_tuples_no_modifications(self):
        assert (
            _has_expr_modifications(self._ctx(raw_params=(), raw_rename_params=()))
            is False
        )


# --- _forward_ctx_params ---


_fwd_captured_ctx: dict = {}


@click.command("_fwd_fake")
@click.option("--name", default=None)
@click.option("--count", type=int, default=None)
@click.option("--flag/--no-flag", default=False)
@click.option("--multi", multiple=True)
@click.option("-i", "--instream", type=click.File("rb"), default="-")
@click.option("--excluded", default=None)
@click.pass_context
def _fwd_fake_cmd(ctx, **kwargs):
    _fwd_captured_ctx["ctx"] = ctx


class TestForwardCtxParams:
    @staticmethod
    def _invoke(*args, exclude=frozenset()):
        _fwd_captured_ctx.clear()
        CliRunner().invoke(_fwd_fake_cmd, list(args), standalone_mode=False)
        return _forward_ctx_params(_fwd_captured_ctx["ctx"], exclude=exclude)

    def test_defaults_produce_empty(self):
        assert self._invoke() == ()

    def test_scalar_option(self):
        result = self._invoke("--name", "foo")
        assert "--name" in result
        assert "foo" in result

    def test_int_option(self):
        result = self._invoke("--count", "42")
        idx = result.index("--count")
        assert result[idx + 1] == "42"

    def test_flag_non_default(self):
        result = self._invoke("--flag")
        assert "--flag" in result

    def test_flag_default_omitted(self):
        result = self._invoke()
        assert "--flag" not in result
        assert "--no-flag" not in result

    def test_multiple_option(self):
        result = self._invoke("--multi", "a", "--multi", "b")
        indices = [i for i, v in enumerate(result) if v == "--multi"]
        assert len(indices) == 2
        assert result[indices[0] + 1] == "a"
        assert result[indices[1] + 1] == "b"

    def test_file_stdin_not_forwarded(self):
        result = self._invoke()
        assert "-i" not in result
        assert "--instream" not in result

    def test_file_real_path_forwarded(self, tmp_path):
        f = tmp_path / "input.bin"
        f.write_bytes(b"")
        result = self._invoke("-i", str(f))
        assert "-i" in result
        assert str(f.resolve()) in result

    def test_exclude(self):
        result = self._invoke("--name", "foo", "--count", "1", exclude={"name"})
        assert "--name" not in result
        assert "--count" in result

    def test_arguments_skipped(self):
        captured = {}

        @click.command("_fwd_with_arg")
        @click.argument("pos")
        @click.option("--opt", default=None)
        @click.pass_context
        def cmd(ctx, pos, opt):
            captured["ctx"] = ctx

        CliRunner().invoke(cmd, ["myarg", "--opt", "val"], standalone_mode=False)
        result = _forward_ctx_params(captured["ctx"])
        assert "myarg" not in result
        assert "--opt" in result
