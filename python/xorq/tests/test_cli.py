import contextlib
import io
import os
import re
import shutil
import subprocess as _subprocess
import sys
import uuid
import zipfile
from itertools import chain
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pandas as pd
import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

import xorq
import xorq.api as xo
from xorq.caching.strategy import SnapshotStrategy
from xorq.catalog.cli import cli as catalog_cli
from xorq.cli import (
    arbitrate_output_format,
    build_command,
    cli,
    maybe_unzip,
    run_command,
)
from xorq.cli_constants import OutputFormats
from xorq.cli_options import (
    cache_dir_option,
    cache_strategy_options,
    limit_option,
    output_options,
    params_option,
    serve_options,
    unbind_options,
)
from xorq.common.utils.io_utils import Peeker
from xorq.common.utils.logging_utils import Run, Runs
from xorq.common.utils.node_utils import (
    find_node,
)
from xorq.common.utils.process_utils import (
    remove_ansi_escape,
    subprocess_run,
)
from xorq.flight.client import (
    FlightClient,
)
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
)
from xorq.init_templates import InitTemplates


build_run_examples_expr_names = (
    ("local_cache.py", "expr"),
    ("multi_engine.py", "expr"),
    ("remote_caching.py", "expr"),
    ("iris_example.py", "expr"),
    ("simple_example.py", "expr"),
    ("deferred_read_csv.py", "pg_expr_replace"),
    ("train_test_splits.py", "train_table"),
    ("train_test_splits.py", "split_column"),
    ("postgres_caching.py", "expr"),
    ("xgboost_udaf.py", "expr"),
    ("expr_scalar_udf.py", "expr"),
    ("bank_marketing.py", "encoded_test"),
    ("flight_udtf_llm_example.py", "expr"),
    ("pyiceberg_backend_simple.py", "expr"),
    ("python_udwf.py", "expr"),
)


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert xorq.__version__ in result.output


def test_build_command_function(tmp_path, fixture_dir):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.py"

    build_command(script_path, "expr", str(builds_dir))
    assert builds_dir.exists()


def test_build_command_relocate_reads(tmp_path: Path) -> None:
    """--relocate-reads flag should plumb through build_command and bundle local files."""
    parquet_path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), parquet_path)

    script = tmp_path / "script.py"
    script.write_text(
        "from xorq.common.utils.defer_utils import deferred_read_parquet\n"
        f"expr = deferred_read_parquet('{parquet_path}')\n"
    )
    builds_dir = tmp_path / "builds"
    build_command(str(script), "expr", str(builds_dir), relocate_reads=True)

    build_dirs = list(builds_dir.iterdir())
    assert len(build_dirs) == 1
    reads_dir = build_dirs[0] / "reads"
    assert reads_dir.exists(), "reads/ directory should be created by relocate_reads"
    assert list(reads_dir.glob("*.parquet"))


def test_build_command(tmp_path, fixture_dir):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(builds_dir),
    ]
    (returncode, _, stderr) = subprocess_run(test_args)

    assert "Building expr" in stderr.decode("ascii")
    assert returncode == 0, stderr
    assert builds_dir.exists()


def test_build_command_emit_build_path_to(tmp_path, fixture_dir):
    builds_dir = tmp_path / "builds"
    emit_path = tmp_path / "build_path.txt"
    script_path = fixture_dir / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(builds_dir),
        "--emit-build-path-to",
        str(emit_path),
    ]
    env = {**os.environ, "OTEL_EXPORTER_CONSOLE_FALLBACK": ""}
    (returncode, stdout, stderr) = subprocess_run(test_args, env=env)
    assert returncode == 0, stderr
    assert emit_path.exists()
    emitted = emit_path.read_text().strip()
    assert emitted, "emit file is empty"
    # stdout still ends with the path for back-compat with shell consumers.
    stdout_last = stdout.decode("ascii").strip().splitlines()[-1]
    assert emitted == stdout_last
    assert Path(emitted).is_dir()


def test_build_command_emit_survives_stdout_pollution(tmp_path, fixture_dir):
    """OTel console fallback flushes span JSON to stdout at process exit;
    the build_path file must still hold the correct path."""
    builds_dir = tmp_path / "builds"
    emit_path = tmp_path / "build_path.txt"
    script_path = fixture_dir / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(builds_dir),
        "--emit-build-path-to",
        str(emit_path),
    ]
    env = {
        **os.environ,
        "OTEL_EXPORTER_CONSOLE_FALLBACK": "1",
        # Force the console fallback even if the caller's environment
        # points OTel at a real collector.
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "",
    }
    (returncode, stdout, stderr) = subprocess_run(test_args, env=env)
    assert returncode == 0, stderr
    assert emit_path.exists()
    emitted = emit_path.read_text().strip()
    assert Path(emitted).is_dir()
    stdout_text = stdout.decode("ascii", errors="replace")
    stdout_has_otel = '"name":' in stdout_text
    if not stdout_has_otel:
        pytest.skip("OTel console fallback did not produce stdout output")


@pytest.mark.slow(level=1)
def test_build_command_with_udtf(tmp_path, fixture_dir):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "udxf_expr.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--builds-dir",
        str(builds_dir),
    ]
    (returncode, _, stderr) = subprocess_run(test_args)
    assert "Building expr" in stderr.decode("ascii")
    assert returncode == 0, stderr
    assert builds_dir.exists()


@pytest.mark.slow(level=1)
def test_build_command_on_notebook(monkeypatch, tmp_path, fixture_dir, capsys):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.ipynb"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--builds-dir",
        str(builds_dir),
    ]
    (returncode, _, stderr) = subprocess_run(test_args)

    assert "Building expr" in stderr.decode("ascii")
    assert returncode == 0, stderr
    assert builds_dir.exists()


@pytest.mark.slow(level=1)
def test_build_command_with_cache_dir(tmp_path, fixture_dir):
    builds_dir = tmp_path / "builds"
    cache_dir = tmp_path / "cache"
    script_path = fixture_dir / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(builds_dir),
        "--cache-dir",
        str(cache_dir),
    ]
    (returncode, _, stderr) = subprocess_run(test_args)

    assert "Building expr" in stderr.decode("ascii")
    assert returncode == 0, stderr
    assert builds_dir.exists()


def test_run_command_raises_on_unbound_expr(tmp_path):
    t = xo.table(schema={"a": "int64"})
    expr = t.filter(t.a > 0)
    build_dir = build_expr(expr, builds_dir=tmp_path)
    with pytest.raises(ValueError, match="Cannot run unbound expression"):
        run_command(build_dir)


@pytest.mark.slow(level=1)
def test_run_command_default(tmp_path, fixture_dir):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        expression_path = match.group()
        test_args = [
            "xorq",
            "run",
            expression_path,
        ]
        (returncode, _, _) = subprocess_run(test_args)

        # test with problematic name (see https://github.com/xorq-labs/xorq/issues/1116)
        test_args = [
            "xorq",
            "run",
            str(
                shutil.move(
                    expression_path,
                    Path(expression_path).parent.joinpath("becb4e71406b.bak"),
                )
            ),
        ]
        (returncode, _, stderr) = subprocess_run(test_args)

        assert returncode == 0, stderr
    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
@pytest.mark.parametrize("output_format", OutputFormats)
def test_run_command(tmp_path, fixture_dir, output_format):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        output_path = tmp_path / f"test.{output_format}"
        expression_path = match.group()
        test_args = [
            "xorq",
            "run",
            expression_path,
            "--output-path",
            str(output_path),
            "--format",
            output_format,
        ]
        (returncode, _, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
def test_run_cached_command(tmp_path, fixture_dir, output_format):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        output_path = tmp_path / f"test.{output_format}"
        expression_path = match.group()
        test_args = [
            "xorq",
            "run-cached",
            expression_path,
            "--output-path",
            str(output_path),
            "--format",
            output_format,
        ]
        (returncode, _, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
def test_run_cached_command_default(tmp_path, fixture_dir):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        expression_path = match.group()
        test_args = [
            "xorq",
            "run-cached",
            expression_path,
        ]
        (returncode, _, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
@pytest.mark.parametrize(
    "cache_type,ttl",
    [
        ("modification-time", None),
        ("snapshot", None),
        ("snapshot", "3600"),
    ],
)
def test_run_cached_command_cache_types(tmp_path, fixture_dir, cache_type, ttl):
    target_dir = tmp_path / "build"
    cache_dir = tmp_path / "cache"
    script_path = fixture_dir / "pipeline.py"

    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        output_path = tmp_path / "test.parquet"
        expression_path = match.group()
        test_args = [
            "xorq",
            "run-cached",
            expression_path,
            "--cache-dir",
            str(cache_dir),
            "--cache-type",
            cache_type,
            "--output-path",
            str(output_path),
        ]
        if ttl is not None:
            test_args.extend(["--ttl", ttl])
        (returncode, _, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
@pytest.mark.xdist_group(name="serve")
@pytest.mark.parametrize(
    "host,port,cache_dir", [(None, None, None), ("localhost", None, "cache")]
)
def test_serve_command(tmp_path, fixture_dir, cache_dir, host, port):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "udxf_pipeline.py"

    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (return_code, stdout, stderr) = subprocess_run(build_args)
    assert return_code == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        expression_path = match.group()

        optional_args = tuple(
            chain.from_iterable(
                (arg, value)
                for arg, value in (
                    ("--cache-dir", str(tmp_path / cache_dir) if cache_dir else None),
                    ("--host", host),
                    ("--port", port),
                )
                if value
            )
        )

        serve_args = ("xorq", "serve-flight-udxf", str(expression_path), *optional_args)

        with serve_process(serve_args) as (proc, peeker):
            port = peek_port(proc, peeker)

            flight_con = xo.flight.connect(host=host, port=int(port))
            assert (
                proc.poll() is None
                and "diamonds_exchange_command" in flight_con.list_exchanges()
            )

    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
def test_run_command_stdout(tmp_path, fixture_dir, output_format):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"
    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        expression_path = match.group()
        test_args = [
            "xorq",
            "run",
            expression_path,
            "--output-path",
            "-",
            "--format",
            output_format,
        ]
        (returncode, stdout, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
        assert stdout
    else:
        raise AssertionError("No expression hash")


def test_run_command_logging(tmp_path):
    """Test that run_command emits expected structured log events with metrics."""
    expr = xo.memtable({"a": [1, 2, 3], "b": [4, 5, 6]}, name="test_table")
    expr_path = build_expr(
        expr, builds_dir=tmp_path / "builds", cache_dir=tmp_path / "cache"
    )
    output_path = tmp_path / "out.parquet"
    runs_dir = tmp_path / "runs"

    with patch(
        "xorq.common.utils.logging_utils.get_xorq_runs_dir", return_value=runs_dir
    ):
        run_command(str(expr_path), str(output_path), "parquet")

    expr_hash = expr_path.name
    run: Run = Runs(expr_dir=runs_dir / expr_hash).runs[0]
    events = run.read_events()
    by_name = {e["event"]: e for e in events}

    # run.start captures all input params
    assert "run.start" in by_name
    start = by_name["run.start"]
    assert start["expr_path"] == str(expr_path)
    assert start["output_path"] == str(output_path)
    assert "output_format" in start
    assert "limit" in start

    # run.expr_loaded has elapsed timing
    assert "run.expr_loaded" in by_name
    assert by_name["run.expr_loaded"]["elapsed_s"] >= 0

    # run.done carries timing + output_format
    assert "run.done" in by_name
    done = by_name["run.done"]
    assert "elapsed_s" in done
    assert "output_format" in done

    # run.output_written carries file metrics for parquet output
    assert "run.output_written" in by_name
    written = by_name["run.output_written"]
    assert "bytes" in written
    assert written["bytes"] > 0
    assert "rows" in written
    assert written["rows"] == 3
    # rows uses pq.read_metadata (parquet footer) — verify it matches the actual file
    assert pq.read_metadata(output_path).num_rows == written["rows"]


def test_run_command_error_logging(tmp_path):
    """Test that run_command records error status in meta and re-raises exceptions."""
    runs_dir = tmp_path / "runs"
    nonexistent_path = tmp_path / "does_not_exist"
    expr_hash = nonexistent_path.name

    with patch(
        "xorq.common.utils.logging_utils.get_xorq_runs_dir", return_value=runs_dir
    ):
        with pytest.raises(Exception):  # noqa: B017
            run_command(str(nonexistent_path), str(tmp_path / "out.parquet"))

    run: Run = Runs(expr_dir=runs_dir / expr_hash).runs[0]
    meta = run.read_meta()
    assert meta["status"] == "error"
    assert "error" in meta


def test_run_command_writes_run_logger(tmp_path):
    expr = xo.memtable({"x": [10, 20, 30]}, name="t")
    expr_path = build_expr(
        expr, builds_dir=tmp_path / "builds", cache_dir=tmp_path / "cache"
    )
    output_path = tmp_path / "out.parquet"
    runs_dir = tmp_path / "runs"

    with patch(
        "xorq.common.utils.logging_utils.get_xorq_runs_dir", return_value=runs_dir
    ):
        run_command(str(expr_path), str(output_path), "parquet")

        expr_hash = expr_path.name
        runs_obj = Runs(expr_dir=runs_dir / expr_hash)
        assert len(runs_obj.list()) == 1, "Expected exactly one run to be recorded"

        run: Run = runs_obj.runs[0]
        assert uuid.UUID(run.run_id).version == 4, "Run ID should be a UUID4"

        meta = run.read_meta()
        assert meta is not None
        assert meta["status"] == "ok"
        assert meta["run_id"] == run.run_id
        assert meta["expr_hash"] == expr_hash
        assert meta["expr_path"] == str(expr_path)
        assert meta["output_format"] == "parquet"
        assert "started_at" in meta
        assert "completed_at" in meta
        assert meta["xorq_version"] == xorq.__version__

        events = run.read_events()
        event_names = [e["event"] for e in events]
        assert "run.start" in event_names
        assert "run.expr_loaded" in event_names
        assert "run.done" in event_names
        assert "run.output_written" in event_names

        loaded_event = next(e for e in events if e["event"] == "run.expr_loaded")
        assert "elapsed_s" in loaded_event
        assert loaded_event["elapsed_s"] >= 0

        written_event = next(e for e in events if e["event"] == "run.output_written")
        assert "bytes" in written_event
        assert written_event["bytes"] > 0


def test_run_command_run_logger_error_status(tmp_path):
    runs_dir = tmp_path / "runs"
    nonexistent_path = tmp_path / "does_not_exist"

    with patch(
        "xorq.common.utils.logging_utils.get_xorq_runs_dir", return_value=runs_dir
    ):
        with pytest.raises(Exception):  # noqa: B017
            run_command(str(nonexistent_path), str(tmp_path / "out.parquet"))

        expr_hash = nonexistent_path.name
        runs_obj = Runs(expr_dir=runs_dir / expr_hash)
        assert len(runs_obj.list()) == 1

        meta = runs_obj.runs[0].read_meta()
        assert meta is not None
        assert meta["status"] == "error"
        assert "error" in meta


def test_run_logger_multiple_runs(tmp_path):
    expr = xo.memtable({"v": [1, 2]}, name="t2")
    expr_path = build_expr(
        expr, builds_dir=tmp_path / "builds", cache_dir=tmp_path / "cache"
    )
    output_path = tmp_path / "out.parquet"
    runs_dir = tmp_path / "runs"

    with patch(
        "xorq.common.utils.logging_utils.get_xorq_runs_dir", return_value=runs_dir
    ):
        run_command(str(expr_path), str(output_path), "parquet")
        run_command(str(expr_path), str(output_path), "parquet")

        expr_hash = expr_path.name
        run_ids = Runs(expr_dir=runs_dir / expr_hash).list()
        assert len(run_ids) == 2
        assert run_ids[0] != run_ids[1]


@pytest.mark.parametrize(
    "expression,message",
    [
        ("integer", "The object integer must be an instance of"),
        ("missing", "Expression missing not found"),
    ],
)
@pytest.mark.slow(level=1)
def test_build_command_bad_expr_name(tmp_path, fixture_dir, expression, message):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        expression,
        "--builds-dir",
        str(builds_dir),
    ]
    (returncode, _, stderr) = subprocess_run(test_args)
    assert returncode != 0
    assert message in stderr.decode("ascii")


@pytest.mark.parametrize(
    ("example", "expr_name"),
    build_run_examples_expr_names,
)
@pytest.mark.slow(level=2)
def test_examples(
    example,
    expr_name,
    examples_dir,
    tmp_path,
):
    # build
    builds_dir = tmp_path / "builds"
    example_path = examples_dir / example
    assert example_path.exists()
    build_args = (
        "xorq",
        "build",
        str(example_path),
        "--expr-name",
        expr_name,
        "--builds-dir",
        str(builds_dir),
    )
    print(" ".join(build_args), file=sys.stderr)
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr
    print(stderr.decode("ascii"), file=sys.stderr)
    expression_path = Path(stdout.decode("ascii").strip().split("\n")[-1])
    # debugging can capture stdout and result in spurious path of "."
    assert expression_path.name and expression_path.exists()

    # run
    output_format = "parquet"
    output_path = expression_path / f"test.{output_format}"
    assert not output_path.exists()
    run_args = (
        "xorq",
        "run",
        str(expression_path),
        "--format",
        output_format,
        "--output-path",
        str(output_path),
    )
    print(" ".join(run_args), file=sys.stderr)
    (returncode, stdout, stderr) = subprocess_run(run_args)
    assert returncode == 0, stderr
    print(stderr, file=sys.stderr)
    assert output_path.exists()


def test_init_command_default(tmpdir):
    path = Path(tmpdir).joinpath("xorq-template-default")
    init_args = (
        "xorq",
        "init",
        "--path",
        str(path),
    )
    print(" ".join(init_args), file=sys.stderr)
    (returncode, stdout, stderr) = subprocess_run(init_args)
    assert returncode == 0, stderr
    assert path.exists()
    assert path.joinpath("pyproject.toml").exists()


@pytest.mark.parametrize("template", InitTemplates)
def test_init_command_sklearn(template, tmpdir):
    path = Path(tmpdir).joinpath(f"xorq-template-{template}")
    init_args = (
        "xorq",
        "init",
        "--path",
        str(path),
        "--template",
        template,
    )
    print(" ".join(init_args), file=sys.stderr)
    (returncode, stdout, stderr) = subprocess_run(init_args)
    assert returncode == 0, stderr
    assert path.exists()
    assert path.joinpath("pyproject.toml").exists()


@pytest.mark.parametrize("template", InitTemplates)
def test_init_command_path_exists(template, tmpdir):
    path = Path(tmpdir).joinpath(f"xorq-template-{template}")
    init_args = (
        "xorq",
        "init",
        "--path",
        str(path),
        "--template",
        template,
    )
    print(" ".join(init_args), file=sys.stderr)
    path.mkdir()
    (returncode, stdout, stderr) = subprocess_run(init_args)
    assert returncode != 0


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", InitTemplates)
def test_init_uv_build_uv_run(template, tmpdir):
    tmpdir = Path(tmpdir)
    path = tmpdir.joinpath(f"xorq-template-{template}")
    init_args = (
        "xorq",
        "init",
        "--path",
        str(path),
        "--template",
        template,
    )
    print(" ".join(init_args), file=sys.stderr)
    (returncode, stdout, stderr) = subprocess_run(init_args)
    assert returncode == 0, stderr
    assert path.exists()
    assert path.joinpath("pyproject.toml").exists()
    assert path.joinpath("requirements.txt").exists()
    # Remove pre-committed requirements.txt; the template's copy may have been
    # exported with a different uv version than CI, causing a sync-check failure.
    path.joinpath("requirements.txt").unlink()

    build_args = (
        "xorq",
        "uv",
        "build",
        str(path.joinpath("expr.py")),
    )
    (returncode, stdout, stderr) = subprocess_run(build_args, text=True)
    assert returncode == 0, stderr
    build_path = Path(stdout.strip().split("\n")[-1])
    assert build_path.exists()

    output_path = tmpdir.joinpath("output")
    run_args = (
        "xorq",
        "uv",
        "run",
        "--output-path",
        str(output_path),
        str(build_path),
    )
    (returncode, stdout, stderr) = subprocess_run(run_args, text=True)
    assert returncode == 0, stderr
    assert output_path.exists()


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
def test_uv_build_with_project_path(tmpdir):
    tmpdir = Path(tmpdir)
    path = tmpdir.joinpath("xorq-template-default")
    init_args = (
        "xorq",
        "init",
        "--path",
        str(path),
    )
    (returncode, stdout, stderr) = subprocess_run(init_args)
    assert returncode == 0, stderr
    (path / "requirements.txt").unlink(missing_ok=True)

    # Move script outside the project dir so upward search would fail
    script_path = tmpdir / "expr.py"
    shutil.copy2(path / "expr.py", script_path)

    build_args = (
        "xorq",
        "uv",
        "build",
        str(script_path),
        "--project-path",
        str(path),
    )
    (returncode, stdout, stderr) = subprocess_run(build_args, text=True)
    assert returncode == 0, stderr
    build_path = Path(stdout.strip().split("\n")[-1])
    assert build_path.exists()


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
def test_uv_build_no_all_extras(tmpdir):
    tmpdir = Path(tmpdir)
    path = tmpdir.joinpath("xorq-template-default")
    (returncode, _, stderr) = subprocess_run(("xorq", "init", "--path", str(path)))
    assert returncode == 0, stderr
    (path / "requirements.txt").unlink(missing_ok=True)
    build_args = (
        "xorq",
        "uv",
        "build",
        str(path / "expr.py"),
        "--no-all-extras",
    )
    (returncode, stdout, stderr) = subprocess_run(build_args, text=True)
    assert returncode == 0, stderr
    build_path = Path(stdout.strip().split("\n")[-1])
    assert build_path.exists()


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
def test_uv_build_with_extra(tmpdir):
    import tomlkit  # noqa: PLC0415

    tmpdir = Path(tmpdir)
    path = tmpdir.joinpath("xorq-template-default")
    (returncode, _, stderr) = subprocess_run(("xorq", "init", "--path", str(path)))
    assert returncode == 0, stderr
    # Add an optional dependency group and re-lock
    pyproject = path / "pyproject.toml"
    data = tomlkit.loads(pyproject.read_text())
    data["project"]["optional-dependencies"] = {"testgroup": ["requests>=2"]}
    pyproject.write_text(tomlkit.dumps(data))
    subprocess_run(("uv", "lock", "--directory", str(path)))
    (path / "requirements.txt").unlink(missing_ok=True)
    build_args = (
        "xorq",
        "uv",
        "build",
        str(path / "expr.py"),
        "--no-all-extras",
        "--extra",
        "testgroup",
    )
    (returncode, stdout, stderr) = subprocess_run(build_args, text=True)
    assert returncode == 0, stderr
    build_path = Path(stdout.strip().split("\n")[-1])
    assert build_path.exists()


serve_hashes = (
    "c0480036de329f7d90bd9ebda3b58fc6",  # batting, rel.Read
    "9f91abf3ff66d64d441724f71cab534e",  # awards_players, rel.Read
    "3ce4de9ff26d12bbd0159061325af767",  # left, ops.Filter
    "dd76e5abc15dc283090fa73338c1ecf3",  # right, ops.DropColumns
)


@pytest.fixture(scope="session")
def pipeline_https_build(tmp_path_factory, fixture_dir):
    builds_dir = tmp_path_factory.mktemp("builds")
    script_path = fixture_dir / "pipeline_https.py"

    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(builds_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args, text=True)

    assert "Building expr" in stderr
    assert returncode == 0, stderr
    assert builds_dir.exists()
    serve_dir = Path(stdout.strip())
    return serve_dir


@contextlib.contextmanager
def serve_process(args):
    proc = _subprocess.Popen(args, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE)
    peeker = Peeker(proc.stdout)
    try:
        yield proc, peeker
    finally:
        proc.terminate()
        proc.wait()


def peek_port(proc, peeker, timeout=60):
    def do_match(buf):
        (*_, line) = remove_ansi_escape(buf.decode("ascii").strip()).rsplit("\n", 1)
        match = re.match(".*on grpc://localhost:(\\d+)$", line)
        return match

    proc.poll()
    if proc.returncode:
        raise Exception(proc.stderr.read())
    try:
        buf = peeker.peek_line_until(do_match, timeout=timeout)
    except TimeoutError as e:
        proc.terminate()
        raise Exception(proc.stderr.read()) from e
    (as_string,) = do_match(buf).groups()
    port = int(as_string)
    return port


def hit_server(port, expr):
    client = FlightClient(port=port)
    (_, rbr) = client.do_exchange("default", expr)
    df = rbr.read_pandas()
    return df


@pytest.mark.slow(level=3)
@pytest.mark.xdist_group(name="serve")
@pytest.mark.parametrize("serve_hash", serve_hashes)
def test_serve_unbound_hash(serve_hash, pipeline_https_build):
    lookup = {
        "dd76e5abc15dc283090fa73338c1ecf3": "xorq.vendor.ibis.expr.operations.DropColumns",
        "3ce4de9ff26d12bbd0159061325af767": "xorq.vendor.ibis.expr.operations.Filter",
    }
    expr = load_expr(pipeline_https_build)
    typ = lookup.get(serve_hash)
    subexpr = find_node(
        expr, hash=serve_hash, tag=None, typs=typ, strategy=SnapshotStrategy()
    ).to_expr()

    serve_args = (
        "xorq",
        "serve-unbound",
        str(pipeline_https_build),
        "--to-unbind-hash",
        serve_hash,
    ) + (("--typ", typ) if typ else ())
    with serve_process(serve_args) as (proc, peeker):
        port = peek_port(proc, peeker)
        actual = hit_server(port=port, expr=subexpr)
        expected = expr.execute()
        (actual, expected) = (
            df.sort_values(list(df.columns), ignore_index=True)
            for df in (actual, expected)
        )
        assert actual.equals(expected)


serve_tags = (
    "read-batting",
    "read-players",
    "batting-filtered",
    "players-filtered",
    # this needs the fix for finding the correct source
    "joined",
)


@pytest.mark.slow(level=3)
@pytest.mark.xdist_group(name="serve")
@pytest.mark.parametrize("serve_tag", serve_tags)
def test_serve_unbound_tag(serve_tag, pipeline_https_build):
    expr = load_expr(pipeline_https_build)
    subexpr = find_node(expr, hash=None, tag=serve_tag).to_expr()

    serve_args = (
        "xorq",
        "serve-unbound",
        str(pipeline_https_build),
        "--to-unbind-tag",
        serve_tag,
    )
    with serve_process(serve_args) as (proc, peeker):
        port = peek_port(proc, peeker)
        actual = hit_server(port=port, expr=subexpr)
        expected = expr.execute()
        (actual, expected) = (
            df.sort_values(list(df.columns), ignore_index=True)
            for df in (actual, expected)
        )
        assert actual.equals(expected)


@pytest.mark.slow(level=1)
@pytest.mark.xdist_group(name="serve")
def test_serve_unbound_tag_get_exchange(pipeline_https_build, parquet_dir):
    batting_url = "https://storage.googleapis.com/letsql-pins/batting/20240711T171118Z-431ef/batting.parquet"
    serve_tag = "read-batting"
    expr = load_expr(pipeline_https_build)

    serve_args = (
        "xorq",
        "serve-unbound",
        str(pipeline_https_build),
        "--to-unbind-tag",
        serve_tag,
    )
    with serve_process(serve_args) as (proc, peeker):
        port = peek_port(proc, peeker)

        flight_backend = xo.flight.connect(port=port)
        f = flight_backend.get_exchange("default")
        actual = xo.deferred_read_parquet(batting_url).pipe(f).execute()

        expected = expr.execute()
        (actual, expected) = (
            df.sort_values(list(df.columns), ignore_index=True)
            for df in (actual, expected)
        )
        assert actual.equals(expected)


@pytest.mark.slow(level=1)
@pytest.mark.xdist_group(name="serve")
def test_serve_unbound_tag_get_exchange_udf(fixture_dir, tmp_path):
    df = pd.DataFrame([float(v) for v in range(10)], columns=["x"])

    serve_tag = "full"

    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline_pandas_udf.py"

    # Capture print output
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        build_command(script_path, "expr", str(builds_dir))

    serve_args = (
        "xorq",
        "serve-unbound",
        str(output.getvalue().strip()),
        "--to-unbind-tag",
        serve_tag,
    )
    with serve_process(serve_args) as (proc, peeker):
        port = peek_port(proc, peeker)

        flight_backend = xo.flight.connect(port=port)
        f = flight_backend.get_exchange("default")
        actual = xo.connect().register(df).select("x").pipe(f).execute()

        assert not actual.empty


@pytest.mark.slow(level=3)
@pytest.mark.xdist_group(name="serve")
def test_serve_penguins_template(tmpdir, tmp_path):
    tmpdir = Path(tmpdir)
    path = tmpdir.joinpath("xorq-template-penguins")
    init_args = (
        "xorq",
        "init",
        "--path",
        str(path),
        "--template",
        "penguins",
    )

    (returncode, stdout, stderr) = subprocess_run(init_args)

    assert returncode == 0, stderr
    assert path.exists()
    assert path.joinpath("pyproject.toml").exists()
    assert path.joinpath("requirements.txt").exists()

    target_dir = tmp_path / "build"
    build_args = [
        "xorq",
        "build",
        str(path / "expr.py"),
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)

    assert "Building expr" in stderr.decode("ascii")
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        serve_hash = "5ee5e0d98754937a205e8be7e0728bb7"  # CachedNode (test split)

        serve_args = (
            "xorq",
            "serve-unbound",
            str(target_dir / match.group()),
            "--to-unbind-hash",
            serve_hash,
        )
        with serve_process(serve_args) as (proc, peeker):
            port = peek_port(proc, peeker)

            # Create sample penguin data using memtable instead of reading from URL
            sample_data = pd.DataFrame(
                {
                    "bill_length_mm": [
                        39.1,
                        39.5,
                        40.3,
                        36.7,
                        39.3,
                        38.9,
                        39.2,
                        34.1,
                        42.0,
                        37.8,
                    ],
                    "bill_depth_mm": [
                        18.7,
                        17.4,
                        18.0,
                        19.3,
                        20.6,
                        17.8,
                        19.6,
                        18.1,
                        20.2,
                        17.1,
                    ],
                    "species": [
                        "Adelie",
                        "Adelie",
                        "Adelie",
                        "Adelie",
                        "Adelie",
                        "Chinstrap",
                        "Chinstrap",
                        "Chinstrap",
                        "Gentoo",
                        "Gentoo",
                    ],
                }
            )

            expr = xo.memtable(sample_data, name="penguins")

            actual = hit_server(port=port, expr=expr)
            assert not actual.empty
            assert actual["predict"].isin(("Adelie", "Chinstrap", "Gentoo")).all()
            assert len(actual) == len(sample_data)
    else:
        raise AssertionError("No expression hash")


# ---------------------------------------------------------------------------
# uv run-cached / uv run-unbound: help-text tests
# ---------------------------------------------------------------------------


def test_uv_run_cached_help():
    result = CliRunner().invoke(cli, ["uv", "run-cached", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--cache-type",
        "--ttl",
        "--limit",
        "--params",
        "--cache-dir",
        "--output-path",
        "--format",
    ):
        assert flag in result.output, f"missing {flag}"


def test_uv_run_unbound_help():
    result = CliRunner().invoke(cli, ["uv", "run-unbound", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--to-unbind-hash",
        "--to-unbind-tag",
        "--typ",
        "--batch-size",
        "--instream",
        "--limit",
        "--output-path",
        "--format",
    ):
        assert flag in result.output, f"missing {flag}"
    assert "stdout" in result.output, "--output-path help text should mention stdout"


def test_catalog_serve_unbound_help():
    result = CliRunner().invoke(catalog_cli, ["serve-unbound", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--to-unbind-hash",
        "--to-unbind-tag",
        "--typ",
        "--host",
        "--port",
        "--prometheus-port",
        "--code",
        "--fuse",
        "--rename-params",
        "--params",
        "--cache-dir",
    ):
        assert flag in result.output, f"missing {flag}"


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
def test_uv_run_cached_roundtrip(tmpdir):
    tmpdir = Path(tmpdir)
    path = tmpdir / "xorq-template-penguins"
    (returncode, _, stderr) = subprocess_run(
        ("xorq", "init", "--path", str(path), "--template", "penguins"), text=True
    )
    assert returncode == 0, stderr
    # Template's requirements.txt may be from a different uv version, causing sync failures
    path.joinpath("requirements.txt").unlink(missing_ok=True)

    (returncode, stdout, stderr) = subprocess_run(
        ("xorq", "uv", "build", str(path / "expr.py")), text=True
    )
    assert returncode == 0, stderr
    # build_command prints the build directory path as the last line of stdout
    build_path = Path(stdout.strip().split("\n")[-1])
    assert build_path.exists(), (
        f"build_path not found: {build_path!r} (stdout={stdout!r})"
    )

    output_path = tmpdir / "cached_output.parquet"
    cache_dir = tmpdir / "cache"
    (returncode, _, stderr) = subprocess_run(
        (
            "xorq",
            "uv",
            "run-cached",
            str(build_path),
            "--cache-type",
            "modification-time",
            "--output-path",
            str(output_path),
            "--cache-dir",
            str(cache_dir),
        ),
        text=True,
    )
    assert returncode == 0, stderr
    assert output_path.exists()
    first_size = output_path.stat().st_size
    assert first_size > 0

    output_path_2 = tmpdir / "cached_output_2.parquet"
    (returncode, _, stderr) = subprocess_run(
        (
            "xorq",
            "uv",
            "run-cached",
            str(build_path),
            "--cache-type",
            "modification-time",
            "--output-path",
            str(output_path_2),
            "--cache-dir",
            str(cache_dir),
        ),
        text=True,
    )
    assert returncode == 0, stderr
    assert output_path_2.exists()
    assert output_path_2.stat().st_size == first_size


_PENGUINS_IPC_SCHEMA = pa.schema(
    [
        ("species", pa.string()),
        ("island", pa.string()),
        ("bill_length_mm", pa.float64()),
        ("bill_depth_mm", pa.float64()),
        ("flipper_length_mm", pa.float64()),
        ("body_mass_g", pa.float64()),
        ("sex", pa.string()),
        ("year", pa.int64()),
    ]
)


def _make_penguins_ipc_bytes():
    batch = pa.record_batch(
        [
            pa.array(["Adelie", "Gentoo", "Adelie"]),
            pa.array(["Torgersen", "Biscoe", "Dream"]),
            pa.array([39.1, 46.1, 36.7]),
            pa.array([18.7, 13.2, 19.3]),
            pa.array([181.0, 211.0, 193.0]),
            pa.array([3750.0, 5200.0, 3450.0]),
            pa.array(["male", "female", "female"]),
            pa.array([2007, 2007, 2007]),
        ],
        schema=_PENGUINS_IPC_SCHEMA,
    )
    buf = io.BytesIO()
    writer = pa.ipc.new_stream(buf, _PENGUINS_IPC_SCHEMA)
    writer.write_batch(batch)
    writer.close()
    return buf.getvalue()


@pytest.fixture(scope="session")
def uv_unbound_build(tmp_path_factory, fixture_dir):
    tmpdir = tmp_path_factory.mktemp("uv-unbound")
    path = tmpdir / "xorq-template-penguins"
    (returncode, _, stderr) = subprocess_run(
        ("xorq", "init", "--path", str(path), "--template", "penguins")
    )
    assert returncode == 0, stderr
    path.joinpath("requirements.txt").unlink(missing_ok=True)

    shutil.copy2(fixture_dir / "pipeline_unbound.py", path / "expr.py")

    (returncode, stdout, stderr) = subprocess_run(
        ("xorq", "uv", "build", str(path / "expr.py")), text=True
    )
    assert returncode == 0, stderr
    build_path = Path(stdout.strip().split("\n")[-1])
    assert build_path.exists()
    return build_path


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
def test_uv_run_unbound_roundtrip_stdin(uv_unbound_build, tmp_path):
    ipc_data = _make_penguins_ipc_bytes()
    output_path = tmp_path / "unbound_output.parquet"

    (returncode, _, stderr) = subprocess_run(
        (
            "xorq",
            "uv",
            "run-unbound",
            str(uv_unbound_build),
            "--to-unbind-tag",
            "source",
            "--output-path",
            str(output_path),
            "--format",
            "parquet",
        ),
        input=ipc_data,
        timeout=120,
    )
    assert returncode == 0, stderr
    assert output_path.exists()
    table = pq.read_table(output_path)
    assert len(table) == 2
    assert set(table.column_names) == {"species", "island", "bill_length_mm"}
    assert all(v.as_py() == "Adelie" for v in table.column("species"))


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
def test_uv_run_unbound_roundtrip_file(uv_unbound_build, tmp_path):
    ipc_data = _make_penguins_ipc_bytes()
    ipc_path = tmp_path / "input.arrow"
    ipc_path.write_bytes(ipc_data)

    output_path = tmp_path / "unbound_output.parquet"

    (returncode, _, stderr) = subprocess_run(
        (
            "xorq",
            "uv",
            "run-unbound",
            str(uv_unbound_build),
            "--to-unbind-tag",
            "source",
            "-i",
            str(ipc_path),
            "--output-path",
            str(output_path),
            "--format",
            "parquet",
        ),
        text=True,
        timeout=120,
    )
    assert returncode == 0, stderr
    assert output_path.exists()
    table = pq.read_table(output_path)
    assert len(table) == 2
    assert set(table.column_names) == {"species", "island", "bill_length_mm"}
    assert all(v.as_py() == "Adelie" for v in table.column("species"))


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
def test_uv_run_unbound_roundtrip_popen(uv_unbound_build, tmp_path):
    ipc_data = _make_penguins_ipc_bytes()
    output_path = tmp_path / "fd_output.parquet"

    proc = _subprocess.Popen(
        (
            "xorq",
            "uv",
            "run-unbound",
            str(uv_unbound_build),
            "--to-unbind-tag",
            "source",
            "--output-path",
            str(output_path),
            "--format",
            "parquet",
        ),
        stdin=_subprocess.PIPE,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(input=ipc_data, timeout=120)
    assert proc.returncode == 0, stderr.decode()
    assert output_path.exists()
    table = pq.read_table(output_path)
    assert len(table) == 2
    assert set(table.column_names) == {"species", "island", "bill_length_mm"}
    assert all(v.as_py() == "Adelie" for v in table.column("species"))


def test_batch_size_forwarded_to_pyarrow_stream(monkeypatch):
    """Verify batch_size flows from run_unbound CLI through to to_pyarrow_stream."""
    mock_stream = MagicMock()
    monkeypatch.setattr("xorq.expr.api.to_pyarrow_stream", mock_stream)

    sentinel = MagicMock()
    arbitrate_output_format(sentinel, "/dev/null", "arrow", batch_size=512)
    mock_stream.assert_called_once_with(sentinel, "/dev/null", chunk_size=512)


def test_batch_size_forwarded_from_run_unbound_cli():
    """Verify --batch-size is wired from the Click command to run_unbound_command."""
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)

    with patch("xorq.cli.run_unbound_command", spy):
        result = CliRunner().invoke(
            cli,
            ["run-unbound", "/fake/build", "--batch-size", "256", "-f", "arrow"],
        )
    assert result.exit_code == 0, result.output
    assert captured.get("batch_size") == 256


def test_batch_size_omitted_when_none(monkeypatch):
    """Verify no chunk_size kwarg when batch_size is None."""
    mock_stream = MagicMock()
    monkeypatch.setattr("xorq.expr.api.to_pyarrow_stream", mock_stream)

    sentinel = MagicMock()
    arbitrate_output_format(sentinel, "/dev/null", "arrow", batch_size=None)
    mock_stream.assert_called_once_with(sentinel, "/dev/null")


# ---------------------------------------------------------------------------
# cli_options.py decorator tests
# ---------------------------------------------------------------------------


def _make_test_command(decorator, cmd_name="test"):
    """Build a minimal Click command using the given decorator(s)."""

    @click.command(cmd_name)
    @decorator
    def cmd(**kwargs):
        for k, v in sorted(kwargs.items()):
            click.echo(f"{k}={v}")

    return cmd


def test_output_options_bare_decorator():
    cmd = _make_test_command(output_options)
    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 0
    assert "output_format=parquet" in result.output
    assert "output_path=None" in result.output


def test_output_options_parametrized():
    @click.command()
    @output_options(output_path_help="Custom help text.")
    def cmd(output_path, output_format):
        click.echo(f"output_format={output_format}")

    result = CliRunner().invoke(cmd, ["--help"])
    assert result.exit_code == 0
    assert "Custom help text." in result.output


def test_output_options_explicit_values():
    cmd = _make_test_command(output_options)
    result = CliRunner().invoke(cmd, ["-o", "/tmp/out.csv", "-f", "csv"])
    assert result.exit_code == 0
    assert "output_format=csv" in result.output
    assert "output_path=/tmp/out.csv" in result.output


def test_limit_option_decorator():
    cmd = _make_test_command(limit_option)
    result = CliRunner().invoke(cmd, ["--limit", "42"])
    assert result.exit_code == 0
    assert "limit=42" in result.output


def test_params_option_dest_name():
    cmd = _make_test_command(params_option)
    result = CliRunner().invoke(cmd, ["--params", "x=1", "--params", "y=2"])
    assert result.exit_code == 0
    assert "raw_params=('x=1', 'y=2')" in result.output


def test_cache_dir_option_decorator():
    cmd = _make_test_command(cache_dir_option)
    result = CliRunner().invoke(cmd, ["--cache-dir", "/tmp/cache"])
    assert result.exit_code == 0
    assert "cache_dir=/tmp/cache" in result.output


def test_cache_strategy_options_decorator():
    cmd = _make_test_command(cache_strategy_options)
    result = CliRunner().invoke(cmd, ["--cache-type", "snapshot", "--ttl", "300"])
    assert result.exit_code == 0
    assert "cache_type=snapshot" in result.output
    assert "ttl=300" in result.output


def test_unbind_options_decorator():
    cmd = _make_test_command(unbind_options)
    result = CliRunner().invoke(
        cmd, ["--to-unbind-hash", "abc", "--to-unbind-tag", "v1", "--typ", "int"]
    )
    assert result.exit_code == 0
    assert "to_unbind_hash=abc" in result.output
    assert "to_unbind_tag=v1" in result.output
    assert "typ=int" in result.output


def test_serve_options_decorator():
    cmd = _make_test_command(serve_options)
    result = CliRunner().invoke(cmd, ["--host", "0.0.0.0", "--port", "8080"])
    assert result.exit_code == 0
    assert "host=0.0.0.0" in result.output
    assert "port=8080" in result.output


def test_maybe_unzip_non_zip_passthrough(tmp_path: Path) -> None:
    build_dir = tmp_path / "my_build"
    build_dir.mkdir()
    with maybe_unzip(str(build_dir)) as p:
        assert p == str(build_dir)


def test_maybe_unzip_valid_zip(tmp_path: Path) -> None:
    build_dir = tmp_path / "my_build"
    build_dir.mkdir()
    (build_dir / "file.txt").write_text("hello")
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(build_dir / "file.txt", "my_build/file.txt")
    with maybe_unzip(str(zip_path)) as p:
        assert p != str(zip_path)
        assert Path(p).is_dir()
        assert (Path(p) / "file.txt").read_text() == "hello"


def test_maybe_unzip_case_insensitive_extension(tmp_path: Path) -> None:
    build_dir = tmp_path / "my_build"
    build_dir.mkdir()
    (build_dir / "file.txt").write_text("hello")
    zip_path = tmp_path / "archive.ZIP"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(build_dir / "file.txt", "my_build/file.txt")
    with maybe_unzip(str(zip_path)) as p:
        assert Path(p).is_dir()
        assert (Path(p) / "file.txt").read_text() == "hello"


def test_maybe_unzip_nonexistent_zip_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "does_not_exist.zip"
    with pytest.raises(FileNotFoundError):
        with maybe_unzip(str(zip_path)) as _:
            pass


def test_maybe_unzip_invalid_zip_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    zip_path.write_text("this is not a zip")
    with pytest.raises(zipfile.BadZipFile):
        with maybe_unzip(str(zip_path)) as _:
            pass
