import re
import shutil
import sys
from itertools import chain
from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
from xorq.cli import (
    build_command,
)
from xorq.common.utils.node_utils import (
    find_node,
)
from xorq.common.utils.process_utils import (
    Popened,
    remove_ansi_escape,
    subprocess_run,
)
from xorq.flight.client import (
    FlightClient,
)
from xorq.ibis_yaml.compiler import (
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


def test_build_command_function(tmp_path, fixture_dir):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.py"

    build_command(script_path, "expr", str(builds_dir))
    assert builds_dir.exists()


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
@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
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
@pytest.mark.parametrize(
    "host,port,cache_dir", [(None, None, None), ("localhost", "5000", "cache")]
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

        serve_process = Popened(serve_args)
        port = peek_port(serve_process)

        flight_con = xo.flight.connect(host=host, port=int(port))
        assert (
            serve_process.popen.poll() is None
            and "diamonds_exchange_command" in flight_con.list_exchanges()
        )
        serve_process.popen.terminate()

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


@pytest.mark.xfail(reason="wait for daniel")
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

    build_args = (
        "xorq",
        "uv-build",
        str(path.joinpath("expr.py")),
    )
    (returncode, stdout, stderr) = subprocess_run(build_args, do_decode=True)
    assert returncode == 0, stderr
    build_path = Path(stdout.strip().split("\n")[-1])
    assert build_path.exists()

    output_path = tmpdir.joinpath("output")
    run_args = (
        "xorq",
        "uv-run",
        "--output-path",
        str(output_path),
        str(build_path),
    )
    (returncode, stdout, stderr) = subprocess_run(run_args, do_decode=True)
    assert returncode == 0, stderr
    assert output_path.exists()


serve_hashes = (
    "323f89d94c90d1dcf0660baefd813658",  # batting, rel.Read
    "e6d438bd87aa3d84babd6f4c2956312f",  # awards_players, rel.Read
    "3c1b0dc766c6d217ab921fc73ebfe933",  # left, ops.Filter
    "c3950f7f9ab98a9e943e02f761db7c2a",  # right, ops.DropColumns
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
    (returncode, stdout, stderr) = subprocess_run(build_args, do_decode=True)

    assert "Building expr" in stderr
    assert returncode == 0, stderr
    assert builds_dir.exists()
    serve_dir = Path(stdout.strip())
    return serve_dir


def peek_port(popened, timeout=60):
    def do_match(buf):
        (*_, line) = remove_ansi_escape(buf.decode("ascii").strip()).rsplit("\n", 1)
        match = re.match(".*on grpc://localhost:(\\d+)$", line)
        return match

    popened.popen.poll()
    if popened.popen.returncode:
        raise Exception(popened.stderr)
    try:
        buf = popened.stdout_peeker.peek_line_until(do_match, timeout=timeout)
    except TimeoutError as e:
        popened.popen.terminate()
        raise Exception(popened.stderr) from e
    (as_string,) = do_match(buf).groups()
    port = int(as_string)
    return port


def hit_server(port, expr):
    client = FlightClient(port=port)
    (_, rbr) = client.do_exchange("default", expr)
    df = rbr.read_pandas()
    return df


@pytest.mark.slow(level=3)
@pytest.mark.parametrize("serve_hash", serve_hashes)
def test_serve_unbound_hash(serve_hash, pipeline_https_build):
    lookup = {
        "c3950f7f9ab98a9e943e02f761db7c2a": "xorq.vendor.ibis.expr.operations.DropColumns",
        "3c1b0dc766c6d217ab921fc73ebfe933": "xorq.vendor.ibis.expr.operations.Filter",
    }
    expr = load_expr(pipeline_https_build)
    typ = lookup.get(serve_hash)
    subexpr = find_node(expr, hash=serve_hash, tag=None, typs=typ).to_expr()

    serve_args = (
        "xorq",
        "serve-unbound",
        str(pipeline_https_build),
        "--to_unbind_hash",
        serve_hash,
    ) + (("--typ", typ) if typ else ())
    serve_popened = Popened(serve_args, deferred=False)
    port = peek_port(serve_popened)
    actual = hit_server(port=port, expr=subexpr)
    expected = expr.execute()
    (actual, expected) = (
        df.sort_values(list(df.columns), ignore_index=True) for df in (actual, expected)
    )
    assert actual.equals(expected)

    serve_popened.popen.terminate()


serve_tags = (
    "read-batting",
    "read-players",
    "batting-filtered",
    "players-filtered",
    # this needs the fix for finding the correct source
    "joined",
)


@pytest.mark.slow(level=3)
@pytest.mark.parametrize("serve_tag", serve_tags)
def test_serve_unbound_tag(serve_tag, pipeline_https_build):
    expr = load_expr(pipeline_https_build)
    subexpr = find_node(expr, hash=None, tag=serve_tag).to_expr()

    serve_args = (
        "xorq",
        "serve-unbound",
        str(pipeline_https_build),
        "--to_unbind_tag",
        serve_tag,
    )
    serve_popened = Popened(serve_args, deferred=False)
    port = peek_port(serve_popened)
    actual = hit_server(port=port, expr=subexpr)
    expected = expr.execute()
    (actual, expected) = (
        df.sort_values(list(df.columns), ignore_index=True) for df in (actual, expected)
    )
    assert actual.equals(expected)

    serve_popened.popen.terminate()


@pytest.mark.slow(level=1)
def test_serve_unbound_tag_get_exchange(pipeline_https_build, parquet_dir):
    batting_url = "https://storage.googleapis.com/letsql-pins/batting/20240711T171118Z-431ef/batting.parquet"
    serve_tag = "read-batting"
    expr = load_expr(pipeline_https_build)

    serve_args = (
        "xorq",
        "serve-unbound",
        str(pipeline_https_build),
        "--to_unbind_tag",
        serve_tag,
    )
    serve_popened = Popened(serve_args, deferred=False)
    port = peek_port(serve_popened)

    flight_backend = xo.flight.connect(port=port)
    f = flight_backend.get_exchange("default")
    actual = xo.deferred_read_parquet(batting_url).pipe(f).execute()

    expected = expr.execute()
    (actual, expected) = (
        df.sort_values(list(df.columns), ignore_index=True) for df in (actual, expected)
    )
    assert actual.equals(expected)

    serve_popened.popen.terminate()


@pytest.mark.slow(level=1)
def test_serve_unbound_tag_get_exchange_udf(fixture_dir, tmp_path):
    import pandas as pd

    df = pd.DataFrame([float(v) for v in range(10)], columns=["x"])

    serve_tag = "full"

    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline_pandas_udf.py"

    import contextlib
    import io

    # Capture print output
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        build_command(script_path, "expr", str(builds_dir))

    serve_args = (
        "xorq",
        "serve-unbound",
        str(output.getvalue().strip()),
        "--to_unbind_tag",
        serve_tag,
    )
    serve_popened = Popened(serve_args, deferred=False)
    port = peek_port(serve_popened)

    flight_backend = xo.flight.connect(port=port)
    f = flight_backend.get_exchange("default")
    actual = xo.connect().register(df).select("x").pipe(f).execute()

    assert not actual.empty

    serve_popened.popen.terminate()


@pytest.mark.slow(level=3)
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
        serve_hash = "b73886b4352e03c64f30681425c795ed"  # RemoteTable

        serve_args = (
            "xorq",
            "serve-unbound",
            str(target_dir / match.group()),
            "--to_unbind_hash",
            serve_hash,
        )
        serve_popened = Popened(serve_args, deferred=False)
        port = peek_port(serve_popened)

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
