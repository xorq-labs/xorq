import datetime
import os
import pdb
import sys
import traceback
from functools import partial, wraps
from pathlib import Path

import click

from xorq.cli_constants import DEFAULT_CACHE_TYPE, DEFAULT_OUTPUT_FORMAT, OutputFormats
from xorq.cli_options import (
    cache_dir_option,
    cache_strategy_options,
    limit_option,
    output_options,
    params_option,
    serve_options,
    unbind_options,
)
from xorq.init_templates import InitTemplates


def _lazy_span(name):
    """Decorator that wraps a function in an OpenTelemetry span, importing lazily."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

            with tracer.start_as_current_span(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def _get_cache_dir(cache_dir):
    if cache_dir is None:
        from xorq.common.utils.caching_utils import get_xorq_cache_dir  # noqa: PLC0415

        cache_dir = get_xorq_cache_dir()
    return cache_dir


class _ClickDate(click.ParamType):
    name = "date"

    def convert(self, value, param, ctx):
        try:
            return datetime.date.fromisoformat(value)
        except (ValueError, TypeError):
            self.fail(
                f"{value!r} is not a valid date (expected YYYY-MM-DD)", param, ctx
            )


def _click_type_for_dtype(dtype):
    """Return the :class:`click.ParamType` corresponding to an ibis *dtype*."""
    import xorq.expr.datatypes as dt  # noqa: PLC0415

    # Rebuilt per-call because the dt import is deferred for startup speed.
    _CLICK_TYPES = {
        dt.Float64: click.FLOAT,
        dt.Float32: click.FLOAT,
        dt.Int64: click.INT,
        dt.Int32: click.INT,
        dt.Int16: click.INT,
        dt.Int8: click.INT,
        dt.String: click.STRING,
        dt.Boolean: click.BOOL,
        dt.Date: _ClickDate(),
        dt.Timestamp: click.DateTime(),
    }
    click_type = _CLICK_TYPES.get(type(dtype))
    if click_type is None:
        raise click.BadParameter(f"Unsupported parameter dtype: {dtype}")
    return click_type


def _parse_cli_params(expr, raw_params: tuple) -> dict:
    """Parse key=value CLI strings into a {name: typed_value} dict.

    Uses Click type converters to coerce and validate each string value
    against the declared parameter dtype. The returned dict is suitable
    for passing directly to :func:`xorq.expr.api.bind_params`.
    """
    if not raw_params:
        return {}

    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.operations import NamedScalarParameter  # noqa: PLC0415

    named = {node.label: node for node in walk_nodes(NamedScalarParameter, expr)}

    params = {}
    errors = []
    for kv in raw_params:
        key, sep, value = kv.partition("=")
        if not sep:
            errors.append(f"Expected key=value, got {kv!r}")
            continue
        if key not in named:
            errors.append(
                f"Unknown parameter {key!r}. Available: {', '.join(named) or '(none)'}"
            )
            continue
        click_type = _click_type_for_dtype(named[key].dtype)
        try:
            params[key] = click_type.convert(value, param=None, ctx=None)
        except click.exceptions.BadParameter as e:
            errors.append(str(e))

    if errors:
        raise click.BadParameter("\n".join(errors))

    return params


def _apply_cli_params(expr, raw_params: tuple):
    """Parse --params CLI strings and bind them to expr; no-op if empty."""
    param_dict = _parse_cli_params(expr, raw_params)
    if not param_dict:
        return expr
    from xorq.expr.api import bind_params  # noqa: PLC0415

    return bind_params(expr, param_dict)


def ensure_build_dir(expr_path):
    build_dir = Path(expr_path)
    if not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {expr_path}")
        sys.exit(2)
    return build_dir


@_lazy_span("cli.uv_build_command")
def uv_build_command(
    script_path,
    expr_name="expr",
    builds_dir="builds",
    cache_dir=None,
    project_path=None,
    pep723=False,
    extras=(),
    all_extras=True,
    debug=False,
):
    if project_path and pep723:
        raise click.UsageError("--project-path and --pep723 are mutually exclusive")

    from xorq.ibis_yaml.packager import PackagedBuilder  # noqa: PLC0415

    builder = PackagedBuilder.from_script_path(
        script_path,
        project_path=project_path,
        pep723=pep723,
        expr_name=expr_name,
        builds_dir=builds_dir,
        cache_dir=cache_dir,
        extras=extras,
        all_extras=all_extras,
        debug=debug,
    )
    builder.build()
    print(builder.build_path)
    return builder


@_lazy_span("cli.uv_run_command")
def uv_run_command(
    expr_path,
    cache_dir=None,
    output_path=None,
    output_format="parquet",
    limit=None,
    raw_params=(),
):
    from xorq.ibis_yaml.packager import (  # noqa: PLC0415
        PackagedRunner,
        validate_params_early,
    )

    try:
        validate_params_early(expr_path, raw_params)
    except ValueError as e:
        raise click.BadParameter(str(e)) from None
    runner = PackagedRunner(
        expr_path,
        cache_dir=cache_dir,
        output_path=output_path,
        output_format=output_format,
        limit=limit,
        raw_params=raw_params,
    )
    runner.run()
    return runner


@_lazy_span("cli.build_command")
def build_command(
    script_path,
    expr_name,
    builds_dir="builds",
    cache_dir=None,
    debug: bool = False,
):
    """
    Generate artifacts from an expression in a given Python script

    Parameters
    ----------
    script_path : Path to the Python script
    expr_name : The name of the expression to build
    builds_dir : Directory where artifacts will be generated
    cache_dir : Directory where the parquet cache files will be generated

    Returns
    -------

    """
    from opentelemetry import trace  # noqa: PLC0415

    import xorq.common.utils.pickle_utils  # noqa: F401, PLC0415
    from xorq.common.utils.import_utils import import_from_path  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415
    from xorq.vendor.ibis import Expr  # noqa: PLC0415

    cache_dir = _get_cache_dir(cache_dir)

    span = trace.get_current_span()
    span.add_event(
        "build.params",
        {
            "script_path": str(script_path),
            "expr_name": expr_name,
        },
    )

    if not os.path.exists(script_path):
        raise ValueError(f"Error: Script not found at {script_path}")
    print(f"Building {expr_name} from {script_path}", file=sys.stderr)
    vars_module = import_from_path(script_path, module_name="__main__")
    match expr := getattr(vars_module, expr_name, None):
        case Expr():
            pass
        case None:
            raise ValueError(f"Expression {expr_name} not found")
        case _:
            raise ValueError(
                f"The object {expr_name} must be an instance of {Expr.__module__}.{Expr.__name__}"
            )

    build_path = build_expr(
        expr, builds_dir=builds_dir, cache_dir=Path(cache_dir), debug=debug
    )
    expr_hash = build_path.name
    span.add_event("build.outputs", {"expr_hash": expr_hash})
    print(
        f"Written '{expr_name}' to {build_path}",
        file=sys.stderr,
    )
    print(build_path)


@_lazy_span("cli.run_command")
def run_command(
    expr_path,
    output_path=None,
    output_format=DEFAULT_OUTPUT_FORMAT,
    cache_dir=None,
    limit=None,
    raw_params=(),
):
    """
    Execute an artifact

    Parameters
    ----------
    expr_path : str
        Path to the expr in the builds dir
    output_path : str
        Path to write output. Defaults to os.devnull
    output_format : OutputFormats | str, optional
        Output format, either "csv", "json", "arrow", or "parquet". Defaults to "parquet"
    cache_dir : Path, optional
        Directory where the parquet cache files will be generated
    limit : int, optional
        Limit number of rows to output. Defaults to None (no limit).

    Returns
    -------

    """
    from opentelemetry import trace  # noqa: PLC0415
    from opentelemetry.trace import StatusCode  # noqa: PLC0415

    from xorq.common.exceptions import UnboundExpressionError  # noqa: PLC0415
    from xorq.common.utils.logging_utils import RunLogger  # noqa: PLC0415
    from xorq.common.utils.profile_utils import timed  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415
    from xorq.ibis_yaml.packager import validate_params_early  # noqa: PLC0415

    try:
        validate_params_early(expr_path, raw_params)
    except ValueError as e:
        raise click.BadParameter(str(e)) from None

    cache_dir = _get_cache_dir(cache_dir)

    span = trace.get_current_span()

    expr_hash = Path(expr_path).name
    run_params = (
        ("expr_hash", expr_hash),
        ("expr_path", str(expr_path)),
        ("output_path", str(output_path)),
        ("output_format", str(output_format)),
        ("limit", limit),
    )

    span.add_event("run.params", {k: v for k, v in run_params if v is not None})

    try:
        with RunLogger.from_expr_hash(expr_hash, params_tuple=run_params) as rl:
            rl.log_event("run.start", dict(run_params))

            with timed() as get_elapsed:
                try:
                    expr = load_expr(
                        expr_path, cache_dir=cache_dir, raise_on_unbound=True
                    )
                except UnboundExpressionError as err:
                    raise UnboundExpressionError(
                        "Cannot run unbound expression"
                        " - compose it with a source first using xorq catalog compose-add"
                    ) from err

                param_dict = _parse_cli_params(expr, raw_params)
                load_metrics = {"elapsed_s": round(get_elapsed(), 3)}
                span.add_event("run.expr_loaded", load_metrics)
                rl.log_event("run.expr_loaded", load_metrics)

            if param_dict:
                from xorq.expr.api import bind_params  # noqa: PLC0415

                expr = bind_params(expr, param_dict)

            if limit is not None:
                expr = expr.limit(limit)

            with timed() as get_elapsed:
                arbitrate_output_format(expr, output_path, output_format)
                execute_metrics = {
                    "elapsed_s": round(get_elapsed(), 3),
                    "output_format": str(output_format),
                }
                span.add_event("run.done", execute_metrics)
                rl.log_event("run.done", execute_metrics)

            file_metrics = RunLogger._compute_file_metrics(output_format, output_path)
            if file_metrics:
                span.add_event("run.output_written", file_metrics)
                rl.log_event("run.output_written", file_metrics)

            span.set_status(StatusCode.OK)
            rl.finalize(status="ok", span_context=span.get_span_context())

    except Exception as e:
        span.set_status(StatusCode.ERROR, str(e))
        span.record_exception(e)
        rl.finalize(status="error", span_context=span.get_span_context())
        raise


@_lazy_span("cli.run_cached_command")
def run_cached_command(
    expr_path,
    output_path=None,
    output_format=DEFAULT_OUTPUT_FORMAT,
    cache_dir=None,
    limit=None,
    cache_type=DEFAULT_CACHE_TYPE,
    ttl=None,
    raw_params=(),
):
    """
    Execute an artifact with a ParquetCache wrapping the top-level expression.

    Parameters
    ----------
    expr_path : str
        Path to the expr in the builds dir
    output_path : str
        Path to write output. Defaults to os.devnull
    output_format : OutputFormats | str, optional
        Output format, either "csv", "json", "arrow", or "parquet". Defaults to "parquet"
    cache_dir : Path, optional
        Directory where the parquet cache files will be generated
    limit : int, optional
        Limit number of rows to output. Defaults to None (no limit).
    cache_type : str
        Cache type: "modification-time" for ParquetCache (default), "snapshot"
        for ParquetSnapshotCache (or ParquetTTLSnapshotCache when --ttl is set).
    ttl : int, optional
        TTL in seconds for snapshot cache type. When set, uses
        ParquetTTLSnapshotCache instead of ParquetSnapshotCache.
    """
    from opentelemetry import trace  # noqa: PLC0415

    from xorq.caching import (  # noqa: PLC0415
        ParquetCache,
        ParquetSnapshotCache,
        ParquetTTLSnapshotCache,
    )
    from xorq.common.utils.logging_utils import RunLogger  # noqa: PLC0415
    from xorq.common.utils.profile_utils import timed  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    cache_dir = _get_cache_dir(cache_dir)

    span = trace.get_current_span()

    expr_hash = Path(expr_path).name
    run_params = (
        ("expr_hash", expr_hash),
        ("expr_path", str(expr_path)),
        ("output_path", str(output_path)),
        ("output_format", str(output_format)),
        ("cache_type", cache_type),
        ("ttl", ttl),
        ("limit", limit),
    )

    with RunLogger.from_expr_hash(expr_hash, params_tuple=run_params, span=span) as rl:
        rl.log_span_event(span, "run_cached.start", dict(run_params))

        with timed() as get_elapsed:
            expr = load_expr(expr_path, cache_dir=cache_dir)
            expr = _apply_cli_params(expr, raw_params)

        rl.log_span_event(
            span, "run_cached.expr_loaded", {"elapsed_s": round(get_elapsed(), 3)}
        )

        match (cache_type, ttl):
            case ("modification-time", None):
                cache = ParquetCache.from_kwargs(base_path=cache_dir)
            case (_, int(seconds)):
                ttl_delta = datetime.timedelta(seconds=seconds)
                cache = ParquetTTLSnapshotCache.from_kwargs(
                    base_path=cache_dir, ttl=ttl_delta
                )
            case ("snapshot", None):
                cache = ParquetSnapshotCache.from_kwargs(base_path=cache_dir)
            case _:
                raise click.BadParameter(
                    f"Unknown cache type: {cache_type!r}. "
                    "Must be 'modification-time' or 'snapshot'."
                )

        expr = expr.cache(cache=cache)

        if limit is not None:
            expr = expr.limit(limit)

        with timed() as get_elapsed:
            arbitrate_output_format(expr, output_path, output_format)

        rl.log_span_event(
            span,
            "run_cached.done",
            {
                "elapsed_s": round(get_elapsed(), 3),
                "output_format": str(output_format),
            },
        )

        file_metrics = RunLogger._compute_file_metrics(output_format, output_path)
        if file_metrics:
            rl.log_span_event(span, "run_cached.output_written", file_metrics)


def arbitrate_output_format(expr, output_path, output_format, batch_size=None):
    match (output_path, output_format):
        case (None, _):
            output_path = os.devnull
        case ("-", OutputFormats.json):
            # FIXME: deal with windows
            output_path = sys.stdout
        case ("-", _):
            output_path = sys.stdout.buffer
        case _:
            pass
    match output_format:
        case OutputFormats.csv:
            expr.to_csv(output_path)
        case OutputFormats.json:
            expr.to_json(output_path)
        case OutputFormats.arrow:
            from xorq.expr.api import to_pyarrow_stream  # noqa: PLC0415

            batch_kwargs = {"chunk_size": batch_size} if batch_size is not None else {}
            to_pyarrow_stream(expr, output_path, **batch_kwargs)
        case OutputFormats.parquet:
            expr.to_parquet(output_path)
        case _:
            raise ValueError(f"Unknown output_format: {output_format}")


@_lazy_span("cli.run_unbound_command")
def run_unbound_command(
    expr_path,
    *,
    to_unbind_hash=None,
    to_unbind_tag=None,
    output_path=None,
    output_format=DEFAULT_OUTPUT_FORMAT,
    cache_dir=None,
    limit=None,
    typ=None,
    instream=sys.stdin.buffer,
    batch_size=None,
):
    """
    Execute an unbound expression by reading Arrow IPC from stdin and binding it

    Parameters
    ----------
    expr_path : str
        Path to the expr in the builds dir
    to_unbind_hash : str, optional
        Hash of the node to unbind
    to_unbind_tag : str, optional
        Tag of the node to unbind
    output_path : str, optional
        Path to write output. Defaults to stdout for arrow, os.devnull otherwise
    output_format : str, optional
        Output format, either "csv", "json", "parquet", or "arrow". Defaults to "parquet"
    cache_dir : Path, optional
        Directory where the parquet cache files will be generated
    limit : int, optional
        Limit number of rows to output. Defaults to None (no limit).
    typ : str, optional
        Type of the node to unbind

    Returns
    -------

    """
    from opentelemetry import trace  # noqa: PLC0415

    from xorq.common.utils.io_utils import maybe_open  # noqa: PLC0415
    from xorq.common.utils.node_utils import expr_to_unbound  # noqa: PLC0415
    from xorq.expr.api import read_pyarrow_stream  # noqa: PLC0415
    from xorq.flight.exchanger import replace_one_unbound  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    cache_dir = _get_cache_dir(cache_dir)

    span = trace.get_current_span()
    span.add_event(
        "run_unbound.params",
        {
            "expr_path": str(expr_path),
            "to_unbind_hash": str(to_unbind_hash),
            "to_unbind_tag": str(to_unbind_tag),
            "output_format": str(output_format),
        },
    )

    # Resolve build identifier
    expr_path = ensure_build_dir(expr_path)
    # Log to stderr to avoid polluting Arrow streams
    print(f"[run-unbound] Loading expression from {expr_path}", file=sys.stderr)

    # Load the expression and make it unbound
    expr = load_expr(expr_path, cache_dir=cache_dir)
    unbound_expr = expr_to_unbound(
        expr, hash=to_unbind_hash, tag=to_unbind_tag, typs=typ
    ).to_expr()

    # Read Arrow IPC from instream
    print("[run-unbound] Reading Arrow IPC from instream...", file=sys.stderr)
    # Create a connection and register the input table
    with maybe_open(instream, "rb") as stream:
        input_expr = read_pyarrow_stream(stream)
        # Replace the unbound node with the input table
        bound_expr = replace_one_unbound(unbound_expr, input_expr)

        if limit is not None:
            bound_expr = bound_expr.limit(limit)
        arbitrate_output_format(
            bound_expr, output_path, output_format, batch_size=batch_size
        )


@_lazy_span("cli.unbind_and_serve_command")
def unbind_and_serve_command(
    expr_path,
    to_unbind_hash=None,
    to_unbind_tag=None,
    host=None,
    port=None,
    prometheus_port=None,
    cache_dir=None,
    typ=None,
):
    import xorq.expr.relations  # noqa: PLC0415
    from xorq.caching.strategy import SnapshotStrategy  # noqa: PLC0415
    from xorq.common.utils.logging_utils import get_print_logger  # noqa: PLC0415
    from xorq.common.utils.node_utils import expr_to_unbound  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    logger = get_print_logger()
    cache_dir = _get_cache_dir(cache_dir)

    # Resolve build path to an actual build directory
    expr_path = ensure_build_dir(expr_path)
    logger.info(f"Loading expression from {expr_path}")
    try:
        # initialize console and optional Prometheus metrics
        from xorq.flight.metrics import setup_console_metrics  # noqa: PLC0415

        setup_console_metrics(prometheus_port=prometheus_port)
    except ImportError:
        logger.warning(
            "Metrics support requires 'opentelemetry-sdk' and console exporter"
        )

    expr = load_expr(expr_path, cache_dir=cache_dir)
    unbound_expr = expr_to_unbound(
        expr,
        hash=to_unbind_hash,
        tag=to_unbind_tag,
        typs=typ,
        strategy=SnapshotStrategy(),
    )
    flight_url = xorq.flight.FlightUrl(host=host, port=port)
    make_server = partial(
        xorq.flight.FlightServer,
        flight_url=flight_url,
    )
    server, _ = xorq.expr.relations.flight_serve_unbound(
        unbound_expr, make_server=make_server
    )
    logger.info(f"Serving expression from '{expr_path}' on {flight_url.to_location()}")
    server.wait()


@_lazy_span("cli.serve_command")
def serve_command(
    expr_path,
    host=None,
    port=None,
    duckdb_path=None,
    prometheus_port=None,
    cache_dir=None,
):
    """
    Serve a built expression via Flight Server

    Parameters
    ----------
    expr_path : str
        Path to the expression directory (output of xorq build)
    host : str
        Host to bind Flight Server
    port : int or None
        Port to bind Flight Server (None for random)
    duckdb_path : str or None
        Path to duckdb cache DB file
    prometheus_port : int or None
        Port to connect to the prometheus server
    cache_dir : str or None
        Path to the dir to store the parquet cache files
    """
    from opentelemetry import trace  # noqa: PLC0415

    from xorq.common.utils.logging_utils import get_print_logger  # noqa: PLC0415
    from xorq.flight import FlightServer  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415
    from xorq.loader import load_backend  # noqa: PLC0415

    logger = get_print_logger()
    cache_dir = _get_cache_dir(cache_dir)

    # Resolve build path to an actual build directory
    expr_path = ensure_build_dir(expr_path)
    span = trace.get_current_span()
    params = {
        "build_path": expr_path,
        "host": host,
        "port": port,
    }
    if duckdb_path is not None:
        params["duckdb_path"] = duckdb_path
    span.add_event("serve.params", params)

    expr_path = Path(expr_path)
    logger.info(f"Loading expression from {expr_path}")
    expr = load_expr(expr_path, cache_dir=cache_dir)

    db_path = Path(duckdb_path or "xorq_serve.db")  # FIXME what should be the default?

    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using duckdb at {db_path}")

    try:
        from xorq.flight.metrics import setup_console_metrics  # noqa: PLC0415

        setup_console_metrics(prometheus_port=prometheus_port)
    except ImportError:
        logger.warning(
            "Metrics support requires 'opentelemetry-sdk' and console exporter"
        )

    server = FlightServer.from_udxf(
        expr,
        make_connection=partial(load_backend("duckdb").connect, str(db_path)),
        port=port,
        host=host,
    )
    location = server.flight_url.to_location()
    server.serve(block=False)
    logger.info(f"Serving expression '{expr_path.stem}' on {location}")
    server.wait()


@_lazy_span("cli.init_command")
def init_command(
    path="./xorq-template",
    template=InitTemplates.default,
    branch=None,
    xorq_spec=None,
    no_lock=False,
):
    from xorq.common.utils.download_utils import (  # noqa: PLC0415
        download_unpacked_xorq_template,
    )
    from xorq.init_templates import (  # noqa: PLC0415
        _LATEST_RE,
        has_latest_placeholder,
        resolve_xorq_spec,
        rewrite_template_xorq_dep,
        run_uv_lock,
    )

    path = download_unpacked_xorq_template(path, template, branch=branch)

    if not has_latest_placeholder(path):
        click.echo(
            f"warning: template `{template}` does not contain `xorq @ LATEST`; "
            "skipping substitution and `uv lock`. Update the template to the "
            "new placeholder format.",
            err=True,
        )
        click.echo(f"initialized xorq template `{template}` to {path}")
        return path

    extras = _extract_latest_extras(path, _LATEST_RE)
    spec = resolve_xorq_spec(override=xorq_spec, extras=extras)
    rewrite_template_xorq_dep(path, spec)
    if not no_lock:
        run_uv_lock(path)
    click.echo(
        f"initialized xorq template `{template}` to {path} (xorq pinned to `{spec}`)"
    )
    return path


def _extract_latest_extras(template_dir: Path, latest_re) -> str:
    import tomlkit  # noqa: PLC0415

    pyproject = Path(template_dir).joinpath("pyproject.toml")
    data = tomlkit.loads(pyproject.read_text())
    for entry in data.get("project", {}).get("dependencies", []):
        m = latest_re.match(str(entry).strip())
        if m:
            return m.group("extras") or ""
    return ""


class PdbGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        if cmd_name == "catalog" and "catalog" not in self.commands:
            _load_catalog_cli()
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx):
        if "catalog" not in self.commands:
            _load_catalog_cli()
        return super().list_commands(ctx)

    def invoke(self, ctx):
        try:
            if ctx.params.get("pdb_runcall"):
                return pdb.runcall(super().invoke, ctx)
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, SystemExit):
            raise
        except Exception as e:
            if ctx.params.get("use_pdb"):
                traceback.print_exception(e)
                pdb.post_mortem(e.__traceback__)
            else:
                traceback.print_exc()
            sys.exit(1)


@click.group(cls=PdbGroup)
@click.version_option(package_name="xorq")
@click.option("--pdb", "use_pdb", is_flag=True, help="Drop into pdb on failure.")
@click.option(
    "--pdb-runcall", "pdb_runcall", is_flag=True, help="Invoke with pdb.runcall."
)
def cli(use_pdb, pdb_runcall):
    pass


@cli.group("uv", invoke_without_command=True)
@click.pass_context
def uv_group(ctx):
    """Commands that use uv to manage a custom Python environment."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@uv_group.command("build")
@click.argument("script_path")
@click.option(
    "-e",
    "--expr-name",
    default="expr",
    help="Name of the expression variable in the Python script.",
)
@click.option(
    "--builds-dir", default="builds", help="Directory for all generated artifacts."
)
@cache_dir_option
@click.option(
    "--project-path",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Explicit project root (default: search upward from script for pyproject.toml).",
)
@click.option(
    "--pep723",
    is_flag=True,
    default=False,
    help="Use PEP 723 inline metadata from the script instead of a project's pyproject.toml.",
)
@click.option(
    "--extra",
    "extras",
    multiple=True,
    help="Optional dependency group to include in requirements (repeatable).",
)
@click.option(
    "--all-extras/--no-all-extras",
    default=True,
    help="Include all optional dependency groups (default: enabled).",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Output SQL files and other debug artifacts.",
)
def uv_build(
    script_path,
    expr_name,
    builds_dir,
    cache_dir,
    project_path,
    pep723,
    extras,
    all_extras,
    debug,
):
    """Build an expression with a custom Python environment."""
    uv_build_command(
        script_path,
        expr_name,
        builds_dir,
        cache_dir,
        project_path=project_path,
        pep723=pep723,
        extras=extras,
        all_extras=all_extras,
        debug=debug,
    )


@uv_group.command("run")
@click.argument("build_path")
@cache_dir_option
@output_options
@limit_option
@params_option
def uv_run(build_path, cache_dir, output_path, output_format, limit, raw_params):
    """Run an expression with a custom Python environment."""
    uv_run_command(
        build_path,
        cache_dir,
        output_path,
        output_format,
        limit=limit,
        raw_params=raw_params,
    )


@uv_group.command("run-cached")
@click.argument("build_path")
@cache_dir_option
@output_options
@limit_option
@cache_strategy_options
@params_option
def uv_run_cached(
    build_path,
    cache_dir,
    output_path,
    output_format,
    limit,
    cache_type,
    ttl,
    raw_params,
):
    """Run a cached expression with a custom Python environment."""
    from xorq.ibis_yaml.packager import (  # noqa: PLC0415
        PackagedCachedRunner,
        validate_params_early,
    )

    try:
        validate_params_early(build_path, raw_params)
    except ValueError as e:
        raise click.BadParameter(str(e)) from None
    runner = PackagedCachedRunner(
        build_path,
        cache_dir=cache_dir,
        output_path=output_path,
        output_format=output_format,
        cache_type=cache_type,
        ttl=ttl,
        limit=limit,
        raw_params=raw_params,
    )
    runner.run()


_UNBOUND_OUTPUT_PATH_HELP = (
    f"Path to write output (default: stdout for arrow, {os.devnull} otherwise)."
)


@uv_group.command("run-unbound")
@click.argument("build_path")
@unbind_options
@output_options(output_path_help=_UNBOUND_OUTPUT_PATH_HELP)
@limit_option
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for Arrow streaming output (default: use table default).",
)
@cache_dir_option
@click.option(
    "-i",
    "--instream",
    type=click.Path(exists=True),
    default=None,
    help="Path to file with Arrow IPC data (default: read from stdin).",
)
def uv_run_unbound(
    build_path,
    to_unbind_hash,
    to_unbind_tag,
    typ,
    output_path,
    output_format,
    limit,
    batch_size,
    cache_dir,
    instream,
):
    """Run an unbound expr with a custom Python environment."""
    from xorq.ibis_yaml.packager import PackagedUnboundRunner  # noqa: PLC0415

    runner = PackagedUnboundRunner(
        build_path,
        cache_dir=cache_dir,
        output_path=output_path,
        output_format=output_format,
        to_unbind_hash=to_unbind_hash,
        to_unbind_tag=to_unbind_tag,
        typ=typ,
        limit=limit,
        batch_size=batch_size,
        instream=instream,
    )
    runner.run()


@cli.command("build")
@click.argument("script_path")
@click.option(
    "-e",
    "--expr-name",
    default="expr",
    help="Name of the expression variable in the Python script.",
)
@click.option(
    "--builds-dir", default="builds", help="Directory for all generated artifacts."
)
@cache_dir_option
@click.option(
    "--debug",
    is_flag=True,
    help="Output SQL files and other debug artifacts.",
)
def build(script_path, expr_name, builds_dir, cache_dir, debug):
    """Generate artifacts from an expression."""
    build_command(script_path, expr_name, builds_dir, cache_dir, debug)


@cli.command("run")
@click.argument("build_path")
@cache_dir_option
@output_options
@limit_option
@params_option
def run(build_path, cache_dir, output_path, output_format, limit, raw_params):
    """Run a build from a builds directory."""
    run_command(build_path, output_path, output_format, cache_dir, limit, raw_params)


@cli.command("run-cached")
@click.argument("build_path")
@cache_dir_option
@output_options
@limit_option
@cache_strategy_options
@params_option
def run_cached(
    build_path,
    cache_dir,
    output_path,
    output_format,
    limit,
    cache_type,
    ttl,
    raw_params,
):
    """Run a build with a ParquetCache wrapping the expression."""
    run_cached_command(
        build_path,
        output_path,
        output_format,
        cache_dir,
        limit,
        cache_type,
        ttl,
        raw_params,
    )


@cli.command("run-unbound")
@click.argument("build_path")
@unbind_options
@output_options(output_path_help=_UNBOUND_OUTPUT_PATH_HELP)
@limit_option
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for Arrow streaming output (default: use table default).",
)
@cache_dir_option
@click.option(
    "-i",
    "--instream",
    type=click.File("rb"),
    default="-",
    help="Stream to read record batches from.",
)
def run_unbound(
    build_path,
    to_unbind_hash,
    to_unbind_tag,
    typ,
    output_path,
    output_format,
    limit,
    batch_size,
    cache_dir,
    instream,
):
    """Run an unbound expr by reading Arrow IPC from stdin."""
    run_unbound_command(
        build_path,
        to_unbind_hash=to_unbind_hash,
        to_unbind_tag=to_unbind_tag,
        output_path=output_path,
        output_format=output_format,
        cache_dir=cache_dir,
        limit=limit,
        typ=typ,
        instream=instream,
        batch_size=batch_size,
    )


@cli.command("serve-unbound")
@click.argument("build_path")
@unbind_options
@serve_options
@cache_dir_option
def serve_unbound(
    build_path,
    to_unbind_hash,
    to_unbind_tag,
    host,
    port,
    cache_dir,
    typ,
    prometheus_port,
):
    """Serve an unbound expr via Flight Server."""
    unbind_and_serve_command(
        build_path,
        to_unbind_hash,
        to_unbind_tag,
        host,
        port,
        prometheus_port,
        cache_dir,
        typ,
    )


@cli.command("serve-flight-udxf")
@click.argument("build_path")
@serve_options
@click.option(
    "--duckdb-path",
    default=None,
    help="Path to duckdb DB (default: <build_path>/xorq_serve.db).",
)
@cache_dir_option
def serve_flight_udxf(build_path, host, port, duckdb_path, prometheus_port, cache_dir):
    """Serve a build via Flight Server."""
    serve_command(build_path, host, port, duckdb_path, prometheus_port, cache_dir)


@cli.command("init")
@click.option(
    "-p",
    "--path",
    default="./xorq-template",
    help="Path to initialize the template.",
)
@click.option(
    "-t",
    "--template",
    type=click.Choice([str(t) for t in InitTemplates]),
    default=str(InitTemplates.default),
    help="Template to use.",
)
@click.option(
    "-b",
    "--branch",
    default=None,
    help="Branch to use for the template.",
)
@click.option(
    "--xorq-spec",
    default=None,
    help=(
        "Full PEP 508 spec to pin xorq to (e.g. 'xorq[duckdb] == 0.3.25' or "
        "'xorq @ git+https://github.com/xorq-labs/xorq@main'). Overrides "
        "auto-detection of the running xorq install."
    ),
)
@click.option(
    "--no-lock",
    is_flag=True,
    default=False,
    help="Skip running `uv lock` in the generated template directory.",
)
def init(path, template, branch, xorq_spec, no_lock):
    """Initialize a xorq project."""
    init_command(path, template, branch, xorq_spec=xorq_spec, no_lock=no_lock)


_COMPLETION_INSTALL_PATHS = {
    "bash": Path("~/.local/share/bash-completion/completions/xorq").expanduser(),
    "zsh": Path("~/.zfunc/_xorq").expanduser(),
    "fish": Path("~/.config/fish/completions/xorq.fish").expanduser(),
}


def _get_completion_source(shell):
    from click.shell_completion import get_completion_class  # noqa: PLC0415

    prog_name = "xorq"
    complete_var = "_XORQ_COMPLETE"
    comp_cls = get_completion_class(shell)
    comp = comp_cls(cli, {}, prog_name, complete_var)
    return comp.source()


def _detect_shell():
    shell_bin = Path(os.environ.get("SHELL", "")).name
    if shell_bin not in _COMPLETION_INSTALL_PATHS:
        raise click.UsageError(
            f"Cannot detect shell from $SHELL={os.environ.get('SHELL')!r}. "
            "Pass the shell name explicitly: bash, zsh, or fish."
        )
    return shell_bin


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
def completion(shell):
    """Output shell completion script.

    SHELL defaults to the value of $SHELL if not provided.

    \b
    Add to your shell config:
      bash:  eval "$(xorq completion bash)"
      zsh:   eval "$(xorq completion zsh)"
      fish:  xorq completion fish | source
    """
    if shell is None:
        shell = _detect_shell()
    click.echo(_get_completion_source(shell), nl=False)


@cli.command("install-completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
def install_completion(shell):
    """Install shell completion script to the standard location.

    SHELL defaults to the value of $SHELL if not provided.

    \b
    Install paths:
      bash:  ~/.local/share/bash-completion/completions/xorq
      zsh:   ~/.zfunc/_xorq  (requires ~/.zfunc in fpath)
      fish:  ~/.config/fish/completions/xorq.fish
    """
    if shell is None:
        shell = _detect_shell()

    install_path = _COMPLETION_INSTALL_PATHS[shell]
    install_path.parent.mkdir(parents=True, exist_ok=True)
    install_path.write_text(_get_completion_source(shell))
    click.echo(f"Installed {shell} completion to {install_path}")
    click.echo(f"Restart your shell or run: source {install_path}")


def _load_catalog_cli():
    from xorq.catalog.cli import cli as _catalog_cli  # noqa: PLC0415

    cli.add_command(_catalog_cli, "catalog")


def main():
    cli()


if __name__ == "__main__":
    main()
