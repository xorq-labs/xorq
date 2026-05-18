import os

import click

from xorq.cli_constants import DEFAULT_CACHE_TYPE, DEFAULT_OUTPUT_FORMAT, OutputFormats


_DEFAULT_OUTPUT_PATH_HELP = f"Path to write output (default: {os.devnull})."


def output_options(fn=None, *, output_path_help=None):
    if output_path_help is None:
        output_path_help = _DEFAULT_OUTPUT_PATH_HELP

    def decorator(fn):
        fn = click.option(
            "-f",
            "--format",
            "output_format",
            type=click.Choice(tuple(f.value for f in OutputFormats)),
            default=DEFAULT_OUTPUT_FORMAT,
            help=f"Output format (default: {DEFAULT_OUTPUT_FORMAT}).",
        )(fn)
        fn = click.option(
            "-o",
            "--output-path",
            default=None,
            help=output_path_help,
        )(fn)
        return fn

    if fn is not None:
        return decorator(fn)
    return decorator


def limit_option(fn):
    fn = click.option(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to output.",
    )(fn)
    return fn


def params_options(fn):
    fn = click.option(
        "-p",
        "--params",
        "raw_params",
        multiple=True,
        help="Parameter as key=value (repeatable, e.g. --params threshold=0.5).",
    )(fn)
    return fn


def cache_dir_option(fn):
    fn = click.option(
        "--cache-dir",
        default=None,
        help="Directory for all generated parquet files cache.",
    )(fn)
    return fn


def cache_strategy_options(fn):
    fn = click.option(
        "--ttl",
        type=int,
        default=None,
        help="TTL in seconds for snapshot cache (uses ParquetTTLSnapshotCache when set).",
    )(fn)
    fn = click.option(
        "--cache-type",
        type=click.Choice(["modification-time", "snapshot"]),
        default=DEFAULT_CACHE_TYPE,
        help=f"Cache strategy: 'modification-time' (ParquetCache) or 'snapshot' (ParquetSnapshotCache). Default: {DEFAULT_CACHE_TYPE}.",
    )(fn)
    return fn


def unbind_options(fn):
    fn = click.option(
        "--typ",
        default=None,
        help="Type of the node to unbind.",
    )(fn)
    fn = click.option(
        "--to_unbind_tag",
        default=None,
        help="Tag of the node to unbind.",
    )(fn)
    fn = click.option(
        "--to_unbind_hash",
        default=None,
        help="Hash of the node to unbind.",
    )(fn)
    return fn


def serve_options(fn):
    fn = click.option(
        "--prometheus-port",
        type=int,
        default=None,
        help="Port to expose Prometheus metrics (default: disabled).",
    )(fn)
    fn = click.option(
        "--port",
        type=int,
        default=None,
        help="Port to bind Flight Server (default: random).",
    )(fn)
    fn = click.option(
        "--host",
        default="localhost",
        help="Host to bind Flight Server (default: localhost).",
    )(fn)
    return fn


def fuse_option(fn):
    fn = click.option(
        "--fuse/--no-fuse",
        default=True,
        help="Enable/disable catalog source fusion (default: enabled).",
    )(fn)
    return fn


def rename_params_option(fn):
    fn = click.option(
        "--rename-params",
        "raw_rename_params",
        multiple=True,
        help="Rename a parameter: entry,old_name,new_name (repeatable).",
    )(fn)
    return fn


def code_option(fn):
    fn = click.option(
        "-c",
        "--code",
        default=None,
        help="Inline Ibis code expression applied to `source`.",
    )(fn)
    return fn


def sync_option(fn):
    fn = click.option(
        "--sync/--no-sync",
        default=True,
        help="Enable/disable git-annex sync after operation (default: enabled).",
    )(fn)
    return fn


def env_options(fn):
    fn = click.option(
        "--env-prefix",
        default=None,
        help="Env var prefix for annex remote (e.g. XORQ_CATALOG_S3_).",
    )(fn)
    fn = click.option(
        "--env-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="Env file for annex remote (e.g. .env.catalog.s3).",
    )(fn)
    return fn


def gcs_option(fn):
    fn = click.option(
        "--gcs",
        is_flag=True,
        help="Apply GCS defaults to S3 remote config.",
    )(fn)
    return fn


def json_option(fn):
    fn = click.option(
        "--json",
        "as_json",
        is_flag=True,
        default=False,
        help="Output as JSON.",
    )(fn)
    return fn
