from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

import click

from xorq.cli_constants import DEFAULT_CACHE_TYPE, DEFAULT_OUTPUT_FORMAT, OutputFormats


_F = TypeVar("_F", bound=Callable)

_DEFAULT_OUTPUT_PATH_SHOW_DEFAULT = f"`{os.devnull}` (discard)"


def output_options(
    fn: _F | None = None,
    *,
    output_path_help: str | None = None,
    output_path_show_default: str | None = None,
) -> _F | Callable[[_F], _F]:
    if output_path_help is None:
        output_path_help = "Path to write output. Use '-' for stdout."
    if output_path_show_default is None:
        output_path_show_default = _DEFAULT_OUTPUT_PATH_SHOW_DEFAULT

    def decorator(fn):
        fn = click.option(
            "-f",
            "--format",
            "output_format",
            type=click.Choice(tuple(f.value for f in OutputFormats)),
            default=DEFAULT_OUTPUT_FORMAT,
            show_default=True,
            help="Output format.",
        )(fn)
        fn = click.option(
            "-o",
            "--output-path",
            default=None,
            show_default=output_path_show_default,
            help=output_path_help,
        )(fn)
        return fn

    if fn is not None:
        return decorator(fn)
    return decorator


limit_option = click.option(
    "--limit",
    type=int,
    default=None,
    show_default="unlimited",
    help="Maximum number of rows to output.",
)


params_option = click.option(
    "-p",
    "--params",
    "raw_params",
    multiple=True,
    help=(
        "Override an expression parameter as key=value"
        " (repeatable, for example --params threshold=0.5)."
    ),
)


cache_dir_option = click.option(
    "--cache-dir",
    default=None,
    show_default="`$XORQ_CACHE_DIR` or `~/.cache/xorq`",
    help="Directory for parquet cache files.",
)


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
        show_default=True,
        help=(
            "Cache strategy: 'modification-time' (ParquetCache)"
            " or 'snapshot' (ParquetSnapshotCache)."
        ),
    )(fn)
    return fn


def unbind_options(fn):
    fn = click.option(
        "--typ",
        default=None,
        help="Type of the node to unbind.",
    )(fn)
    fn = click.option(
        "--to-unbind-tag",
        default=None,
        show_default="inferred",
        help="Tag of the node to unbind (alternative to --to-unbind-hash).",
    )(fn)
    fn = click.option(
        "--to-unbind-hash",
        default=None,
        show_default="inferred",
        help="Hash of the node to unbind.",
    )(fn)
    return fn


def serve_options(fn):
    fn = click.option(
        "--prometheus-port",
        type=int,
        default=None,
        show_default="off",
        help="Port to expose Prometheus metrics.",
    )(fn)
    fn = click.option(
        "--port",
        type=int,
        default=None,
        show_default="random",
        help="Port to bind the Flight server.",
    )(fn)
    fn = click.option(
        "--host",
        default="localhost",
        show_default=True,
        help="Host to bind the Flight server.",
    )(fn)
    return fn


fuse_option = click.option(
    "--fuse/--no-fuse",
    default=True,
    show_default=True,
    help="Enable catalog source fusion.",
)


rename_params_option = click.option(
    "--rename-params",
    "raw_rename_params",
    multiple=True,
    help="Rename a parameter on a specific entry: entry,old_name,new_name (repeatable).",
)


code_option = click.option(
    "-c",
    "--code",
    default=None,
    help="Inline Ibis code expression applied to `source`.",
)


sync_option = click.option(
    "--sync/--no-sync",
    default=True,
    show_default=True,
    help="Push the catalog to its remotes after the operation.",
)


def env_options(fn):
    fn = click.option(
        "--env-prefix",
        default=None,
        help=(
            "Env-var prefix for the annex remote"
            " (for example XORQ_CATALOG_S3_; mutually exclusive with --env-file)."
        ),
    )(fn)
    fn = click.option(
        "--env-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help=(
            "Env file for the annex remote"
            " (for example .env.catalog.s3; mutually exclusive with --env-prefix)."
        ),
    )(fn)
    return fn


gcs_option = click.option(
    "--gcs",
    is_flag=True,
    help="Apply GCS defaults to S3 config (annex remote or content store).",
)


content_store_option = click.option(
    "--content-store",
    "content_store_type",
    type=click.Choice(["s3", "directory"]),
    default=None,
    help="Create a pointer-backend catalog with the given content store type.",
)


def cs_env_options(fn):
    fn = click.option(
        "--cs-env-prefix",
        default=None,
        help=(
            "Env-var prefix for the content store"
            " (e.g. XORQ_CONTENT_STORE_S3_; mutually exclusive with --cs-env-file)."
        ),
    )(fn)
    fn = click.option(
        "--cs-env-file",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help=(
            "Env file for the content store (mutually exclusive with --cs-env-prefix)."
        ),
    )(fn)
    return fn


def cs_field_options(fn):
    fn = click.option(
        "--cs-directory",
        default=None,
        help="Local directory for a directory content store.",
    )(fn)
    fn = click.option(
        "--cs-protocol",
        type=click.Choice(["http", "https"]),
        default=None,
        help="Protocol for S3-compatible endpoint.",
    )(fn)
    fn = click.option(
        "--cs-port",
        type=int,
        default=None,
        help="Port for S3-compatible endpoint.",
    )(fn)
    fn = click.option(
        "--cs-host",
        default=None,
        help="Host for S3-compatible endpoint (e.g. minio.example.com).",
    )(fn)
    fn = click.option(
        "--cs-prefix",
        default=None,
        help="Key prefix within the S3 bucket.",
    )(fn)
    fn = click.option(
        "--cs-region",
        default=None,
        help="AWS region for the S3 bucket.",
    )(fn)
    fn = click.option(
        "--cs-catalog-id",
        default=None,
        help="Catalog ID for content store key namespacing (auto-generated if omitted).",
    )(fn)
    fn = click.option(
        "--cs-bucket",
        default=None,
        help="S3 bucket name for the content store.",
    )(fn)
    return fn


json_option = click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output as JSON.",
)
