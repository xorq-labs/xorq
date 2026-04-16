import tempfile
from contextlib import contextmanager

from xorq.catalog.zip_utils import (
    extract_build_zip_context,
    make_zip_context,
)


@contextmanager
def build_expr_context(expr):
    from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        build_dir = build_expr(expr, builds_dir=td)
        yield build_dir


@contextmanager
def build_expr_context_zip(expr):
    with build_expr_context(expr) as build_dir:
        with make_zip_context(build_dir) as zip_path:
            yield zip_path


def load_expr_from_zip(zip_path, lazy=False, read_only_parquet_metadata=False):
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    with extract_build_zip_context(zip_path) as build_dir:
        expr = load_expr(
            build_dir,
            lazy=lazy,
            read_only_parquet_metadata=read_only_parquet_metadata,
            resolve_all_reads=True,
        )
        return expr
