import tempfile
from contextlib import contextmanager

from xorq.catalog.tar_utils import (
    extract_build_tgz_context,
    make_tgz_context,
)


@contextmanager
def build_expr_context(expr):
    from xorq.ibis_yaml.compiler import build_expr

    with tempfile.TemporaryDirectory() as td:
        build_dir = build_expr(expr, builds_dir=td)
        yield build_dir


@contextmanager
def build_expr_context_tgz(expr):
    with build_expr_context(expr) as build_dir:
        with make_tgz_context(build_dir) as tgz_path:
            yield tgz_path


def load_expr_from_tgz(tgz_path):
    from xorq.ibis_yaml.compiler import load_expr

    with extract_build_tgz_context(tgz_path) as build_dir:
        expr = load_expr(build_dir)
        return expr
