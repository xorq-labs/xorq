"""Compose a cataloged unbound transform with a deferred-read CSV source.

Companion runnable for
``docs/tutorials/analytics_tutorials/compose_unbound_with_source.qmd``.

The tutorial frames this as cloning an ``internal_tool_catalog`` (which holds
the unbound ``exp-smooth`` transform) and binding it against a CSV on disk
using ``xorq catalog compose``. This script reproduces the underlying
mechanics in pure Python: it builds the unbound exponential-smoothing
transform, points a ``deferred_read_csv`` at a temp CSV, and replaces the
``UnboundTable`` placeholder with that source — which is what
``xorq catalog compose`` does internally before persisting the result.
"""

import csv
import tempfile
from pathlib import Path

import pyarrow as pa

import xorq.api as xo
from xorq.common.utils.graph_utils import replace_unbound
from xorq.expr.udf import pyarrow_udwf
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor import ibis
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.types.core import ExprMetadata


ALPHA = 0.9


@pyarrow_udwf(
    schema=ibis.schema({"a": float}),
    return_type=ibis.dtype(float),
    alpha=ALPHA,
)
def exp_smooth(self, values: list[pa.Array], num_rows: int) -> pa.Array:
    results = []
    curr_value = 0.0
    arr = values[0]
    for idx in range(num_rows):
        if idx == 0:
            curr_value = arr[idx].as_py()
        else:
            curr_value = (
                arr[idx].as_py() * self.alpha + curr_value * (1.0 - self.alpha)
            )
        results.append(curr_value)
    return pa.array(results)


def make_unbound_transform():
    """Build the unbound exp-smooth transform stored in the catalog."""
    unbound = ops.UnboundTable(
        name="placeholder", schema=ibis.schema({"a": float})
    ).to_expr()
    return unbound.mutate(exp_smooth=exp_smooth.on_expr(unbound))


def make_local_csv(directory: Path) -> Path:
    """Create the on-disk CSV the tutorial reads from."""
    path = directory / "readings.csv"
    rows = [1.0, 2.1, 2.9, 4.0, 5.1, 6.0, 6.9, 8.0]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["a"])
        for value in rows:
            writer.writerow([value])
    return path


def make_source_expr(csv_path: Path):
    """The on-the-fly source expression from the tutorial."""
    return xo.deferred_read_csv(str(csv_path))


def compose(source_expr, transform_expr):
    """Replace the UnboundTable in the transform with the source.

    This is what `xorq catalog compose` performs before persisting the
    result back to the catalog.
    """
    return replace_unbound(transform_expr, source_expr)


def expected_smoothed(values, alpha=ALPHA):
    smoothed = []
    curr = 0.0
    for idx, value in enumerate(values):
        curr = value if idx == 0 else value * alpha + curr * (1.0 - alpha)
        smoothed.append(curr)
    return smoothed


def main():
    transform_expr = make_unbound_transform()

    # The unbound transform is what the catalog stores: it has the shape of
    # a transform but no source attached.
    transform_meta = ExprMetadata.from_expr(transform_expr)
    assert transform_meta.kind == ExprKind.UnboundExpr
    assert dict(transform_meta.schema_in) == {"a": ibis.dtype("float64")}

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = make_local_csv(Path(tmpdir))
        source_expr = make_source_expr(csv_path)

        composed = compose(source_expr, transform_expr)
        result = xo.execute(composed).sort_values("a").reset_index(drop=True)

    expected_a = [1.0, 2.1, 2.9, 4.0, 5.1, 6.0, 6.9, 8.0]
    assert result["a"].tolist() == expected_a
    assert result["exp_smooth"].round(6).tolist() == [
        round(v, 6) for v in expected_smoothed(expected_a)
    ]


if __name__ == "__main__":
    main()
elif __name__ == "__pytest_main__":
    main()
    pytest_examples_passed = True
