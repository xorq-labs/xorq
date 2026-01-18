import pandas as pd
import pytest

import xorq.vendor.ibis as ibis
from xorq.common.utils.graph_utils import bfs, walk_nodes
from xorq.common.utils.node_utils import gen_downstream
from xorq.vendor.ibis.expr.operations import InMemoryTable


def create_expr(depth=3):
    """Create expression with shared subgraphs for benchmarking."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    )
    t = ibis.memtable(df)

    expr = t.mutate(x=t.a + t.b, y=t.b + t.c)

    for i in range(depth):
        expr = expr.mutate(
            **{
                f"d{i}": expr.x * 2 + expr.y,
                f"e{i}": expr.x + expr.y * 3,
                f"f{i}": expr.x * expr.y,
            }
        )

    return expr[[col for col in expr.columns if col.startswith(("d", "e", "f"))]][:5]


@pytest.mark.benchmark
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_gen_downstream_performance(depth):
    expr = create_expr(depth)

    source = next(iter(walk_nodes(InMemoryTable, expr)))

    downstream = list(gen_downstream(expr, source))

    unique_nodes = len(bfs(expr).invert())
    visit_ratio = len(downstream) / unique_nodes

    assert visit_ratio <= 2.0, f"Expected linear visits, got {visit_ratio:.1f}x"
