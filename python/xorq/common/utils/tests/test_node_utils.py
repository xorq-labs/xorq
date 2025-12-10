import dask
import pandas as pd
import pytest
import toolz

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import (
    SourceCache,
)
from xorq.common.utils.graph_utils import (
    walk_nodes,
)
from xorq.common.utils.node_utils import (
    expr_to_unbound,
    find_by_expr_hash,
    replace_by_expr_hash,
)


try_find_by_expr_hash = toolz.excepts(Exception, find_by_expr_hash)


def make_exprs():
    on = ("playerID", "yearID", "lgID")
    batting = xo.examples.batting.fetch()
    batting_tagged = batting.tag("batting")
    batting_cached = batting_tagged.cache(storage=SourceCache.from_kwargs())
    awards_players = xo.examples.awards_players.fetch()
    expr = batting_cached[list(on) + ["G"]].join(
        awards_players[list(on) + ["awardID"]],
        predicates=on,
    )
    expr_tagged = expr.tag("join_expr")
    expr_cached = expr_tagged.cache(storage=SourceCache.from_kwargs())
    return {
        "batting": batting,
        "batting_cached": batting_cached,
        "awards_players": awards_players,
        "expr": expr,
        "expr_cached": expr_cached,
    }


@pytest.mark.parametrize(
    "to_find_name",
    (
        "batting",
        "batting_cached",
        "awards_players",
        "expr",
        "expr_cached",
    ),
)
def test_find_by_expr_hash(to_find_name):
    dct = make_exprs()
    (expr_cached, to_find) = (dct[k] for k in ("expr_cached", to_find_name))
    to_find_hash = dask.base.tokenize(to_find)
    typs = (type(to_find.op()),)
    result = find_by_expr_hash(expr_cached, to_find_hash, typs=typs)
    assert result


@pytest.mark.parametrize(
    "to_replace_name",
    (
        "batting",
        "batting_cached",
        "awards_players",
        # "expr",
        "expr_cached",
    ),
)
def test_replace_by_expr_hash(to_replace_name):
    dct = make_exprs()
    (expr_cached, to_replace) = (dct[k] for k in ("expr_cached", to_replace_name))
    to_replace_hash = dask.base.tokenize(to_replace)
    typs = (type(to_replace.op()),)
    schema = to_replace.schema()
    replace_with = xo.memtable(
        pd.DataFrame(columns=tuple(schema)).astype(dict(schema.to_pandas())),
        schema=schema,
        name=to_replace_name,
    ).op()
    replace_with_typs = (type(replace_with),)

    found = walk_nodes(replace_with_typs, expr_cached)
    assert not found
    found = walk_nodes(typs, expr_cached)
    assert found

    to_replace_hash = dask.base.tokenize(to_replace)
    replaced = replace_by_expr_hash(
        expr_cached, to_replace_hash, replace_with, typs=typs
    )
    found = walk_nodes(replace_with_typs, replaced)
    assert found
    found = try_find_by_expr_hash(replaced, to_replace_hash, typs=typs)
    assert not found


@pytest.mark.parametrize(
    "to_replace_name",
    (
        "batting",
        "batting_cached",
        "awards_players",
        # "expr",
        "expr_cached",
    ),
)
def test_unbind_expr_hash(to_replace_name):
    dct = make_exprs()
    (expr_cached, to_replace) = (dct[k] for k in ("expr_cached", to_replace_name))

    typs = (type(to_replace.op()),)
    replace_with = ops.UnboundTable(name="unbound", schema=to_replace.schema())
    replace_with_typs = (type(replace_with),)

    found = walk_nodes(replace_with_typs, expr_cached)
    assert not found
    found = walk_nodes(typs, expr_cached)
    assert found

    to_replace_hash = dask.base.tokenize(to_replace)
    replaced = replace_by_expr_hash(
        expr_cached, to_replace_hash, replace_with, typs=typs
    )

    found = walk_nodes(replace_with_typs, replaced)
    assert found
    found = try_find_by_expr_hash(replaced, to_replace_hash, typs=typs)
    assert not found


@pytest.mark.parametrize(
    "expr_to_hash,tag_value",
    (
        ("batting", None),
        ("batting_cached", None),
        ("awards_players", None),
        ("expr_cached", None),
        (None, "batting"),
        (None, "join_expr"),
    ),
)
def test_expr_to_unbound(expr_to_hash, tag_value):
    dct = make_exprs()

    expr_cached = dct["expr_cached"]
    (to_replace_hash, to_replace_tag) = (
        (dask.base.tokenize(dct[expr_to_hash]), None)
        if expr_to_hash
        else (None, tag_value)
    )

    unbound = expr_to_unbound(expr_cached, to_replace_hash, to_replace_tag, None)
    assert unbound is not None
    assert tuple(walk_nodes(ops.UnboundTable, unbound))
