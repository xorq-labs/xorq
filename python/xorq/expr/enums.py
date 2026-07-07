from __future__ import annotations

from xorq.common.compat import StrEnum


class Traversal(StrEnum):
    """How a tier-1 transform pass walks the graph -- the single source of
    ``replace_nodes`` vs ``op.replace`` (see ``xorq.expr.transform``).

    DESCEND recurses into opaque sub-exprs (``replace_nodes``); only safe for
    pure structural rewrites. BOUNDARY stops at opaque nodes (``op.replace``);
    required for effectful passes (so a side effect fires once, at this execution
    boundary) and for passes resolved at the boundary (deferred reads). Choosing
    DESCEND for an effectful pass double-materializes -- a mistake this enum
    makes impossible outside these two values.

    Stopping at opaque nodes is not a coverage gap: each opaque interior
    (RemoteTable, CachedNode, Flight*, ExprScalarUDF) re-enters the transform at
    its own execution boundary (caching resolves and re-transforms the cached
    parent; into_backend/flight re-pull via ``to_pyarrow_batches``), so nodes
    nested inside it still get transformed -- exactly once. DESCEND would fire
    them a second time.
    """

    DESCEND = "descend"
    BOUNDARY = "boundary"
