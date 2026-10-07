from __future__ import annotations

from xorq.common.compat import StrEnum


class Traversal(StrEnum):
    """How a tier-1 transform pass walks the graph -- the single source of
    ``replace_nodes`` vs ``op.replace`` (see ``xorq.expr.transform``).

    DESCEND recurses into opaque sub-exprs (``replace_nodes``); only safe for
    pure structural rewrites. BOUNDARY stops at opaque nodes (``op.replace``);
    required for effectful passes (so a side effect fires once, at this execution
    boundary) and for passes resolved at the boundary (deferred reads). Choosing
    DESCEND for an effectful pass double-materializes; the record names the
    choice, so a review can see it.

    Pure is necessary for DESCEND, not sufficient. A pure rewrite that changes a
    payload's identity (``remove_tags`` drops ``HashingTag``) must also be
    BOUNDARY when any BOUNDARY pass keys that payload (``cache`` keys
    ``CachedNode.parent``), wherever the rewrite sits in the table: descending
    hands the keying pass a payload other than the one written, and the key
    diverges from ``ls.get_key()``. The owning pass keys the payload as written;
    the payload's own nested transform then strips it.
    ``bind_params`` is the deliberate exception; the ``_PASSES`` header in
    ``xorq.expr.api`` records why.

    Stopping at opaque nodes is not a coverage gap: each opaque interior
    (RemoteTable, CachedNode, Flight*, ExprScalarUDF) re-enters the transform at
    its own execution boundary (caching resolves and re-transforms the cached
    parent; into_backend/flight re-pull via ``to_pyarrow_batches``), so nodes
    nested inside it still get transformed -- exactly once. DESCEND would fire
    them a second time.
    """

    DESCEND = "descend"
    BOUNDARY = "boundary"
