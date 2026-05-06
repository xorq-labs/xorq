import xorq.vendor.ibis.expr.operations as ops


class PyIcebergTable(ops.DatabaseTable):
    """A table that is bound to a PyIceberg backend."""

    snapshot_id: int | None = None
