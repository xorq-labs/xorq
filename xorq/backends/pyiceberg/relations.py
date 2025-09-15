from typing import Optional

import xorq.vendor.ibis.expr.operations as ops


class PyIcebergTable(ops.DatabaseTable):
    """A table that is bound to a PyIceberg backend."""

    snapshot_id: Optional[int] = None
