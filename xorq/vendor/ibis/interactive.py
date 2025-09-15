from __future__ import annotations

import xorq.vendor.ibis.examples as ex
from xorq.vendor import ibis
from xorq.vendor.ibis import deferred as _
from xorq.vendor.ibis import selectors as s
from xorq.vendor.ibis import udf


ibis.options.interactive = True

__all__ = ["_", "ex", "ibis", "s", "udf"]
