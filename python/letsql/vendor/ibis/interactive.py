from __future__ import annotations

import letsql.vendor.ibis.examples as ex
from letsql.vendor import ibis
from letsql.vendor.ibis import deferred as _
from letsql.vendor.ibis import selectors as s
from letsql.vendor.ibis import udf


ibis.options.interactive = True

__all__ = ["_", "ex", "ibis", "s", "udf"]
