from __future__ import annotations

from typing import Any

from xorq.vendor.ibis.backends.bigquery import Backend as IbisBigQueryBackend


class Backend(IbisBigQueryBackend):
    pass


def connect(*args: Any, **kwargs: Any) -> Backend:
    con = Backend()
    return con.connect(*args, **kwargs)
