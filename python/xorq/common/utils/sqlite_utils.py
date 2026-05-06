from __future__ import annotations

from typing import Any

import adbc_driver_sqlite.dbapi
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)

from xorq.backends.sqlite import (
    Backend as SQLiteBackend,
)
from xorq.common.utils.adbc_utils import ADBCBase


def get_sqlite_stats(dt: Any) -> tuple[int, int | None]:
    (con, name) = (dt.source, dt.name)
    sql = f"SELECT COUNT(*), MAX(id) FROM '{name}'"
    (count, max_id) = con.sql(sql).execute()
    return (count, max_id)


@frozen
class SQLiteADBC(ADBCBase):
    con = field(validator=instance_of(SQLiteBackend))

    @property
    def uri(self) -> str:
        return self.con.uri

    @property
    def conn(self) -> Any:
        return self.get_conn()

    def get_conn(self, **kwargs: Any) -> Any:
        return adbc_driver_sqlite.dbapi.connect(self.uri)
