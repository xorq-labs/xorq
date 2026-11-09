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


def get_sqlite_stats(dt):
    # rowid stands in for a missing id column; views and WITHOUT ROWID
    # tables have no rowid either
    stamp = "id" if "id" in map(str.lower, dt.schema) else "rowid"
    sql = f"SELECT COUNT(*), MAX({stamp}) FROM '{dt.name}'"
    return dt.source.raw_sql(sql).fetchone()


@frozen
class SQLiteADBC(ADBCBase):
    con = field(validator=instance_of(SQLiteBackend))

    @property
    def uri(self):
        return self.con.uri

    @property
    def conn(self):
        return self.get_conn()

    def get_conn(self, **kwargs):
        return adbc_driver_sqlite.dbapi.connect(self.uri)
