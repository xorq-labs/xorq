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
from xorq.vendor.ibis.backends.sqlite import _quote


def get_sqlite_stats(dt):
    (query, name) = (dt.source.con.execute, dt.name)
    # an unqualified table compiles to an unqualified read, which SQLite
    # resolves to temp before main: stamp the schema that read hits
    (schema,) = (
        (dt.namespace.database,)
        if dt.namespace.database
        else query(
            "SELECT schema FROM pragma_table_list(?)"
            " WHERE schema IN ('temp', 'main') ORDER BY schema = 'main' LIMIT 1",
            (name,),
        ).fetchone()
    )
    # qualified so the stats read hits that schema, not whatever the bare
    # name would resolve to
    table = f"{_quote(schema)}.{_quote(name)}"
    ((kind, without_rowid),) = query(
        "SELECT type, wr FROM pragma_table_list(?) WHERE schema = ? COLLATE NOCASE",
        (name, schema),
    ).fetchall()
    match ("id" in map(str.lower, dt.schema), kind, without_rowid):
        case (True, _, _):
            stamp = "id"
        case (_, "view", _):
            # a view stores no rows of its own: key on its row count and
            # its definition
            return query(
                f"SELECT COUNT(*), (SELECT sql FROM {_quote(schema)}.sqlite_master"
                f" WHERE name = ?) FROM {table}",
                (name,),
            ).fetchone()
        case (_, _, 0):
            stamp = "rowid"
        case _:
            ((stamp,),) = query(
                "SELECT name FROM pragma_table_info(?, ?) WHERE pk = 1", (name, schema)
            ).fetchall()
    return query(f"SELECT COUNT(*), MAX({_quote(stamp)}) FROM {table}").fetchone()


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
