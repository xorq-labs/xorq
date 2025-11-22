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


def get_sqlite_stats(dt):
    (con, name) = (dt.source, dt.name)
    sql = f"SELECT COUNT(*), MAX(id) FROM '{name}'"
    (count, max_id) = con.sql(sql).execute()
    return (count, max_id)


@frozen
class SQLiteADBC:
    con = field(validator=instance_of(SQLiteBackend))

    @property
    def uri(self):
        return self.con.uri

    def get_conn(self, **kwargs):
        return adbc_driver_sqlite.dbapi.connect(self.uri)

    @property
    def conn(self):
        return self.get_conn()

    def adbc_ingest(
        self, table_name, record_batch_reader, mode="create", temporary=False, **kwargs
    ):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.adbc_ingest(
                    table_name,
                    record_batch_reader,
                    mode=mode,
                    temporary=temporary,
                    **kwargs,
                )
            # must commit!
            conn.commit()
