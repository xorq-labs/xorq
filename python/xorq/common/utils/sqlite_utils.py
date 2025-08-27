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


@frozen
class PgADBC:
    con = field(validator=instance_of(SQLiteBackend))

    def get_uri(self, **kwargs):
        params = {**self.params, **kwargs}
        uri = f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
        return uri

    @property
    def uri(self):
        return self.get_uri()

    def get_conn(self, **kwargs):
        return adbc_driver_sqlite.dbapi.connect(self.get_uri(**kwargs))

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
