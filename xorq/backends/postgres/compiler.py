from __future__ import annotations

from xorq.vendor.ibis.backends.sql.compilers.postgres import (
    PostgresCompiler as IbisPostgresCompiler,
)


_UNIX_EPOCH = "1970-01-01T00:00:00Z"


class PostgresCompiler(IbisPostgresCompiler):
    __slots__ = ()


compiler = PostgresCompiler()
