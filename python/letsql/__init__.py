from __future__ import annotations

from letsql.main import Backend

ba = Backend()
ba.do_connect()


def con():
    ba.do_connect()
    return ba


read_parquet = ba.read_parquet
read_csv = ba.read_csv
register = ba.register
memtable = ba.memtable
