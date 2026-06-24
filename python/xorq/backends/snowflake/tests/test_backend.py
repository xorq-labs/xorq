import contextlib
import operator

import pytest

import xorq.api as xo
import xorq.expr.relations as rel
from xorq.backends.snowflake.tests.conftest import inside_temp_schema
from xorq.common.utils.graph_utils import (
    find_all_sources,
    walk_nodes,
)


@pytest.mark.snowflake
def test_con_equality_read(temp_catalog, temp_db, parquet_dir):
    # moved out of backends/tests/test_backend.py: read_parquet's CREATE TEMP
    # STAGE is rejected on the read-only SNOWFLAKE_SAMPLE_DATA default database,
    # so each connection is switched into a writable temp schema. A fresh
    # connection per call keeps the Read ops (and their sources) distinct.
    on = "playerID"
    name = "batting"
    with contextlib.ExitStack() as stack:

        def connect():
            con = xo.snowflake.connect_env_keypair(
                database="SNOWFLAKE_SAMPLE_DATA",
                schema="TPCH_SF1",
                create_object_udfs=False,
            )
            stack.enter_context(inside_temp_schema(con, temp_catalog, temp_db))
            return con

        ts = t0, t1 = tuple(
            xo.deferred_read_parquet(
                parquet_dir.joinpath(f"{name}.parquet"),
                con=connect(),
                # must give a name so Read ops could be equal
                table_name=name,
            )
            .filter(operator.eq(xo._.yearID, year))
            .select(on)
            for year in (2014, 2015)
        )
        joined = t0.join(t1.into_backend(t0._find_backend()), on)

        result = joined.execute()
        assert len(result) == 1270

        actual = find_all_sources(joined)
        expected = tuple(t._find_backend() for t in ts)
        assert actual == expected

        (r0, *rest) = walk_nodes(rel.Read, joined)
        assert r0 and rest
