import xorq as xo
from xorq import _
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
)


csv_name = "iris"
csv_path = xo.options.pins.get_path(csv_name)


# we can work with a pandas expr without having read it yet
pd_con = xo.pandas.connect()
expr = deferred_read_csv(con=pd_con, path=csv_path, table_name=csv_name).filter(
    _.sepal_length > 6
)
# tables is empty
assert csv_name not in pd_con.tables
# and now we can execute
print(len(xo.execute(expr)))
assert csv_name in pd_con.tables


# we can even work with postgres!
pg = xo.postgres.connect_env()
expr = deferred_read_csv(con=pg, path=csv_path, table_name=csv_name).filter(
    _.sepal_length > 6
)
# tables is empty
assert csv_name not in pg.tables
# and now we can execute
print(len(xo.execute(expr)))
assert csv_name in pg.tables

# NOTE: we can't re-run the expr in postgres
try:
    xo.execute(expr)
    raise RuntimeError("We shouldn't be able to get here!")
except Exception as e:
    assert f'relation "{csv_name}" already exists' in str(e)
# UNLESS we set the create_table mode to "replace"
expr = deferred_read_csv(
    con=pg, path=csv_path, table_name=csv_name, mode="replace"
).filter(_.sepal_length > 6)
print(len(xo.execute(expr)))

# don't forget to clean up
pg.drop_table(csv_name)
