import xorq.api as xo
from xorq.api import _
from xorq.caching import ParquetCache


pg = xo.postgres.connect_examples()
con = xo.connect()
storage = ParquetCache.from_kwargs(source=con)


expr = (
    pg.table("batting")
    .mutate(row_number=xo.row_number().over(group_by=[_.playerID], order_by=[_.yearID]))
    .filter(_.row_number == 1)
    .cache(storage=storage)
)


if __name__ == "__pytest_main__":
    print(f"{expr.ls.get_key()} exists?: {expr.ls.exists()}")
    res = xo.execute(expr)
    print(res)
    print(f"{expr.ls.get_key()} exists?: {expr.ls.exists()}")
    pytest_examples_passed = True
