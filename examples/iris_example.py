from pathlib import Path

import xorq as xo
from xorq.caching import ParquetStorage


t = xo.examples.iris.fetch()
con = t.op().source
storage = ParquetStorage(source=con, path=Path.cwd())

expr = t.filter([t.species == "Adelie"]).cache(storage=storage)


if __name__ == "__pytest_main__":
    (op,) = expr.ls.cached_nodes
    path = storage.get_loc(op.to_expr().ls.get_key())
    print(f"{path} exists?: {path.exists()}")
    result = xo.execute(expr)
    print(f"{path} exists?: {path.exists()}")
    pytest_examples_passed = True
