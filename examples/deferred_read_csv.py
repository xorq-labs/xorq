import argparse

import xorq.api as xo
from xorq.api import _


csv_name = "iris"
csv_path = xo.options.pins.get_path(csv_name)


# we can work with a pandas expr without having read it yet
pd_con = xo.pandas.connect()
pd_expr = xo.deferred_read_csv(con=pd_con, path=csv_path, table_name=csv_name).filter(
    _.sepal_length > 6
)

# we can even work with postgres!
pg = xo.postgres.connect_env()
pg_expr = xo.deferred_read_csv(con=pg, path=csv_path, table_name=csv_name).filter(
    _.sepal_length > 6
)

# NOTE: we can't re-run the expr in postgres
# UNLESS we set the create_table mode to "replace"
pg_expr_replace = xo.deferred_read_csv(
    con=pg, path=csv_path, table_name=csv_name, mode="replace"
).filter(_.sepal_length > 6)


def parse_args(override=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true")
    # parse_known_args: so that this script can still be run via pytest inside runpy
    (args, *rest) = parser.parse_known_args(override)
    return args


def test_pd_expr():
    # tables is empty
    assert csv_name not in pd_con.tables
    # and now we can execute
    print(len(xo.execute(pd_expr)))
    assert csv_name in pd_con.tables


def test_pg_expr():
    # tables is empty
    assert csv_name not in pg.tables
    # and now we can execute
    print(len(xo.execute(pg_expr)))
    assert csv_name in pg.tables

    try:
        xo.execute(pg_expr)
        raise RuntimeError("We shouldn't be able to get here!")
    except Exception as e:
        assert f'relation "{csv_name}" already exists' in str(e)


def test_pg_expr_replace():
    assert csv_name in pg.tables
    print(len(xo.execute(pg_expr_replace)))


def cleanup():
    # don't forget to clean up
    pg.drop_table(csv_name)
    assert csv_name not in pg.tables


def main():
    test_pd_expr()
    test_pg_expr()
    test_pg_expr_replace()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    if args.cleanup:
        cleanup()
    else:
        main()
elif __name__ == "__pytest_main__":
    main()
    pytest_examples_passed = True
