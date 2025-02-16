from functools import cache

import letsql as ls


@cache
def cached_penguins():
    import pandas as pd

    url = "https://raw.githubusercontent.com/mesejo/palmerpenguins/master/palmerpenguins/data/penguins.csv"
    return pd.read_csv(url)


class Example:
    def __init__(self, name, load):
        self.con = None
        self.load = load
        self.name = name

    def fetch(self, table_name=None):
        if self.con is None:
            self.con = ls.connect()
        return self.con.register(self.load(), table_name=table_name or self.name)


penguins = Example("penguins", cached_penguins)


def __getattr__(name):
    from letsql.vendor.ibis import examples as ibex

    return getattr(ibex, name)
