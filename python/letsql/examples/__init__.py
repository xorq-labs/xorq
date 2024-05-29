from functools import cache

import letsql as ls

from palmerpenguins import load_penguins


@cache
def cached_penguins():
    return load_penguins()


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
