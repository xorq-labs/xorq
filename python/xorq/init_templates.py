from xorq.common.utils import classproperty


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


default_branch = "main"


class InitTemplates(StrEnum):
    cached_fetcher = "cached-fetcher"
    sklearn = "sklearn"
    penguins = "penguins"

    @classproperty
    def default(self):
        return self.cached_fetcher

    def get_default_branch(template, default_branch=default_branch):
        return dict(templates_branches).get(template, default_branch)


# NOTE: These are commit hashes from when the template update occurred
templates_branches = (
    (InitTemplates.cached_fetcher, "7d3a3419b271d7354afb114b7a84c56d38a75053"),
    (InitTemplates.sklearn, "f706aa1025a74e9153c28d168f70eb9a9a903847"),
    (InitTemplates.penguins, "aa091f4bbf024b613fec72fdadaba1d407ed1171"),
)
