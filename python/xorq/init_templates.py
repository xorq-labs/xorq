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
    (InitTemplates.cached_fetcher, "13696303173f138853cc77ae09f8e82a1e17afd4"),
    (InitTemplates.sklearn, "a915733da8fe69408e3254aa51e539017e0ac92a"),
    (InitTemplates.penguins, "034e12236e4935a62616253a7b096f7f29b92134"),
)
