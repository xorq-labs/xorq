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
    (InitTemplates.cached_fetcher, "3db774af377cf5b3f0ba1f5b788c7ef7e768042a"),
    (InitTemplates.sklearn, "ee4a2ac1314e8181a49eb3974e2f65516771e0fc"),
    (InitTemplates.penguins, "f6e8417ca5efaeef7f94efb46ed1ef237dfe7862"),
)
