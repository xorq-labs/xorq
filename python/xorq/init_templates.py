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
    (InitTemplates.cached_fetcher, "023ffe1fa1e3f317d2830bfffc9d542f27aaa1fb"),
    (InitTemplates.sklearn, "d94dd60fe2328af70ba5652effb442e022719186"),
    (InitTemplates.penguins, "152a99294d502dacc9fcce8bd33904948f089f39"),
)
