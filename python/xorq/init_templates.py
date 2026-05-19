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
    (InitTemplates.cached_fetcher, "2c6c69f62e29ffd762d95b440a013de3a11c4750"),
    (InitTemplates.sklearn, "f63e5658d250f5b42b24ddbd1bcdb2f0d2e0d830"),
    (InitTemplates.penguins, "6ef79e76c27ef1f2cfd396c7424b976d06df4770"),
)
