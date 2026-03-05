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
    (InitTemplates.cached_fetcher, "873507c2bfdc2a0bc5109ec809969b378cfc8ff4"),
    (InitTemplates.sklearn, "17f4e5e37c7879594960e8864dcb2e1ac829692f"),
    (InitTemplates.penguins, "29cd09605b60e938fd05b139fdead64504bb8279"),
)
