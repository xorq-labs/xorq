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
    (InitTemplates.cached_fetcher, "832543bcee23a6db0f835ddee64364583e156a94"),
    (InitTemplates.sklearn, "64f8a2b03d582773a1bfa5879c2befba6d23f675"),
    (InitTemplates.penguins, "56680c2fdb5745ae1a9c45394f28cf3707e26670"),
)
