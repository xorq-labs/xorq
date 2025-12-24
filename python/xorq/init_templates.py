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


templates_branches = (
    (InitTemplates.cached_fetcher, "832543bcee23a6db0f835ddee64364583e156a94"),
    (InitTemplates.sklearn, "29366e03ec07b1e8e9ee9ae8a5f99fbf40e0e07c"),
    (InitTemplates.penguins, "089e1694792437ba56f51e69c73ede724c649286"),
)
