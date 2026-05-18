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
    (InitTemplates.cached_fetcher, "574dbc1d8eb5636633592bda43e5f2e92ae94f46"),
    (InitTemplates.sklearn, "3449258697314e12e3473044a3fabcc3a0b8cbad"),
    (InitTemplates.penguins, "9424469bf9128a39f9c58f6904fb747b721eaaf1"),
)
