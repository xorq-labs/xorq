import os
import re
from pathlib import Path

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)


env_templates_dir = Path(__file__).parent.parent.parent.parent.parent.joinpath(
    "env-templates"
)


def parse_env_file(env_file):
    pattern = "(export )?([^=]+)=(.*)"
    lines = Path(env_file).read_text().split("\n")
    matches = filter(None, (re.match(pattern, line) for line in lines))
    dct = {name: value for name, value in (match.groups()[1:] for match in matches)}
    return dct


@frozen
class EnvConfigable:
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key, default=None):
        return getattr(self, key, default)

    @property
    def varnames(self):
        return tuple(self.get_defaults())

    @classmethod
    def get_env_overrides(cls, mapper=str.upper):
        dct = {
            name: value
            for (name, value) in (
                (attr.name, os.environ.get(mapper(attr.name)))
                for attr in cls.__attrs_attrs__
            )
            if value is not None
        }
        return dct

    @classmethod
    def get_defaults(cls):
        dct = {attr.name: attr.default for attr in cls.__attrs_attrs__}
        return dct

    @classmethod
    def from_env(cls, **kwargs):
        return cls(**(kwargs | cls.get_env_overrides()))

    @classmethod
    def from_kwargs(cls, *args, **kwargs):
        fields = {
            name: field(
                default=os.environ.get(name, ""), validator=optional(instance_of(str))
            )
            for name in args
        } | {
            name: field(
                default=value,
                validator=optional(instance_of(str if value is None else type(value))),
            )
            for name, value in kwargs.items()
        }
        return frozen(
            type(
                "EnvConfig",
                (cls,),
                fields,
            )
        )

    @classmethod
    def from_env_file(cls, env_file):
        env_file = Path(env_file).resolve()
        return cls.from_kwargs(
            **(
                parse_env_file(env_file)
                | {
                    "env_file": env_file,
                }
            )
        )
