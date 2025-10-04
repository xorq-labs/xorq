import itertools
import os
import re
import shlex
from pathlib import Path

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)


env_templates_dir = Path(__file__).parents[2].joinpath("env_templates")


def parse_env_file(env_file):
    def make_lexer(path):
        lex = shlex.shlex(Path(path).read_text(), posix=True)
        lex.whitespace_split = True
        return lex

    def get_before_token_after(lexer):
        return (lexer.lineno, lexer.get_token(), lexer.lineno)

    def gen_lines(gen):
        tokens = ()
        for before, token, after in gen:
            if token is None:
                break
            tokens += (token,)
            if before != after:
                yield " ".join(tokens)
                tokens = ()

    matches = (
        re.match(
            "(export )?([^=]+)=(.*)",
            line,
            flags=re.DOTALL,
        )
        for line in gen_lines(
            map(get_before_token_after, itertools.repeat(make_lexer(env_file)))
        )
    )
    dct = {
        name: value
        for name, value in (match.groups()[1:] for match in filter(None, matches))
    }
    return dct


@frozen
class EnvConfigable:
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def clone(self, **overrides):
        return type(self)(
            **({varname: self[varname] for varname in self.varnames} | overrides)
        )

    def maybe_process_env_var(self, obj):
        from xorq.vendor.ibis.backends.profiles import (
            maybe_process_env_var,
        )

        return maybe_process_env_var(obj, self)

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
    def subclass_from_kwargs(cls, *args, **kwargs):
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
    def subclass_from_env_file(cls, env_file):
        env_file = Path(env_file).resolve()
        return cls.subclass_from_kwargs(
            **(
                parse_env_file(env_file)
                | {
                    "env_file": env_file,
                }
            )
        )
