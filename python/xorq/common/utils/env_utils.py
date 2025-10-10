import itertools
import os
import re
import shlex
from pathlib import Path

import toolz
from attr import (
    evolve,
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)


env_templates_dir = Path(__file__).parents[2].joinpath("env_templates")


compiled_env_var_substitution_re = re.compile(r"^(?:\${(.*)}$)|(?:\$(.*))$")
compiled_env_var_setting_re = re.compile(
    "(?:export )?([^=]+)=(.*)",
    flags=re.DOTALL,
)


def parse_env_file(env_file, compiled_re=compiled_env_var_setting_re):
    def gen_shlex_lines(path):
        def make_lexer(path):
            lex = shlex.shlex(Path(path).read_text(), posix=True)
            lex.whitespace_split = True
            return lex

        def get_before_token_after(lexer):
            return (lexer.lineno, lexer.get_token(), lexer.lineno)

        tokens = ()
        for before, token, after in map(
            get_before_token_after,
            itertools.repeat(make_lexer(path)),
        ):
            if token is None:
                if tokens:
                    # single line env file never triggers `before != after`
                    yield " ".join(tokens)
                break
            tokens += (token,)
            if before != after:
                yield " ".join(tokens)
                tokens = ()

    matches = map(
        compiled_re.match,
        gen_shlex_lines(env_file),
    )
    dct = dict(match.groups() for match in filter(None, matches))
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

    clone = evolve

    def maybe_substitute_env_var(self, obj):
        return maybe_substitute_env_var(obj, self)

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


@toolz.curry
def maybe_substitute_env_var(obj, ctx=os.environ):
    if isinstance(obj, str) and (match := compiled_env_var_substitution_re.match(obj)):
        # this will match on "$"/"${}" and then raise on env_value is None
        env_var = next(filter(None, match.groups()), None)
        return ctx[env_var]
    else:
        return obj


def filter_existing_env_vars(dct, ctx):
    f = toolz.excepts(ValueError, maybe_substitute_env_var(ctx=ctx))
    dct = {k: v for k, v in dct.items() if (v is None or f(v) is not None)}
    return dct


def maybe_substitute_env_vars(dct: dict, ctx=os.environ) -> dict:
    return toolz.valmap(
        maybe_substitute_env_var(ctx=ctx),
        dct,
    )
