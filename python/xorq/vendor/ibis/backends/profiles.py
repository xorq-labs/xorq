import itertools
import json
import os
import re
from pathlib import Path

import dask
import toolz
import yaml
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)

import xorq as xo
from xorq.common.utils.inspect_utils import get_arguments


compiled_env_var_re = re.compile("^(?:\${(.*)}$)|(?:\$(.*))$")


@frozen
class Profiles:
    profile_dir = field(validator=optional(instance_of(Path)), default=None)

    def __attrs_post_init__(self):
        if self.profile_dir is None:
            object.__setattr__(self, "profile_dir", xo.options.profiles.profile_dir)
        if not self.profile_dir.exists():
            self.profile_dir.mkdir(exist_ok=True, parents=True)

    def get(self, name):
        return Profile.load(name, profile_dir=self.profile_dir)

    def __getattr__(self, stem):
        try:
            return self.get(
                next(el.name for el in self.profile_dir.iterdir() if el.stem == stem)
            )
        except Exception:
            return object.__getattribute__(self, stem)

    def __getitem__(self, stem):
        return self.get(
            next(el.name for el in self.profile_dir.iterdir() if el.stem == stem)
        )

    def __dir__(self):
        return tuple(el for el in self.list() if el.isidentifier())

    def list(self):
        return tuple(el.stem for el in self.profile_dir.iterdir())

    def _ipython_key_completions_(self):
        return self.list()


@frozen
class Profile:
    con_name = field(validator=instance_of(str))
    kwargs_tuple = field(validator=instance_of(tuple))
    idx = field(validator=instance_of(int), factory=itertools.count().__next__)

    @con_name.validator
    def validate_con_name(self, attr, value):
        assert next(
            (ep for ep in xo._load_entry_points() if ep.name == value),
            None,
        )

    def __attrs_post_init__(self):
        # Sort kwargs_tuple after initialization
        object.__setattr__(self, "kwargs_tuple", tuple(sorted(self.kwargs_tuple)))

    @property
    def kwargs_dict(self):
        return {k: v for k, v in self.kwargs_tuple}

    @property
    def hash_name(self):
        dask_hash = dask.base.tokenize(
            json.dumps(
                toolz.dissoc(self.as_dict(), "idx"),
            )
        )
        return f"{dask_hash}_{self.idx}"

    def get_con(self, **kwargs):
        """Create a connection using this profile's parameters."""
        _kwargs = dict(self.kwargs_tuple) | kwargs
        connect = getattr(xo.load_backend(self.con_name), "connect")
        con = connect(**_kwargs)
        return con

    def clone(self, idx=None, **kwargs):
        idx = idx if idx is not None else self.idx
        kwargs_tuple = tuple(
            {
                **dict(self.kwargs_tuple),
                **kwargs,
            }.items()
        )
        return type(self)(
            con_name=self.con_name,
            kwargs_tuple=kwargs_tuple,
            idx=idx,
        )

    def as_dict(self):
        return {
            name: getattr(self, name)
            for name in (attr.name for attr in self.__attrs_attrs__)
        }

    def as_json(self):
        return json.dumps(self.as_dict())

    def as_yaml(self):
        return yaml.safe_dump(
            dict(con_name=self.con_name, kwargs_dict=self.kwargs_dict, idx=self.idx)
        )

    def save(self, profile_dir=None, alias=None, clobber=False):
        path = self.get_path(self.hash_name, profile_dir=profile_dir)
        if not path.exists():
            path.write_text(self.as_yaml())
        if alias:
            alias_path = self.get_path(alias, profile_dir=profile_dir)
            if alias_path.exists():
                if not clobber:
                    raise ValueError
                alias_path.unlink()
            alias_path.symlink_to(path)
            return alias_path
        return path

    def almost_equals(self, other):
        return self.clone(idx=-1) == other.clone(idx=-1)

    @classmethod
    def get_path(cls, name, profile_dir=None):
        profile_dir = profile_dir or xo.options.profiles.profile_dir
        profile_dir.mkdir(exist_ok=True, parents=True)
        path = profile_dir.joinpath(name).with_suffix(".yaml")
        return path

    @classmethod
    def load(cls, name, profile_dir=None):
        path = cls.get_path(name, profile_dir=profile_dir)
        env = yaml.safe_load(path.read_text())
        con_name = env.get("con_name")
        kwargs_dict = env.get("kwargs_dict")
        idx = env.get("idx")
        sorted_kwargs = tuple(sorted(kwargs_dict.items()))
        return cls(con_name=con_name, kwargs_tuple=sorted_kwargs, idx=idx)

    @classmethod
    def from_con(cls, con, *args, **kwargs):
        """Create a Profile from a connection, preserving env var references if possible."""

        def get_combined_arguments():
            # these are the env-mapped values
            arguments0 = get_arguments(
                con.do_connect, *con._con_args, **con._con_kwargs
            )
            # these are the "raw" values (if passed)
            arguments1 = toolz.valfilter(
                bool, get_arguments(con.do_connect, *args, **kwargs)
            )
            assert not arguments0.get("args")
            assert not arguments1.get("args")
            arguments = toolz.dissoc(arguments0 | arguments1, "args")
            return arguments

        if con.name == "xorq_flight":
            return None

        kwargs_name = "config" if con.name == "duckdb" else "kwargs"
        arguments = get_combined_arguments()
        kwargs = toolz.dissoc(arguments, kwargs_name) | arguments.get(kwargs_name, {})

        # Fix port type if needed
        if (
            "port" in kwargs
            and kwargs["port"] is not None
            and isinstance(kwargs["port"], str)
        ):
            kwargs["port"] = int(kwargs["port"])

        return cls(con_name=con.name, kwargs_tuple=tuple(sorted(kwargs.items())))


def maybe_process_env_var(obj):
    if isinstance(obj, str) and (match := compiled_env_var_re.match(obj)):
        # this will match on "$"/"${}" and then raise on env_value is None
        env_var = next(filter(None, match.groups()), None)
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise ValueError(f"env var {env_var} not found")
        else:
            return env_value
    else:
        return obj


# TODO: find a better home for this
def parse_env_vars(kwargs_dict: dict) -> dict:
    processed_kwargs = {}
    missing_vars = []

    env_matches = {
        k: next(filter(None, match.groups()))
        for k, match in (
            (k, compiled_env_var_re.match(v))
            for k, v in kwargs_dict.items()
            if isinstance(v, str)
        )
        if match
    }
    missing_vars = tuple(
        env_var for env_var in env_matches.values() if env_var not in os.environ
    )
    # Strict mode: raise error if any env vars are missing
    if missing_vars:
        missing_list = ", ".join(f"'{var}'" for var in missing_vars)
        raise ValueError(f"Environment variable(s) {missing_list} not set")

    env_kwargs = {k: os.environ[env_var] for k, env_var in env_matches.items()}
    processed_kwargs = kwargs_dict | env_kwargs
    return processed_kwargs
