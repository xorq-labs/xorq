import itertools
import json
import os
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
from envyaml import EnvYAML

import xorq as xo
from xorq.common.utils.inspect_utils import get_arguments


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
        pre_hash = json.loads(self.as_json())
        idx = pre_hash.get("idx")
        pre_hash = toolz.dissoc(pre_hash, "idx")
        dask_hash = dask.base.tokenize(json.dumps(pre_hash))
        return f"{dask_hash}_{idx}"

    def get_con(self, **kwargs):
        kwargs = {
            **kwargs,
            **dict(self.kwargs_tuple),
        }
        connect = getattr(xo.load_backend(self.con_name), "connect")
        con = connect(**kwargs)
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

    def elide_secrets(self):
        # TODO: Needs more secret keys for different backends
        SECRET_KEYS = ("password", "secret")

        new_kwargs = {}
        for k, v in self.kwargs_dict.items():
            found_env = None
            # check to see if the value is an env variable
            # and that it exists before we save it
            if not v:
                continue
            if isinstance(v, str):
                if v.startswith("${") and v.endswith("}"):
                    env_name = v[2:-1]
                    if env_name in os.environ:
                        new_kwargs[k] = f"${{{env_name}}}"
                        continue

            for env_name, env_value in os.environ.items():
                if env_value == v:
                    found_env = env_name
                    break
            if found_env:
                new_kwargs[k] = f"${{{found_env}}}"
            else:
                if k in SECRET_KEYS:
                    new_kwargs[k] = "***elided***"
                else:
                    new_kwargs[k] = v
        return self.clone(**new_kwargs)

    def save(self, profile_dir=None, alias=None, clobber=False):
        path = self.get_path(self.hash_name, profile_dir=profile_dir)
        if not path.exists():
            elided = self.elide_secrets()
            path.write_text(elided.as_yaml())
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
        env = EnvYAML(path)
        con_name = env.get("con_name")
        kwargs_dict = env.get("kwargs_dict")
        idx = env.get("idx")
        sorted_kwargs = tuple(sorted(kwargs_dict.items()))
        return cls(con_name=con_name, kwargs_tuple=sorted_kwargs, idx=idx)

    @classmethod
    def from_con(cls, con):
        if con.name == "xorq_flight":
            return None
        kwargs_name = "config" if con.name == "duckdb" else "kwargs"
        arguments = get_arguments(con.do_connect, *con._con_args, **con._con_kwargs)
        assert not arguments.get("args")
        kwargs = {
            **toolz.dissoc(arguments, "args", kwargs_name),
            **arguments.get(kwargs_name, {}),
        }
        if "port" in kwargs and kwargs["port"] is not None:
            kwargs["port"] = int(kwargs["port"])
        return cls(con_name=con.name, kwargs_tuple=tuple(kwargs.items()))
