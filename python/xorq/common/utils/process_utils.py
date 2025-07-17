import functools
import operator
from subprocess import (
    PIPE,
    Popen,
)

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    or_,
)


@functools.cache
def in_nix_shell():
    return bool(Popened.check_output("echo $IN_NIX_SHELL").strip())


def assert_not_in_nix_shell():
    if in_nix_shell():
        raise ValueError("in nix shell")


try_decode_ascii = toolz.excepts(
    AttributeError, operator.methodcaller("decode", "ascii")
)


@frozen
class Popened:
    args = field(
        validator=or_(
            deep_iterable(instance_of(str), instance_of(tuple)), instance_of(str)
        )
    )
    kwargs_tuple = field(validator=instance_of(tuple), default=())
    capturing = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        if isinstance(self.args, str):
            assert self.kwargs.get("shell")

    @property
    def kwargs(self):
        return dict(self.kwargs_tuple)

    @property
    @functools.cache
    def popen(self):
        popen = non_blocking_subprocess_run(self.args, **self.kwargs)
        return popen

    @property
    @functools.cache
    def communicated(self):
        return self.popen.communicate()

    @property
    def _stdout(self):
        return self.communicated[0]

    @property
    def _stderr(self):
        return self.communicated[1]

    @property
    def stdout(self):
        return try_decode_ascii(self._stdout)

    @property
    def stderr(self):
        return try_decode_ascii(self._stderr)

    @property
    def returncode(self):
        # ensure we have executed
        self.communicated
        return self.popen.returncode

    @classmethod
    def check_output(cls, args, shell=True, **kwargs):
        self = cls(args, tuple(({"shell": shell} | kwargs).items()))
        assert not self.returncode
        return self.stdout


def non_blocking_subprocess_run(args, stdout=PIPE, stderr=PIPE, **kwargs):
    return Popen(args, stdout=stdout, stderr=stderr, **kwargs)


def subprocess_run(args, do_decode=False, **kwargs):
    popened = non_blocking_subprocess_run(args, **kwargs)
    (stdout, stderr) = popened.communicate()
    if do_decode:
        (stdout, stderr) = (try_decode_ascii(el) for el in popened.communicate())
    return (popened.returncode, stdout, stderr)
