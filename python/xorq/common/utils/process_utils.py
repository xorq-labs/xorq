import functools
import operator
import os
import re
import subprocess
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
    return bool(os.environ.get("IN_NIX_SHELL"))


def assert_not_in_nix_shell():
    if in_nix_shell():
        raise ValueError("in nix shell")


try_decode_utf8 = toolz.excepts(
    AttributeError, operator.methodcaller("decode", "utf-8")
)


@frozen(eq=False)
class Popened:
    args = field(
        validator=or_(
            deep_iterable(instance_of(str), instance_of(tuple)), instance_of(str)
        ),
    )
    kwargs_tuple = field(validator=instance_of(tuple), default=())
    deferred = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        if isinstance(self.args, str):
            if not self.kwargs.get("shell"):
                raise ValueError("string args require shell=True")
        if not self.deferred:
            self.popen

    @property
    def kwargs(self):
        return dict(self.kwargs_tuple)

    @functools.cached_property
    def popen(self):
        popen = non_blocking_subprocess_run(self.args, **self.kwargs)
        return popen

    @functools.cached_property
    def stdout_peeker(self):
        from xorq.common.utils.io_utils import Peeker  # noqa: PLC0415

        return Peeker(self.popen.stdout) if self.popen.stdout else None

    def peek_stdout(self, size):
        return self.stdout_peeker.peek(size)

    @functools.cached_property
    def communicated(self):
        # Read already-peeked bytes from the BytesIO buffer (non-blocking)
        peeked = self.stdout_peeker.buf.read() if self.stdout_peeker else b""
        # Always drain both pipes concurrently to avoid deadlock
        (_stdout, _stderr) = self.popen.communicate()
        if peeked:
            _stdout = peeked + (_stdout or b"")
        return (_stdout, _stderr)

    def wait(self):
        self.communicated
        self.popen.wait()

    @property
    def _stdout(self):
        return self.communicated[0]

    @property
    def _stderr(self):
        return self.communicated[1]

    @property
    def stdout(self):
        return try_decode_utf8(self._stdout)

    @property
    def stderr(self):
        return try_decode_utf8(self._stderr)

    @property
    def returncode(self):
        # ensure we have executed
        self.communicated
        return self.popen.returncode

    @classmethod
    def check_output(cls, args, shell=True, **kwargs):
        self = cls(args, tuple(({"shell": shell} | kwargs).items()))
        if self.returncode:
            raise subprocess.CalledProcessError(
                self.returncode, args, self._stdout, self._stderr
            )
        return self.stdout


non_blocking_subprocess_run = functools.partial(Popen, stdout=PIPE, stderr=PIPE)


def subprocess_run(args, text=False, **kwargs):
    result = subprocess.run(
        args,
        capture_output=True,
        encoding="utf-8" if text else None,
        **kwargs,
    )
    return (result.returncode, result.stdout, result.stderr)


# https://stackoverflow.com/a/14693789
ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
remove_ansi_escape = functools.partial(ansi_escape.sub, "")
