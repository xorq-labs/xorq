import functools
from subprocess import (
    PIPE,
    Popen,
)

from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
)


@frozen
class Popened:
    args = field(validator=deep_iterable(instance_of(str), instance_of(tuple)))

    @property
    @functools.cache
    def popen(self):
        popen = non_blocking_subprocess_run(self.args)
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
        return self._stdout.decode("ascii")

    @property
    def stderr(self):
        return self._stderr.decode("ascii")

    @property
    def returncode(self):
        return self.popen.returncode


def non_blocking_subprocess_run(args):
    return Popen(args, stdout=PIPE, stderr=PIPE)


def subprocess_run(args, do_decode=False):
    popened = non_blocking_subprocess_run(args)
    (stdout, stderr) = popened.communicate()
    if do_decode:
        (stdout, stderr) = (el.decode("ascii") for el in popened.communicate())
    return (popened.returncode, stdout, stderr)
